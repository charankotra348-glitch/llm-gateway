"""The HTTP surface: OpenAI-compatible endpoints over the route table."""

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import anyio
from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.ext.asyncio import async_sessionmaker

from gateway.cache import ResponseCache, plan
from gateway.config import Settings, get_settings
from gateway.db import make_engine, make_sessions
from gateway.errors import BudgetExceeded, GatewayError, InvalidApiKey, InvalidRequest, RateLimitExceeded, error_body
from gateway.keys import KeyCache
from gateway.ledger import CACHE_PROVIDER, LedgerWriter, month_usage
from gateway.limits import RateLimiter
from gateway.models import ApiKey, RequestLog
from gateway.pricing import PriceTable
from gateway.providers.base import Provider
from gateway.redaction import StreamRestorer, Vault, redact
from gateway.resilience import Attempt, Resilience, ResiliencePolicy
from gateway.routes import Route, RouteTable
from gateway.schemas import ChatCompletionRequest, ChatRequest, Chunk, Completion, normalize

log = logging.getLogger("gateway")
if not log.handlers:            # one JSON line per request on stdout, whatever uvicorn's log config is
    _stdout = logging.StreamHandler()
    _stdout.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(_stdout)
    log.setLevel(logging.INFO)
    log.propagate = False


class Gateway:
    """Everything a request needs: providers, routes, prices, the database, and the in-process state
    (circuit breakers, rate-limit buckets, the response cache)."""

    def __init__(self, providers: dict[str, Provider], routes: RouteTable, sessions: async_sessionmaker,
                 *, prices: PriceTable | None = None, resilience: Resilience | None = None,
                 limiter: RateLimiter | None = None, cache: ResponseCache | None = None, redact_pii: bool = True,
                 keys: KeyCache | None = None, ledger: LedgerWriter | None = None, engine=None):
        self.providers = providers
        self.routes = routes
        self.sessions = sessions
        self.prices = prices or PriceTable({})
        # `is None`, not `or`: an empty ResponseCache has len 0, so it is falsy
        self.resilience = resilience if resilience is not None else Resilience()
        self.limiter = limiter if limiter is not None else RateLimiter(default_rpm=0)
        self.cache = cache if cache is not None else ResponseCache(ttl_s=0)
        self.redact_pii = redact_pii
        self.keys = keys if keys is not None else KeyCache(ttl_s=0)
        self.ledger = ledger if ledger is not None else LedgerWriter(sessions)
        self._engine = engine

    def unpriced_targets(self) -> list[str]:
        return [f"{r}: {t.provider}/{t.model}" for r in self.routes.names() for t in self.routes.get(r).targets
                if not self.prices.knows(t.provider, t.model)]

    async def start(self) -> None:
        await self.ledger.start()

    async def aclose(self) -> None:
        await self.ledger.stop()                # write out queued rows before anything closes
        for p in self.providers.values():
            await p.aclose()
        if self._engine is not None:
            await self._engine.dispose()


def build_gateway(settings: Settings) -> Gateway:
    providers: dict[str, Provider] = {}
    if settings.enable_fake:
        from gateway.providers.fake import FakeProvider
        providers["fake"] = FakeProvider(chunk_delay_s=settings.fake_chunk_delay_s)
        providers["flaky"] = FakeProvider(failure_rate=settings.flaky_failure_rate,
                                          chunk_delay_s=settings.fake_chunk_delay_s)
    if settings.gemini_api_key:
        from gateway.providers.gemini import GeminiProvider
        providers["gemini"] = GeminiProvider(settings.gemini_api_key)
    if settings.bedrock_region:
        from gateway.providers.bedrock import BedrockProvider
        providers["bedrock"] = BedrockProvider(settings.bedrock_region)
    if settings.ollama_url:
        from gateway.providers.ollama import OllamaProvider
        providers["ollama"] = OllamaProvider(settings.ollama_url)
    routes = RouteTable.load(settings.routes_file, known_providers=set(providers))
    engine = make_engine(settings.database_url)
    sessions = make_sessions(engine)
    policy = ResiliencePolicy(
        attempt_timeout_s=settings.request_timeout_s, total_timeout_s=settings.total_timeout_s,
        attempts_per_target=settings.attempts_per_target, backoff_s=settings.retry_backoff_s,
        breaker_threshold=settings.breaker_threshold, breaker_cooldown_s=settings.breaker_cooldown_s)
    return Gateway(providers, routes, sessions, prices=PriceTable.load(settings.prices_file),
                   resilience=Resilience(policy), limiter=RateLimiter(settings.default_rpm),
                   cache=ResponseCache(settings.cache_ttl_s, settings.cache_max_entries),
                   redact_pii=settings.redact_pii, keys=KeyCache(settings.key_cache_ttl_s),
                   ledger=LedgerWriter(sessions, batched=settings.ledger_batched), engine=engine)


async def authenticate(request: Request) -> ApiKey:
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise InvalidApiKey("Missing API key. Send it as 'Authorization: Bearer gw_...'.")
    gw: Gateway = request.app.state.gateway
    key = await gw.keys.find(gw.sessions, token.strip())
    if key is None:
        raise InvalidApiKey("Incorrect or revoked API key.")
    return key


def create_app(gateway: Gateway | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.gateway = gateway or build_gateway(get_settings())
        await app.state.gateway.start()
        for target in app.state.gateway.unpriced_targets():
            log.warning(json.dumps({"event": "unpriced_target", "target": target,
                                    "note": "requests here are not counted against budgets"}))
        yield
        await app.state.gateway.aclose()

    app = FastAPI(title="LLM Gateway", lifespan=lifespan)

    @app.exception_handler(GatewayError)
    async def gateway_error(_: Request, exc: GatewayError):
        # The OpenAI SDKs honour x-should-retry; without it they retry every 429, budgets included.
        # retry-after (rate limits) tells them how long to wait first.
        headers = {"x-should-retry": "true" if exc.retryable else "false", **getattr(exc, "headers", {})}
        return JSONResponse(error_body(exc), status_code=exc.status, headers=headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError):
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"] if p != "body")
        return JSONResponse(error_body(InvalidRequest(f"{where}: {first['msg']}" if where else first["msg"])),
                            status_code=400, headers={"x-should-retry": "false"})

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    @app.get("/v1/models")
    async def models(request: Request, key: ApiKey = Depends(authenticate)):
        gw: Gateway = request.app.state.gateway
        return {"object": "list",
                "data": [{"id": n, "object": "model", "created": 0, "owned_by": "llm-gateway"}
                         for n in gw.routes.names()]}

    @app.get("/v1/status")
    async def status(request: Request, key: ApiKey = Depends(authenticate)):
        """Circuit breaker state of every target that has been called. Not part of the OpenAI API."""
        gw: Gateway = request.app.state.gateway
        return {"targets": gw.resilience.states()}

    @app.get("/v1/usage")
    async def usage(request: Request, key: ApiKey = Depends(authenticate)):
        """This key's spend in the current calendar month (UTC). Not part of the OpenAI API."""
        gw: Gateway = request.app.state.gateway
        async with gw.sessions() as session:
            u = await month_usage(session, key.id)
        budget = key.monthly_budget_micros
        return {"key": key.prefix, "name": key.name,
                "estimated_spend_usd": u.spend_micros / 1e6,
                "monthly_budget_usd": None if budget is None else budget / 1e6,
                "requests": u.requests, "unpriced_requests": u.unpriced_requests, "cache_hits": u.cache_hits}

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest, request: Request, key: ApiKey = Depends(authenticate)):
        gw: Gateway = request.app.state.gateway
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        started = time.perf_counter()

        limit = gw.limiter.check(key.id, key.rate_limit_rpm)
        if limit is not None and not limit.allowed:
            raise RateLimitExceeded(f"Rate limit of {limit.limit} requests per minute reached for this key. "
                                    f"Try again in {limit.retry_after_s:.1f}s.", limit.retry_after_s)
        headers = {"x-request-id": request_id}
        if limit is not None:
            headers |= {"x-ratelimit-limit-requests": str(limit.limit),
                        "x-ratelimit-remaining-requests": str(limit.remaining)}

        route = gw.routes.get(body.model)
        headers["x-gateway-route"] = route.name
        chat = normalize(body)

        # Streams skip the cache: an answer is cached once it is complete, and a stream is consumed as it goes.
        cache = plan(key.id, route.name, chat, request.headers.get("cache-control")) \
            if gw.cache.enabled and not body.stream else None
        if cache and cache.read and (hit := gw.cache.get(cache.key)) is not None:
            completion, target = hit
            await _record(gw, request_id, key, route.name, [Attempt(CACHE_PROVIDER, target.model, "ok")], "ok",
                          completion, latency_ms=(time.perf_counter() - started) * 1000)
            return _response(request_id, completion, headers | {
                "x-gateway-provider": CACHE_PROVIDER, "x-gateway-attempts": "0", "x-gateway-cache": "hit"})

        # Cache hits are free, so the budget is only checked for requests that will reach a provider.
        if key.monthly_budget_micros is not None:
            # A soft limit: requests already in flight can take the key slightly past it.
            async with gw.sessions() as session:
                spent = (await month_usage(session, key.id)).spend_micros
            if spent >= key.monthly_budget_micros:
                raise BudgetExceeded(f"This key's monthly budget of ${key.monthly_budget_micros / 1e6:.2f} "
                                     f"is used up (estimated spend ${spent / 1e6:.4f}). It resets on the 1st (UTC).")

        # The cache works on the original request; only what goes to a provider is redacted.
        outbound, vault = redact(chat) if gw.redact_pii else (chat, Vault())
        if gw.redact_pii:
            headers["x-gateway-redactions"] = str(vault.total)

        if body.stream:
            if gw.cache.enabled:
                headers["x-gateway-cache"] = "bypass"
            include_usage = bool(body.stream_options and body.stream_options.include_usage)
            return await _stream(gw, request_id, key, route, outbound, vault, headers, started, include_usage)

        try:
            result = await gw.resilience.complete(route, outbound, gw.providers)
        except GatewayError as exc:
            attempts = getattr(exc, "attempts", [])
            await _record(gw, request_id, key, route.name, attempts, exc.code, None,
                          latency_ms=(time.perf_counter() - started) * 1000, vault=vault)
            raise
        completion, target = result.completion, result.target
        completion = completion.model_copy(update={"text": vault.restore(completion.text)})
        await _record(gw, request_id, key, route.name, result.attempts, "ok", completion,
                      latency_ms=(time.perf_counter() - started) * 1000, vault=vault)
        if cache and cache.write:
            gw.cache.put(cache.key, (completion, target))
        headers |= {"x-gateway-provider": target.provider, "x-gateway-attempts": str(result.calls)}
        if cache is not None:
            headers["x-gateway-cache"] = "miss" if cache.read else cache.reason
        return _response(request_id, completion, headers)

    return app


async def _stream(gw: Gateway, request_id: str, key: ApiKey, route: Route, chat: ChatRequest, vault: Vault,
                  headers: dict[str, str], started: float, include_usage: bool) -> StreamingResponse:
    """Server-sent events in OpenAI's chunk format.

    Failures before the first chunk are ordinary HTTP errors (and are failed over like any request).
    After it, the status line has been sent, so a failure becomes an error event that the OpenAI SDK
    raises as an exception, and the stream ends without [DONE]."""
    try:
        result = await gw.resilience.stream(route, chat, gw.providers)
    except GatewayError as exc:
        await _record(gw, request_id, key, route.name, getattr(exc, "attempts", []), exc.code, None,
                      latency_ms=(time.perf_counter() - started) * 1000, vault=vault)
        raise
    target, created = result.target, int(time.time())
    restorer = StreamRestorer(vault)

    def event(payload: dict) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    def chunk(delta: dict, finish_reason: str | None = None) -> str:
        return event({"id": request_id, "object": "chat.completion.chunk", "created": created, "model": target.model,
                      "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]})

    async def events():
        parts: list[str] = []
        final: Chunk | None = None
        outcome = "client_disconnected"     # unless the stream gets to the end or fails
        try:
            yield chunk({"role": "assistant", "content": ""})
            async for piece in result.chunks():
                if piece.final:
                    final = piece
                    break
                if text := restorer.feed(piece.text):
                    parts.append(text)
                    yield chunk({"content": text})
            if text := restorer.flush():
                parts.append(text)
                yield chunk({"content": text})
            yield chunk({}, final.finish_reason)
            if include_usage:
                yield event({"id": request_id, "object": "chat.completion.chunk", "created": created,
                             "model": target.model, "choices": [], "usage": _usage(final)})
            yield "data: [DONE]\n\n"
            outcome = "ok"
        except GatewayError as exc:
            outcome = exc.code
            yield event(error_body(exc))
        finally:
            completion = None
            if final is not None:
                completion = Completion(text="".join(parts), model=final.model or target.model,
                                        finish_reason=final.finish_reason, input_tokens=final.input_tokens,
                                        output_tokens=final.output_tokens)
            with anyio.CancelScope(shield=True):     # still record if the client went away mid-stream
                await _record(gw, request_id, key, route.name, result.attempts, outcome, completion,
                              latency_ms=(time.perf_counter() - started) * 1000, vault=vault)

    return StreamingResponse(events(), media_type="text/event-stream", headers=headers | {
        "x-gateway-provider": target.provider, "x-gateway-attempts": str(result.calls), "cache-control": "no-cache"})


def _usage(completion: Completion | Chunk) -> dict | None:
    if completion.input_tokens is None or completion.output_tokens is None:
        return None
    return {"prompt_tokens": completion.input_tokens, "completion_tokens": completion.output_tokens,
            "total_tokens": completion.input_tokens + completion.output_tokens}


def _response(request_id: str, completion: Completion, headers: dict[str, str]) -> JSONResponse:
    usage = _usage(completion)
    return JSONResponse(
        {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": completion.model,
            "choices": [{"index": 0, "finish_reason": completion.finish_reason,
                         "message": {"role": "assistant", "content": completion.text}}],
            "usage": usage,
        },
        headers=headers,
    )


async def _record(gw: Gateway, request_id: str, key: ApiKey, route: str, attempts: list[Attempt], outcome: str,
                  completion: Completion | None, *, latency_ms: float, vault: Vault | None = None) -> None:
    """Log line + ledger row. A ledger failure is logged, never turned into a failed response:
    the provider has already been paid for the answer. With the batched writer, the row is queued
    and written in the background.

    Requests that never reached a provider (every circuit open) are not recorded."""
    calls = [a for a in attempts if a.outcome != "circuit_open"]
    if not calls:
        log.info(json.dumps({"event": "chat_completion", "id": request_id, "key": key.prefix, "route": route,
                             "outcome": outcome, "attempts": [a.__dict__ for a in attempts]}))
        return
    provider, model = calls[-1].provider, calls[-1].model     # who answered, or the last one tried
    cached = provider == CACHE_PROVIDER
    tokens_in = completion.input_tokens if completion else None
    tokens_out = completion.output_tokens if completion else None
    if cached or not completion:
        cost = 0                    # a cache hit costs nothing; failed calls are not billed
    else:
        cost = gw.prices.cost_micros(provider, model, tokens_in, tokens_out)
    line = {"event": "chat_completion", "id": request_id, "key": key.prefix, "route": route,
            "provider": provider, "model": model, "outcome": outcome, "latency_ms": round(latency_ms, 1),
            "input_tokens": tokens_in, "output_tokens": tokens_out,
            "cost_usd": None if cost is None else cost / 1e6, "attempts": [a.__dict__ for a in attempts],
            "redacted": dict(vault.counts) if vault else {}}     # kinds and counts only, never the values
    try:
        await gw.ledger.write(RequestLog(
            id=request_id, key_id=key.id, route=route, provider=provider, model=model, outcome=outcome,
            input_tokens=tokens_in, output_tokens=tokens_out, cost_micros=cost, latency_ms=latency_ms,
            attempts=0 if cached else len(calls), redactions=vault.total if vault else 0,
            created_at=datetime.now(UTC)))          # the request's time, not the time a batch is written
    except Exception as exc:  # noqa: BLE001
        line["ledger_error"] = type(exc).__name__
        log.error(json.dumps(line))
        return
    log.info(json.dumps(line))


app = create_app()
