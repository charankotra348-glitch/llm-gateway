"""The HTTP surface: OpenAI-compatible endpoints over the route table."""

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import async_sessionmaker

from gateway.cache import ResponseCache, plan
from gateway.config import Settings, get_settings
from gateway.db import make_engine, make_sessions
from gateway.errors import BudgetExceeded, GatewayError, InvalidApiKey, InvalidRequest, RateLimitExceeded, error_body
from gateway.keys import find_key
from gateway.ledger import CACHE_PROVIDER, month_usage
from gateway.limits import RateLimiter
from gateway.models import ApiKey, RequestLog
from gateway.pricing import PriceTable
from gateway.providers.base import Provider
from gateway.resilience import Attempt, Resilience, ResiliencePolicy
from gateway.routes import RouteTable
from gateway.schemas import ChatCompletionRequest, Completion, normalize

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
                 limiter: RateLimiter | None = None, cache: ResponseCache | None = None, engine=None):
        self.providers = providers
        self.routes = routes
        self.sessions = sessions
        self.prices = prices or PriceTable({})
        # `is None`, not `or`: an empty ResponseCache has len 0, so it is falsy
        self.resilience = resilience if resilience is not None else Resilience()
        self.limiter = limiter if limiter is not None else RateLimiter(default_rpm=0)
        self.cache = cache if cache is not None else ResponseCache(ttl_s=0)
        self._engine = engine

    def unpriced_targets(self) -> list[str]:
        return [f"{r}: {t.provider}/{t.model}" for r in self.routes.names() for t in self.routes.get(r).targets
                if not self.prices.knows(t.provider, t.model)]

    async def aclose(self) -> None:
        for p in self.providers.values():
            await p.aclose()
        if self._engine is not None:
            await self._engine.dispose()


def build_gateway(settings: Settings) -> Gateway:
    providers: dict[str, Provider] = {}
    if settings.enable_fake:
        from gateway.providers.fake import FakeProvider
        providers["fake"] = FakeProvider()
        providers["flaky"] = FakeProvider(failure_rate=settings.flaky_failure_rate)
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
    policy = ResiliencePolicy(
        attempt_timeout_s=settings.request_timeout_s, total_timeout_s=settings.total_timeout_s,
        attempts_per_target=settings.attempts_per_target, backoff_s=settings.retry_backoff_s,
        breaker_threshold=settings.breaker_threshold, breaker_cooldown_s=settings.breaker_cooldown_s)
    return Gateway(providers, routes, make_sessions(engine), prices=PriceTable.load(settings.prices_file),
                   resilience=Resilience(policy), limiter=RateLimiter(settings.default_rpm),
                   cache=ResponseCache(settings.cache_ttl_s, settings.cache_max_entries), engine=engine)


async def authenticate(request: Request) -> ApiKey:
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise InvalidApiKey("Missing API key. Send it as 'Authorization: Bearer gw_...'.")
    gw: Gateway = request.app.state.gateway
    async with gw.sessions() as session:
        key = await find_key(session, token.strip())
    if key is None:
        raise InvalidApiKey("Incorrect or revoked API key.")
    return key


def create_app(gateway: Gateway | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.gateway = gateway or build_gateway(get_settings())
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
        if body.stream:
            raise InvalidRequest("stream=true is not supported yet")
        chat = normalize(body)

        cache = plan(key.id, route.name, chat, request.headers.get("cache-control")) if gw.cache.enabled \
            else None
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

        try:
            result = await gw.resilience.complete(route, chat, gw.providers)
        except GatewayError as exc:
            attempts = getattr(exc, "attempts", [])
            await _record(gw, request_id, key, route.name, attempts, exc.code, None,
                          latency_ms=(time.perf_counter() - started) * 1000)
            raise
        completion, target = result.completion, result.target
        await _record(gw, request_id, key, route.name, result.attempts, "ok", completion,
                      latency_ms=(time.perf_counter() - started) * 1000)
        if cache and cache.write:
            gw.cache.put(cache.key, (completion, target))
        headers |= {"x-gateway-provider": target.provider, "x-gateway-attempts": str(result.calls)}
        if cache is not None:
            headers["x-gateway-cache"] = "miss" if cache.read else cache.reason
        return _response(request_id, completion, headers)

    return app


def _response(request_id: str, completion: Completion, headers: dict[str, str]) -> JSONResponse:
    usage = None
    if completion.input_tokens is not None and completion.output_tokens is not None:
        usage = {"prompt_tokens": completion.input_tokens,
                 "completion_tokens": completion.output_tokens,
                 "total_tokens": completion.input_tokens + completion.output_tokens}
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
                  completion: Completion | None, *, latency_ms: float) -> None:
    """Log line + ledger row. A ledger failure is logged, never turned into a failed response:
    the provider has already been paid for the answer.

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
            "cost_usd": None if cost is None else cost / 1e6, "attempts": [a.__dict__ for a in attempts]}
    try:
        async with gw.sessions() as session:
            session.add(RequestLog(id=request_id, key_id=key.id, route=route, provider=provider, model=model,
                                   outcome=outcome, input_tokens=tokens_in, output_tokens=tokens_out,
                                   cost_micros=cost, latency_ms=latency_ms, attempts=0 if cached else len(calls)))
            await session.commit()
    except Exception as exc:  # noqa: BLE001
        line["ledger_error"] = type(exc).__name__
        log.error(json.dumps(line))
        return
    log.info(json.dumps(line))


app = create_app()
