"""The HTTP surface: OpenAI-compatible endpoints over the route table."""

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from gateway.config import Settings, get_settings
from gateway.errors import GatewayError, InvalidRequest, error_body
from gateway.providers.base import Provider
from gateway.routes import RouteTable
from gateway.schemas import ChatCompletionRequest, normalize

log = logging.getLogger("gateway")
if not log.handlers:            # one JSON line per request on stdout, whatever uvicorn's log config is
    _stdout = logging.StreamHandler()
    _stdout.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(_stdout)
    log.setLevel(logging.INFO)
    log.propagate = False


class Gateway:
    """Everything a request needs: providers by name, the route table, limits."""

    def __init__(self, providers: dict[str, Provider], routes: RouteTable, *, timeout_s: float = 30.0):
        self.providers = providers
        self.routes = routes
        self.timeout_s = timeout_s

    async def aclose(self) -> None:
        for p in self.providers.values():
            await p.aclose()


def build_gateway(settings: Settings) -> Gateway:
    providers: dict[str, Provider] = {}
    if settings.enable_fake:
        from gateway.providers.fake import FakeProvider
        providers["fake"] = FakeProvider()
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
    return Gateway(providers, routes, timeout_s=settings.request_timeout_s)


def create_app(gateway: Gateway | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.gateway = gateway or build_gateway(get_settings())
        yield
        await app.state.gateway.aclose()

    app = FastAPI(title="LLM Gateway", lifespan=lifespan)

    @app.exception_handler(GatewayError)
    async def gateway_error(_: Request, exc: GatewayError):
        return JSONResponse(error_body(exc), status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError):
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"] if p != "body")
        return JSONResponse(error_body(InvalidRequest(f"{where}: {first['msg']}" if where else first["msg"])),
                            status_code=400)

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    @app.get("/v1/models")
    async def models(request: Request):
        gw: Gateway = request.app.state.gateway
        return {"object": "list",
                "data": [{"id": n, "object": "model", "created": 0, "owned_by": "llm-gateway"}
                         for n in gw.routes.names()]}

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest, request: Request):
        gw: Gateway = request.app.state.gateway
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        started = time.perf_counter()
        route = gw.routes.get(body.model)
        if body.stream:
            raise InvalidRequest("stream=true is not supported yet")
        chat = normalize(body)
        target = route.targets[0]       # fallback across targets arrives with Day 3
        provider = gw.providers[target.provider]
        outcome, completion = "error", None
        try:
            completion = await provider.complete(chat, model=target.model, timeout_s=gw.timeout_s)
            outcome = "ok"
        except GatewayError as exc:
            outcome = exc.code
            raise
        finally:
            log.info(json.dumps({
                "event": "chat_completion", "id": request_id, "route": route.name,
                "provider": target.provider, "model": target.model, "outcome": outcome,
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "input_tokens": completion.input_tokens if completion else None,
                "output_tokens": completion.output_tokens if completion else None,
            }))

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
            headers={"x-gateway-route": route.name, "x-gateway-provider": target.provider},
        )

    return app


app = create_app()
