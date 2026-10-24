"""stream=true: the OpenAI chunk format, failover before the first chunk, failures after it, and the
provider adapters' streaming."""

import json

import httpx
import openai
import pytest
from botocore.exceptions import ClientError
from fastapi.testclient import TestClient
from sqlalchemy import select

from gateway.cache import ResponseCache
from gateway.db import make_engine, make_sessions
from gateway.errors import ProviderAuthError, ProviderRateLimited, ProviderTimeout, ProviderUnavailable
from gateway.limits import RateLimiter
from gateway.main import Gateway, create_app
from gateway.models import RequestLog
from gateway.pricing import PriceTable
from gateway.providers.bedrock import BedrockProvider
from gateway.providers.fake import FakeProvider
from gateway.providers.gemini import GeminiProvider
from gateway.providers.ollama import OllamaProvider
from gateway.resilience import Resilience, ResiliencePolicy
from gateway.routes import RouteTable
from gateway.schemas import ChatRequest, Message

MSG = [{"role": "user", "content": "hello there"}]
REQ = ChatRequest(messages=[Message(role="user", text="hi")])


@pytest.fixture
def stack(database_url, make_key):
    """Route `smart` = a -> b (both fakes, $1 per token); returns (sdk, a, b, client, key)."""
    a, b = FakeProvider(), FakeProvider()
    routes = RouteTable.from_dict({"routes": {"smart": [{"provider": "a", "model": "a-1"},
                                                        {"provider": "b", "model": "b-1"}]}})
    price = {"input_per_mtok": 1_000_000, "output_per_mtok": 1_000_000}
    engine = make_engine(database_url)
    gw = Gateway({"a": a, "b": b}, routes, make_sessions(engine),
                 prices=PriceTable.from_dict({"prices": {"a": {"a-1": price}, "b": {"b-1": price}}}),
                 resilience=Resilience(ResiliencePolicy(backoff_s=0, attempt_timeout_s=0.5)),
                 limiter=RateLimiter(0), cache=ResponseCache(ttl_s=3600), engine=engine)
    _, key = make_key()
    with TestClient(create_app(gw)) as client:
        sdk = openai.OpenAI(base_url="http://testserver/v1", api_key=key, http_client=client, max_retries=0)
        yield sdk, a, b, client, key


def ledger(db):
    async def q(s):
        return list(await s.scalars(select(RequestLog)))
    return db(q)


# --------------------------------------------------------------------------- the wire format
def test_stream_through_the_sdk(stack, db):
    sdk, a, _, _, _ = stack
    chunks = list(sdk.chat.completions.create(model="smart", messages=MSG, stream=True))
    assert chunks[0].choices[0].delta.role == "assistant"
    assert "".join(c.choices[0].delta.content or "" for c in chunks) == "echo: hello there"
    assert [c.choices[0].delta.content for c in chunks[1:-1]] == ["echo:", " hello", " there"]
    assert chunks[-1].choices[0].finish_reason == "stop"
    assert len({c.id for c in chunks}) == 1 and chunks[0].id.startswith("chatcmpl-")
    assert all(c.object == "chat.completion.chunk" and c.model == "a-1" for c in chunks)
    [row] = ledger(db)
    assert (row.outcome, row.provider, row.input_tokens, row.output_tokens) == ("ok", "a", 2, 3)
    assert row.cost_micros == 5_000_000


def test_include_usage_adds_a_usage_chunk(stack):
    sdk, *_ = stack
    chunks = list(sdk.chat.completions.create(model="smart", messages=MSG, stream=True,
                                              stream_options={"include_usage": True}))
    last = chunks[-1]
    assert last.choices == [] and (last.usage.prompt_tokens, last.usage.completion_tokens) == (2, 3)
    assert all(c.usage is None for c in chunks[:-1])


def test_raw_sse_ends_with_done(stack):
    _, _, _, client, key = stack
    r = client.post("/v1/chat/completions", headers={"authorization": f"Bearer {key}"},
                    json={"model": "smart", "messages": MSG, "stream": True})
    assert r.headers["content-type"].startswith("text/event-stream")
    events = [line for line in r.text.split("\n") if line.startswith("data: ")]
    assert events[-1] == "data: [DONE]"
    assert all(json.loads(e[6:])["object"] == "chat.completion.chunk" for e in events[:-1])


def test_max_tokens_streams_a_length_finish(stack):
    sdk, *_ = stack
    chunks = list(sdk.chat.completions.create(model="smart", messages=MSG, stream=True, max_tokens=2))
    assert "".join(c.choices[0].delta.content or "" for c in chunks) == "echo: hello"
    assert chunks[-1].choices[0].finish_reason == "length"


# --------------------------------------------------------------------------- failures
def test_failure_before_the_first_chunk_fails_over(stack, db):
    sdk, a, b, _, _ = stack
    a.script = [ProviderTimeout("slow")]
    raw = sdk.chat.completions.with_raw_response.create(model="smart", messages=MSG, stream=True)
    assert raw.headers["x-gateway-provider"] == "b" and raw.headers["x-gateway-attempts"] == "2"
    assert "".join(c.choices[0].delta.content or "" for c in raw.parse()) == "echo: hello there"
    assert ledger(db)[0].provider == "b"


def test_slow_first_chunk_fails_over(stack):
    sdk, a, b, _, _ = stack
    a.latency_s = 5                          # attempt timeout is 0.5 s
    raw = sdk.chat.completions.with_raw_response.create(model="smart", messages=MSG, stream=True)
    assert raw.headers["x-gateway-provider"] == "b"


def test_failure_before_anything_is_sent_is_a_normal_http_error(stack, db):
    sdk, a, b, _, _ = stack
    a.script = [ProviderRateLimited("busy")]
    b.script = [ProviderAuthError("bad key")]
    with pytest.raises(openai.APIStatusError) as exc:
        sdk.chat.completions.create(model="smart", messages=MSG, stream=True)
    assert exc.value.status_code == 502 and exc.value.body["code"] == "provider_auth_error"
    assert ledger(db)[0].outcome == "provider_auth_error"


def test_failure_mid_stream_ends_it_with_an_error_the_sdk_raises(stack, db):
    sdk, a, b, _, _ = stack
    a.fail_after_chunks = 2
    received = []
    with pytest.raises(openai.APIError, match="dropped the stream"):
        for c in sdk.chat.completions.create(model="smart", messages=MSG, stream=True):
            received.append(c.choices[0].delta.content or "")
    assert "".join(received) == "echo: hello"       # what was sent stays sent
    assert b.calls == []                              # too late to switch providers
    [row] = ledger(db)
    assert (row.outcome, row.provider, row.cost_micros) == ("provider_unavailable", "a", 0)


def test_a_stream_that_stalls_is_cut_off(stack, db):
    sdk, a, _, _, _ = stack
    a.chunk_delay_s = 5                       # longer than the 0.5 s per-chunk limit
    with pytest.raises(openai.APIError, match="stopped sending"):
        list(sdk.chat.completions.create(model="smart", messages=MSG, stream=True))
    assert ledger(db)[0].outcome == "provider_timeout"


def test_mid_stream_failures_count_toward_the_breaker(stack):
    sdk, a, *_ = stack
    a.fail_after_chunks = 1
    for _ in range(5):
        with pytest.raises(openai.APIError):
            list(sdk.chat.completions.create(model="smart", messages=MSG, stream=True))
    raw = sdk.chat.completions.with_raw_response.create(model="smart", messages=MSG, stream=True)
    assert raw.headers["x-gateway-provider"] == "b"      # a's circuit opened


# --------------------------------------------------------------------------- with the other features
def test_streams_bypass_the_cache(stack):
    sdk, a, *_ = stack
    for _ in range(2):
        raw = sdk.chat.completions.with_raw_response.create(model="smart", messages=MSG, stream=True, temperature=0)
        assert raw.headers["x-gateway-cache"] == "bypass"
        list(raw.parse())
    assert len(a.calls) == 2


def test_budget_applies_to_streams(stack, make_key):
    sdk, _, _, client, _ = stack
    _, key = make_key(budget_usd=1)
    capped = openai.OpenAI(base_url="http://testserver/v1", api_key=key, http_client=client, max_retries=0)
    list(capped.chat.completions.create(model="smart", messages=MSG, stream=True))
    with pytest.raises(openai.RateLimitError):
        capped.chat.completions.create(model="smart", messages=MSG, stream=True)


# --------------------------------------------------------------------------- adapters
async def collect(stream):
    return [c async for c in stream]


@pytest.mark.anyio
async def test_gemini_stream():
    events = [
        {"candidates": [{"content": {"parts": [{"text": "thinking", "thought": True}]}}]},
        {"candidates": [{"content": {"parts": [{"text": "Hel"}]}}]},
        {"candidates": [{"content": {"parts": [{"text": "lo"}]}, "finishReason": "STOP"}],
         "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 2, "thoughtsTokenCount": 3},
         "modelVersion": "gemini-2.5-flash-001"},
    ]
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        return httpx.Response(200, text="".join(f"data: {json.dumps(e)}\r\n\r\n" for e in events),
                              headers={"content-type": "text/event-stream"})

    p = GeminiProvider("k", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    chunks = await collect(p.stream(REQ, model="gemini-2.5-flash", timeout_s=5))
    assert ":streamGenerateContent" in seen["url"] and "alt=sse" in seen["url"]
    assert [c.text for c in chunks[:-1]] == ["Hel", "lo"]
    last = chunks[-1]
    assert (last.finish_reason, last.input_tokens, last.output_tokens, last.model) == (
        "stop", 4, 5, "gemini-2.5-flash-001")


@pytest.mark.anyio
async def test_gemini_stream_errors():
    p = GeminiProvider("k", client=httpx.AsyncClient(transport=httpx.MockTransport(
        lambda r: httpx.Response(429, json={"error": {"message": "quota"}}))))
    with pytest.raises(ProviderRateLimited):
        await collect(p.stream(REQ, model="m", timeout_s=5))
    cut = GeminiProvider("k", client=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(
        200, text='data: {"candidates": [{"content": {"parts": [{"text": "Hi"}]}}]}\n\n'))))
    with pytest.raises(ProviderUnavailable, match="without finishing"):
        await collect(cut.stream(REQ, model="m", timeout_s=5))


@pytest.mark.anyio
async def test_ollama_stream():
    lines = [{"message": {"content": "Hel"}, "done": False}, {"message": {"content": "lo"}, "done": False},
             {"message": {"content": ""}, "done": True, "done_reason": "stop", "model": "llama3.2",
              "prompt_eval_count": 7, "eval_count": 2}]
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, text="\n".join(json.dumps(x) for x in lines))

    p = OllamaProvider(client=httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://o"))
    chunks = await collect(p.stream(REQ, model="llama3.2", timeout_s=5))
    assert seen["body"]["stream"] is True
    assert [c.text for c in chunks[:-1]] == ["Hel", "lo"]
    assert (chunks[-1].finish_reason, chunks[-1].input_tokens, chunks[-1].output_tokens) == ("stop", 7, 2)


class FakeBedrock:
    def __init__(self, events):
        self.events = events

    def converse_stream(self, **kwargs):
        self.kwargs = kwargs
        return {"stream": iter(self.events)}


@pytest.mark.anyio
async def test_bedrock_stream():
    client = FakeBedrock([{"messageStart": {"role": "assistant"}},
                          {"contentBlockDelta": {"delta": {"text": "Hel"}}},
                          {"contentBlockDelta": {"delta": {"text": "lo"}}},
                          {"messageStop": {"stopReason": "max_tokens"}},
                          {"metadata": {"usage": {"inputTokens": 9, "outputTokens": 2}}}])
    chunks = await collect(BedrockProvider(client=client).stream(REQ, model="nova", timeout_s=5))
    assert client.kwargs["modelId"] == "nova"
    assert [c.text for c in chunks[:-1]] == ["Hel", "lo"]
    assert (chunks[-1].finish_reason, chunks[-1].input_tokens, chunks[-1].output_tokens) == ("length", 9, 2)


@pytest.mark.anyio
async def test_bedrock_stream_error_mid_way():
    def events():
        yield {"contentBlockDelta": {"delta": {"text": "Hel"}}}
        raise ClientError({"Error": {"Code": "ThrottlingException", "Message": "slow down"}}, "ConverseStream")

    with pytest.raises(ProviderRateLimited):
        await collect(BedrockProvider(client=FakeBedrock(events())).stream(REQ, model="nova", timeout_s=5))
