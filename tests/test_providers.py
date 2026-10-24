import json

import httpx
import pytest
from botocore.exceptions import ReadTimeoutError
from botocore.stub import Stubber

from gateway.errors import (
    ProviderAuthError,
    ProviderRateLimited,
    ProviderRejected,
    ProviderTimeout,
    ProviderUnavailable,
)
from gateway.providers.bedrock import BedrockProvider
from gateway.providers.fake import FakeProvider
from gateway.providers.gemini import GeminiProvider
from gateway.providers.ollama import OllamaProvider
from gateway.schemas import ChatRequest, Message

pytestmark = pytest.mark.anyio

REQ = ChatRequest(
    messages=[Message(role="system", text="sys"), Message(role="user", text="hi"),
              Message(role="assistant", text="hello"), Message(role="user", text="again")],
    temperature=0.3, top_p=0.9, max_tokens=64, stop=["END"],
)


def mock_client(handler, **kw) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), **kw)


# --------------------------------------------------------------------------- Gemini
async def test_gemini_request_and_response():
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "candidates": [{"content": {"parts": [{"text": "thinking...", "thought": True}, {"text": "Hi!"}]},
                            "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 3, "thoughtsTokenCount": 5},
            "modelVersion": "gemini-2.5-flash",
        })

    p = GeminiProvider("k", client=mock_client(handler))
    out = await p.complete(REQ.model_copy(update={"json_mode": True}), model="gemini-2.5-flash", timeout_s=5)
    assert seen["url"].endswith("/models/gemini-2.5-flash:generateContent")
    body = seen["body"]
    assert body["systemInstruction"] == {"parts": [{"text": "sys"}]}
    assert [c["role"] for c in body["contents"]] == ["user", "model", "user"]
    assert body["generationConfig"] == {"temperature": 0.3, "topP": 0.9, "maxOutputTokens": 64,
                                        "stopSequences": ["END"], "responseMimeType": "application/json"}
    assert out.text == "Hi!"                       # thought parts are not the answer
    assert (out.input_tokens, out.output_tokens) == (10, 8)   # thinking billed as output


@pytest.mark.parametrize("reason, expected", [("MAX_TOKENS", "length"), ("SAFETY", "content_filter"),
                                              ("STOP", "stop")])
async def test_gemini_finish_reasons(reason, expected):
    p = GeminiProvider("k", client=mock_client(lambda r: httpx.Response(200, json={
        "candidates": [{"content": {"parts": [{"text": "x"}]}, "finishReason": reason}]})))
    out = await p.complete(REQ, model="m", timeout_s=5)
    assert out.finish_reason == expected
    assert out.input_tokens is None and out.output_tokens is None


@pytest.mark.parametrize("status, body, error", [
    (429, {"error": {"message": "quota", "status": "RESOURCE_EXHAUSTED"}}, ProviderRateLimited),
    (400, {"error": {"message": "API key not valid. Please pass a valid API key."}}, ProviderAuthError),
    (403, {"error": {"message": "denied"}}, ProviderAuthError),
    (503, {"error": {"message": "overloaded"}}, ProviderUnavailable),
    (504, {"error": {"message": "deadline", "status": "DEADLINE_EXCEEDED"}}, ProviderTimeout),
    (400, {"error": {"message": "bad stop sequence"}}, ProviderRejected),
])
async def test_gemini_errors(status, body, error):
    p = GeminiProvider("k", client=mock_client(lambda r: httpx.Response(status, json=body)))
    with pytest.raises(error) as exc:
        await p.complete(REQ, model="m", timeout_s=5)
    assert exc.value.retryable == (error in (ProviderRateLimited, ProviderUnavailable, ProviderTimeout))


async def test_gemini_auth_error_does_not_leak_the_key():
    p = GeminiProvider("secret-key", client=mock_client(
        lambda r: httpx.Response(400, json={"error": {"message": "API key not valid: secret-key"}})))
    with pytest.raises(ProviderAuthError) as exc:
        await p.complete(REQ, model="m", timeout_s=5)
    assert "secret-key" not in exc.value.message


async def test_gemini_timeout_and_connect_errors():
    def slow(request):
        raise httpx.ReadTimeout("slow", request=request)

    def down(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ProviderTimeout):
        await GeminiProvider("k", client=mock_client(slow)).complete(REQ, model="m", timeout_s=5)
    with pytest.raises(ProviderUnavailable):
        await GeminiProvider("k", client=mock_client(down)).complete(REQ, model="m", timeout_s=5)


# --------------------------------------------------------------------------- Ollama
async def test_ollama_request_and_response():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"model": "llama3.2", "message": {"role": "assistant", "content": "{}"},
                                         "done": True, "done_reason": "length",
                                         "prompt_eval_count": 20, "eval_count": 64})

    p = OllamaProvider(client=mock_client(handler, base_url="http://ollama"))
    out = await p.complete(REQ.model_copy(update={"json_mode": True}), model="llama3.2", timeout_s=5)
    body = seen["body"]
    assert body["stream"] is False and body["format"] == "json"
    assert body["options"] == {"temperature": 0.3, "top_p": 0.9, "num_predict": 64, "stop": ["END"]}
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert (out.text, out.finish_reason, out.input_tokens, out.output_tokens) == ("{}", "length", 20, 64)


async def test_ollama_errors():
    def down(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ProviderUnavailable):
        await OllamaProvider(client=mock_client(down, base_url="http://o")).complete(REQ, model="m", timeout_s=5)
    missing = mock_client(lambda r: httpx.Response(404, json={"error": "model not found"}), base_url="http://o")
    with pytest.raises(ProviderRejected):
        await OllamaProvider(client=missing).complete(REQ, model="m", timeout_s=5)


# --------------------------------------------------------------------------- Bedrock
def bedrock_client():
    import boto3
    return boto3.client("bedrock-runtime", region_name="us-east-1", aws_access_key_id="x",
                        aws_secret_access_key="x")


async def test_bedrock_request_and_response():
    client = bedrock_client()
    with Stubber(client) as stub:
        stub.add_response("converse", {
            "output": {"message": {"role": "assistant", "content": [{"text": "Hi"}]}},
            "stopReason": "max_tokens",
            "usage": {"inputTokens": 12, "outputTokens": 64, "totalTokens": 76},
            "metrics": {"latencyMs": 100},
        }, {
            "modelId": "us.amazon.nova-2-lite-v1:0",
            "messages": [{"role": "user", "content": [{"text": "hi"}]},
                         {"role": "assistant", "content": [{"text": "hello"}]},
                         {"role": "user", "content": [{"text": "again"}]}],
            "inferenceConfig": {"maxTokens": 64, "temperature": 0.3, "topP": 0.9, "stopSequences": ["END"]},
            "system": [{"text": "sys"}],
        })
        out = await BedrockProvider(client=client).complete(REQ, model="us.amazon.nova-2-lite-v1:0", timeout_s=5)
    assert (out.text, out.finish_reason, out.input_tokens, out.output_tokens) == ("Hi", "length", 12, 64)


async def test_bedrock_json_mode_adds_instruction():
    class Recorder:
        def converse(self, **kwargs):
            self.kwargs = kwargs
            return {"output": {"message": {"content": [{"text": "{}"}]}}, "stopReason": "end_turn"}

    rec = Recorder()
    out = await BedrockProvider(client=rec).complete(
        ChatRequest(messages=[Message(role="user", text="x")], json_mode=True), model="m", timeout_s=5)
    assert "JSON" in rec.kwargs["system"][0]["text"]
    assert out.input_tokens is None              # no usage reported -> none invented


@pytest.mark.parametrize("code, status, error", [
    ("ThrottlingException", 429, ProviderRateLimited),
    ("AccessDeniedException", 403, ProviderAuthError),
    ("ServiceUnavailableException", 503, ProviderUnavailable),
    ("ModelTimeoutException", 408, ProviderTimeout),
    ("ValidationException", 400, ProviderRejected),
])
async def test_bedrock_errors(code, status, error):
    client = bedrock_client()
    with Stubber(client) as stub:
        stub.add_client_error("converse", service_error_code=code, http_status_code=status)
        with pytest.raises(error):
            await BedrockProvider(client=client).complete(REQ, model="m", timeout_s=5)


async def test_bedrock_read_timeout():
    class Slow:
        def converse(self, **kwargs):
            raise ReadTimeoutError(endpoint_url="https://bedrock")

    with pytest.raises(ProviderTimeout):
        await BedrockProvider(client=Slow()).complete(REQ, model="m", timeout_s=5)


# --------------------------------------------------------------------------- Fake
async def test_fake_timeout_and_failures():
    with pytest.raises(ProviderTimeout):
        await FakeProvider(latency_s=0.2).complete(REQ, model="m", timeout_s=0.05)
    flaky = FakeProvider(failure_rate=0.5, seed=1)
    results = []
    for _ in range(40):
        try:
            await flaky.complete(REQ, model="m", timeout_s=1)
            results.append(True)
        except ProviderUnavailable:
            results.append(False)
    assert 10 < results.count(False) < 30
