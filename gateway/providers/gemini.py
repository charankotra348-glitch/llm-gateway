"""Google Gemini through its REST API (generateContent, and streamGenerateContent for streams)."""

import json
from collections.abc import AsyncIterator

import httpx

from gateway.errors import (
    ProviderAuthError,
    ProviderError,
    ProviderRateLimited,
    ProviderRejected,
    ProviderTimeout,
    ProviderUnavailable,
)
from gateway.schemas import ChatRequest, Chunk, Completion, FinishReason

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
FILTERED = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY"}


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str, *, client: httpx.AsyncClient | None = None):
        if not api_key:
            raise ValueError("Gemini needs GEMINI_API_KEY")
        self._client = client or httpx.AsyncClient(headers={"x-goog-api-key": api_key})

    async def complete(self, request: ChatRequest, *, model: str, timeout_s: float) -> Completion:
        try:
            resp = await self._client.post(f"{API_ROOT}/{model}:generateContent", json=_body(request),
                                           timeout=timeout_s)
        except httpx.TimeoutException as exc:
            raise ProviderTimeout("Gemini timed out") from exc
        except httpx.TransportError as exc:
            raise ProviderUnavailable("could not reach Gemini") from exc
        if resp.status_code >= 400:
            raise _error_for(resp)

        data = resp.json()
        candidate = (data.get("candidates") or [{}])[0]
        prompt, output = _usage(data)
        return Completion(
            text=_text(candidate),
            model=data.get("modelVersion") or model,
            finish_reason=_finish(candidate.get("finishReason")),
            input_tokens=prompt,
            output_tokens=output,
        )

    async def stream(self, request: ChatRequest, *, model: str, timeout_s: float) -> AsyncIterator[Chunk]:
        """streamGenerateContent with alt=sse: each event is a partial GenerateContentResponse.
        The usage figures are running totals, so the last one seen is the answer's."""
        reason, usage, version = None, (None, None), model
        try:
            async with self._client.stream("POST", f"{API_ROOT}/{model}:streamGenerateContent",
                                           params={"alt": "sse"}, json=_body(request), timeout=timeout_s) as resp:
                if resp.status_code >= 400:
                    await resp.aread()
                    raise _error_for(resp)
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = json.loads(line[5:])
                    candidate = (data.get("candidates") or [{}])[0]
                    reason = candidate.get("finishReason") or reason
                    version = data.get("modelVersion") or version
                    if data.get("usageMetadata"):
                        usage = _usage(data)
                    if text := _text(candidate):
                        yield Chunk(text=text)
        except httpx.TimeoutException as exc:
            raise ProviderTimeout("Gemini timed out") from exc
        except httpx.TransportError as exc:
            raise ProviderUnavailable("lost the connection to Gemini") from exc
        if reason is None:
            raise ProviderUnavailable("Gemini ended the stream without finishing")
        yield Chunk(finish_reason=_finish(reason), model=version, input_tokens=usage[0], output_tokens=usage[1])

    async def aclose(self) -> None:
        await self._client.aclose()


def _body(request: ChatRequest) -> dict:
    body: dict = {
        "contents": [
            {"role": "model" if m.role == "assistant" else "user", "parts": [{"text": m.text}]}
            for m in request.conversation
        ],
        "generationConfig": {},
    }
    if request.system:
        body["systemInstruction"] = {"parts": [{"text": request.system}]}
    config = body["generationConfig"]
    if request.temperature is not None:
        config["temperature"] = request.temperature
    if request.top_p is not None:
        config["topP"] = request.top_p
    if request.max_tokens is not None:
        config["maxOutputTokens"] = request.max_tokens
    if request.stop:
        config["stopSequences"] = request.stop
    if request.json_mode:
        config["responseMimeType"] = "application/json"
    return body


def _text(candidate: dict) -> str:
    parts = (candidate.get("content") or {}).get("parts") or []
    return "".join(p.get("text", "") for p in parts if not p.get("thought"))


def _usage(data: dict) -> tuple[int | None, int | None]:
    usage = data.get("usageMetadata") or {}
    prompt, output = usage.get("promptTokenCount"), usage.get("candidatesTokenCount")
    if prompt is None or output is None:
        return None, None
    return prompt, output + (usage.get("thoughtsTokenCount") or 0)   # thinking tokens are billed as output


def _finish(reason: str | None) -> FinishReason:
    if reason == "MAX_TOKENS":
        return "length"
    if reason in FILTERED:
        return "content_filter"
    return "stop"


def _error_for(resp: httpx.Response) -> ProviderError:
    try:
        detail = resp.json().get("error", {})
    except ValueError:
        detail = {}
    message = str(detail.get("message") or resp.text[:200] or f"HTTP {resp.status_code}")
    status = str(detail.get("status") or "")
    if resp.status_code == 429:
        return ProviderRateLimited(f"Gemini rate limit: {message}")
    if resp.status_code == 504 or status == "DEADLINE_EXCEEDED":
        return ProviderTimeout(f"Gemini: {message}")
    if resp.status_code in (401, 403) or "api key" in message.lower():
        return ProviderAuthError("Gemini refused the gateway's API key")
    if resp.status_code >= 500:
        return ProviderUnavailable(f"Gemini: {message}")
    return ProviderRejected(f"Gemini: {message}")
