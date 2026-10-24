"""Google Gemini through its REST API (generateContent)."""

import httpx

from gateway.errors import (
    ProviderAuthError,
    ProviderError,
    ProviderRateLimited,
    ProviderRejected,
    ProviderTimeout,
    ProviderUnavailable,
)
from gateway.schemas import ChatRequest, Completion, FinishReason

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
FILTERED = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY"}


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str, *, client: httpx.AsyncClient | None = None):
        if not api_key:
            raise ValueError("Gemini needs GEMINI_API_KEY")
        self._client = client or httpx.AsyncClient(headers={"x-goog-api-key": api_key})

    async def complete(self, request: ChatRequest, *, model: str, timeout_s: float) -> Completion:
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

        try:
            resp = await self._client.post(f"{API_ROOT}/{model}:generateContent", json=body, timeout=timeout_s)
        except httpx.TimeoutException as exc:
            raise ProviderTimeout("Gemini timed out") from exc
        except httpx.TransportError as exc:
            raise ProviderUnavailable("could not reach Gemini") from exc
        if resp.status_code >= 400:
            raise _error_for(resp)

        data = resp.json()
        candidate = (data.get("candidates") or [{}])[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        usage = data.get("usageMetadata") or {}
        prompt, output = usage.get("promptTokenCount"), usage.get("candidatesTokenCount")
        if prompt is not None and output is not None:
            output += usage.get("thoughtsTokenCount") or 0   # thinking tokens are billed as output
        else:
            prompt = output = None
        return Completion(
            text="".join(p.get("text", "") for p in parts if not p.get("thought")),
            model=data.get("modelVersion") or model,
            finish_reason=_finish(candidate.get("finishReason")),
            input_tokens=prompt,
            output_tokens=output,
        )

    async def aclose(self) -> None:
        await self._client.aclose()


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
