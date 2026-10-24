"""Local models through Ollama's chat API: free, for development and as a last fallback."""

import httpx

from gateway.errors import ProviderRejected, ProviderTimeout, ProviderUnavailable
from gateway.schemas import ChatRequest, Completion


class OllamaProvider:
    name = "ollama"

    def __init__(self, base_url: str = "http://localhost:11434", *, client: httpx.AsyncClient | None = None):
        self._client = client or httpx.AsyncClient(base_url=base_url)

    async def complete(self, request: ChatRequest, *, model: str, timeout_s: float) -> Completion:
        messages = [{"role": m.role, "content": m.text} for m in request.messages]
        options: dict = {}
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.top_p is not None:
            options["top_p"] = request.top_p
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        if request.stop:
            options["stop"] = request.stop
        payload: dict = {"model": model, "messages": messages, "stream": False, "options": options}
        if request.json_mode:
            payload["format"] = "json"

        try:
            resp = await self._client.post("/api/chat", json=payload, timeout=timeout_s)
        except httpx.TimeoutException as exc:
            raise ProviderTimeout("Ollama timed out") from exc
        except httpx.TransportError as exc:
            raise ProviderUnavailable("could not reach Ollama; is it running?") from exc
        if resp.status_code >= 500:
            raise ProviderUnavailable(f"Ollama: HTTP {resp.status_code}")
        if resp.status_code >= 400:   # e.g. 404: model not pulled
            raise ProviderRejected(f"Ollama: {resp.text[:200]}")

        data = resp.json()
        return Completion(
            text=(data.get("message") or {}).get("content", ""),
            model=data.get("model") or model,
            finish_reason="length" if data.get("done_reason") == "length" else "stop",
            input_tokens=data.get("prompt_eval_count"),
            output_tokens=data.get("eval_count"),
        )

    async def aclose(self) -> None:
        await self._client.aclose()
