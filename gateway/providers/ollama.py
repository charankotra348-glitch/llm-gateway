"""Local models through Ollama's chat API: free, for development and as a last fallback."""

import json
from collections.abc import AsyncIterator

import httpx

from gateway.errors import ProviderRejected, ProviderTimeout, ProviderUnavailable
from gateway.schemas import ChatRequest, Chunk, Completion


class OllamaProvider:
    name = "ollama"

    def __init__(self, base_url: str = "http://localhost:11434", *, client: httpx.AsyncClient | None = None):
        self._client = client or httpx.AsyncClient(base_url=base_url)

    async def complete(self, request: ChatRequest, *, model: str, timeout_s: float) -> Completion:
        try:
            resp = await self._client.post("/api/chat", json=_payload(request, model, stream=False), timeout=timeout_s)
        except httpx.TimeoutException as exc:
            raise ProviderTimeout("Ollama timed out") from exc
        except httpx.TransportError as exc:
            raise ProviderUnavailable("could not reach Ollama; is it running?") from exc
        _check(resp)
        data = resp.json()
        return Completion(
            text=(data.get("message") or {}).get("content", ""),
            model=data.get("model") or model,
            finish_reason="length" if data.get("done_reason") == "length" else "stop",
            input_tokens=data.get("prompt_eval_count"),
            output_tokens=data.get("eval_count"),
        )

    async def stream(self, request: ChatRequest, *, model: str, timeout_s: float) -> AsyncIterator[Chunk]:
        """With stream: true, Ollama sends one JSON object per line; the last has done: true and the counts."""
        try:
            async with self._client.stream("POST", "/api/chat", json=_payload(request, model, stream=True),
                                           timeout=timeout_s) as resp:
                if resp.status_code >= 400:
                    await resp.aread()
                    _check(resp)
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if data.get("error"):
                        raise ProviderUnavailable(f"Ollama: {data['error']}"[:200])
                    if text := (data.get("message") or {}).get("content", ""):
                        yield Chunk(text=text)
                    if data.get("done"):
                        yield Chunk(finish_reason="length" if data.get("done_reason") == "length" else "stop",
                                    model=data.get("model") or model, input_tokens=data.get("prompt_eval_count"),
                                    output_tokens=data.get("eval_count"))
                        return
        except httpx.TimeoutException as exc:
            raise ProviderTimeout("Ollama timed out") from exc
        except httpx.TransportError as exc:
            raise ProviderUnavailable("lost the connection to Ollama") from exc
        raise ProviderUnavailable("Ollama ended the stream without finishing")

    async def aclose(self) -> None:
        await self._client.aclose()


def _payload(request: ChatRequest, model: str, *, stream: bool) -> dict:
    options: dict = {}
    if request.temperature is not None:
        options["temperature"] = request.temperature
    if request.top_p is not None:
        options["top_p"] = request.top_p
    if request.max_tokens is not None:
        options["num_predict"] = request.max_tokens
    if request.stop:
        options["stop"] = request.stop
    payload: dict = {"model": model, "messages": [{"role": m.role, "content": m.text} for m in request.messages],
                     "stream": stream, "options": options}
    if request.json_mode:
        payload["format"] = "json"
    return payload


def _check(resp: httpx.Response) -> None:
    if resp.status_code >= 500:
        raise ProviderUnavailable(f"Ollama: HTTP {resp.status_code}")
    if resp.status_code >= 400:   # e.g. 404: model not pulled
        raise ProviderRejected(f"Ollama: {resp.text[:200]}")
