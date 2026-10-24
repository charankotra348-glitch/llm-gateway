"""A provider that never leaves the process: for tests, local development and load tests.

`latency_s` and `failure_rate` let the load and chaos tests (Day 6) shape its behaviour;
`seed` makes failures reproducible. `script` is a queue of exceptions to raise on the next calls,
one per call, for tests that need an exact sequence ("fail, fail, then answer").

Streaming sends the reply a word at a time, `chunk_delay_s` apart. `fail_after_chunks` makes a
stream break after that many chunks, for testing failures in the middle of an answer.
"""

import asyncio
import random
from collections.abc import AsyncIterator

from gateway.errors import ProviderTimeout, ProviderUnavailable
from gateway.schemas import ChatRequest, Chunk, Completion


class FakeProvider:
    name = "fake"

    def __init__(self, *, reply: str | None = None, latency_s: float = 0.0, failure_rate: float = 0.0,
                 seed: int | None = None, script: list[Exception] | None = None, chunk_delay_s: float = 0.0,
                 fail_after_chunks: int | None = None):
        self.reply = reply
        self.latency_s = latency_s
        self.failure_rate = failure_rate
        self._random = random.Random(seed)
        self.script = list(script or [])
        self.chunk_delay_s = chunk_delay_s
        self.fail_after_chunks = fail_after_chunks
        self.calls: list[tuple[str, ChatRequest]] = []

    async def complete(self, request: ChatRequest, *, model: str, timeout_s: float) -> Completion:
        await self._start(request, model, timeout_s)
        return self._answer(request, model)

    async def stream(self, request: ChatRequest, *, model: str, timeout_s: float) -> AsyncIterator[Chunk]:
        await self._start(request, model, timeout_s)
        answer = self._answer(request, model)
        words = answer.text.split(" ")
        for i, word in enumerate(words):
            if self.fail_after_chunks is not None and i >= self.fail_after_chunks:
                raise ProviderUnavailable("fake provider dropped the stream")
            if i and self.chunk_delay_s:
                await asyncio.sleep(self.chunk_delay_s)
            yield Chunk(text=word if i == 0 else " " + word)
        yield Chunk(finish_reason=answer.finish_reason, model=model, input_tokens=answer.input_tokens,
                    output_tokens=answer.output_tokens)

    async def _start(self, request: ChatRequest, model: str, timeout_s: float) -> None:
        self.calls.append((model, request))
        if self.script:
            raise self.script.pop(0)
        if self.latency_s > timeout_s:
            await asyncio.sleep(timeout_s)
            raise ProviderTimeout(f"fake provider took longer than {timeout_s}s")
        if self.latency_s:
            await asyncio.sleep(self.latency_s)
        if self.failure_rate and self._random.random() < self.failure_rate:
            raise ProviderUnavailable("fake provider failure")

    def _answer(self, request: ChatRequest, model: str) -> Completion:
        last_user = next(m.text for m in reversed(request.messages) if m.role == "user")
        text = self.reply if self.reply is not None else f"echo: {last_user}"
        if request.max_tokens is not None:
            words = text.split()
            if len(words) > request.max_tokens:
                return Completion(text=" ".join(words[: request.max_tokens]), model=model, finish_reason="length",
                                  input_tokens=_count(request), output_tokens=request.max_tokens)
        return Completion(text=text, model=model, input_tokens=_count(request), output_tokens=len(text.split()))

    async def aclose(self) -> None:
        return None


def _count(request: ChatRequest) -> int:
    """Whitespace words, standing in for tokens. The fake is the only place counts are made up,
    and it is never used for real traffic."""
    return sum(len(m.text.split()) for m in request.messages)
