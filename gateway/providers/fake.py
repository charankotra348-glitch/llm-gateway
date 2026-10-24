"""A provider that never leaves the process: for tests, local development and load tests.

`latency_s` and `failure_rate` let the load and chaos tests (Day 6) shape its behaviour;
`seed` makes failures reproducible.
"""

import asyncio
import random

from gateway.errors import ProviderTimeout, ProviderUnavailable
from gateway.schemas import ChatRequest, Completion


class FakeProvider:
    name = "fake"

    def __init__(self, *, reply: str | None = None, latency_s: float = 0.0, failure_rate: float = 0.0,
                 seed: int | None = None):
        self.reply = reply
        self.latency_s = latency_s
        self.failure_rate = failure_rate
        self._random = random.Random(seed)
        self.calls: list[tuple[str, ChatRequest]] = []

    async def complete(self, request: ChatRequest, *, model: str, timeout_s: float) -> Completion:
        self.calls.append((model, request))
        if self.latency_s > timeout_s:
            await asyncio.sleep(timeout_s)
            raise ProviderTimeout(f"fake provider took longer than {timeout_s}s")
        if self.latency_s:
            await asyncio.sleep(self.latency_s)
        if self.failure_rate and self._random.random() < self.failure_rate:
            raise ProviderUnavailable("fake provider failure")

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
