from collections.abc import AsyncIterator
from typing import Protocol

from gateway.schemas import ChatRequest, Chunk, Completion


class Provider(Protocol):
    """One model provider. Implementations translate ChatRequest to the provider's API and
    raise gateway.errors.Provider* errors: never a provider SDK's own exception."""

    name: str

    async def complete(self, request: ChatRequest, *, model: str, timeout_s: float) -> Completion: ...

    def stream(self, request: ChatRequest, *, model: str, timeout_s: float) -> AsyncIterator[Chunk]:
        """Text chunks as the model writes them, then one final Chunk with finish_reason set.
        `timeout_s` bounds the wait for each chunk (the gateway also enforces it)."""
        ...

    async def aclose(self) -> None: ...
