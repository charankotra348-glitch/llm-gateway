from typing import Protocol

from gateway.schemas import ChatRequest, Completion


class Provider(Protocol):
    """One model provider. Implementations translate ChatRequest to the provider's API and
    raise gateway.errors.Provider* errors: never a provider SDK's own exception."""

    name: str

    async def complete(self, request: ChatRequest, *, model: str, timeout_s: float) -> Completion: ...

    async def aclose(self) -> None: ...
