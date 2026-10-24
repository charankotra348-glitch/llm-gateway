"""The OpenAI chat-completions wire format, and the provider-neutral request inside the gateway.

Only what the gateway supports is modelled; unknown fields are ignored, because SDKs send
fields a given gateway feature may not use (`user`, `metadata`, `seed`, ...).
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from gateway.errors import InvalidRequest

Role = Literal["system", "user", "assistant"]


# --------------------------------------------------------------------------- incoming
class WireMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None = None


class ResponseFormat(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["text", "json_object", "json_schema"] = "text"


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="ignore")

    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = Field(min_length=1)
    messages: list[WireMessage] = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    max_tokens: int | None = Field(default=None, gt=0)
    max_completion_tokens: int | None = Field(default=None, gt=0)
    stop: str | list[str] | None = None
    n: int = 1
    stream: bool = False
    stream_options: StreamOptions | None = None
    response_format: ResponseFormat | None = None


# --------------------------------------------------------------------------- inside the gateway
class Message(BaseModel):
    role: Role
    text: str


class ChatRequest(BaseModel):
    """Provider-neutral: every adapter translates from this, never from the wire format."""

    messages: list[Message]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: list[str] = []
    json_mode: bool = False

    @property
    def system(self) -> str | None:
        parts = [m.text for m in self.messages if m.role == "system"]
        return "\n\n".join(parts) if parts else None

    @property
    def conversation(self) -> list[Message]:
        return [m for m in self.messages if m.role != "system"]


FinishReason = Literal["stop", "length", "content_filter"]


class Completion(BaseModel):
    """What a provider returns: raw facts only. Token counts are None when not reported."""

    text: str
    model: str
    finish_reason: FinishReason = "stop"
    input_tokens: int | None = None
    output_tokens: int | None = None


class Chunk(BaseModel):
    """One piece of a streamed answer. Providers yield text chunks, then exactly one final chunk
    with `finish_reason` set (and the token counts, when the provider reports them)."""

    text: str = ""
    finish_reason: FinishReason | None = None
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None

    @property
    def final(self) -> bool:
        return self.finish_reason is not None


def _text_of(message: WireMessage) -> str:
    if message.content is None:
        return ""
    if isinstance(message.content, str):
        return message.content
    texts = []
    for part in message.content:
        if part.get("type") != "text":
            raise InvalidRequest(f"content part of type {part.get('type')!r} is not supported; send text parts")
        texts.append(str(part.get("text", "")))
    return "".join(texts)


def normalize(req: ChatCompletionRequest) -> ChatRequest:
    """Wire request -> provider-neutral request, refusing what the gateway can't honour."""
    if req.n != 1:
        raise InvalidRequest("n must be 1")
    if req.response_format and req.response_format.type == "json_schema":
        raise InvalidRequest("response_format json_schema is not supported; use json_object")
    messages = []
    for m in req.messages:
        if m.role == "tool":
            raise InvalidRequest("tool messages are not supported")
        role: Role = "system" if m.role in ("system", "developer") else m.role
        messages.append(Message(role=role, text=_text_of(m)))
    if not any(m.role == "user" for m in messages):
        raise InvalidRequest("messages must include at least one user message")
    stop = [req.stop] if isinstance(req.stop, str) else list(req.stop or [])
    return ChatRequest(
        messages=messages,
        temperature=req.temperature,
        top_p=req.top_p,
        max_tokens=req.max_completion_tokens or req.max_tokens,
        stop=stop,
        json_mode=bool(req.response_format and req.response_format.type == "json_object"),
    )
