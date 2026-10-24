"""Amazon Bedrock through the Converse API. Credentials come from the AWS chain (a role on
AWS, `aws configure` on a laptop), never from gateway settings.

boto3 is synchronous, so calls run in a worker thread to keep the event loop free.
"""

import anyio
import boto3
from botocore.config import Config
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    ConnectTimeoutError,
    EndpointConnectionError,
    NoCredentialsError,
    ReadTimeoutError,
)

from gateway.errors import (
    ProviderAuthError,
    ProviderError,
    ProviderRateLimited,
    ProviderRejected,
    ProviderTimeout,
    ProviderUnavailable,
)
from gateway.schemas import ChatRequest, Completion, FinishReason

JSON_INSTRUCTION = "Reply with only one JSON object: no prose before or after it, no code fences."
RATE_LIMITED = {"ThrottlingException", "ModelNotReadyException", "TooManyRequestsException"}
UNAVAILABLE = {"ServiceUnavailableException", "InternalServerException", "ModelErrorException"}
AUTH = {"AccessDeniedException", "UnrecognizedClientException", "ExpiredTokenException",
        "InvalidSignatureException"}


class BedrockProvider:
    name = "bedrock"

    def __init__(self, region: str = "us-east-1", *, client=None, session=None):
        self._region = region
        self._session = session or boto3.session.Session()
        self._fixed_client = client
        self._clients: dict[float, object] = {}

    def _client_for(self, timeout_s: float):
        if self._fixed_client is not None:
            return self._fixed_client
        key = round(timeout_s, 1)   # botocore timeouts are per client
        if key not in self._clients:
            self._clients[key] = self._session.client(
                "bedrock-runtime", region_name=self._region,
                config=Config(connect_timeout=min(10.0, timeout_s), read_timeout=timeout_s,
                              retries={"total_max_attempts": 1, "mode": "standard"}),  # the gateway retries
            )
        return self._clients[key]

    async def complete(self, request: ChatRequest, *, model: str, timeout_s: float) -> Completion:
        system = request.system or ""
        if request.json_mode:
            system = f"{system}\n\n{JSON_INSTRUCTION}".strip()
        inference: dict = {}
        if request.max_tokens is not None:
            inference["maxTokens"] = request.max_tokens
        if request.temperature is not None:
            inference["temperature"] = request.temperature
        if request.top_p is not None:
            inference["topP"] = request.top_p
        if request.stop:
            inference["stopSequences"] = request.stop
        kwargs: dict = {
            "modelId": model,
            "messages": [{"role": m.role, "content": [{"text": m.text}]} for m in request.conversation],
        }
        if inference:
            kwargs["inferenceConfig"] = inference
        if system:
            kwargs["system"] = [{"text": system}]

        client = self._client_for(timeout_s)
        try:
            # abandon_on_cancel: the gateway's own deadline can give up on the thread instead of waiting for it
            response = await anyio.to_thread.run_sync(lambda: client.converse(**kwargs), abandon_on_cancel=True)
        except ClientError as exc:
            raise _error_for(exc) from exc
        except (ReadTimeoutError, ConnectTimeoutError) as exc:
            raise ProviderTimeout("Bedrock timed out") from exc
        except NoCredentialsError as exc:
            raise ProviderAuthError("no AWS credentials for Bedrock") from exc
        except (EndpointConnectionError, BotoCoreError) as exc:
            raise ProviderUnavailable("could not reach Bedrock") from exc

        blocks = response.get("output", {}).get("message", {}).get("content", [])
        usage = response.get("usage") or {}
        return Completion(
            text="".join(b.get("text", "") for b in blocks),
            model=model,
            finish_reason=_finish(response.get("stopReason")),
            input_tokens=usage.get("inputTokens"),
            output_tokens=usage.get("outputTokens"),
        )

    async def aclose(self) -> None:
        return None


def _finish(reason: str | None) -> FinishReason:
    if reason in ("max_tokens", "model_context_window_exceeded"):
        return "length"
    if reason in ("guardrail_intervened", "content_filtered"):
        return "content_filter"
    return "stop"


def _error_for(exc: ClientError) -> ProviderError:
    code = exc.response.get("Error", {}).get("Code", "")
    status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
    message = str(exc.response.get("Error", {}).get("Message", ""))[:200]
    if code in RATE_LIMITED or status == 429:
        return ProviderRateLimited(f"Bedrock: {message or code}")
    if code == "ModelTimeoutException" or status in (408, 504):
        return ProviderTimeout(f"Bedrock: {message or code}")
    if code in AUTH or status in (401, 403):
        return ProviderAuthError("Bedrock refused the gateway's AWS credentials")
    if code in UNAVAILABLE or status >= 500:
        return ProviderUnavailable(f"Bedrock: {message or code}")
    return ProviderRejected(f"Bedrock: {message or code}")
