"""Errors, and how each one reaches the client in OpenAI's error format.

Provider errors say whether retrying could help (`retryable`); the fallback logic (resilience.py)
uses that. The HTTP status tells the client whose fault it was: 4xx = the request,
5xx = the gateway or a provider behind it.
"""


class GatewayError(Exception):
    status = 500
    type = "server_error"
    code = "internal_error"
    retryable = False

    def __init__(self, message: str = ""):
        super().__init__(message or self.code)
        self.message = message or self.code


# --------------------------------------------------------------------------- the client's request
class InvalidRequest(GatewayError):
    status = 400
    type = "invalid_request_error"
    code = "invalid_request"


class InvalidApiKey(GatewayError):
    status = 401
    type = "invalid_request_error"
    code = "invalid_api_key"


class BudgetExceeded(GatewayError):
    """The key's monthly budget is spent. 429 like OpenAI's insufficient_quota, but never worth retrying."""

    status = 429
    type = "insufficient_quota"
    code = "budget_exceeded"


class ModelNotFound(GatewayError):
    status = 404
    type = "invalid_request_error"
    code = "model_not_found"


# --------------------------------------------------------------------------- a provider behind us
class ProviderError(GatewayError):
    status = 502
    type = "provider_error"
    code = "provider_error"


class ProviderTimeout(ProviderError):
    status = 504
    code = "provider_timeout"
    retryable = True


class ProviderRateLimited(ProviderError):
    status = 429
    code = "provider_rate_limited"
    retryable = True


class ProviderUnavailable(ProviderError):
    code = "provider_unavailable"
    retryable = True


class ProviderAuthError(ProviderError):
    """The gateway's own credentials for a provider are wrong: not the client's fault."""

    code = "provider_auth_error"


class ProviderRejected(ProviderError):
    """The provider refused the request as invalid (bad parameter, unknown model)."""

    status = 400
    type = "invalid_request_error"
    code = "provider_rejected_request"


def error_body(exc: GatewayError) -> dict:
    return {"error": {"message": exc.message, "type": exc.type, "code": exc.code, "param": None}}
