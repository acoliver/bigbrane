"""Gateway error types. Every error carries the HTTP status and OpenAI-style error code it maps to."""
from __future__ import annotations

from dataclasses import dataclass


class GatewayError(Exception):
    status = 500
    code = "gateway_error"

    def __init__(self, message: str, *, status: int | None = None, code: str | None = None):
        super().__init__(message)
        if status is not None:
            self.status = status
        if code is not None:
            self.code = code

    @property
    def message(self) -> str:
        return str(self)


class BadRequest(GatewayError):
    status = 400
    code = "invalid_request_error"


class Unauthorized(GatewayError):
    status = 401
    code = "invalid_api_key"


class LoginRequired(GatewayError):
    """Stored credentials are missing or unusable. Only a new interactive login fixes this."""

    status = 401
    code = "login_required"

    def __init__(self, provider: str, reason: str):
        self.provider = provider
        super().__init__(f"{provider} login needed ({reason}). Run: python -m hybrid.gateway login {provider}")


@dataclass(frozen=True)
class UpstreamDetail:
    """What a provider said when it refused a call. The text is the provider's own; it is scrubbed where it leaves the process."""

    http_status: int
    message: str
    code: str | None
    error_type: str | None
    retry_after_s: float | None = None     # the Retry-After header, in seconds from receipt, when the provider sent one
    request_id: str | None = None          # upstream x-request-id; OpenAI asks apps to keep it for support
    root_error_code: str | None = None     # x-openai-ide-root-error-code
    authorization_error: str | None = None  # x-openai-authorization-error

    def diagnostics(self) -> dict[str, str]:
        """The upstream request id and authorization headers that were present, for logs and the usage meter."""
        pairs = (("request_id", self.request_id), ("root_error_code", self.root_error_code),
                 ("authorization_error", self.authorization_error))
        return {name: value for name, value in pairs if value}


class UpstreamError(GatewayError):
    """The provider answered with an error, or the call failed in transit."""

    status = 502
    code = "upstream_error"

    def __init__(self, message: str, *, status: int | None = None, code: str | None = None,
                 detail: UpstreamDetail | None = None):
        super().__init__(message, status=status, code=code)
        self.detail = detail


class RefreshFailed(UpstreamError):
    """A refresh failed in a way that may pass (5xx, an unknown 4xx). ``details`` holds what the token endpoint said,
    without any credential (``oauth_common.refresh_failure_details``)."""

    code = "refresh_failed"

    def __init__(self, message: str, *, status: int | None = None, details: dict | None = None):
        super().__init__(message, status=status, code="refresh_failed")
        self.details = details or {}


class StoreError(GatewayError):
    """The token store is unreadable, unsafe (owner, mode, type) or could not be written."""

    status = 500
    code = "store_error"


class StoreConflict(StoreError):
    """The stored record changed while a login was in progress; the newer record was kept."""

    status = 409
    code = "store_conflict"


class AlreadyRunning(GatewayError):
    code = "already_running"
