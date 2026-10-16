"""Exception types for hybrid/llm."""


class LLMError(Exception):
    """Base class for every error raised by hybrid/llm. ``call_id`` names the transcript of the failed call when
    LLMClient.complete raised it."""

    call_id: str | None = None


class ConfigError(LLMError):
    """models.toml is invalid or a configured resource is missing."""


class BudgetExceeded(LLMError):
    """A per-run call or token budget would be or has been exceeded."""


class TransportError(LLMError):
    """Subprocess failure, HTTP 5xx/429, connection error or timeout. Triggers fallback."""


class OutputExhausted(TransportError):
    """The backend answered with no visible text because the reply reached its output cap (finish_reason=length or
    stop_reason=max_tokens): on Claude with adaptive thinking the whole cap went to thinking. ``usage`` is what the
    backend reported for the spent tokens, so the budget still counts them. Falls back like any transport failure;
    LLMClient first retries the same entry once at a lower effort (FIX-5)."""

    def __init__(self, message: str, *, usage: dict | None = None):
        super().__init__(message)
        self.usage = usage


class BackendError(LLMError):
    """The backend answered, but the answer is unusable (for example HTTP 4xx). No fallback."""


class ToolPolicyViolation(LLMError):
    """The llxprt run used tools that the call's policy does not allow. No fallback."""


class SchemaValidationError(LLMError):
    """Model output failed JSON extraction or schema validation after the retry. No fallback."""
