from typing import Optional


class LLMError(Exception):
    """Base exception for all Language Model operations."""

    def __init__(
        self,
        message: str,
        error_code: str = "LLM_ERROR",
        retryable: bool = False,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.retryable = retryable
        self.cause = cause


class LLMAuthenticationError(LLMError):
    """Raised when LLM provider rejects authentication (401/403). Non-retryable."""

    def __init__(
        self,
        message: str = "LLM authentication failed. Check LLM_API_KEY.",
        cause: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_code="LLM_AUTHENTICATION_ERROR",
            retryable=False,
            cause=cause,
        )


class LLMConfigurationError(LLMError):
    """Raised when LLM provider configuration is missing or invalid. Non-retryable."""

    def __init__(
        self,
        message: str = "LLM configuration error.",
        cause: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_code="LLM_CONFIGURATION_ERROR",
            retryable=False,
            cause=cause,
        )


class LLMRateLimitError(LLMError):
    """Raised when LLM provider returns 429 Rate Limit. Retryable."""

    def __init__(
        self,
        message: str = "LLM rate limit exceeded.",
        cause: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_code="LLM_RATE_LIMIT",
            retryable=True,
            cause=cause,
        )


class LLMTimeoutError(LLMError):
    """Raised when LLM provider request times out. Retryable."""

    def __init__(
        self,
        message: str = "LLM request timed out.",
        cause: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_code="LLM_TIMEOUT",
            retryable=True,
            cause=cause,
        )


class LLMNetworkError(LLMError):
    """Raised on socket/connection failure communicating with LLM provider. Retryable."""

    def __init__(
        self,
        message: str = "LLM network connection error.",
        cause: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_code="LLM_NETWORK_ERROR",
            retryable=True,
            cause=cause,
        )


class LLMProviderError(LLMError):
    """Raised when LLM provider returns a 5xx server error."""

    def __init__(
        self,
        message: str = "LLM provider encountered an error.",
        retryable: bool = True,
        cause: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_code="LLM_PROVIDER_ERROR",
            retryable=retryable,
            cause=cause,
        )


class LLMInvalidResponseError(LLMError):
    """Raised when LLM returns unparseable or schema-violating output."""

    def __init__(
        self,
        message: str = "LLM returned invalid or malformed structured response.",
        retryable: bool = False,
        cause: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_code="LLM_INVALID_RESPONSE",
            retryable=retryable,
            cause=cause,
        )


class LLMContextLengthError(LLMError):
    """Raised when input transcript exceeds the LLM context window limit."""

    def __init__(
        self,
        message: str = "Transcript exceeds the maximum LLM context length.",
        cause: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_code="LLM_CONTEXT_LENGTH_EXCEEDED",
            retryable=False,
            cause=cause,
        )
