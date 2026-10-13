from typing import Optional


class STTError(Exception):
    """Base exception for all Speech-to-Text operations."""

    def __init__(
        self,
        message: str,
        error_code: str = "STT_ERROR",
        retryable: bool = False,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.retryable = retryable
        self.cause = cause


class STTAuthenticationError(STTError):
    """Raised when STT provider rejects authentication (401/403). Non-retryable."""

    def __init__(self, message: str = "STT authentication failed. Check STT_API_KEY.", cause: Optional[Exception] = None):
        super().__init__(
            message=message,
            error_code="STT_AUTHENTICATION_ERROR",
            retryable=False,
            cause=cause,
        )


class STTRateLimitError(STTError):
    """Raised when STT provider returns 429 Rate Limit. Retryable."""

    def __init__(self, message: str = "STT rate limit exceeded.", cause: Optional[Exception] = None):
        super().__init__(
            message=message,
            error_code="STT_RATE_LIMIT",
            retryable=True,
            cause=cause,
        )


class STTTimeoutError(STTError):
    """Raised when STT provider request times out. Retryable."""

    def __init__(self, message: str = "STT request timed out.", cause: Optional[Exception] = None):
        super().__init__(
            message=message,
            error_code="STT_TIMEOUT",
            retryable=True,
            cause=cause,
        )


class STTNetworkError(STTError):
    """Raised on socket/connection failure communicating with STT provider. Retryable."""

    def __init__(self, message: str = "STT network connection error.", cause: Optional[Exception] = None):
        super().__init__(
            message=message,
            error_code="STT_NETWORK_ERROR",
            retryable=True,
            cause=cause,
        )


class STTProviderError(STTError):
    """Raised when STT provider returns 5xx error or invalid payload."""

    def __init__(
        self,
        message: str = "STT provider encountered an error.",
        retryable: bool = True,
        cause: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_code="STT_PROVIDER_ERROR",
            retryable=retryable,
            cause=cause,
        )



class STTConfigurationError(STTError):
    """Raised when STT configuration is invalid or missing required credentials. Non-retryable."""

    def __init__(self, message: str = "STT configuration error.", cause: Optional[Exception] = None):
        super().__init__(
            message=message,
            error_code="STT_CONFIGURATION_ERROR",
            retryable=False,
            cause=cause,
        )
