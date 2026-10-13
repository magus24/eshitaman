from typing import Optional


class RAGError(Exception):
    """Base exception for all RAG-related operations."""

    def __init__(
        self,
        message: str,
        error_code: str = "RAG_ERROR",
        status_code: int = 500,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.cause = cause

    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"


class EmbeddingError(RAGError):
    """Base exception for embedding operations."""

    def __init__(
        self,
        message: str,
        error_code: str = "EMBEDDING_ERROR",
        status_code: int = 500,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, error_code, status_code, cause)


class EmbeddingAuthenticationError(EmbeddingError):
    """Authentication or authorization failure with embedding provider (401/403)."""

    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message, error_code="EMBEDDING_AUTH_FAILED", status_code=401, cause=cause)


class EmbeddingConfigurationError(EmbeddingError):
    """Configuration error (missing API key, invalid model, invalid dimensions)."""

    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message, error_code="EMBEDDING_CONFIG_ERROR", status_code=500, cause=cause)


class EmbeddingRateLimitError(EmbeddingError):
    """Rate limit exceeded with embedding provider (429)."""

    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message, error_code="EMBEDDING_RATE_LIMIT", status_code=429, cause=cause)


class EmbeddingTimeoutError(EmbeddingError):
    """Network or execution timeout during embedding request."""

    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message, error_code="EMBEDDING_TIMEOUT", status_code=504, cause=cause)


class EmbeddingNetworkError(EmbeddingError):
    """Low-level network or connection failure during embedding generation."""

    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message, error_code="EMBEDDING_NETWORK_ERROR", status_code=503, cause=cause)


class EmbeddingProviderError(EmbeddingError):
    """Embedding provider returned 5xx server error."""

    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message, error_code="EMBEDDING_PROVIDER_ERROR", status_code=502, cause=cause)


class VectorStoreError(RAGError):
    """Base exception for vector database / store operations."""

    def __init__(
        self,
        message: str,
        error_code: str = "VECTOR_STORE_ERROR",
        status_code: int = 500,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, error_code, status_code, cause)


class VectorStoreConnectionError(VectorStoreError):
    """Vector database connection failure or unreachable service."""

    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message, error_code="VECTOR_STORE_CONNECTION_FAILED", status_code=503, cause=cause)


class NoRelevantContextError(RAGError):
    """Raised when similarity search returns no chunks above the relevance threshold."""

    def __init__(
        self,
        message: str = "No sufficiently relevant context chunks found in recording",
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, error_code="NO_RELEVANT_CONTEXT", status_code=404, cause=cause)
