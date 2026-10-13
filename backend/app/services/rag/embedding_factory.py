from typing import Optional
from app.core.config import settings
from app.services.rag.embedding_base import BaseEmbeddingProvider
from app.services.rag.embedding_mock import MockEmbeddingProvider
from app.services.rag.embedding_openai import OpenAIEmbeddingProvider
from app.services.rag.exceptions import EmbeddingConfigurationError


class EmbeddingProviderFactory:
    """Factory to instantiate embedding providers based on environment configuration."""

    @staticmethod
    def create_provider(provider_type: Optional[str] = None) -> BaseEmbeddingProvider:
        provider = (provider_type or settings.EMBEDDING_PROVIDER).lower()

        if provider == "mock":
            return MockEmbeddingProvider(
                dimension=settings.EMBEDDING_DIMENSION,
                model_name=settings.EMBEDDING_MODEL,
            )
        elif provider in ("openai", "openai-compatible"):
            return OpenAIEmbeddingProvider(
                api_key=settings.EMBEDDING_API_KEY,
                base_url=settings.EMBEDDING_API_BASE_URL,
                model=settings.EMBEDDING_MODEL,
                dimension=settings.EMBEDDING_DIMENSION,
                timeout_seconds=settings.EMBEDDING_TIMEOUT_SECONDS,
                max_retries=settings.EMBEDDING_MAX_RETRIES,
            )
        else:
            raise EmbeddingConfigurationError(
                f"Unknown EMBEDDING_PROVIDER '{provider}'. Supported providers: 'mock', 'openai'."
            )
