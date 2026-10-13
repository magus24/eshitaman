from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddingProvider(ABC):
    """Abstract base class for provider-agnostic text embedding generation."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the embedding provider (e.g. 'mock', 'openai')."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier (e.g. 'mock-embed-v1', 'text-embedding-3-small')."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of generated vectors (e.g. 1536)."""
        pass

    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """
        Generates an embedding vector for a single string.

        :param text: Text string to embed.
        :return: Normalized list of floats representing the vector.
        """
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embedding vectors for a batch of strings.

        :param texts: List of text strings to embed.
        :return: List of normalized vector floats.
        """
        pass
