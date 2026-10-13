import hashlib
import math
import re
from typing import List, Optional
from app.services.rag.embedding_base import BaseEmbeddingProvider
from app.services.rag.exceptions import (
    EmbeddingAuthenticationError,
    EmbeddingProviderError,
    EmbeddingRateLimitError,
    EmbeddingTimeoutError,
)

WORD_REGEX = re.compile(r"[\w']+", re.UNICODE)


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """
    Deterministic offline embedding provider for development and testing.
    Uses hash-based random projection with L2 normalization:
    - Same text always yields the identical vector.
    - Texts sharing vocabulary or n-grams yield higher cosine similarity.
    - Unrelated texts yield near-zero cosine similarity.
    - Supports simulated transient and permanent errors for test coverage.
    """

    def __init__(
        self,
        dimension: int = 1536,
        model_name: str = "mock-embedding-v1",
        force_error: Optional[str] = None,
        fail_times: int = 0,
    ):
        self._dimension = dimension
        self._model = model_name
        self._force_error = force_error
        self._fail_times = fail_times
        self._failed_count = 0
        self.call_count = 0

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    def _check_injected_error(self) -> None:
        if self._force_error == "auth":
            raise EmbeddingAuthenticationError("Mock embedding API key rejected (401)")
        elif self._force_error == "timeout":
            raise EmbeddingTimeoutError("Mock embedding request timed out")
        elif self._force_error == "rate_limit":
            if self._failed_count < self._fail_times:
                self._failed_count += 1
                raise EmbeddingRateLimitError(f"Mock embedding rate limit (429) hit (Attempt {self._failed_count})")
        elif self._force_error == "server":
            if self._failed_count < self._fail_times:
                self._failed_count += 1
                raise EmbeddingProviderError("Mock embedding internal server error (500)")

    def _generate_vector(self, text: str) -> List[float]:
        """Generates a deterministic pseudo-semantic normalized vector for input text."""
        vec = [0.0] * self._dimension
        words = WORD_REGEX.findall(text.lower())

        if not words:
            # Deterministic non-zero unit vector for empty strings
            vec[0] = 1.0
            return vec

        for word in words:
            # Hash word into bucket indices and signs
            h_int = int(hashlib.sha256(word.encode("utf-8")).hexdigest()[:8], 16)
            idx1 = h_int % self._dimension
            idx2 = (h_int // self._dimension) % self._dimension
            sign = 1.0 if (h_int & 1) else -1.0

            vec[idx1] += sign * 1.0
            vec[idx2] += (1.0 if (h_int & 2) else -1.0) * 0.5

            # Also add character 3-grams for fuzzy/multilingual matching
            if len(word) >= 3:
                for i in range(len(word) - 2):
                    tri = word[i : i + 3]
                    tri_hash = int(hashlib.md5(tri.encode("utf-8")).hexdigest()[:6], 16)
                    tri_idx = tri_hash % self._dimension
                    vec[tri_idx] += 0.25

        # Normalize with L2 norm
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            return [x / norm for x in vec]
        return [1.0 / math.sqrt(self._dimension)] * self._dimension

    async def embed_text(self, text: str) -> List[float]:
        self.call_count += 1
        self._check_injected_error()
        return self._generate_vector(text)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        self.call_count += 1
        self._check_injected_error()
        return [self._generate_vector(t) for t in texts]
