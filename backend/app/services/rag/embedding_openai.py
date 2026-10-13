import asyncio
import logging
from typing import List, Optional
import httpx
from app.core.config import settings
from app.services.rag.embedding_base import BaseEmbeddingProvider
from app.services.rag.exceptions import (
    EmbeddingAuthenticationError,
    EmbeddingConfigurationError,
    EmbeddingNetworkError,
    EmbeddingProviderError,
    EmbeddingRateLimitError,
    EmbeddingTimeoutError,
)

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """
    Production embedding provider for OpenAI and OpenAI-compatible embedding endpoints
    (e.g., text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002).
    Includes automatic retries with exponential backoff for transient failures and rate limits.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        dimension: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
        max_retries: Optional[int] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        self._api_key = api_key or settings.EMBEDDING_API_KEY
        self._base_url = (base_url or settings.EMBEDDING_API_BASE_URL).rstrip("/")
        self._model = model or settings.EMBEDDING_MODEL
        self._dimension = dimension or settings.EMBEDDING_DIMENSION
        self._timeout = timeout_seconds or settings.EMBEDDING_TIMEOUT_SECONDS
        self._max_retries = max_retries if max_retries is not None else settings.EMBEDDING_MAX_RETRIES
        self._client = http_client

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    async def _send_request(self, inputs: List[str]) -> List[List[float]]:
        if not self._api_key:
            raise EmbeddingConfigurationError(
                "EMBEDDING_API_KEY is not configured. Please set EMBEDDING_API_KEY or OPENAI_API_KEY."
            )

        url = f"{self._base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "input": inputs,
        }

        last_error = None
        for attempt in range(1, self._max_retries + 1):
            try:
                if self._client is not None:
                    response = await self._client.post(
                        url,
                        headers=headers,
                        json=payload,
                        timeout=self._timeout,
                    )
                else:
                    async with httpx.AsyncClient(timeout=self._timeout) as client:
                        response = await client.post(
                            url,
                            headers=headers,
                            json=payload,
                        )

                if response.status_code in (401, 403):
                    raise EmbeddingAuthenticationError(
                        f"Embedding API authentication failed ({response.status_code}): {response.text}"
                    )

                if response.status_code == 429:
                    raise EmbeddingRateLimitError(
                        f"Embedding API rate limit exceeded (Attempt {attempt}/{self._max_retries})"
                    )

                if response.status_code >= 500:
                    raise EmbeddingProviderError(
                        f"Embedding API server error ({response.status_code}): {response.text}"
                    )

                if response.status_code != 200:
                    raise EmbeddingProviderError(
                        f"Embedding API returned unexpected status {response.status_code}: {response.text}"
                    )

                data = response.json()
                items = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
                vectors = [item["embedding"] for item in items]
                if not vectors or len(vectors) != len(inputs):
                    raise EmbeddingProviderError(
                        f"Mismatch between inputs count ({len(inputs)}) and returned embeddings ({len(vectors)})"
                    )
                return vectors

            except (EmbeddingAuthenticationError, EmbeddingConfigurationError):
                raise
            except EmbeddingRateLimitError as e:
                last_error = e
                if attempt < self._max_retries:
                    await asyncio.sleep(0.5 * (2 ** (attempt - 1)))
                else:
                    raise
            except (httpx.TimeoutException, asyncio.TimeoutError) as e:
                last_error = EmbeddingTimeoutError(f"Embedding request timed out: {e}", cause=e)
                if attempt < self._max_retries:
                    await asyncio.sleep(0.5 * (2 ** (attempt - 1)))
                else:
                    raise last_error
            except (httpx.NetworkError, httpx.ConnectError) as e:
                last_error = EmbeddingNetworkError(f"Embedding network error: {e}", cause=e)
                if attempt < self._max_retries:
                    await asyncio.sleep(0.5 * (2 ** (attempt - 1)))
                else:
                    raise last_error
            except EmbeddingProviderError as e:
                last_error = e
                if attempt < self._max_retries:
                    await asyncio.sleep(0.5 * (2 ** (attempt - 1)))
                else:
                    raise
            except Exception as e:
                raise EmbeddingProviderError(f"Unexpected error in OpenAI embedding request: {e}", cause=e)

        if last_error:
            raise last_error
        raise EmbeddingProviderError("Failed to generate embeddings after max retries")

    async def embed_text(self, text: str) -> List[float]:
        res = await self._send_request([text])
        return res[0]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return await self._send_request(texts)
