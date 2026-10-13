import asyncio
import logging
from typing import Optional
import httpx
from app.core.config import settings
from app.services.stt.models import Transcript
from app.services.llm.base import BaseLLMProvider
from app.services.llm.models import AIAnalysis
from app.services.llm.parser import LLMResponseParser
from app.services.llm.prompts import SYSTEM_PROMPT, build_user_prompt
from app.services.llm.exceptions import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMContextLengthError,
    LLMError,
    LLMNetworkError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)

logger = logging.getLogger(__name__)


class OpenAILLMProvider(BaseLLMProvider):
    """
    Production-ready LLM provider supporting OpenAI and any OpenAI-compatible Chat Completions API.
    Handles multilingual Uzbek/Russian/English comprehension, technical code-switching preservation,
    structured JSON output enforcement, and exponential backoff retry for transient errors.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        max_retries: Optional[int] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        self._api_key = api_key or settings.LLM_API_KEY
        self._base_url = (base_url or settings.LLM_API_BASE_URL).rstrip("/")
        self._model = model or settings.LLM_MODEL
        self._timeout = timeout_seconds or settings.LLM_TIMEOUT_SECONDS
        self._max_retries = max_retries if max_retries is not None else settings.LLM_MAX_RETRIES
        self._client = http_client

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model

    async def analyze_transcript(
        self,
        transcript: Transcript,
        context_instructions: Optional[str] = None,
    ) -> AIAnalysis:
        if not self._api_key:
            raise LLMConfigurationError(
                "LLM_API_KEY is not configured. Please set LLM_API_KEY in your .env file."
            )

        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        user_content = build_user_prompt(transcript)
        if context_instructions:
            user_content += f"\n\nQo'shimcha ko'rsatma: {context_instructions}"

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
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

                # 1. Permanent Authentication Failure (401/403) - DO NOT RETRY
                if response.status_code in (401, 403):
                    raise LLMAuthenticationError(
                        f"LLM API authentication failed ({response.status_code}): {response.text}"
                    )

                # 2. Rate Limit (429) - Retryable
                if response.status_code == 429:
                    raise LLMRateLimitError(
                        f"LLM API rate limit exceeded (Attempt {attempt}/{self._max_retries})"
                    )

                # 3. Context Length Exceeded (400 with specific code)
                if response.status_code == 400 and "context_length_exceeded" in response.text.lower():
                    raise LLMContextLengthError(f"Transcript exceeded model context length: {response.text}")

                # 4. Client Error (4xx) - Permanent, DO NOT RETRY
                if 400 <= response.status_code < 500:
                    raise LLMProviderError(
                        f"LLM API client error ({response.status_code}): {response.text}",
                        retryable=False,
                    )

                # 5. Server Error (5xx) - Retryable
                if response.status_code >= 500:
                    raise LLMProviderError(
                        f"LLM API server error ({response.status_code}, Attempt {attempt}/{self._max_retries})"
                    )

                res_json = response.json()
                choices = res_json.get("choices", [])
                if not choices:
                    raise LLMProviderError("LLM response contained no choices.")

                raw_message_text = choices[0].get("message", {}).get("content", "")

                return LLMResponseParser.parse_analysis_json(
                    raw_text=raw_message_text,
                    recording_id=transcript.recording_id,
                    provider_name=self.provider_name,
                    model_name=self.model_name,
                    language=transcript.language or "uz",
                )

            except (LLMAuthenticationError, LLMContextLengthError):
                raise
            except LLMProviderError as e:
                if not e.retryable:
                    raise
                last_error = e
            except LLMRateLimitError as e:
                last_error = e
            except httpx.TimeoutException as e:
                last_error = LLMTimeoutError(
                    f"LLM API request timed out after {self._timeout}s (Attempt {attempt}/{self._max_retries})",
                    cause=e,
                )
            except httpx.RequestError as e:
                last_error = LLMNetworkError(
                    f"LLM API network connection error (Attempt {attempt}/{self._max_retries}): {e}",
                    cause=e,
                )
            except Exception as e:
                last_error = LLMProviderError(f"Unexpected LLM API error: {e}", cause=e)

            # Exponential backoff: 0.5s, 1.0s, 2.0s
            if attempt < self._max_retries:
                backoff = 0.5 * (2 ** (attempt - 1))
                logger.warning(f"LLM attempt {attempt} failed ({last_error}). Retrying in {backoff}s...")
                await asyncio.sleep(backoff)

        raise last_error or LLMProviderError("LLM analysis failed after maximum retries.")

    async def generate_rag_answer(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        if not self._api_key:
            raise LLMConfigurationError(
                "LLM_API_KEY is not configured. Please set LLM_API_KEY in your .env file."
            )

        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        sys_content = system_prompt or (
            "You are a helpful and strict educational AI assistant. "
            "Answer strictly based on the provided context."
        )

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": sys_content},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
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
                    raise LLMAuthenticationError(
                        f"LLM API authentication failed ({response.status_code}): {response.text}"
                    )

                if response.status_code == 429:
                    raise LLMRateLimitError(
                        f"LLM API rate limit exceeded (Attempt {attempt}/{self._max_retries})"
                    )

                if response.status_code == 400 and "context_length_exceeded" in response.text.lower():
                    raise LLMContextLengthError(f"Prompt exceeded model context length: {response.text}")

                if 400 <= response.status_code < 500:
                    raise LLMProviderError(
                        f"LLM API client error ({response.status_code}): {response.text}",
                        retryable=False,
                    )

                if response.status_code >= 500:
                    raise LLMProviderError(
                        f"LLM API server error ({response.status_code}): {response.text}",
                        retryable=True,
                    )

                res_json = response.json()
                choices = res_json.get("choices", [])
                if not choices:
                    raise LLMProviderError("Empty choices returned by LLM API")

                content = choices[0].get("message", {}).get("content", "").strip()
                return content

            except (LLMAuthenticationError, LLMContextLengthError):
                raise
            except LLMProviderError as e:
                if not e.retryable:
                    raise
                last_error = e
            except LLMRateLimitError as e:
                last_error = e
            except httpx.TimeoutException as e:
                last_error = LLMTimeoutError(
                    f"LLM API request timed out after {self._timeout}s (Attempt {attempt}/{self._max_retries})",
                    cause=e,
                )
            except httpx.RequestError as e:
                last_error = LLMNetworkError(
                    f"LLM API network connection error (Attempt {attempt}/{self._max_retries}): {e}",
                    cause=e,
                )
            except Exception as e:
                last_error = LLMProviderError(f"Unexpected LLM API error: {e}", cause=e)

            if attempt < self._max_retries:
                backoff = 0.5 * (2 ** (attempt - 1))
                logger.warning(f"LLM RAG attempt {attempt} failed ({last_error}). Retrying in {backoff}s...")
                await asyncio.sleep(backoff)

        raise last_error or LLMProviderError("LLM RAG generation failed after maximum retries.")
