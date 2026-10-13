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
    LLMNetworkError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)

logger = logging.getLogger(__name__)


class GroqLLMProvider(BaseLLMProvider):
    """
    Groq Cloud LLM provider implementing fast OpenAI-compatible Chat Completions.
    Designed for Competition MVP (Stage 10) high-speed inference on LLaMA-3 models.
    Supports structured JSON extraction, multilingual Uzbek/Russian/English,
    and grounded RAG synthesis with strict safety fallbacks.
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
        self._api_key = api_key if api_key is not None else (settings.GROQ_API_KEY or settings.LLM_API_KEY)
        self._base_url = (base_url or settings.GROQ_BASE_URL or "https://api.groq.com/openai/v1").rstrip("/")
        self._model = model or settings.GROQ_MODEL or "llama-3.3-70b-versatile"
        self._timeout = timeout_seconds or settings.GROQ_TIMEOUT_SECONDS or 60.0
        self._max_retries = max_retries if max_retries is not None else settings.GROQ_MAX_RETRIES or 3
        self._client = http_client

    @property
    def provider_name(self) -> str:
        return "groq"

    @property
    def model_name(self) -> str:
        return self._model

    async def analyze_transcript(
        self,
        transcript: Transcript,
        context_instructions: Optional[str] = None,
    ) -> AIAnalysis:
        """
        Extracts structured AIAnalysis (summary, key_points, explanation, terms, questions)
        from a lecture transcript using Groq's JSON mode chat completions.
        """
        if not self._api_key or self._api_key.strip() in ("", "your-llm-api-key-here"):
            raise LLMConfigurationError(
                "GROQ_API_KEY is not configured. Please set GROQ_API_KEY in your .env file or environment."
            )

        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        user_content = build_user_prompt(transcript)
        if context_instructions:
            user_content += f"\n\nQo'shimcha ko'rsatma: {context_instructions}"

        # Groq OpenAI-compatible Chat Completions payload with JSON mode
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

                # 1. Authentication Failure (401/403) - DO NOT RETRY
                if response.status_code in (401, 403):
                    masked_status = response.status_code
                    raise LLMAuthenticationError(
                        f"Groq API authentication failed ({masked_status}). Please verify your GROQ_API_KEY."
                    )

                # 2. Rate Limit (429) - Retryable with backoff
                if response.status_code == 429:
                    raise LLMRateLimitError(
                        f"Groq API rate limit exceeded (Attempt {attempt}/{self._max_retries})"
                    )

                # 3. Context Length Exceeded (400)
                if response.status_code == 400 and "context_length_exceeded" in response.text.lower():
                    raise LLMContextLengthError(
                        f"Transcript exceeded Groq model context length ({self._model})."
                    )

                # 4. Client Error (4xx) - Permanent, DO NOT RETRY
                if 400 <= response.status_code < 500:
                    raise LLMProviderError(
                        f"Groq API client error ({response.status_code}): {response.text}",
                        retryable=False,
                    )

                # 5. Server Error (5xx) - Retryable
                if response.status_code >= 500:
                    raise LLMProviderError(
                        f"Groq API server error ({response.status_code}, Attempt {attempt}/{self._max_retries})",
                        retryable=True,
                    )

                res_json = response.json()
                choices = res_json.get("choices", [])
                if not choices:
                    raise LLMProviderError("Groq response contained no choices.")

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
                    f"Groq API request timed out after {self._timeout}s (Attempt {attempt}/{self._max_retries})",
                    cause=e,
                )
            except httpx.RequestError as e:
                last_error = LLMNetworkError(
                    f"Groq API network connection error (Attempt {attempt}/{self._max_retries}): {e}",
                    cause=e,
                )
            except Exception as e:
                last_error = LLMProviderError(f"Unexpected Groq API error: {e}", cause=e)

            # Exponential backoff: 0.5s, 1.0s, 2.0s
            if attempt < self._max_retries:
                backoff = 0.5 * (2 ** (attempt - 1))
                logger.warning(f"Groq attempt {attempt} failed ({last_error}). Retrying in {backoff}s...")
                await asyncio.sleep(backoff)

        raise last_error or LLMProviderError("Groq analysis failed after maximum retries.")

    async def generate_rag_answer(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generates a grounded natural language answer given a RAG prompt containing lecture context and user question.
        """
        if not self._api_key or self._api_key.strip() in ("", "your-llm-api-key-here"):
            raise LLMConfigurationError(
                "GROQ_API_KEY is not configured. Please set GROQ_API_KEY in your .env file or environment."
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
                        f"Groq API authentication failed ({response.status_code}). Please verify your GROQ_API_KEY."
                    )

                if response.status_code == 429:
                    raise LLMRateLimitError(
                        f"Groq API rate limit exceeded (Attempt {attempt}/{self._max_retries})"
                    )

                if response.status_code == 400 and "context_length_exceeded" in response.text.lower():
                    raise LLMContextLengthError(f"Prompt exceeded Groq model context length: {response.text}")

                if 400 <= response.status_code < 500:
                    raise LLMProviderError(
                        f"Groq API client error ({response.status_code}): {response.text}",
                        retryable=False,
                    )

                if response.status_code >= 500:
                    raise LLMProviderError(
                        f"Groq API server error ({response.status_code}): {response.text}",
                        retryable=True,
                    )

                res_json = response.json()
                choices = res_json.get("choices", [])
                if not choices:
                    raise LLMProviderError("Empty choices returned by Groq API")

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
                    f"Groq API request timed out after {self._timeout}s (Attempt {attempt}/{self._max_retries})",
                    cause=e,
                )
            except httpx.RequestError as e:
                last_error = LLMNetworkError(
                    f"Groq API network connection error (Attempt {attempt}/{self._max_retries}): {e}",
                    cause=e,
                )
            except Exception as e:
                last_error = LLMProviderError(f"Unexpected Groq API error: {e}", cause=e)

            if attempt < self._max_retries:
                backoff = 0.5 * (2 ** (attempt - 1))
                logger.warning(f"Groq RAG attempt {attempt} failed ({last_error}). Retrying in {backoff}s...")
                await asyncio.sleep(backoff)

        raise last_error or LLMProviderError("Groq RAG generation failed after maximum retries.")
