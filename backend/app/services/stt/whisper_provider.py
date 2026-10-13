import asyncio
import logging
from pathlib import Path
from typing import Optional
import httpx
from app.core.config import settings
from app.services.stt.base import BaseSTTProvider
from app.services.stt.models import (
    ChunkTranscriptionResult,
    TranscriptSegment,
    TranscriptWord,
)
from app.services.stt.exceptions import (
    STTAuthenticationError,
    STTConfigurationError,
    STTNetworkError,
    STTProviderError,
    STTRateLimitError,
    STTTimeoutError,
)

logger = logging.getLogger(__name__)


class WhisperSTTProvider(BaseSTTProvider):
    """
    Real Speech-to-Text provider utilizing OpenAI Whisper (or OpenAI-compatible API).
    Provides native multilingual support for Uzbek, Russian, and English,
    preserving code-switching and returning segment and word-level timestamps.
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
        self._api_key = api_key or settings.STT_API_KEY
        self._base_url = (base_url or settings.STT_API_BASE_URL).rstrip("/")
        self._model = model or settings.STT_MODEL
        self._timeout = timeout_seconds or settings.STT_TIMEOUT_SECONDS
        self._max_retries = max_retries if max_retries is not None else settings.STT_MAX_RETRIES
        self._client = http_client

    @property
    def provider_name(self) -> str:
        return "whisper"

    @property
    def model_name(self) -> str:
        return self._model

    async def transcribe_chunk(
        self,
        audio_path: Path,
        chunk_id: str,
        language_hint: Optional[str] = None,
    ) -> ChunkTranscriptionResult:
        if not self._api_key:
            raise STTConfigurationError(
                "STT_API_KEY is not configured. Please set STT_API_KEY in your .env file."
            )

        if not audio_path.exists():
            raise STTProviderError(f"Audio chunk file does not exist: {audio_path}")

        url = f"{self._base_url}/audio/transcriptions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
        }

        # Prepare request payload
        # Note: Do not hardcode language unless explicitly provided as hint
        lang = language_hint or settings.STT_LANGUAGE_HINT

        # Retry loop for transient failures (timeouts, network, 429, 5xx)
        last_error = None
        for attempt in range(1, self._max_retries + 1):
            try:
                # Read audio bytes for multipart upload
                audio_bytes = audio_path.read_bytes()
                files = {
                    "file": (audio_path.name, audio_bytes, "audio/wav"),
                }
                data = {
                    "model": self._model,
                    "response_format": "verbose_json",
                    "timestamp_granularities[]": ["segment", "word"],
                }
                if lang:
                    data["language"] = lang

                # Use injected client if present (e.g. in tests) or create local client
                if self._client is not None:
                    response = await self._client.post(
                        url,
                        headers=headers,
                        files=files,
                        data=data,
                        timeout=self._timeout,
                    )
                else:
                    async with httpx.AsyncClient(timeout=self._timeout) as client:
                        response = await client.post(
                            url,
                            headers=headers,
                            files=files,
                            data=data,
                        )

                # 1. Permanent Authentication Failure (401/403) - DO NOT RETRY
                if response.status_code in (401, 403):
                    raise STTAuthenticationError(
                        f"Whisper API authentication failed ({response.status_code}): {response.text}"
                    )

                # 2. Rate Limit (429) - Retryable
                if response.status_code == 429:
                    raise STTRateLimitError(
                        f"Whisper API rate limit reached (Attempt {attempt}/{self._max_retries})"
                    )

                # 3. Client Error (4xx other than 401/403/429) - Permanent, DO NOT RETRY
                if 400 <= response.status_code < 500:
                    raise STTProviderError(
                        f"Whisper API client error ({response.status_code}): {response.text}",
                        retryable=False,
                    )

                # 4. Server Error (5xx) - Retryable
                if response.status_code >= 500:
                    raise STTProviderError(
                        f"Whisper API server error ({response.status_code}, Attempt {attempt}/{self._max_retries})"
                    )

                # Parse JSON output
                res_data = response.json()
                return self._parse_whisper_response(res_data, chunk_id)

            except (STTAuthenticationError,):
                # Non-retryable
                raise
            except STTProviderError as e:
                if not e.retryable:
                    raise
                last_error = e
            except STTRateLimitError as e:
                last_error = e
            except httpx.TimeoutException as e:
                last_error = STTTimeoutError(
                    f"Whisper API request timed out after {self._timeout}s (Attempt {attempt}/{self._max_retries})",
                    cause=e,
                )
            except httpx.RequestError as e:
                last_error = STTNetworkError(
                    f"Whisper API network connection error (Attempt {attempt}/{self._max_retries}): {e}",
                    cause=e,
                )
            except Exception as e:
                last_error = STTProviderError(f"Unexpected Whisper API error: {e}", cause=e)

            # Exponential backoff: 0.5s, 1.0s, 2.0s
            if attempt < self._max_retries:
                backoff = 0.5 * (2 ** (attempt - 1))
                logger.warning(f"Whisper STT attempt {attempt} failed ({last_error}). Retrying in {backoff}s...")
                await asyncio.sleep(backoff)

        raise last_error or STTProviderError("Whisper transcription failed after maximum retries.")

    def _parse_whisper_response(self, data: dict, chunk_id: str) -> ChunkTranscriptionResult:
        """Parses OpenAI Whisper verbose_json structure into typed chunk transcription result."""
        full_text = data.get("text", "").strip()
        detected_language = data.get("language", "uz")
        raw_duration = float(data.get("duration", 0.0))
        duration_ms = int(raw_duration * 1000)

        # Parse segment-level timestamps
        raw_segments = data.get("segments", [])
        segments: list[TranscriptSegment] = []

        for idx, seg in enumerate(raw_segments):
            seg_id = f"seg_{chunk_id}_{idx:03d}"
            start_ms = int(float(seg.get("start", 0.0)) * 1000)
            end_ms = int(float(seg.get("end", 0.0)) * 1000)
            text = seg.get("text", "").strip()
            # If segment provides confidence (avg_logprob or no_speech_prob)
            avg_logprob = seg.get("avg_logprob")
            confidence = round(pow(2.71828, avg_logprob), 3) if avg_logprob is not None else None

            # Parse word-level timestamps if available
            words: list[TranscriptWord] = []
            for w in seg.get("words", []):
                w_start = int(float(w.get("start", 0.0)) * 1000)
                w_end = int(float(w.get("end", 0.0)) * 1000)
                words.append(
                    TranscriptWord(
                        word=w.get("word", ""),
                        start_ms=w_start,
                        end_ms=w_end,
                        confidence=w.get("probability"),
                    )
                )

            segments.append(
                TranscriptSegment(
                    id=seg_id,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    text=text,
                    language=seg.get("language", detected_language),
                    confidence=confidence,
                    chunk_id=chunk_id,
                    words=words,
                )
            )

        # If Whisper returned full text but 0 segments, construct a fallback segment
        if not segments and full_text:
            segments.append(
                TranscriptSegment(
                    id=f"seg_{chunk_id}_000",
                    start_ms=0,
                    end_ms=duration_ms,
                    text=full_text,
                    language=detected_language,
                    chunk_id=chunk_id,
                )
            )

        return ChunkTranscriptionResult(
            chunk_id=chunk_id,
            text=full_text,
            language=detected_language,
            duration_ms=duration_ms,
            segments=segments,
        )
