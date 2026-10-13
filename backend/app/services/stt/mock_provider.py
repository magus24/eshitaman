from pathlib import Path
from typing import Dict, List, Optional
from app.services.stt.base import BaseSTTProvider
from app.services.stt.models import (
    ChunkTranscriptionResult,
    TranscriptSegment,
    TranscriptWord,
)
from app.services.stt.exceptions import (
    STTAuthenticationError,
    STTProviderError,
    STTTimeoutError,
)

# Deterministic multilingual transcript responses showcasing Uzbek code-switching
DEFAULT_MOCK_UTTERANCES = [
    {
        "text": "Bugun machine learning haqida gaplashamiz.",
        "language": "uz",
        "segments": [
            {
                "start_ms": 1000,
                "end_ms": 4500,
                "text": "Bugun machine learning haqida gaplashamiz.",
                "language": "uz",
                "confidence": 0.94,
            }
        ],
    },
    {
        "text": "Bu API backend bilan bog'lanadi va ma'lumotlarni uzatadi.",
        "language": "uz",
        "segments": [
            {
                "start_ms": 800,
                "end_ms": 5200,
                "text": "Bu API backend bilan bog'lanadi va ma'lumotlarni uzatadi.",
                "language": "uz",
                "confidence": 0.92,
            }
        ],
    },
    {
        "text": "Model accuracy 95 percent bo'ldi, natijalar juda yaxshi.",
        "language": "uz",
        "segments": [
            {
                "start_ms": 1200,
                "end_ms": 5800,
                "text": "Model accuracy 95 percent bo'ldi, natijalar juda yaxshi.",
                "language": "uz",
                "confidence": 0.95,
            }
        ],
    },
    {
        "text": "Сейчас мы рассмотрим database architecture va microservices.",
        "language": "ru",
        "segments": [
            {
                "start_ms": 500,
                "end_ms": 4800,
                "text": "Сейчас мы рассмотрим database architecture va microservices.",
                "language": "ru",
                "confidence": 0.91,
            }
        ],
    },
    {
        "text": "Python'da FastAPI ishlatyapmiz, performance juda yuqori.",
        "language": "uz",
        "segments": [
            {
                "start_ms": 900,
                "end_ms": 5100,
                "text": "Python'da FastAPI ishlatyapmiz, performance juda yuqori.",
                "language": "uz",
                "confidence": 0.96,
            }
        ],
    },
]


class MockSTTProvider(BaseSTTProvider):
    """
    Deterministic mock STT provider for offline development and tests.
    Produces authentic Uzbek multilingual speech transcripts (Uzbek + Russian + English).
    Supports error injection for testing retries and cost-control recovery.
    """

    def __init__(
        self,
        custom_utterances: Optional[List[Dict]] = None,
        force_error: Optional[str] = None,
        fail_chunks: Optional[List[str]] = None,
        fail_times_per_chunk: int = 0,
    ):
        self._utterances = custom_utterances or DEFAULT_MOCK_UTTERANCES
        self._force_error = force_error
        self._fail_chunks = set(fail_chunks or [])
        self._fail_times_per_chunk = fail_times_per_chunk
        self._chunk_fail_counts: Dict[str, int] = {}
        self.call_count = 0

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return "mock-multilingual-v1"

    async def transcribe_chunk(
        self,
        audio_path: Path,
        chunk_id: str,
        language_hint: Optional[str] = None,
    ) -> ChunkTranscriptionResult:
        self.call_count += 1

        # Simulate injected global error
        if self._force_error == "auth":
            raise STTAuthenticationError("Invalid Mock API Key")
        elif self._force_error == "timeout":
            raise STTTimeoutError("Mock timeout exceeded")
        elif self._force_error == "server":
            raise STTProviderError("Mock 500 Internal Error")

        # Simulate per-chunk failure with recovery after N attempts
        if chunk_id in self._fail_chunks:
            current_fails = self._chunk_fail_counts.get(chunk_id, 0)
            if current_fails < self._fail_times_per_chunk:
                self._chunk_fail_counts[chunk_id] = current_fails + 1
                raise STTProviderError(f"Simulated transient error for {chunk_id}")

        # Derive index from chunk_id (e.g. chunk_rec_0001 -> 1)
        idx = 0
        try:
            parts = chunk_id.split("_")
            idx = int(parts[-1])
        except (ValueError, IndexError):
            idx = self.call_count - 1

        chosen = self._utterances[idx % len(self._utterances)]

        # Map to TranscriptSegment objects (chunk-local)
        segments: List[TranscriptSegment] = []
        for i, s in enumerate(chosen.get("segments", [])):
            seg_id = f"seg_{chunk_id}_{i:03d}"
            # Create word objects if words are present
            words = []
            text = s["text"]
            word_list = text.split()
            word_dur = (s["end_ms"] - s["start_ms"]) // max(1, len(word_list))
            for w_idx, w in enumerate(word_list):
                w_start = s["start_ms"] + w_idx * word_dur
                w_end = min(s["end_ms"], w_start + word_dur)
                words.append(
                    TranscriptWord(
                        word=w,
                        start_ms=w_start,
                        end_ms=w_end,
                        confidence=s.get("confidence", 0.9),
                    )
                )

            segments.append(
                TranscriptSegment(
                    id=seg_id,
                    start_ms=s["start_ms"],
                    end_ms=s["end_ms"],
                    text=s["text"],
                    language=s.get("language", chosen.get("language", "uz")),
                    confidence=s.get("confidence", 0.9),
                    chunk_id=chunk_id,
                    words=words,
                )
            )

        return ChunkTranscriptionResult(
            chunk_id=chunk_id,
            text=chosen["text"],
            language=chosen.get("language", "uz"),
            duration_ms=segments[-1].end_ms if segments else 5000,
            segments=segments,
        )
