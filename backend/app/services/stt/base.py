from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
from app.services.stt.models import ChunkTranscriptionResult


class BaseSTTProvider(ABC):
    """Abstract base class for provider-agnostic Speech-to-Text inference."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the STT provider (e.g. 'mock', 'whisper', 'deepgram')."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the underlying speech model (e.g. 'mock-model', 'whisper-1')."""
        pass

    @abstractmethod
    async def transcribe_chunk(
        self,
        audio_path: Path,
        chunk_id: str,
        language_hint: Optional[str] = None,
    ) -> ChunkTranscriptionResult:
        """
        Transcribes a single audio chunk.
        
        :param audio_path: Path to the canonical WAV audio chunk.
        :param chunk_id: Unique identifier of the chunk.
        :param language_hint: Optional language code (e.g. 'uz', 'ru', 'en').
                              If None, provider should automatically detect language
                              or perform multilingual transcription.
        :return: ChunkTranscriptionResult containing text, language, and chunk-local segments.
        """
        pass
