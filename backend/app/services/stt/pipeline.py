import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from app.core.config import settings
from app.schemas.recording import RecordingStatus
from app.schemas.processing import AudioChunk
from app.services.storage.base import BaseStorage
from app.services.stt.base import BaseSTTProvider
from app.services.stt.factory import STTProviderFactory
from app.services.stt.merger import TranscriptMerger
from app.services.stt.models import (
    ChunkTranscriptionResult,
    Transcript,
)
from app.services.stt.exceptions import (
    STTError,
)


class TranscriptionPipeline:
    """
    Orchestrates end-to-end speech-to-text pipeline:
    1. Validates processing readiness (chunks exist)
    2. Enforces idempotency (reuses existing transcript)
    3. Employs chunk-level cost control (does not re-transcribe already completed chunks)
    4. Invokes STT provider with retry capability
    5. Converts chunk timestamps to global recording timestamps and deduplicates overlaps
    6. Persists transcript/transcript.json and updates metadata
    """

    def __init__(
        self,
        storage: BaseStorage,
        provider: Optional[BaseSTTProvider] = None,
        merger: Optional[TranscriptMerger] = None,
    ):
        self.storage = storage
        self.provider = provider or STTProviderFactory.create_provider()
        self.merger = merger or TranscriptMerger()

    async def transcribe_recording(
        self,
        recording_id: str,
        language_hint: Optional[str] = None,
    ) -> Transcript:
        # 1. Check recording metadata
        meta = await self.storage.get_recording_metadata(recording_id)
        if not meta:
            raise STTError(
                f"Recording '{recording_id}' not found",
                error_code="RECORDING_NOT_FOUND",
            )

        audio_path = await self.storage.get_audio_path(recording_id)
        session_dir = (
            self.storage._get_session_dir(recording_id)
            if hasattr(self.storage, "_get_session_dir")
            else (audio_path.parent if audio_path else None)
        )
        if not session_dir or not session_dir.exists():
            raise STTError(
                f"Storage directory for recording '{recording_id}' not found",
                error_code="RECORDING_NOT_FOUND",
            )


        transcript_dir = session_dir / "transcript"
        transcript_path = transcript_dir / "transcript.json"

        # 2. Idempotency check: if already transcribed and transcript exists on disk
        if meta.get("status") == RecordingStatus.TRANSCRIBED.value and transcript_path.exists():
            try:
                with open(transcript_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return Transcript(**data)
            except Exception:
                pass  # If transcript.json is corrupted, fall through to re-transcribe

        # 3. Check processing status
        status = meta.get("status")
        if status not in (
            RecordingStatus.PROCESSED.value,
            RecordingStatus.TRANSCRIBING.value,
            RecordingStatus.TRANSCRIBED.value,
        ):
            raise STTError(
                f"Recording '{recording_id}' is in '{status}' status. It must be in 'processed' status before transcription.",
                error_code="RECORDING_NOT_PROCESSED",
            )

        # 4. Load chunks metadata
        processed_dir = session_dir / "processed"
        chunks_json_path = processed_dir / "chunks.json"
        if not chunks_json_path.exists():
            raise STTError(
                f"Processed chunks metadata not found for recording '{recording_id}'",
                error_code="CHUNKS_NOT_FOUND",
            )

        try:
            with open(chunks_json_path, "r", encoding="utf-8") as f:
                chunks_raw = json.load(f)
            chunks: List[AudioChunk] = [AudioChunk(**c) for c in chunks_raw]
        except Exception as e:
            raise STTError(f"Failed to parse chunks.json: {e}", error_code="CHUNKS_PARSE_ERROR", cause=e)

        if not chunks:
            raise STTError(
                f"Recording '{recording_id}' has 0 audio chunks",
                error_code="CHUNKS_EMPTY",
            )

        # 5. Mark status as transcribing
        meta["status"] = RecordingStatus.TRANSCRIBING.value
        await self.storage.update_recording_metadata(recording_id, meta)

        transcript_dir.mkdir(parents=True, exist_ok=True)
        chunk_cache_path = transcript_dir / "chunk_transcripts.json"

        # 6. Load cached chunk transcriptions (Cost Control)
        cached_chunk_results: Dict[str, dict] = {}
        if chunk_cache_path.exists():
            try:
                with open(chunk_cache_path, "r", encoding="utf-8") as f:
                    cached_chunk_results = json.load(f)
            except Exception:
                cached_chunk_results = {}

        try:
            # 7. Transcribe each chunk (skipping already completed chunks)
            chunk_items: List[Tuple[AudioChunk, ChunkTranscriptionResult]] = []

            for chunk in chunks:
                chunk_file = Path(chunk.file_path)
                if not chunk_file.exists():
                    # Fallback to local relative resolution inside session dir
                    chunk_file = processed_dir / "chunks" / Path(chunk.file_path).name

                if not chunk_file.exists():
                    raise STTError(
                        f"Audio chunk file '{chunk.file_path}' does not exist on disk",
                        error_code="CHUNK_FILE_MISSING",
                    )

                # Cost control check: reuse cached chunk result if available
                if chunk.chunk_id in cached_chunk_results:
                    c_res = ChunkTranscriptionResult(**cached_chunk_results[chunk.chunk_id])
                else:
                    c_res = await self.provider.transcribe_chunk(
                        audio_path=chunk_file,
                        chunk_id=chunk.chunk_id,
                        language_hint=language_hint,
                    )
                    # Cache chunk result incrementally to protect against partial pipeline failures
                    cached_chunk_results[chunk.chunk_id] = c_res.model_dump()
                    with open(chunk_cache_path, "w", encoding="utf-8") as f:
                        json.dump(cached_chunk_results, f, indent=2, ensure_ascii=False)

                chunk_items.append((chunk, c_res))

            # 8. Merge and translate timestamps to global recording timeline
            raw_dur = meta.get("duration_seconds")
            if raw_dur is None and isinstance(meta.get("audio"), dict):
                raw_dur = meta["audio"].get("duration_seconds")
            dur_sec = float(raw_dur) if raw_dur is not None else 0.0
            total_duration_ms = int(dur_sec * 1000)
            if total_duration_ms == 0:
                total_duration_ms = chunks[-1].end_ms if chunks else 0

            transcript = self.merger.merge(
                recording_id=recording_id,
                chunk_items=chunk_items,
                provider_name=self.provider.provider_name,
                model_name=self.provider.model_name,
                total_duration_ms=total_duration_ms,
            )

            # 9. Save full transcript to transcript/transcript.json
            with open(transcript_path, "w", encoding="utf-8") as f:
                json.dump(transcript.model_dump(), f, indent=2, ensure_ascii=False)

            # 10. Update recording metadata to transcribed
            meta["status"] = RecordingStatus.TRANSCRIBED.value
            meta["error_code"] = None
            meta["error_message"] = None
            meta["transcript"] = {
                "language": transcript.language,
                "segments_count": len(transcript.segments),
                "duration_ms": transcript.duration_ms,
            }
            await self.storage.update_recording_metadata(recording_id, meta)

            return transcript

        except STTError as e:
            meta["status"] = RecordingStatus.FAILED.value
            meta["error_code"] = e.error_code
            meta["error_message"] = e.message
            await self.storage.update_recording_metadata(recording_id, meta)
            raise
        except Exception as e:
            meta["status"] = RecordingStatus.FAILED.value
            meta["error_code"] = "TRANSCRIPTION_FAILED"
            meta["error_message"] = str(e)
            await self.storage.update_recording_metadata(recording_id, meta)
            raise STTError(f"Transcription failed: {e}", cause=e)

    async def get_transcript(self, recording_id: str) -> Optional[Transcript]:
        """Retrieves the full transcript.json for a recording."""
        session_dir = (
            self.storage._get_session_dir(recording_id)
            if hasattr(self.storage, "_get_session_dir")
            else None
        )
        if not session_dir or not session_dir.exists():
            audio_path = await self.storage.get_audio_path(recording_id)
            session_dir = audio_path.parent if audio_path else None

        if not session_dir or not session_dir.exists():
            return None

        transcript_path = session_dir / "transcript" / "transcript.json"
        if not transcript_path.exists():
            return None

        with open(transcript_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return Transcript(**data)

