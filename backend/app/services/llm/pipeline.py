import json
import logging
from pathlib import Path
from typing import Optional
from app.core.config import settings
from app.schemas.recording import RecordingStatus
from app.services.storage.base import BaseStorage
from app.services.stt.models import Transcript
from app.services.llm.base import BaseLLMProvider
from app.services.llm.chunker import TranscriptChunker
from app.services.llm.factory import LLMProviderFactory
from app.services.llm.models import AIAnalysis, AITerm
from app.services.llm.exceptions import LLMError

logger = logging.getLogger(__name__)


class AIUnderstandingPipeline:
    """
    Orchestrates the AI understanding / LLM transcript analysis pipeline:
    1. Validates recording and transcript presence
    2. Enforces idempotency (reuses existing analysis.json)
    3. Handles context window chunking for long lectures
    4. Executes LLM analysis with retry capability
    5. Persists analysis/analysis.json and updates session metadata
    """

    def __init__(
        self,
        storage: BaseStorage,
        provider: Optional[BaseLLMProvider] = None,
        chunker: Optional[TranscriptChunker] = None,
    ):
        self.storage = storage
        self.provider = provider or LLMProviderFactory.create_provider()
        self.chunker = chunker or TranscriptChunker(max_chars_per_window=settings.LLM_MAX_CHUNK_CHARACTERS)

    async def analyze_recording(
        self,
        recording_id: str,
        context_instructions: Optional[str] = None,
    ) -> AIAnalysis:
        # 1. Check recording metadata
        meta = await self.storage.get_recording_metadata(recording_id)
        if not meta:
            raise LLMError(
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
            raise LLMError(
                f"Storage directory for recording '{recording_id}' not found",
                error_code="RECORDING_NOT_FOUND",
            )

        analysis_dir = session_dir / "analysis"
        analysis_path = analysis_dir / "analysis.json"

        # 2. Idempotency check: if already analyzed and analysis.json exists on disk
        if meta.get("status") == RecordingStatus.ANALYZED.value and analysis_path.exists():
            try:
                with open(analysis_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return AIAnalysis(**data)
            except Exception:
                pass  # If analysis.json is unreadable, fall through to re-analyze

        # 3. Check status: recording must be transcribed first
        status = meta.get("status")
        if status not in (
            RecordingStatus.TRANSCRIBED.value,
            RecordingStatus.ANALYZING.value,
            RecordingStatus.ANALYZED.value,
        ):
            raise LLMError(
                f"Recording '{recording_id}' is in '{status}' status. It must be in 'transcribed' status before AI analysis.",
                error_code="RECORDING_NOT_TRANSCRIBED",
            )

        # 4. Load transcript
        transcript_path = session_dir / "transcript" / "transcript.json"
        if not transcript_path.exists():
            raise LLMError(
                f"Transcript for recording '{recording_id}' not found. Please transcribe the recording first.",
                error_code="TRANSCRIPT_NOT_FOUND",
            )

        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                t_data = json.load(f)
            transcript = Transcript(**t_data)
        except Exception as e:
            raise LLMError(f"Failed to read transcript.json: {e}", error_code="TRANSCRIPT_PARSE_ERROR", cause=e)

        if not transcript.text and not transcript.segments:
            raise LLMError(
                f"Transcript for recording '{recording_id}' is empty.",
                error_code="TRANSCRIPT_EMPTY",
            )

        # 5. Mark status as analyzing
        meta["status"] = RecordingStatus.ANALYZING.value
        await self.storage.update_recording_metadata(recording_id, meta)

        try:
            # 6. Execute analysis (direct or chunked for long recordings)
            if self.chunker.needs_chunking(transcript):
                analysis = await self._analyze_long_transcript(transcript, context_instructions)
            else:
                analysis = await self.provider.analyze_transcript(
                    transcript=transcript,
                    context_instructions=context_instructions,
                )

            # 7. Persist analysis to analysis/analysis.json
            analysis_dir.mkdir(parents=True, exist_ok=True)
            with open(analysis_path, "w", encoding="utf-8") as f:
                json.dump(analysis.model_dump(), f, indent=2, ensure_ascii=False)

            # 8. Update session metadata to analyzed
            meta["status"] = RecordingStatus.ANALYZED.value
            meta["error_code"] = None
            meta["error_message"] = None
            meta["analysis"] = {
                "summary": analysis.summary,
                "key_points_count": len(analysis.key_points),
                "terms_count": len(analysis.terms),
                "questions_count": len(analysis.questions),
                "language": analysis.language,
                "provider": analysis.provider,
                "model": analysis.model,
            }
            await self.storage.update_recording_metadata(recording_id, meta)

            return analysis

        except LLMError as e:
            meta["status"] = RecordingStatus.FAILED.value
            meta["error_code"] = e.error_code
            meta["error_message"] = e.message
            await self.storage.update_recording_metadata(recording_id, meta)
            raise
        except Exception as e:
            meta["status"] = RecordingStatus.FAILED.value
            meta["error_code"] = "AI_ANALYSIS_FAILED"
            meta["error_message"] = str(e)
            await self.storage.update_recording_metadata(recording_id, meta)
            raise LLMError(f"AI analysis failed: {e}", cause=e)

    async def _analyze_long_transcript(
        self,
        transcript: Transcript,
        context_instructions: Optional[str] = None,
    ) -> AIAnalysis:
        """Handles long transcripts via windowed context blocks and hierarchical synthesis."""
        blocks = self.chunker.partition_segments(transcript.segments)
        partial_analyses: list[AIAnalysis] = []

        for idx, block in enumerate(blocks):
            sub_transcript = Transcript(
                recording_id=f"{transcript.recording_id}_part{idx}",
                schema_version=transcript.schema_version,
                language=transcript.language,
                duration_ms=block[-1].end_ms - block[0].start_ms if block else 0,
                provider=transcript.provider,
                model=transcript.model,
                text=" ".join(s.text for s in block),
                segments=block,
            )
            part_res = await self.provider.analyze_transcript(
                transcript=sub_transcript,
                context_instructions=context_instructions,
            )
            partial_analyses.append(part_res)

        # Synthesize partial analyses into one unified AIAnalysis
        all_key_points = []
        seen_points = set()
        for p in partial_analyses:
            for pt in p.key_points:
                norm = pt.strip().lower()
                if norm not in seen_points:
                    seen_points.add(norm)
                    all_key_points.append(pt)

        all_terms = []
        seen_terms = set()
        for p in partial_analyses:
            for t in p.terms:
                t_norm = t.term.strip().lower()
                if t_norm not in seen_terms:
                    seen_terms.add(t_norm)
                    all_terms.append(t)

        all_questions = []
        seen_questions = set()
        for p in partial_analyses:
            for q in p.questions:
                q_norm = q.strip().lower()
                if q_norm not in seen_questions:
                    seen_questions.add(q_norm)
                    all_questions.append(q)

        unified_summary = " ".join(p.summary for p in partial_analyses)
        unified_explanation = "\n\n".join(p.explanation for p in partial_analyses)

        all_seg_ids = [s.id for s in transcript.segments]

        return AIAnalysis(
            recording_id=transcript.recording_id,
            schema_version=1,
            language=transcript.language or "uz",
            summary=unified_summary,
            key_points=all_key_points[:7],
            explanation=unified_explanation,
            terms=all_terms[:10],
            questions=all_questions[:5],
            source_segment_ids=all_seg_ids,
            provider=self.provider.provider_name,
            model=self.provider.model_name,
        )

    async def get_analysis(self, recording_id: str) -> Optional[AIAnalysis]:
        """Retrieves persistent analysis.json for a recording."""
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

        analysis_path = session_dir / "analysis" / "analysis.json"
        if not analysis_path.exists():
            return None

        with open(analysis_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return AIAnalysis(**data)
