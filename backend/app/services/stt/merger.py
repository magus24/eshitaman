import re
from typing import Counter, List, Tuple
from app.schemas.processing import AudioChunk
from app.services.stt.models import (
    ChunkTranscriptionResult,
    Transcript,
    TranscriptSegment,
    TranscriptWord,
)


class TranscriptMerger:
    """
    Transforms chunk-local transcription results into a unified global transcript:
    1. Translates chunk-relative timestamps into global recording timestamps.
    2. Sorts segments chronologically.
    3. Deduplicates boundary overlaps across adjacent chunks without removing legitimate speech repetitions.
    4. Determines dominant recording language.
    5. Assembles full continuous transcript text.
    """

    def __init__(self, overlap_tolerance_ms: int = 2500):
        self.overlap_tolerance_ms = overlap_tolerance_ms

    def merge(
        self,
        recording_id: str,
        chunk_items: List[Tuple[AudioChunk, ChunkTranscriptionResult]],
        provider_name: str = "mock",
        model_name: str = "mock-model",
        total_duration_ms: int = 0,
    ) -> Transcript:
        if not chunk_items:
            return Transcript(
                recording_id=recording_id,
                language="uz",
                duration_ms=total_duration_ms,
                provider=provider_name,
                model=model_name,
                text="",
                segments=[],
            )

        # 1. Translate local timestamps to global recording timestamps
        global_segments: List[TranscriptSegment] = []
        languages_detected: List[str] = []

        for chunk_meta, chunk_result in chunk_items:
            chunk_start_ms = chunk_meta.start_ms
            if chunk_result.language:
                languages_detected.append(chunk_result.language)

            for seg in chunk_result.segments:
                g_start = chunk_start_ms + seg.start_ms
                g_end = chunk_start_ms + seg.end_ms

                # Translate word timestamps
                g_words: List[TranscriptWord] = []
                for w in seg.words:
                    g_words.append(
                        TranscriptWord(
                            word=w.word,
                            start_ms=chunk_start_ms + w.start_ms,
                            end_ms=chunk_start_ms + w.end_ms,
                            confidence=w.confidence,
                        )
                    )

                global_segments.append(
                    TranscriptSegment(
                        id=seg.id,
                        start_ms=g_start,
                        end_ms=g_end,
                        text=seg.text.strip(),
                        language=seg.language or chunk_result.language,
                        confidence=seg.confidence,
                        chunk_id=chunk_meta.chunk_id,
                        words=g_words,
                    )
                )

        # 2. Sort by start timestamp, then end timestamp
        global_segments.sort(key=lambda s: (s.start_ms, s.end_ms))

        # 3. Deduplicate boundary overlaps
        merged_segments: List[TranscriptSegment] = []
        for current_seg in global_segments:
            if not current_seg.text:
                continue

            if not merged_segments:
                merged_segments.append(current_seg)
                continue

            last_seg = merged_segments[-1]

            # Check if current_seg is a boundary duplicate of last_seg
            if self._is_overlap_duplicate(last_seg, current_seg):
                # If current_seg has higher confidence or richer word timestamps, replace
                if (current_seg.confidence or 0.0) > (last_seg.confidence or 0.0) and current_seg.words:
                    merged_segments[-1] = current_seg
                # Otherwise, ignore current duplicate
                continue

            merged_segments.append(current_seg)

        # Re-index segment IDs neatly (seg_0000, seg_0001, ...)
        final_segments: List[TranscriptSegment] = []
        for idx, seg in enumerate(merged_segments):
            final_segments.append(
                TranscriptSegment(
                    id=f"seg_{idx:04d}",
                    start_ms=seg.start_ms,
                    end_ms=seg.end_ms,
                    text=seg.text,
                    language=seg.language,
                    confidence=seg.confidence,
                    chunk_id=seg.chunk_id,
                    words=seg.words,
                )
            )

        # 4. Determine overall dominant language
        for s in final_segments:
            if s.language:
                languages_detected.append(s.language)

        if languages_detected:
            dominant_language = Counter(languages_detected).most_common(1)[0][0]
        else:
            dominant_language = "uz"

        # 5. Assemble full text
        full_text = " ".join(s.text for s in final_segments if s.text).strip()

        # Compute max duration
        calc_duration = total_duration_ms
        if final_segments and final_segments[-1].end_ms > calc_duration:
            calc_duration = final_segments[-1].end_ms

        return Transcript(
            recording_id=recording_id,
            schema_version=1,
            language=dominant_language,
            duration_ms=calc_duration,
            provider=provider_name,
            model=model_name,
            text=full_text,
            segments=final_segments,
        )

    def _is_overlap_duplicate(self, prev: TranscriptSegment, curr: TranscriptSegment) -> bool:
        """
        Determines whether curr is an overlap duplicate of prev.
        Conditions:
        - Different chunk_id (from consecutive overlapping chunks)
        - Timestamp distance is within overlap tolerance
        - Normalized text matches or heavily overlaps
        """
        if prev.chunk_id == curr.chunk_id:
            return False

        # Check timestamp proximity (starts within overlap tolerance)
        start_diff = abs(curr.start_ms - prev.start_ms)
        overlap_span = min(prev.end_ms, curr.end_ms) - max(prev.start_ms, curr.start_ms)

        if start_diff > self.overlap_tolerance_ms and overlap_span <= 0:
            return False

        # Normalize texts for comparison (strip punctuation and whitespace, lowercase)
        norm_prev = re.sub(r"[^\w\s]", "", prev.text.lower()).strip()
        norm_curr = re.sub(r"[^\w\s]", "", curr.text.lower()).strip()

        if not norm_prev or not norm_curr:
            return False

        # Exact match
        if norm_prev == norm_curr:
            return True

        # High substring inclusion (e.g. truncated sentence at end of chunk 1 completed in chunk 2)
        if norm_prev in norm_curr and len(norm_prev) >= 0.7 * len(norm_curr):
            return True
        if norm_curr in norm_prev and len(norm_curr) >= 0.7 * len(norm_prev):
            return True

        return False
