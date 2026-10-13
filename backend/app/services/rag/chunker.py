import re
from typing import List, Optional
from app.core.config import settings
from app.services.stt.models import Transcript, TranscriptSegment
from app.services.rag.models import RAGChunk

# Sentence split regex supporting Uzbek (o', g' quotes), Russian, and English punctuation
SENTENCE_SPLIT_REGEX = re.compile(r'(?<=[.!?])\s+')


class SemanticChunker:
    """
    Structure-aware semantic chunker for speech transcripts.
    Preserves:
    1. STT segment boundaries and timestamps (start_ms, end_ms).
    2. Sentence and clause integrity across Uzbek, Russian, English, and code-switched text.
    3. Topic continuity with deterministic, bounded segment overlap.
    """

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        overlap_tokens: Optional[int] = None,
    ):
        self.max_tokens = max_tokens or settings.RAG_CHUNK_MAX_TOKENS
        self.overlap_tokens = overlap_tokens or settings.RAG_CHUNK_OVERLAP_TOKENS

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate across multilingual texts based on whitespace splitting."""
        if not text:
            return 0
        return len(text.strip().split())

    def chunk_transcript(
        self,
        transcript: Transcript,
        recording_id: Optional[str] = None,
    ) -> List[RAGChunk]:
        """
        Transforms a timestamped Transcript into coherent, overlapping RAGChunk objects.

        :param transcript: The STT Transcript with timestamped segments.
        :param recording_id: Recording ID for chunk scoping (defaults to transcript.recording_id).
        :return: List of RAGChunk instances with exact start/end ms and segment lineage.
        """
        rec_id = recording_id or transcript.recording_id
        segments = [s for s in transcript.segments if s.text and s.text.strip()]

        if not segments:
            # Fallback if transcript text exists but segments list is empty
            if transcript.text and transcript.text.strip():
                return [
                    RAGChunk(
                        chunk_id="chunk_1",
                        recording_id=rec_id,
                        text=transcript.text.strip(),
                        start_ms=0,
                        end_ms=transcript.duration_ms or 0,
                        segment_ids=[],
                        metadata={"source": "unsegmented_transcript"},
                    )
                ]
            return []

        chunks: List[RAGChunk] = []
        current_segments: List[TranscriptSegment] = []
        current_token_count = 0
        chunk_counter = 1

        for seg in segments:
            seg_tokens = self._estimate_tokens(seg.text)

            # If adding this segment exceeds max_tokens and we already have accumulated segments
            if current_segments and (current_token_count + seg_tokens > self.max_tokens):
                # Emit current accumulated chunk
                chunk = self._build_chunk(rec_id, f"chunk_{chunk_counter}", current_segments)
                chunks.append(chunk)
                chunk_counter += 1

                # Deterministic overlap strategy:
                # Select trailing segments from current_segments whose sum of tokens is <= overlap_tokens
                overlap_segments: List[TranscriptSegment] = []
                overlap_tokens_acc = 0
                for s in reversed(current_segments):
                    t = self._estimate_tokens(s.text)
                    if overlap_tokens_acc + t <= self.overlap_tokens:
                        overlap_segments.insert(0, s)
                        overlap_tokens_acc += t
                    else:
                        break

                # Ensure overlap is never the entire previous chunk
                if len(overlap_segments) >= len(current_segments):
                    overlap_segments = overlap_segments[-1:] if len(overlap_segments) > 1 else []

                current_segments = list(overlap_segments)
                current_token_count = sum(self._estimate_tokens(s.text) for s in current_segments)

            # If a single segment itself exceeds max_tokens, split it by sentence
            if seg_tokens > self.max_tokens:
                sub_segments = self._split_large_segment(seg)
                for sub_seg in sub_segments:
                    sub_tokens = self._estimate_tokens(sub_seg.text)
                    if current_segments and (current_token_count + sub_tokens > self.max_tokens):
                        chunk = self._build_chunk(rec_id, f"chunk_{chunk_counter}", current_segments)
                        chunks.append(chunk)
                        chunk_counter += 1
                        current_segments = []
                        current_token_count = 0

                    current_segments.append(sub_seg)
                    current_token_count += sub_tokens
            else:
                current_segments.append(seg)
                current_token_count += seg_tokens

        # Emit any remaining segments
        if current_segments:
            chunk = self._build_chunk(rec_id, f"chunk_{chunk_counter}", current_segments)
            chunks.append(chunk)

        return chunks

    def _split_large_segment(self, segment: TranscriptSegment) -> List[TranscriptSegment]:
        """Splits an unusually long segment at sentence boundaries, interpolating timestamps."""
        sentences = [s.strip() for s in SENTENCE_SPLIT_REGEX.split(segment.text) if s.strip()]
        if len(sentences) <= 1:
            return [segment]

        total_chars = max(len(segment.text), 1)
        total_duration = max(segment.end_ms - segment.start_ms, 0)

        seg_id = getattr(segment, "id", getattr(segment, "segment_id", "seg"))
        result: List[TranscriptSegment] = []
        cur_start = segment.start_ms

        for i, s in enumerate(sentences):
            fraction = len(s) / total_chars
            cur_end = cur_start + int(total_duration * fraction)
            if i == len(sentences) - 1:
                cur_end = segment.end_ms

            result.append(
                TranscriptSegment(
                    id=f"{seg_id}_sub_{i}",
                    chunk_id=segment.chunk_id,
                    start_ms=cur_start,
                    end_ms=max(cur_end, cur_start),
                    text=s,
                    confidence=segment.confidence,
                )
            )
            cur_start = cur_end

        return result

    @staticmethod
    def _build_chunk(
        recording_id: str,
        chunk_id: str,
        segments: List[TranscriptSegment],
    ) -> RAGChunk:
        """Constructs a RAGChunk from a group of segments."""
        combined_text = " ".join(s.text.strip() for s in segments)
        start_ms = segments[0].start_ms
        end_ms = segments[-1].end_ms
        segment_ids = [getattr(s, "id", getattr(s, "segment_id", "")) for s in segments]

        return RAGChunk(
            id=f"{recording_id}_{chunk_id}",
            chunk_id=chunk_id,
            recording_id=recording_id,
            text=combined_text,
            start_ms=start_ms,
            end_ms=end_ms,
            segment_ids=segment_ids,
            metadata={
                "segments_count": len(segments),
                "word_count": len(combined_text.split()),
            },
        )
