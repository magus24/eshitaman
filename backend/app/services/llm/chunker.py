from typing import List
from app.services.stt.models import Transcript, TranscriptSegment


class TranscriptChunker:
    """
    Slices long transcripts into coherent context-window blocks
    to avoid exceeding LLM context limits and support hierarchical summarization.
    """

    def __init__(self, max_chars_per_window: int = 16000, overlap_segments: int = 2):
        self.max_chars_per_window = max_chars_per_window
        self.overlap_segments = overlap_segments

    def needs_chunking(self, transcript: Transcript) -> bool:
        """Determines if the transcript text length exceeds the single context window limit."""
        total_len = len(transcript.text) if transcript.text else sum(len(s.text) for s in transcript.segments)
        return total_len > self.max_chars_per_window

    def partition_segments(self, segments: List[TranscriptSegment]) -> List[List[TranscriptSegment]]:
        """
        Partitions an ordered list of segments into blocks that fit within the character limit,
        maintaining a small overlap between adjacent blocks for context continuity.
        """
        if not segments:
            return []

        blocks: List[List[TranscriptSegment]] = []
        current_block: List[TranscriptSegment] = []
        current_char_count = 0

        for seg in segments:
            seg_len = len(seg.text) + len(seg.id) + 4
            if current_block and (current_char_count + seg_len > self.max_chars_per_window):
                blocks.append(current_block)
                # Keep last N segments as overlap for the next block
                overlap = (
                    current_block[-self.overlap_segments:]
                    if len(current_block) >= self.overlap_segments
                    else list(current_block)
                )
                current_block = list(overlap)
                current_char_count = sum(len(s.text) + len(s.id) + 4 for s in current_block)

            current_block.append(seg)
            current_char_count += seg_len

        if current_block:
            blocks.append(current_block)

        return blocks
