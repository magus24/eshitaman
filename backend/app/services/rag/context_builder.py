from typing import List, Tuple
from app.services.rag.models import RAGChunk


class ContextBuilder:
    """
    Formats retrieved transcript chunks into a structured, chronological context block
    for the LLM, preserving chunk identifiers and human-readable timestamp ranges.
    """

    def __init__(self, max_context_chars: int = 12000):
        self.max_context_chars = max_context_chars

    def build_context(self, chunks: List[RAGChunk]) -> str:
        """
        Builds formatted context text sorted chronologically by start_ms.

        :param chunks: Retrieved RAGChunk objects.
        :return: Formatted multi-chunk context string.
        """
        if not chunks:
            return ""

        # Sort chronologically by speech timeline
        sorted_chunks = sorted(chunks, key=lambda c: c.start_ms)
        blocks: List[str] = []
        total_chars = 0

        for chunk in sorted_chunks:
            time_label = chunk.timestamp_display
            block = f"[{time_label}] (ID: {chunk.chunk_id})\n{chunk.text.strip()}"
            if total_chars + len(block) > self.max_context_chars and blocks:
                break
            blocks.append(block)
            total_chars += len(block) + 2

        return "\n\n".join(blocks)
