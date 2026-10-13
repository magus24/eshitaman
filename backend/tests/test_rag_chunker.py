import pytest
from app.services.rag.chunker import SemanticChunker
from app.services.stt.models import Transcript, TranscriptSegment


def test_chunker_empty_transcript():
    chunker = SemanticChunker(max_tokens=50, overlap_tokens=10)
    transcript = Transcript(
        recording_id="rec_empty",
        duration_ms=0,
        text="",
        segments=[],
    )
    chunks = chunker.chunk_transcript(transcript)
    assert chunks == []


def test_chunker_single_segment_boundary_and_timestamps():
    chunker = SemanticChunker(max_tokens=100, overlap_tokens=20)
    segments = [
        TranscriptSegment(
            id="seg_0",
            chunk_id="audio_0",
            start_ms=1000,
            end_ms=4500,
            text="Bugungi darsda biz sun'iy intellekt asoslarini o'rganamiz.",
            confidence=0.98,
        )
    ]
    transcript = Transcript(
        recording_id="rec_single",
        duration_ms=5000,
        text="Bugungi darsda biz sun'iy intellekt asoslarini o'rganamiz.",
        segments=segments,
    )
    chunks = chunker.chunk_transcript(transcript)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.recording_id == "rec_single"
    assert c.start_ms == 1000
    assert c.end_ms == 4500
    assert c.start_seconds == 1.0
    assert c.end_seconds == 4.5
    assert c.timestamp_display == "00:01 – 00:04"
    assert c.segment_ids == ["seg_0"]


def test_chunker_multilingual_and_code_switching():
    chunker = SemanticChunker(max_tokens=25, overlap_tokens=5)
    segments = [
        TranscriptSegment(
            id="s1",
            chunk_id="c0",
            start_ms=0,
            end_ms=3000,
            text="Biz backend arxitekturasini FastAPI yordamida quramiz.",
        ),
        TranscriptSegment(
            id="s2",
            chunk_id="c0",
            start_ms=3100,
            end_ms=6500,
            text="Векторный поиск осуществляется через pgvector базу данных.",
        ),
        TranscriptSegment(
            id="s3",
            chunk_id="c1",
            start_ms=6600,
            end_ms=10000,
            text="This enables high-performance semantic retrieval for multilingual queries.",
        ),
    ]
    transcript = Transcript(
        recording_id="rec_multi",
        duration_ms=10000,
        text="Biz backend arxitekturasini... Векторный поиск... This enables...",
        segments=segments,
    )
    chunks = chunker.chunk_transcript(transcript)
    assert len(chunks) >= 1
    # Check all languages are preserved in output chunks
    full_chunk_text = " ".join(c.text for c in chunks)
    assert "FastAPI" in full_chunk_text
    assert "Векторный поиск" in full_chunk_text
    assert "multilingual" in full_chunk_text


def test_chunker_deterministic_overlap():
    # Force small token limit to trigger multi-chunking with overlap
    chunker = SemanticChunker(max_tokens=8, overlap_tokens=4)
    segments = [
        TranscriptSegment(id="s1", chunk_id="c0", start_ms=0, end_ms=2000, text="Birinchi gap darslik haqida so'zlaydi."),
        TranscriptSegment(id="s2", chunk_id="c0", start_ms=2000, end_ms=4000, text="Ikkinchi qism juda muhim atamani o'z ichiga oladi."),
        TranscriptSegment(id="s3", chunk_id="c0", start_ms=4000, end_ms=6000, text="Uchinchi bo'lim yakuniy xulosa beradi."),
    ]
    transcript = Transcript(
        recording_id="rec_overlap",
        duration_ms=6000,
        text="...",
        segments=segments,
    )
    chunks = chunker.chunk_transcript(transcript)
    assert len(chunks) >= 2
    # Ensure sequential chunks maintain time order
    for i in range(len(chunks) - 1):
        assert chunks[i].start_ms <= chunks[i + 1].start_ms
        assert chunks[i].end_ms <= chunks[i + 1].end_ms


def test_chunker_splits_huge_single_segment_by_sentence():
    chunker = SemanticChunker(max_tokens=10, overlap_tokens=2)
    huge_text = (
        "Birinchi fikr juda muhim. Ikkinchi fikr ham tushuntirildi. "
        "Uchinchi qoida esa dasturlashda qo'llaniladi. To'rtinchi fikr xulosa qildi."
    )
    segments = [
        TranscriptSegment(
            id="big_seg",
            chunk_id="c0",
            start_ms=0,
            end_ms=40000,
            text=huge_text,
        )
    ]
    transcript = Transcript(
        recording_id="rec_huge",
        duration_ms=40000,
        text=huge_text,
        segments=segments,
    )
    chunks = chunker.chunk_transcript(transcript)
    assert len(chunks) > 1
    # Check that sentences were not chopped in half abruptly
    assert any("Birinchi fikr" in c.text for c in chunks)
    assert any("To'rtinchi fikr" in c.text for c in chunks)
