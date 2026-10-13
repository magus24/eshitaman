import asyncio
import json
import shutil
import tempfile
from pathlib import Path
import pytest

from app.schemas.recording import RecordingStatus
from app.services.storage.local import LocalStorage
from app.services.stt.models import Transcript, TranscriptSegment
from app.services.rag.chunker import SemanticChunker
from app.services.rag.context_builder import ContextBuilder
from app.services.rag.embedding_mock import MockEmbeddingProvider
from app.services.rag.pipeline import RAGPipeline
from app.services.rag.vector_store import InMemoryVectorStore
from app.services.llm.mock_provider import MockLLMProvider


@pytest.fixture
def temp_storage():
    temp_dir = Path(tempfile.mkdtemp(prefix="eshitaman_rag_test_"))
    storage = LocalStorage(base_path=temp_dir / "storage" / "recordings")
    yield storage
    shutil.rmtree(temp_dir, ignore_errors=True)


def create_mock_transcribed_recording(storage: LocalStorage, recording_id: str, segments_text: list[str]) -> Path:
    session_dir = storage._get_session_dir(recording_id)
    session_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "recording_id": recording_id,
        "filename": "lecture.m4a",
        "file_size_bytes": 1024,
        "mime_type": "audio/m4a",
        "status": RecordingStatus.TRANSCRIBED.value,
        "transcript": {
            "status": "completed",
            "language": "uz",
        },
    }
    with open(session_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    segments = []
    t_ms = 0
    for i, t in enumerate(segments_text):
        segments.append(
            TranscriptSegment(
                id=f"seg_{i}",
                chunk_id=f"chunk_{i}",
                start_ms=t_ms,
                end_ms=t_ms + 5000,
                text=t,
                confidence=0.98,
            )
        )
        t_ms += 5100

    transcript = Transcript(
        recording_id=recording_id,
        duration_ms=t_ms,
        language="uz",
        text=" ".join(segments_text),
        segments=segments,
    )

    t_dir = session_dir / "transcript"
    t_dir.mkdir(parents=True, exist_ok=True)
    with open(t_dir / "transcript.json", "w", encoding="utf-8") as f:
        json.dump(transcript.model_dump(), f)

    return session_dir


def test_rag_pipeline_index_and_answer(temp_storage):
    async def _test():
        rec_id = "rec_ai_lecture"
        segments = [
            "Bugungi ma'ruzamizda biz neyron tarmoqlari va gradient tushish algoritmini o'rganamiz.",
            "Gradient descent bu yo'qotish funksiyasini minimallashtirish uchun ishlatiladigan asosiy optimallash usulidir.",
            "Modelning o'rganish tezligi ya'ni learning rate qiymati 0.01 qilib belgilandi.",
        ]
        create_mock_transcribed_recording(temp_storage, rec_id, segments)

        emb_provider = MockEmbeddingProvider()
        store = InMemoryVectorStore()
        llm = MockLLMProvider()

        pipeline = RAGPipeline(
            storage=temp_storage,
            embedding_provider=emb_provider,
            vector_store=store,
            llm_provider=llm,
            min_similarity=0.1,
        )

        # 1. Index
        idx_resp = await pipeline.index_recording(rec_id)
        assert idx_resp.status == "indexed"
        assert idx_resp.chunks_count >= 1

        # Check session metadata was updated to indexed
        meta = await temp_storage.get_recording_metadata(rec_id)
        assert meta["status"] == "indexed"
        assert meta["rag"]["chunks_count"] == idx_resp.chunks_count

        # 2. Ask Grounded Question
        answer_resp = await pipeline.answer_question(
            recording_id=rec_id,
            question="Gradient descent algoritmi nima uchun ishlatiladi?",
        )
        assert answer_resp.recording_id == rec_id
        assert len(answer_resp.sources) > 0
        # Traceability check: source has timestamps
        assert answer_resp.sources[0].start_time >= 0.0
        assert answer_resp.sources[0].end_time > answer_resp.sources[0].start_time
        assert answer_resp.sources[0].relevance > 0.0
        # Verify grounded answer returned
        assert len(answer_resp.answer) > 20

    asyncio.run(_test())


def test_rag_pipeline_no_hallucination_safe_answer(temp_storage):
    async def _test():
        rec_id = "rec_python_ai"
        segments = [
            "Python dasturlash tilida ma'lumotlar tuzilmalari: ro'yxatlar va lug'atlar.",
            "FastAPI frameworki yordamida tezkor asinxron API yaratish mumkin.",
        ]
        create_mock_transcribed_recording(temp_storage, rec_id, segments)

        emb_provider = MockEmbeddingProvider()
        store = InMemoryVectorStore()
        llm = MockLLMProvider()

        pipeline = RAGPipeline(
            storage=temp_storage,
            embedding_provider=emb_provider,
            vector_store=store,
            llm_provider=llm,
            min_similarity=0.1,
        )
        await pipeline.index_recording(rec_id)

        # Ask about JavaScript closure which is nowhere in the context
        res = await pipeline.answer_question(
            recording_id=rec_id,
            question="JavaScript closure qanday ishlaydi va scope qanday saqlanadi?",
        )
        ans_lower = res.answer.lower()
        assert (
            "yetarli ma'lumot mavjud emas" in ans_lower
            or "yetarli ma'lumot topilmadi" in ans_lower
            or "hech qanday ma'lumot mavjud emas" in ans_lower
        )

    asyncio.run(_test())


def test_rag_pipeline_idempotent_indexing(temp_storage):
    async def _test():
        rec_id = "rec_idempotent"
        segments = ["Birinchi jumlalar to'plami.", "Ikkinchi jumlalar to'plami."]
        create_mock_transcribed_recording(temp_storage, rec_id, segments)

        emb_provider = MockEmbeddingProvider()
        store = InMemoryVectorStore()

        pipeline = RAGPipeline(
            storage=temp_storage,
            embedding_provider=emb_provider,
            vector_store=store,
        )

        # First index
        r1 = await pipeline.index_recording(rec_id)
        first_count = await store.count(rec_id)
        assert first_count == r1.chunks_count

        # Second index (should be a no-op, re-using existing index without duplicating)
        r2 = await pipeline.index_recording(rec_id)
        second_count = await store.count(rec_id)
        assert second_count == first_count
        assert r2.chunks_count == r1.chunks_count

    asyncio.run(_test())
