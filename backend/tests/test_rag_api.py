import json
import shutil
import tempfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.recording import RecordingStatus
from app.services.storage.local import LocalStorage
import app.services.storage as storage_module
from app.api.v1 import recordings as recordings_api
from app.services.stt.models import Transcript, TranscriptSegment
from app.services.rag.pipeline import RAGPipeline
from app.services.rag.vector_store import InMemoryVectorStore
from app.services.rag.embedding_mock import MockEmbeddingProvider
from app.services.llm.mock_provider import MockLLMProvider


@pytest.fixture
def client_with_storage():
    temp_dir = Path(tempfile.mkdtemp(prefix="eshitaman_api_rag_"))
    test_storage = LocalStorage(base_path=temp_dir / "storage" / "recordings")
    vector_store = InMemoryVectorStore()
    embedding_provider = MockEmbeddingProvider()
    llm_provider = MockLLMProvider()

    orig_storage = storage_module.storage_service
    orig_api_storage = recordings_api.storage_service
    orig_get_rag = recordings_api.get_rag_pipeline

    storage_module.storage_service = test_storage
    recordings_api.storage_service = test_storage

    # Patch get_rag_pipeline to use our test fixtures
    def mock_get_rag_pipeline():
        return RAGPipeline(
            storage=test_storage,
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            llm_provider=llm_provider,
            min_similarity=0.1,
        )

    recordings_api.get_rag_pipeline = mock_get_rag_pipeline

    test_client = TestClient(app)
    yield test_client, test_storage

    storage_module.storage_service = orig_storage
    recordings_api.storage_service = orig_api_storage
    recordings_api.get_rag_pipeline = orig_get_rag
    shutil.rmtree(temp_dir, ignore_errors=True)


def seed_recording(storage: LocalStorage, recording_id: str, texts: list[str], status_str: str = "transcribed"):
    session_dir = storage._get_session_dir(recording_id)
    session_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "recording_id": recording_id,
        "filename": "audio.m4a",
        "file_size_bytes": 1024,
        "mime_type": "audio/m4a",
        "status": status_str,
        "created_at": "2026-09-26T10:00:00Z",
    }
    with open(session_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    if texts:
        segments = []
        t_ms = 0
        for i, text in enumerate(texts):
            segments.append(
                TranscriptSegment(
                    id=f"seg_{i}",
                    chunk_id=f"c_{i}",
                    start_ms=t_ms,
                    end_ms=t_ms + 4000,
                    text=text,
                    confidence=0.99,
                )
            )
            t_ms += 4100

        transcript = Transcript(
            recording_id=recording_id,
            duration_ms=t_ms,
            language="uz",
            text=" ".join(texts),
            segments=segments,
        )
        t_dir = session_dir / "transcript"
        t_dir.mkdir(parents=True, exist_ok=True)
        with open(t_dir / "transcript.json", "w", encoding="utf-8") as f:
            json.dump(transcript.model_dump(), f)


def test_api_index_not_found(client_with_storage):
    client, _ = client_with_storage
    res = client.post("/api/v1/recordings/non_existent_rec/index")
    assert res.status_code == 404


def test_api_index_untranscribed(client_with_storage):
    client, storage = client_with_storage
    seed_recording(storage, "rec_raw", texts=[], status_str="uploaded")
    res = client.post("/api/v1/recordings/rec_raw/index")
    assert res.status_code == 400


def test_api_index_and_chat_success(client_with_storage):
    client, storage = client_with_storage
    rec_id = "rec_lecture_101"
    texts = [
        "Bugungi ma'ruza kompyuter arxitekturasi va kesh xotira darajalari haqida.",
        "L1 kesh protsessorga eng yaqin va eng tezkor xotira hisoblanadi.",
        "L2 va L3 keshlari esa hajmi kattaroq, lekin kechikishi ko'proq.",
    ]
    seed_recording(storage, rec_id, texts)

    # 1. Index
    idx_res = client.post(f"/api/v1/recordings/{rec_id}/index")
    assert idx_res.status_code == 200
    idx_data = idx_res.json()
    assert idx_data["recording_id"] == rec_id
    assert idx_data["status"] == "indexed"
    assert idx_data["chunks_count"] >= 1

    # 2. Chat with question
    chat_payload = {
        "question": "L1 kesh xotirasi nima uchun muhim va qanday xususiyatga ega?",
        "top_k": 3,
    }
    chat_res = client.post(f"/api/v1/recordings/{rec_id}/chat", json=chat_payload)
    assert chat_res.status_code == 200
    chat_data = chat_res.json()
    assert chat_data["recording_id"] == rec_id
    assert len(chat_data["answer"]) > 10
    assert len(chat_data["sources"]) > 0

    # Source check
    src = chat_data["sources"][0]
    assert "start_time" in src
    assert "end_time" in src
    assert "text" in src
    assert "relevance" in src


def test_api_chat_empty_question_returns_422(client_with_storage):
    client, storage = client_with_storage
    seed_recording(storage, "rec_q422", ["Test matn"])
    client.post("/api/v1/recordings/rec_q422/index")

    res = client.post("/api/v1/recordings/rec_q422/chat", json={"question": "   "})
    assert res.status_code == 422


def test_api_strict_cross_recording_isolation(client_with_storage):
    """
    CRITICAL MULTI-RECORDING SECURITY TEST VIA API:
    Create Recording A (History of Samarkand) and Recording B (Quantum Physics).
    Querying Recording B for 'Samarqand' must NEVER return chunks from Recording A.
    """
    client, storage = client_with_storage

    rec_a = "rec_samarkand_history"
    texts_a = [
        "Samarqand shahri qadimiy Ipak yo'lining markazida joylashgan yirik obidadir.",
        "Registon maydoni uchta madrasadan iborat: Ulug'bek, Sherdor va Tillakori.",
    ]
    seed_recording(storage, rec_a, texts_a)
    client.post(f"/api/v1/recordings/{rec_a}/index")

    rec_b = "rec_quantum_physics"
    texts_b = [
        "Kvant mexanikasi zarrachalarning to'lqin xususiyatlarini o'rganadi.",
        "Shryodinger tenglamasi kvant holatining vaqt bo'yicha evolyutsiyasini ifodalaydi.",
    ]
    seed_recording(storage, rec_b, texts_b)
    client.post(f"/api/v1/recordings/{rec_b}/index")

    # Ask Recording B about Samarqand
    res_b = client.post(
        f"/api/v1/recordings/{rec_b}/chat",
        json={"question": "Samarqand Registon maydoni haqida nimalar ma'lum?"},
    )
    assert res_b.status_code == 200
    data_b = res_b.json()
    # Must NOT have sources from recording A
    for src in data_b["sources"]:
        assert "Samarqand" not in src["text"]
        assert "Registon" not in src["text"]

    # Ask Recording A about Samarqand
    res_a = client.post(
        f"/api/v1/recordings/{rec_a}/chat",
        json={"question": "Samarqand Registon maydoni haqida nimalar ma'lum?"},
    )
    assert res_a.status_code == 200
    data_a = res_a.json()
    assert any("Registon" in src["text"] for src in data_a["sources"])
