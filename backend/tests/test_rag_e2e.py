"""
Stage 9 End-to-End (E2E) Test:
Synthetic Audio -> Upload -> Process -> Transcribe (Mock) -> Index (RAG) -> Chat (Grounded Q&A with Sources) -> Idempotency
"""

import json
import shutil
import tempfile
import wave
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.recording import RecordingStatus
from app.services.storage.local import LocalStorage
import app.services.storage as storage_module
from app.api.v1 import recordings as recordings_api
from app.services.rag.pipeline import RAGPipeline
from app.services.rag.vector_store import InMemoryVectorStore
from app.services.rag.embedding_mock import MockEmbeddingProvider
from app.services.llm.mock_provider import MockLLMProvider

client = TestClient(app)


def generate_test_wav(file_path: Path, duration_s: float = 12.0, sample_rate: int = 16000) -> Path:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    num_samples = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, num_samples, endpoint=False)
    tone = np.sin(2 * np.pi * 440.0 * t) * 0.5
    waveform = (tone * 32767).astype(np.int16)

    with wave.open(str(file_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(waveform.tobytes())

    return file_path


@pytest.fixture
def rag_e2e_storage(monkeypatch):
    temp_dir = Path(tempfile.mkdtemp(prefix="eshitaman_rag_e2e_"))
    storage = LocalStorage(base_path=temp_dir / "storage" / "recordings")
    vector_store = InMemoryVectorStore()
    embedding_provider = MockEmbeddingProvider()
    llm_provider = MockLLMProvider()

    monkeypatch.setattr(storage_module, "storage_service", storage)
    monkeypatch.setattr(recordings_api, "storage_service", storage)

    def mock_get_rag_pipeline():
        return RAGPipeline(
            storage=storage,
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            llm_provider=llm_provider,
            min_similarity=0.05,
        )

    monkeypatch.setattr(recordings_api, "get_rag_pipeline", mock_get_rag_pipeline)

    yield storage, temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_full_rag_pipeline_e2e(rag_e2e_storage):
    storage, temp_dir = rag_e2e_storage
    recording_id = "rec_e2e_rag_lecture_001"
    wav_path = temp_dir / "lecture.wav"
    generate_test_wav(wav_path, duration_s=12.0)

    # 1. Upload audio
    with open(wav_path, "rb") as f:
        res_upload = client.post(
            "/api/v1/recordings",
            data={
                "recording_id": recording_id,
                "duration_seconds": "12.0",
                "sample_rate": "16000",
                "channel_count": "1",
            },
            files={"file": ("lecture.wav", f, "audio/wav")},
        )
    assert res_upload.status_code == 201
    assert res_upload.json()["status"] == "uploaded"

    # 2. Audio Processing (VAD, Normalization, Chunking)
    res_proc = client.post(f"/api/v1/recordings/{recording_id}/process")
    assert res_proc.status_code == 200
    assert res_proc.json()["status"] == "processed"

    # 3. STT Transcription (Mock provider)
    res_trans = client.post(
        f"/api/v1/recordings/{recording_id}/transcribe",
        json={"language_hint": "uz"},
    )
    assert res_trans.status_code == 200
    assert res_trans.json()["status"] == "transcribed"
    assert res_trans.json()["segments_count"] > 0

    # 4. Index for RAG
    res_index = client.post(f"/api/v1/recordings/{recording_id}/index")
    assert res_index.status_code == 200
    index_data = res_index.json()
    assert index_data["recording_id"] == recording_id
    assert index_data["status"] == "indexed"
    assert index_data["chunks_count"] > 0
    assert index_data["embedding_dimension"] == 1536

    # 5. Check recording status updated to indexed
    res_rec = client.get(f"/api/v1/recordings/{recording_id}")
    assert res_rec.status_code == 200
    rec_meta = res_rec.json()
    assert rec_meta["status"] == "indexed"
    assert rec_meta["rag"] is not None
    assert rec_meta["rag"]["chunks_count"] == index_data["chunks_count"]

    # 6. Verify disk layout: session_dir / "rag" / "chunks.json"
    session_dir = storage._get_session_dir(recording_id)
    chunks_file = session_dir / "rag" / "chunks.json"
    assert chunks_file.exists()
    with open(chunks_file, "r", encoding="utf-8") as f:
        disk_chunks = json.load(f)
    assert len(disk_chunks) == index_data["chunks_count"]
    assert "start_ms" in disk_chunks[0]
    assert "end_ms" in disk_chunks[0]
    assert "embedding" in disk_chunks[0]

    # 7. Chat with Grounded Question
    res_chat = client.post(
        f"/api/v1/recordings/{recording_id}/chat",
        json={"question": "Machine learning va model accuracy haqida nimalar aytildi?", "top_k": 3},
    )
    assert res_chat.status_code == 200
    chat_data = res_chat.json()
    assert chat_data["recording_id"] == recording_id
    assert len(chat_data["answer"]) > 0
    assert len(chat_data["sources"]) > 0

    # Check sources have valid citation metadata
    first_src = chat_data["sources"][0]
    assert first_src["start_time"] >= 0.0
    assert first_src["end_time"] >= first_src["start_time"]
    assert len(first_src["text"]) > 0
    assert first_src["relevance"] > 0.0

    # 8. Test Idempotency: second index returns immediately without re-embedding
    res_reindex = client.post(f"/api/v1/recordings/{recording_id}/index")
    assert res_reindex.status_code == 200
    reindex_data = res_reindex.json()
    assert reindex_data["status"] == "indexed"
    assert reindex_data["chunks_count"] == index_data["chunks_count"]
    assert "already indexed" in reindex_data["message"].lower()
