"""
Stage 8 End-to-End (E2E) Test:
Synthetic Audio -> Upload -> Process -> Transcribe (Mock) -> AI Analysis (Mock) -> Verification -> Idempotency
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
def e2e_storage(monkeypatch):
    temp_dir = Path(tempfile.mkdtemp(prefix="eshitaman_llm_e2e_"))
    storage = LocalStorage(base_path=temp_dir / "storage" / "recordings")
    monkeypatch.setattr(storage_module, "storage_service", storage)
    monkeypatch.setattr(recordings_api, "storage_service", storage)
    yield storage, temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_full_llm_pipeline_e2e(e2e_storage):
    storage, temp_dir = e2e_storage
    recording_id = "rec_e2e_ai_lecture_001"
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

    # 4. Trigger AI Understanding (Mock provider)
    res_analyze = client.post(
        f"/api/v1/recordings/{recording_id}/analyze",
        json={"context_instructions": "O'zbek tilida talabalar uchun qisqacha xulosa"},
    )
    assert res_analyze.status_code == 200
    analysis_resp = res_analyze.json()
    assert analysis_resp["recording_id"] == recording_id
    assert analysis_resp["status"] == "analyzed"
    assert len(analysis_resp["summary"]) > 0
    assert analysis_resp["key_points_count"] >= 3
    assert analysis_resp["terms_count"] >= 3
    assert analysis_resp["questions_count"] >= 3

    # 5. GET recording metadata and verify full status transition
    res_rec = client.get(f"/api/v1/recordings/{recording_id}")
    assert res_rec.status_code == 200
    rec_meta = res_rec.json()
    assert rec_meta["status"] == "analyzed"
    assert rec_meta["transcript"] is not None
    assert rec_meta["analysis"] is not None
    assert rec_meta["analysis"]["summary"] == analysis_resp["summary"]
    assert rec_meta["analysis"]["key_points_count"] == analysis_resp["key_points_count"]

    # 6. GET full AIAnalysis schema
    res_full = client.get(f"/api/v1/recordings/{recording_id}/analysis")
    assert res_full.status_code == 200
    full_analysis = res_full.json()
    assert full_analysis["recording_id"] == recording_id
    assert full_analysis["schema_version"] == 1
    assert full_analysis["language"] in ("uz", "ru", "en")
    assert len(full_analysis["key_points"]) == analysis_resp["key_points_count"]
    assert len(full_analysis["terms"]) == analysis_resp["terms_count"]
    assert len(full_analysis["questions"]) == analysis_resp["questions_count"]
    assert len(full_analysis["explanation"]) > 0

    # Verify source segment tracking is preserved
    assert len(full_analysis["source_segment_ids"]) > 0
    for term in full_analysis["terms"]:
        assert "term" in term
        assert "explanation" in term
        assert isinstance(term.get("source_segment_ids"), list)

    # 7. Verify persistent disk storage layout
    session_dir = storage._get_session_dir(recording_id)
    analysis_file = session_dir / "analysis" / "analysis.json"
    assert analysis_file.exists()
    with open(analysis_file, "r", encoding="utf-8") as f:
        disk_data = json.load(f)
        assert disk_data["recording_id"] == recording_id
        assert disk_data["summary"] == full_analysis["summary"]

    # 8. Test Idempotency: repeated POST /analyze should return immediately from cache
    res_idemp = client.post(f"/api/v1/recordings/{recording_id}/analyze")
    assert res_idemp.status_code == 200
    idemp_data = res_idemp.json()
    assert idemp_data["status"] == "analyzed"
    assert "already analyzed" in idemp_data["message"].lower()
    assert idemp_data["summary"] == analysis_resp["summary"]
