"""
Stage 7 End-to-End (E2E) Test:
Synthetic Audio -> Upload -> Process -> Transcribe (Mock) -> Metadata & Transcript Verification -> Idempotency
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
from app.core.config import settings
from app.schemas.recording import RecordingStatus
from app.services.storage.local import LocalStorage
import app.services.storage as storage_module
from app.api.v1 import recordings as recordings_api

client = TestClient(app)


def generate_multichunk_wav(file_path: Path, duration_s: float = 65.0, sample_rate: int = 16000) -> Path:
    """Generates synthetic audio long enough to trigger multiple chunks with overlap."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    num_samples = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, num_samples, endpoint=False)
    
    # Alternating 4s speech-like tone (440Hz) and 1s silence
    tone = np.sin(2 * np.pi * 440.0 * t) * 0.6
    pattern = ((t % 5.0) < 4.0).astype(np.float32)
    waveform = (tone * pattern * 32767).astype(np.int16)

    with wave.open(str(file_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(waveform.tobytes())

    return file_path


@pytest.fixture
def e2e_storage(monkeypatch):
    temp_dir = Path(tempfile.mkdtemp(prefix="eshitaman_e2e_"))
    storage = LocalStorage(base_path=temp_dir / "storage" / "recordings")
    monkeypatch.setattr(storage_module, "storage_service", storage)
    monkeypatch.setattr(recordings_api, "storage_service", storage)
    yield storage, temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_full_stt_pipeline_e2e(e2e_storage):
    storage, temp_dir = e2e_storage
    recording_id = "rec_e2e_stt_lecture_001"
    wav_path = temp_dir / "synthetic_lecture.wav"
    generate_multichunk_wav(wav_path, duration_s=65.0)

    # 1. Upload audio
    with open(wav_path, "rb") as f:
        res_upload = client.post(
            "/api/v1/recordings",
            data={
                "recording_id": recording_id,
                "duration_seconds": "65.0",
                "sample_rate": "16000",
                "channel_count": "1",
            },
            files={"file": ("synthetic_lecture.wav", f, "audio/wav")},
        )
    assert res_upload.status_code == 201
    upload_data = res_upload.json()
    assert upload_data["recording_id"] == recording_id
    assert upload_data["status"] == "uploaded"

    # 2. Process audio (VAD, Normalization, Chunking)
    res_process = client.post(f"/api/v1/recordings/{recording_id}/process")
    assert res_process.status_code == 200
    process_data = res_process.json()
    assert process_data["status"] == "processed"
    assert process_data["chunks_count"] >= 2  # 65s produces multiple 30s chunks

    # 3. Transcribe audio with MockSTTProvider
    res_transcribe = client.post(
        f"/api/v1/recordings/{recording_id}/transcribe",
        json={"language_hint": "uz"},
    )
    assert res_transcribe.status_code == 200
    trans_data = res_transcribe.json()
    assert trans_data["recording_id"] == recording_id
    assert trans_data["status"] == "transcribed"
    assert trans_data["segments_count"] > 0
    assert trans_data["language"] in ("uz", "ru", "en")

    # 4. GET recording status & metadata
    res_recording = client.get(f"/api/v1/recordings/{recording_id}")
    assert res_recording.status_code == 200
    rec_meta = res_recording.json()
    assert rec_meta["status"] == "transcribed"
    assert rec_meta["transcript"] is not None
    assert rec_meta["transcript"]["segments_count"] == trans_data["segments_count"]
    assert rec_meta["transcript"]["duration_ms"] > 0

    # 5. GET full transcript
    res_transcript = client.get(f"/api/v1/recordings/{recording_id}/transcript")
    assert res_transcript.status_code == 200
    full_transcript = res_transcript.json()
    assert full_transcript["recording_id"] == recording_id
    assert full_transcript["schema_version"] == 1
    assert len(full_transcript["segments"]) == trans_data["segments_count"]
    assert full_transcript["text"] != ""

    # Verify global timestamps and segment linking
    prev_start_ms = -1
    seen_chunk_ids = set()
    for seg in full_transcript["segments"]:
        assert seg["id"].startswith("seg_")
        assert seg["start_ms"] >= 0
        assert seg["end_ms"] > seg["start_ms"]
        # Timestamps must be sorted globally
        assert seg["start_ms"] >= prev_start_ms
        prev_start_ms = seg["start_ms"]
        # Chunks must be linked
        assert seg["chunk_id"] is not None
        seen_chunk_ids.add(seg["chunk_id"])

    # Must span multiple chunks
    assert len(seen_chunk_ids) >= 2

    # 6. Verify transcript.json and storage files directly on disk
    session_dir = storage._get_session_dir(recording_id)
    transcript_disk_path = session_dir / "transcript" / "transcript.json"
    chunk_cache_path = session_dir / "transcript" / "chunk_transcripts.json"
    assert transcript_disk_path.exists()
    assert chunk_cache_path.exists()

    with open(transcript_disk_path, "r", encoding="utf-8") as f:
        disk_data = json.load(f)
        assert disk_data["recording_id"] == recording_id
        assert disk_data["schema_version"] == 1
        assert len(disk_data["segments"]) == len(full_transcript["segments"])

    # 7. Test Idempotency: second POST /transcribe should return instantly without duplicate work
    res_idempotent = client.post(f"/api/v1/recordings/{recording_id}/transcribe")
    assert res_idempotent.status_code == 200
    idemp_data = res_idempotent.json()
    assert idemp_data["status"] == "transcribed"
    assert "already transcribed" in idemp_data["message"].lower()
    assert idemp_data["segments_count"] == trans_data["segments_count"]
