import shutil
import tempfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings
from app.services.storage.local import LocalStorage
import app.services.storage as storage_module

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_temp_storage(monkeypatch):
    """Use a temporary directory for storage during each test."""
    temp_dir = Path(tempfile.mkdtemp(prefix="eshitaman_test_storage_"))
    test_storage = LocalStorage(base_path=temp_dir / "recordings")
    monkeypatch.setattr(storage_module, "storage_service", test_storage)
    # Also patch where endpoints import storage_service
    from app.api.v1 import recordings
    monkeypatch.setattr(recordings, "storage_service", test_storage)

    yield test_storage

    shutil.rmtree(temp_dir, ignore_errors=True)


def test_successful_audio_upload_m4a(setup_temp_storage):
    """Test successful upload of an M4A audio recording with metadata."""
    fake_audio = b"\x00\x00\x00\x20ftypM4A \x00\x00\x00\x00M4A mp42isom" + b"\x00" * 1024
    recording_id = "rec_test_12345"

    response = client.post(
        "/api/v1/recordings",
        data={
            "recording_id": recording_id,
            "duration_seconds": "45.5",
            "duration_millis": "45500",
            "mime_type": "audio/m4a",
            "codec": "aac_lc",
            "sample_rate": "44100",
            "bit_rate": "128000",
            "channel_count": "1",
        },
        files={"file": ("recording.m4a", fake_audio, "audio/m4a")},
    )

    assert response.status_code == 201
    data = response.json()
    assert data["recording_id"] == recording_id
    assert data["status"] == "uploaded"
    assert data["file_size"] == len(fake_audio)
    assert data["duration"] == 45.5
    assert data["mime_type"] == "audio/m4a"

    # Verify storage
    audio_path = setup_temp_storage.base_path / recording_id / "audio.m4a"
    meta_path = setup_temp_storage.base_path / recording_id / "metadata.json"
    assert audio_path.exists()
    assert meta_path.exists()
    assert audio_path.read_bytes() == fake_audio


def test_successful_audio_upload_webm(setup_temp_storage):
    """Test successful upload of a WebM audio recording (web platform target)."""
    fake_audio = b"\x1A\x45\xDF\xA3" + b"\x00" * 512
    recording_id = "rec_webm_54321"

    response = client.post(
        "/api/v1/recordings",
        data={
            "recording_id": recording_id,
            "duration_seconds": "15.0",
            "mime_type": "audio/webm;codecs=opus",
        },
        files={"file": ("audio.webm", fake_audio, "audio/webm")},
    )

    assert response.status_code == 201
    data = response.json()
    assert data["recording_id"] == recording_id
    assert data["status"] == "uploaded"
    assert data["file_size"] == len(fake_audio)


def test_empty_audio_file_rejected():
    """Empty files must be rejected with 400."""
    response = client.post(
        "/api/v1/recordings",
        data={"recording_id": "rec_empty"},
        files={"file": ("empty.m4a", b"", "audio/m4a")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_unsupported_extension_rejected():
    """Executable or text files must be rejected."""
    response = client.post(
        "/api/v1/recordings",
        data={"recording_id": "rec_malicious"},
        files={"file": ("malicious.exe", b"malicious binary payload", "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "unsupported file extension" in response.json()["detail"].lower()


def test_unsupported_mime_type_rejected():
    """Unsupported MIME type must be rejected."""
    response = client.post(
        "/api/v1/recordings",
        data={"recording_id": "rec_mime_bad"},
        files={"file": ("sample.m4a", b"some bytes", "application/pdf")},
    )
    assert response.status_code == 400
    assert "unsupported audio mime type" in response.json()["detail"].lower()


def test_oversized_audio_file_rejected(monkeypatch):
    """Files exceeding maximum allowed size must be rejected with 413."""
    monkeypatch.setattr(settings, "MAX_AUDIO_SIZE_MB", 1)  # 1 MB limit
    huge_data = b"X" * (1024 * 1024 + 100)

    response = client.post(
        "/api/v1/recordings",
        data={"recording_id": "rec_oversized"},
        files={"file": ("huge.m4a", huge_data, "audio/m4a")},
    )
    assert response.status_code == 413
    assert "exceeds maximum allowed size" in response.json()["detail"].lower()


def test_duplicate_recording_id_rejected():
    """Attempting to upload with an existing recording ID returns 409 Conflict."""
    payload = b"\x00\x00\x00\x20ftypM4A " + b"\x00" * 100
    recording_id = "rec_duplicate_id"

    # First upload
    res1 = client.post(
        "/api/v1/recordings",
        data={"recording_id": recording_id},
        files={"file": ("first.m4a", payload, "audio/m4a")},
    )
    assert res1.status_code == 201

    # Second upload with same ID
    res2 = client.post(
        "/api/v1/recordings",
        data={"recording_id": recording_id},
        files={"file": ("second.m4a", payload, "audio/m4a")},
    )
    assert res2.status_code == 409
    assert "already exists" in res2.json()["detail"].lower()


def test_invalid_recording_id_path_traversal():
    """Path traversal attempt in recording_id must be blocked."""
    response = client.post(
        "/api/v1/recordings",
        data={"recording_id": "../../etc_passwd"},
        files={"file": ("test.m4a", b"bytes", "audio/m4a")},
    )
    assert response.status_code == 400
    assert "alphanumeric" in response.json()["detail"].lower()


def test_get_uploaded_recording_status():
    """Retrieve uploaded recording status by ID."""
    recording_id = "rec_status_check"
    client.post(
        "/api/v1/recordings",
        data={"recording_id": recording_id, "duration_seconds": "12.3"},
        files={"file": ("audio.m4a", b"audio payload bytes", "audio/m4a")},
    )

    response = client.get(f"/api/v1/recordings/{recording_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["recording_id"] == recording_id
    assert data["status"] == "uploaded"
    assert data["duration"] == 12.3


def test_get_nonexistent_recording():
    """Querying a non-existent recording returns 404."""
    response = client.get("/api/v1/recordings/rec_does_not_exist_999")
    assert response.status_code == 404
