import json
import shutil
import tempfile
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings
from app.schemas.recording import RecordingStatus
from app.schemas.processing import AudioInfo, SpeechSegment
from app.services.audio.exceptions import (
    AudioProcessingError,
    AudioInspectionError,
    AudioNormalizationError,
    FFmpegNotFoundError,
    VADError,
    ChunkingError,
)
from app.services.audio.inspector import AudioInspector
from app.services.audio.normalizer import AudioNormalizer
from app.services.audio.vad import EnergyVAD
from app.services.audio.chunker import AudioChunker
from app.services.audio.pipeline import AudioProcessingPipeline
from app.services.storage.local import LocalStorage
import app.services.storage as storage_module
from app.api.v1 import recordings as recordings_api

client = TestClient(app)


def generate_wav(
    file_path: Path,
    duration_s: float = 2.0,
    sample_rate: int = 16000,
    channels: int = 1,
    amplitude: float = 0.5,
    freq: float = 440.0,
) -> Path:
    """Generate a valid 16-bit PCM WAV file programmatically."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    num_samples = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, num_samples, endpoint=False)
    waveform = (np.sin(2 * np.pi * freq * t) * amplitude * 32767).astype(np.int16)

    if channels == 2:
        waveform = np.column_stack((waveform, waveform)).flatten()

    with wave.open(str(file_path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(waveform.tobytes())

    return file_path


def generate_speech_silence_wav(file_path: Path, sample_rate: int = 16000) -> Path:
    """
    Creates audio pattern:
    0.0s - 1.0s: silence
    1.0s - 3.0s: speech (tone)
    3.0s - 4.5s: silence
    4.5s - 6.5s: speech (tone)
    6.5s - 7.5s: silence
    Total: 7.5s
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    silence_1s = np.zeros(int(1.0 * sample_rate), dtype=np.int16)
    silence_1_5s = np.zeros(int(1.5 * sample_rate), dtype=np.int16)

    t2 = np.linspace(0, 2.0, int(2.0 * sample_rate), endpoint=False)
    speech_2s = (np.sin(2 * np.pi * 300.0 * t2) * 0.7 * 32767).astype(np.int16)

    full_signal = np.concatenate([
        silence_1s,
        speech_2s,
        silence_1_5s,
        speech_2s,
        silence_1s,
    ])

    with wave.open(str(file_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(full_signal.tobytes())

    return file_path


@pytest.fixture
def temp_dir():
    d = Path(tempfile.mkdtemp(prefix="eshitaman_audio_test_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def temp_storage(temp_dir, monkeypatch):
    storage = LocalStorage(base_path=temp_dir / "storage" / "recordings")
    monkeypatch.setattr(storage_module, "storage_service", storage)
    monkeypatch.setattr(recordings_api, "storage_service", storage)
    return storage


# =========================================================================
# 1. AUDIO INSPECTION TESTS
# =========================================================================

def test_inspect_valid_wav(temp_dir):
    wav_path = temp_dir / "sample.wav"
    generate_wav(wav_path, duration_s=3.0, sample_rate=16000, channels=1)

    inspector = AudioInspector()
    info = inspector.inspect(wav_path)

    assert info.duration_seconds == 3.0
    assert info.sample_rate == 16000
    assert info.channels == 1
    assert info.codec == "pcm_s16le"
    assert info.container == "wav"
    assert info.file_size_bytes == wav_path.stat().st_size
    assert info.bitrate == 16000 * 1 * 2 * 8


def test_inspect_corrupted_wav(temp_dir):
    corrupted_path = temp_dir / "corrupted.wav"
    corrupted_path.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt \x00\x00\x00")

    inspector = AudioInspector()
    with pytest.raises(AudioInspectionError):
        inspector.inspect(corrupted_path)


def test_inspect_empty_file(temp_dir):
    empty_path = temp_dir / "empty.wav"
    empty_path.write_bytes(b"")

    inspector = AudioInspector()
    with pytest.raises(AudioInspectionError) as exc:
        inspector.inspect(empty_path)
    assert "empty" in str(exc.value).lower()


def test_inspect_missing_file(temp_dir):
    missing_path = temp_dir / "does_not_exist.wav"
    inspector = AudioInspector()
    with pytest.raises(AudioInspectionError):
        inspector.inspect(missing_path)


def test_inspect_compressed_without_ffprobe(temp_dir):
    """When ffprobe is unavailable, inspecting compressed files raises FFmpegNotFoundError."""
    m4a_path = temp_dir / "audio.m4a"
    m4a_path.write_bytes(b"dummy m4a bytes")

    mock_ffmpeg = MagicMock()
    mock_ffmpeg.is_ffprobe_available.return_value = False
    mock_ffmpeg.inspect_audio.side_effect = FFmpegNotFoundError()

    inspector = AudioInspector(ffmpeg_service=mock_ffmpeg)
    with pytest.raises(FFmpegNotFoundError):
        inspector.inspect(m4a_path)


def test_inspect_compressed_with_ffprobe(temp_dir):
    """When ffprobe is available and succeeds, AudioInfo is returned."""
    m4a_path = temp_dir / "audio.m4a"
    m4a_path.write_bytes(b"dummy m4a bytes")

    mock_ffmpeg = MagicMock()
    mock_ffmpeg.inspect_audio.return_value = AudioInfo(
        duration_seconds=12.5,
        sample_rate=44100,
        channels=2,
        codec="aac",
        container="mov,mp4,m4a",
        bitrate=128000,
        file_size_bytes=len(b"dummy m4a bytes"),
    )

    inspector = AudioInspector(ffmpeg_service=mock_ffmpeg)
    info = inspector.inspect(m4a_path)
    assert info.duration_seconds == 12.5
    assert info.sample_rate == 44100
    assert info.codec == "aac"


# =========================================================================
# 2. AUDIO NORMALIZATION TESTS
# =========================================================================

def test_normalization_peak_scaling(temp_dir):
    input_wav = temp_dir / "quiet.wav"
    output_wav = temp_dir / "normalized.wav"

    # Quiet sine wave (peak ~ 0.1 * 32767 = 3276)
    generate_wav(input_wav, duration_s=1.0, amplitude=0.1)

    normalizer = AudioNormalizer(target_peak_dbfs=-1.0, max_gain_factor=10.0)
    normalizer.normalize(input_wav, output_wav)

    assert output_wav.exists()
    assert output_wav.stat().st_size > 0

    # Verify output peak is close to target peak (-1.0 dBFS = ~29204)
    with wave.open(str(output_wav), "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        data = np.frombuffer(frames, dtype=np.int16)
        peak = np.max(np.abs(data))
        assert peak > 28000
        assert peak <= 32767  # Absolutely no clipping


def test_normalization_preserves_original_file(temp_dir):
    input_wav = temp_dir / "original.wav"
    output_wav = temp_dir / "norm.wav"
    generate_wav(input_wav, duration_s=1.0, amplitude=0.2)
    original_bytes = input_wav.read_bytes()

    normalizer = AudioNormalizer()
    normalizer.normalize(input_wav, output_wav)

    # Original must not be modified
    assert input_wav.read_bytes() == original_bytes


def test_normalization_pure_silence(temp_dir):
    input_wav = temp_dir / "silence.wav"
    output_wav = temp_dir / "norm_silence.wav"
    input_wav.parent.mkdir(parents=True, exist_ok=True)

    # Pure zeros
    with wave.open(str(input_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(np.zeros(16000, dtype=np.int16).tobytes())

    normalizer = AudioNormalizer()
    normalizer.normalize(input_wav, output_wav)

    with wave.open(str(output_wav), "rb") as wf:
        data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        assert np.max(np.abs(data)) == 0


def test_normalization_missing_file(temp_dir):
    normalizer = AudioNormalizer()
    with pytest.raises(AudioNormalizationError):
        normalizer.normalize(temp_dir / "missing.wav", temp_dir / "out.wav")


# =========================================================================
# 3. VAD / SILENCE DETECTION TESTS
# =========================================================================

def test_vad_pure_silence(temp_dir):
    wav_path = temp_dir / "pure_silence.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(np.zeros(16000 * 3, dtype=np.int16).tobytes())

    vad = EnergyVAD(energy_threshold=0.015)
    segments = vad.detect_speech_segments(wav_path)
    assert len(segments) == 0


def test_vad_continuous_speech(temp_dir):
    wav_path = temp_dir / "continuous_speech.wav"
    generate_wav(wav_path, duration_s=4.0, amplitude=0.6, freq=300.0)

    vad = EnergyVAD(energy_threshold=0.015)
    segments = vad.detect_speech_segments(wav_path)

    assert len(segments) == 1
    assert segments[0].start_ms == 0
    assert segments[0].end_ms >= 3900


def test_vad_mixed_speech_and_silence(temp_dir):
    wav_path = temp_dir / "mixed.wav"
    generate_speech_silence_wav(wav_path)

    vad = EnergyVAD(energy_threshold=0.015, min_speech_duration_ms=250, min_silence_duration_ms=300)
    segments = vad.detect_speech_segments(wav_path)

    # Should detect 2 distinct speech segments
    assert len(segments) == 2
    # First speech segment around 1.0s - 3.0s (1000ms - 3000ms)
    assert 800 <= segments[0].start_ms <= 1200
    assert 2800 <= segments[0].end_ms <= 3300

    # Second speech segment around 4.5s - 6.5s (4500ms - 6500ms)
    assert 4300 <= segments[1].start_ms <= 4700
    assert 6300 <= segments[1].end_ms <= 6700


def test_vad_detect_silence_segments(temp_dir):
    wav_path = temp_dir / "mixed_silence.wav"
    generate_speech_silence_wav(wav_path)

    vad = EnergyVAD(energy_threshold=0.015)
    silences = vad.detect_silence_segments(wav_path, total_duration_ms=7500)

    # Silence at beginning, middle, and end
    assert len(silences) >= 2


# =========================================================================
# 4. CHUNKING TESTS
# =========================================================================

def test_chunking_short_audio(temp_dir):
    """Audio shorter than chunk duration produces exactly 1 chunk."""
    wav_path = temp_dir / "short.wav"
    generate_wav(wav_path, duration_s=10.0)

    chunks_dir = temp_dir / "chunks"
    chunker = AudioChunker(chunk_duration_s=30.0, overlap_s=2.0)
    chunks = chunker.chunk("rec_short", wav_path, chunks_dir)

    assert len(chunks) == 1
    assert chunks[0].index == 0
    assert chunks[0].start_ms == 0
    assert chunks[0].end_ms == 10000
    assert chunks[0].duration_ms == 10000
    assert Path(chunks[0].file_path).exists()


def test_chunking_long_audio_with_overlap(temp_dir):
    """Audio longer than 30s produces multiple chunks with overlap."""
    wav_path = temp_dir / "long.wav"
    # 70 seconds audio
    generate_wav(wav_path, duration_s=70.0)

    chunks_dir = temp_dir / "chunks"
    chunker = AudioChunker(chunk_duration_s=30.0, overlap_s=2.0)
    chunks = chunker.chunk("rec_long", wav_path, chunks_dir)

    assert len(chunks) >= 3
    # Check consecutive chunk overlap
    for i in range(len(chunks) - 1):
        assert chunks[i + 1].start_ms < chunks[i].end_ms
        overlap_ms = chunks[i].end_ms - chunks[i + 1].start_ms
        assert overlap_ms > 0

    # Check chunks.json was created
    chunks_json = chunks_dir.parent / "chunks.json"
    assert chunks_json.exists()
    with open(chunks_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
        assert len(meta) == len(chunks)


def test_chunking_boundary_alignment_with_vad(temp_dir):
    """Chunk boundaries adjust to silence gaps when speech segments are provided."""
    wav_path = temp_dir / "speech_and_silence.wav"
    generate_wav(wav_path, duration_s=60.0)

    # Speech from 0-28s, silence 28-32s, speech 32-60s
    speech_segments = [
        SpeechSegment(start_ms=0, end_ms=28000, duration_ms=28000),
        SpeechSegment(start_ms=32000, end_ms=60000, duration_ms=28000),
    ]

    chunks_dir = temp_dir / "chunks"
    chunker = AudioChunker(chunk_duration_s=30.0, overlap_s=2.0, boundary_search_window_s=3.0)
    chunks = chunker.chunk("rec_aligned", wav_path, chunks_dir, speech_segments=speech_segments)

    # The split should be placed in the silence gap (around 30000ms), not cutting speech
    first_chunk_end = chunks[0].end_ms
    assert 28000 <= first_chunk_end <= 32000


# =========================================================================
# 5. PIPELINE TESTS
# =========================================================================

import asyncio


def test_full_pipeline_success(temp_storage, temp_dir):
    async def _run():
        recording_id = "rec_pipeline_success"
        wav_path = temp_dir / "source.wav"
        generate_wav(wav_path, duration_s=15.0, sample_rate=16000, channels=1)

        # Save to storage
        await temp_storage.save_recording(
            recording_id=recording_id,
            audio_content=wav_path.read_bytes(),
            filename="audio.wav",
            metadata={
                "recording_id": recording_id,
                "filename": "audio.wav",
                "file_size": wav_path.stat().st_size,
                "file_size_bytes": wav_path.stat().st_size,
                "duration_seconds": 15.0,
                "mime_type": "audio/wav",
                "codec": "pcm_s16le",
                "sample_rate": 16000,
                "channel_count": 1,
                "status": RecordingStatus.UPLOADED.value,
            },
        )

        pipeline = AudioProcessingPipeline(storage=temp_storage)
        result = await pipeline.process_recording(recording_id)

        assert result.status == "processed"
        assert result.recording_id == recording_id
        assert result.chunks_count >= 1
        assert Path(result.processed_audio_path).exists()

        # Check updated metadata in storage
        meta = await temp_storage.get_recording_metadata(recording_id)
        assert meta["status"] == RecordingStatus.PROCESSED.value
        assert "audio" in meta
        assert meta["audio"]["sample_rate"] == 16000
        assert "processing" in meta
        assert meta["processing"]["chunks_count"] == result.chunks_count

    asyncio.run(_run())


def test_pipeline_idempotency(temp_storage, temp_dir):
    """Calling process on already processed recording returns existing result without re-processing."""
    async def _run():
        recording_id = "rec_idempotent"
        wav_path = temp_dir / "source.wav"
        generate_wav(wav_path, duration_s=8.0)

        await temp_storage.save_recording(
            recording_id=recording_id,
            audio_content=wav_path.read_bytes(),
            filename="audio.wav",
            metadata={
                "recording_id": recording_id,
                "filename": "audio.wav",
                "file_size": wav_path.stat().st_size,
                "status": RecordingStatus.UPLOADED.value,
            },
        )

        pipeline = AudioProcessingPipeline(storage=temp_storage)
        # First run
        res1 = await pipeline.process_recording(recording_id)
        assert res1.status == "processed"

        # Second run (idempotency check)
        res2 = await pipeline.process_recording(recording_id)
        assert res2.status == "processed"
        assert res2.chunks_count == res1.chunks_count

    asyncio.run(_run())


def test_pipeline_nonexistent_recording(temp_storage):
    async def _run():
        pipeline = AudioProcessingPipeline(storage=temp_storage)
        with pytest.raises(AudioProcessingError) as exc:
            await pipeline.process_recording("rec_missing_999")
        assert exc.value.error_code == "RECORDING_NOT_FOUND"

    asyncio.run(_run())


def test_pipeline_missing_audio_file(temp_storage):
    async def _run():
        recording_id = "rec_no_audio"
        # Create session directory with metadata only, no audio.*
        session_dir = temp_storage._get_session_dir(recording_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        meta = {"recording_id": recording_id, "status": "uploaded"}
        with open(session_dir / "metadata.json", "w") as f:
            json.dump(meta, f)

        pipeline = AudioProcessingPipeline(storage=temp_storage)
        with pytest.raises(AudioProcessingError) as exc:
            await pipeline.process_recording(recording_id)
        assert exc.value.error_code == "AUDIO_FILE_MISSING"

        # Check status changed to failed
        updated_meta = await temp_storage.get_recording_metadata(recording_id)
        assert updated_meta["status"] == "failed"

    asyncio.run(_run())



# =========================================================================
# 6. API TESTS
# =========================================================================

def test_api_process_recording_success(temp_storage, temp_dir):
    recording_id = "rec_api_process_ok"
    wav_path = temp_dir / "api_audio.wav"
    generate_wav(wav_path, duration_s=12.0)

    # Upload first
    up_res = client.post(
        "/api/v1/recordings",
        data={"recording_id": recording_id, "duration_seconds": "12.0"},
        files={"file": ("api_audio.wav", wav_path.read_bytes(), "audio/wav")},
    )
    assert up_res.status_code == 201

    # POST /process
    proc_res = client.post(f"/api/v1/recordings/{recording_id}/process")
    assert proc_res.status_code == 200
    data = proc_res.json()
    assert data["recording_id"] == recording_id
    assert data["status"] == "processed"
    assert data["chunks_count"] >= 1
    assert data["duration_seconds"] == 12.0

    # GET /recordings/{id} verifies expanded processing info
    get_res = client.get(f"/api/v1/recordings/{recording_id}")
    assert get_res.status_code == 200
    get_data = get_res.json()
    assert get_data["status"] == "processed"
    assert get_data["audio"] is not None
    assert get_data["audio"]["sample_rate"] == 16000
    assert get_data["processing"] is not None
    assert get_data["processing"]["chunks_count"] == data["chunks_count"]


def test_api_process_idempotent(temp_storage, temp_dir):
    recording_id = "rec_api_idempotent"
    wav_path = temp_dir / "audio.wav"
    generate_wav(wav_path, duration_s=5.0)

    client.post(
        "/api/v1/recordings",
        data={"recording_id": recording_id},
        files={"file": ("audio.wav", wav_path.read_bytes(), "audio/wav")},
    )

    # First call
    r1 = client.post(f"/api/v1/recordings/{recording_id}/process")
    assert r1.status_code == 200

    # Second call (idempotent)
    r2 = client.post(f"/api/v1/recordings/{recording_id}/process")
    assert r2.status_code == 200
    assert r2.json()["status"] == "processed"
    assert "already processed" in r2.json()["message"].lower()


def test_api_process_not_found():
    res = client.post("/api/v1/recordings/rec_not_found_xyz/process")
    assert res.status_code == 404


def test_ffmpeg_conversion_command_args(temp_dir):
    """Verify FFmpeg conversion passes canonical parameters (16kHz, mono, pcm_s16le)."""
    from app.services.audio.ffmpeg import FFmpegService
    svc = FFmpegService(ffmpeg_path="dummy_ffmpeg", ffprobe_path="dummy_ffprobe")

    input_file = temp_dir / "in.m4a"
    input_file.write_bytes(b"dummy")
    output_file = temp_dir / "out.wav"

    with patch("shutil.which", return_value="dummy_ffmpeg"):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            # Simulate output file creation
            def side_effect(*args, **kwargs):
                output_file.write_bytes(b"dummy wav data")
                return MagicMock(returncode=0, stdout="", stderr="")
            mock_run.side_effect = side_effect

            svc.convert_to_canonical_wav(input_file, output_file, sample_rate=16000, channels=1)

            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert "-acodec" in cmd
            assert cmd[cmd.index("-acodec") + 1] == "pcm_s16le"
            assert "-ac" in cmd
            assert cmd[cmd.index("-ac") + 1] == "1"
            assert "-ar" in cmd
            assert cmd[cmd.index("-ar") + 1] == "16000"


def test_chunking_exact_30s_audio(temp_dir):
    wav_path = temp_dir / "exact_30s.wav"
    generate_wav(wav_path, duration_s=30.0)

    chunks_dir = temp_dir / "chunks_30s"
    chunker = AudioChunker(chunk_duration_s=30.0, overlap_s=2.0)
    chunks = chunker.chunk("rec_30s", wav_path, chunks_dir)

    assert len(chunks) == 1
    assert chunks[0].duration_ms == 30000


def test_chunking_exact_60s_audio(temp_dir):
    wav_path = temp_dir / "exact_60s.wav"
    generate_wav(wav_path, duration_s=60.0)

    chunks_dir = temp_dir / "chunks_60s"
    chunker = AudioChunker(chunk_duration_s=30.0, overlap_s=2.0)
    chunks = chunker.chunk("rec_60s", wav_path, chunks_dir)

    assert len(chunks) >= 2
    for c in chunks:
        assert Path(c.file_path).exists()


def test_pipeline_corrupted_audio(temp_storage, temp_dir):
    async def _run():
        recording_id = "rec_corrupted"
        corrupt_path = temp_dir / "corrupt.wav"
        corrupt_path.write_bytes(b"NOT A WAV FILE CONTENT AT ALL")

        await temp_storage.save_recording(
            recording_id=recording_id,
            audio_content=corrupt_path.read_bytes(),
            filename="audio.wav",
            metadata={"recording_id": recording_id, "status": "uploaded"},
        )

        pipeline = AudioProcessingPipeline(storage=temp_storage)
        with pytest.raises(AudioProcessingError):
            await pipeline.process_recording(recording_id)

        meta = await temp_storage.get_recording_metadata(recording_id)
        assert meta["status"] == "failed"
        assert meta["error_code"] is not None

    asyncio.run(_run())


def test_api_process_failed_state_retrieval(temp_storage, temp_dir):
    recording_id = "rec_api_failure"
    corrupt_path = temp_dir / "bad.wav"
    corrupt_path.write_bytes(b"INVALID DATA")

    client.post(
        "/api/v1/recordings",
        data={"recording_id": recording_id},
        files={"file": ("bad.wav", corrupt_path.read_bytes(), "audio/wav")},
    )

    # Process fails
    proc_res = client.post(f"/api/v1/recordings/{recording_id}/process")
    assert proc_res.status_code == 422
    assert "error_code" in proc_res.json()["detail"]

    # Status in GET is failed
    get_res = client.get(f"/api/v1/recordings/{recording_id}")
    assert get_res.status_code == 200
    assert get_res.json()["status"] == "failed"
    assert get_res.json()["error_code"] is not None

