import asyncio
import json
import shutil
import tempfile
import wave
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings
from app.schemas.recording import RecordingStatus
from app.schemas.processing import AudioChunk
from app.services.stt.base import BaseSTTProvider
from app.services.stt.exceptions import (
    STTAuthenticationError,
    STTConfigurationError,
    STTError,
    STTNetworkError,
    STTProviderError,
    STTRateLimitError,
    STTTimeoutError,
)
from app.services.stt.factory import STTProviderFactory
from app.services.stt.merger import TranscriptMerger
from app.services.stt.mock_provider import MockSTTProvider
from app.services.stt.models import (
    ChunkTranscriptionResult,
    Transcript,
    TranscriptSegment,
    TranscriptWord,
)
from app.services.stt.pipeline import TranscriptionPipeline
from app.services.stt.whisper_provider import WhisperSTTProvider
from app.services.storage.local import LocalStorage
import app.services.storage as storage_module
from app.api.v1 import recordings as recordings_api

client = TestClient(app)


def generate_wav(
    file_path: Path,
    duration_s: float = 2.0,
    sample_rate: int = 16000,
) -> Path:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    num_samples = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, num_samples, endpoint=False)
    waveform = (np.sin(2 * np.pi * 440.0 * t) * 0.5 * 32767).astype(np.int16)

    with wave.open(str(file_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(waveform.tobytes())

    return file_path


@pytest.fixture
def temp_dir():
    d = Path(tempfile.mkdtemp(prefix="eshitaman_stt_test_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def temp_storage(temp_dir, monkeypatch):
    storage = LocalStorage(base_path=temp_dir / "storage" / "recordings")
    monkeypatch.setattr(storage_module, "storage_service", storage)
    monkeypatch.setattr(recordings_api, "storage_service", storage)
    return storage


# =========================================================================
# 1. PROVIDER TESTS
# =========================================================================

def test_mock_provider_deterministic_multilingual(temp_dir):
    wav_path = temp_dir / "chunk.wav"
    generate_wav(wav_path, duration_s=3.0)

    provider = MockSTTProvider()

    async def _test():
        res0 = await provider.transcribe_chunk(wav_path, "chunk_0000")
        assert "machine learning" in res0.text.lower()
        assert res0.language == "uz"
        assert len(res0.segments) >= 1
        assert res0.segments[0].words

        res1 = await provider.transcribe_chunk(wav_path, "chunk_0001")
        assert "api backend" in res1.text.lower()

        res3 = await provider.transcribe_chunk(wav_path, "chunk_0003")
        assert "database architecture" in res3.text.lower()
        assert res3.language == "ru"

    asyncio.run(_test())


def test_whisper_provider_successful_transcription(temp_dir):
    wav_path = temp_dir / "chunk.wav"
    generate_wav(wav_path, duration_s=2.0)

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "text": "Bugun machine learning haqida gaplashamiz.",
        "language": "uz",
        "duration": 4.5,
        "segments": [
            {
                "id": 0,
                "start": 1.0,
                "end": 4.5,
                "text": "Bugun machine learning haqida gaplashamiz.",
                "avg_logprob": -0.15,
                "words": [
                    {"word": "Bugun", "start": 1.0, "end": 1.8, "probability": 0.98},
                    {"word": "machine", "start": 1.8, "end": 2.5, "probability": 0.95},
                ],
            }
        ],
    }
    mock_client.post.return_value = mock_response

    provider = WhisperSTTProvider(api_key="test-api-key", http_client=mock_client)

    async def _test():
        result = await provider.transcribe_chunk(wav_path, "chunk_0000")
        assert result.text == "Bugun machine learning haqida gaplashamiz."
        assert result.language == "uz"
        assert len(result.segments) == 1
        assert result.segments[0].start_ms == 1000
        assert result.segments[0].end_ms == 4500
        assert len(result.segments[0].words) == 2

    asyncio.run(_test())


def test_whisper_provider_auth_error_no_retry(temp_dir):
    wav_path = temp_dir / "chunk.wav"
    generate_wav(wav_path)

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 401
    mock_response.text = "Incorrect API key provided"
    mock_client.post.return_value = mock_response

    provider = WhisperSTTProvider(api_key="bad-key", max_retries=3, http_client=mock_client)

    async def _test():
        with pytest.raises(STTAuthenticationError):
            await provider.transcribe_chunk(wav_path, "chunk_0000")
        # Ensure it was called only once (no retries on 401)
        assert mock_client.post.call_count == 1

    asyncio.run(_test())


def test_whisper_provider_client_400_no_retry(temp_dir):
    wav_path = temp_dir / "chunk.wav"
    generate_wav(wav_path)

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 400
    mock_response.text = "Invalid format or parameters"
    mock_client.post.return_value = mock_response

    provider = WhisperSTTProvider(api_key="valid-key", max_retries=3, http_client=mock_client)

    async def _test():
        with pytest.raises(STTProviderError) as exc:
            await provider.transcribe_chunk(wav_path, "chunk_0000")
        assert exc.value.retryable is False
        assert mock_client.post.call_count == 1

    asyncio.run(_test())


def test_whisper_provider_rate_limit_retries_and_fails(temp_dir):
    wav_path = temp_dir / "chunk.wav"
    generate_wav(wav_path)

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 429
    mock_client.post.return_value = mock_response

    provider = WhisperSTTProvider(api_key="key", max_retries=2, http_client=mock_client)

    async def _test():
        with pytest.raises(STTRateLimitError):
            await provider.transcribe_chunk(wav_path, "chunk_0000")
        assert mock_client.post.call_count == 2

    asyncio.run(_test())


def test_whisper_provider_transient_error_recovers(temp_dir):
    """First attempt 500, second attempt 200 -> should succeed after retry."""
    wav_path = temp_dir / "chunk.wav"
    generate_wav(wav_path)

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    res_fail = MagicMock(spec=httpx.Response)
    res_fail.status_code = 500

    res_ok = MagicMock(spec=httpx.Response)
    res_ok.status_code = 200
    res_ok.json.return_value = {"text": "Recovered text", "language": "uz", "segments": []}

    mock_client.post.side_effect = [res_fail, res_ok]

    provider = WhisperSTTProvider(api_key="key", max_retries=3, http_client=mock_client)

    async def _test():
        res = await provider.transcribe_chunk(wav_path, "chunk_0000")
        assert res.text == "Recovered text"
        assert mock_client.post.call_count == 2

    asyncio.run(_test())


def test_factory_creation():
    mock_p = STTProviderFactory.create_provider("mock")
    assert isinstance(mock_p, MockSTTProvider)

    with patch.object(settings, "STT_API_KEY", "dummy-key"):
        whisper_p = STTProviderFactory.create_provider("whisper")
        assert isinstance(whisper_p, WhisperSTTProvider)

    with pytest.raises(STTConfigurationError):
        STTProviderFactory.create_provider("non_existent_provider_xyz")


# =========================================================================
# 2. TIMESTAMP & GLOBAL CONVERSION TESTS
# =========================================================================

def test_global_timestamp_conversion():
    """Verify chunk_offset + local_timestamp = global_timestamp."""
    merger = TranscriptMerger()

    chunk1 = AudioChunk(
        chunk_id="chunk_0000",
        recording_id="rec_1",
        index=0,
        start_ms=0,
        end_ms=30000,
        duration_ms=30000,
        file_path="/path/chunk_0.wav",
        file_size_bytes=1000,
    )
    chunk2 = AudioChunk(
        chunk_id="chunk_0001",
        recording_id="rec_1",
        index=1,
        start_ms=28000,  # 2s overlap
        end_ms=58000,
        duration_ms=30000,
        file_path="/path/chunk_1.wav",
        file_size_bytes=1000,
    )

    res1 = ChunkTranscriptionResult(
        chunk_id="chunk_0000",
        text="Birinchi jumla.",
        language="uz",
        segments=[
            TranscriptSegment(
                id="s1",
                start_ms=1000,
                end_ms=4000,
                text="Birinchi jumla.",
                words=[TranscriptWord(word="Birinchi", start_ms=1000, end_ms=2000)],
            )
        ],
    )
    res2 = ChunkTranscriptionResult(
        chunk_id="chunk_0001",
        text="Ikkinchi jumla.",
        language="uz",
        segments=[
            TranscriptSegment(
                id="s2",
                start_ms=5000,  # Relative to chunk2 (starts at 28000)
                end_ms=9000,
                text="Ikkinchi jumla.",
                words=[TranscriptWord(word="Ikkinchi", start_ms=5000, end_ms=7000)],
            )
        ],
    )

    transcript = merger.merge(
        recording_id="rec_1",
        chunk_items=[(chunk1, res1), (chunk2, res2)],
        total_duration_ms=58000,
    )

    assert len(transcript.segments) == 2
    # Segment 1: global start = 0 + 1000 = 1000, end = 0 + 4000 = 4000
    assert transcript.segments[0].start_ms == 1000
    assert transcript.segments[0].end_ms == 4000
    assert transcript.segments[0].words[0].start_ms == 1000

    # Segment 2: global start = 28000 + 5000 = 33000, end = 28000 + 9000 = 37000
    assert transcript.segments[1].start_ms == 33000
    assert transcript.segments[1].end_ms == 37000
    assert transcript.segments[1].words[0].start_ms == 33000


# =========================================================================
# 3. MERGE & DEDUPLICATION TESTS
# =========================================================================

def test_merge_overlap_deduplication():
    """Identical phrase falling into overlapping chunk region must be deduplicated."""
    merger = TranscriptMerger(overlap_tolerance_ms=3000)

    chunk1 = AudioChunk(
        chunk_id="chunk_0000",
        recording_id="rec_test",
        index=0,
        start_ms=0,
        end_ms=32000,
        duration_ms=32000,
        file_path="chunk_0.wav",
        file_size_bytes=100,
    )
    chunk2 = AudioChunk(
        chunk_id="chunk_0001",
        recording_id="rec_test",
        index=1,
        start_ms=29000,  # 3s overlap with chunk 1
        end_ms=60000,
        duration_ms=31000,
        file_path="chunk_1.wav",
        file_size_bytes=100,
    )

    # Chunk 1 ends with "Bu API backend bilan bog'lanadi." at global 29500 - 31500
    res1 = ChunkTranscriptionResult(
        chunk_id="chunk_0000",
        text="Birinchi qism. Bu API backend bilan bog'lanadi.",
        segments=[
            TranscriptSegment(id="s1", start_ms=1000, end_ms=15000, text="Birinchi qism."),
            TranscriptSegment(id="s2", start_ms=29500, end_ms=31500, text="Bu API backend bilan bog'lanadi."),
        ],
    )
    # Chunk 2 begins with the same phrase at chunk local 500 - 2500 (global 29500 - 31500)
    res2 = ChunkTranscriptionResult(
        chunk_id="chunk_0001",
        text="Bu API backend bilan bog'lanadi. Yangi ma'lumotlar.",
        segments=[
            TranscriptSegment(id="s3", start_ms=500, end_ms=2500, text="Bu API backend bilan bog'lanadi."),
            TranscriptSegment(id="s4", start_ms=4000, end_ms=12000, text="Yangi ma'lumotlar."),
        ],
    )

    merged = merger.merge("rec_test", [(chunk1, res1), (chunk2, res2)], total_duration_ms=60000)

    # Should have 3 unique segments, not 4 (the duplicate was dropped)
    assert len(merged.segments) == 3
    texts = [s.text for s in merged.segments]
    assert texts.count("Bu API backend bilan bog'lanadi.") == 1


def test_merge_preserves_legitimate_repeated_speech():
    """Speaker intentionally repeating a sentence at distinct times must be preserved."""
    merger = TranscriptMerger(overlap_tolerance_ms=2000)

    chunk1 = AudioChunk(
        chunk_id="chunk_0000",
        recording_id="rec_repeat",
        index=0,
        start_ms=0,
        end_ms=30000,
        duration_ms=30000,
        file_path="chunk_0.wav",
        file_size_bytes=100,
    )
    # Both repeated sentences occur within chunk 1 at distinct times (5s and 22s)
    res1 = ChunkTranscriptionResult(
        chunk_id="chunk_0000",
        text="Rahmat. ... Rahmat.",
        segments=[
            TranscriptSegment(id="s1", start_ms=5000, end_ms=6500, text="Rahmat."),
            TranscriptSegment(id="s2", start_ms=22000, end_ms=23500, text="Rahmat."),
        ],
    )

    merged = merger.merge("rec_repeat", [(chunk1, res1)], total_duration_ms=30000)
    assert len(merged.segments) == 2
    assert merged.segments[0].text == "Rahmat."
    assert merged.segments[1].text == "Rahmat."


# =========================================================================
# 4. LANGUAGE & CODE-SWITCHING TESTS
# =========================================================================

def test_multilingual_code_switching_preservation():
    """Verify uz, ru, en mixed speech is preserved verbatim without translation."""
    merger = TranscriptMerger()
    chunk1 = AudioChunk(
        chunk_id="chunk_0000",
        recording_id="rec_multi",
        index=0,
        start_ms=0,
        end_ms=30000,
        duration_ms=30000,
        file_path="chunk_0.wav",
        file_size_bytes=100,
    )

    res = ChunkTranscriptionResult(
        chunk_id="chunk_0000",
        text="Bugun API architecture haqida gaplashamiz. Сейчас мы рассмотрим database. Model accuracy is 95 percent.",
        language="uz",
        segments=[
            TranscriptSegment(id="s1", start_ms=1000, end_ms=5000, text="Bugun API architecture haqida gaplashamiz.", language="uz"),
            TranscriptSegment(id="s2", start_ms=6000, end_ms=10000, text="Сейчас мы рассмотрим database.", language="ru"),
            TranscriptSegment(id="s3", start_ms=11000, end_ms=16000, text="Model accuracy is 95 percent.", language="en"),
        ],
    )

    transcript = merger.merge("rec_multi", [(chunk1, res)], total_duration_ms=30000)
    assert transcript.language == "uz"
    assert transcript.segments[0].language == "uz"
    assert transcript.segments[1].language == "ru"
    assert transcript.segments[2].language == "en"

    # Verbatim text preservation (no forced translation)
    assert "Bugun API architecture haqida gaplashamiz." in transcript.text
    assert "Сейчас мы рассмотрим database." in transcript.text
    assert "Model accuracy is 95 percent." in transcript.text


# =========================================================================
# 5. PIPELINE & COST CONTROL TESTS
# =========================================================================

def test_pipeline_full_transcription_success(temp_storage, temp_dir):
    async def _run():
        recording_id = "rec_pipe_stt_ok"
        session_dir = temp_storage._get_session_dir(recording_id)
        session_dir.mkdir(parents=True, exist_ok=True)

        # Create audio and processed chunks
        audio_path = session_dir / "audio.wav"
        generate_wav(audio_path, duration_s=60.0)

        chunks_dir = session_dir / "processed" / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)
        chunk0_path = chunks_dir / "chunk_0000.wav"
        chunk1_path = chunks_dir / "chunk_0001.wav"
        generate_wav(chunk0_path, duration_s=30.0)
        generate_wav(chunk1_path, duration_s=30.0)

        chunks_meta = [
            {
                "chunk_id": "chunk_rec_0000",
                "recording_id": recording_id,
                "index": 0,
                "start_ms": 0,
                "end_ms": 30000,
                "duration_ms": 30000,
                "file_path": str(chunk0_path.as_posix()),
                "file_size_bytes": chunk0_path.stat().st_size,
            },
            {
                "chunk_id": "chunk_rec_0001",
                "recording_id": recording_id,
                "index": 1,
                "start_ms": 28000,
                "end_ms": 58000,
                "duration_ms": 30000,
                "file_path": str(chunk1_path.as_posix()),
                "file_size_bytes": chunk1_path.stat().st_size,
            },
        ]
        with open(session_dir / "processed" / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks_meta, f)

        # Metadata with status 'processed'
        meta = {
            "recording_id": recording_id,
            "status": RecordingStatus.PROCESSED.value,
            "duration_seconds": 60.0,
            "file_size_bytes": audio_path.stat().st_size,
        }
        with open(session_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

        mock_provider = MockSTTProvider()
        pipeline = TranscriptionPipeline(storage=temp_storage, provider=mock_provider)

        transcript = await pipeline.transcribe_recording(recording_id)

        assert transcript.recording_id == recording_id
        assert transcript.language == "uz"
        assert len(transcript.segments) >= 2
        assert (session_dir / "transcript" / "transcript.json").exists()

        # Check metadata updated to 'transcribed'
        updated_meta = await temp_storage.get_recording_metadata(recording_id)
        assert updated_meta["status"] == RecordingStatus.TRANSCRIBED.value
        assert updated_meta["transcript"] is not None
        assert updated_meta["transcript"]["segments_count"] == len(transcript.segments)

    asyncio.run(_run())


def test_pipeline_idempotency(temp_storage, temp_dir):
    """Calling transcribe on already transcribed recording returns existing transcript without re-calling provider."""
    async def _run():
        recording_id = "rec_idemp_stt"
        session_dir = temp_storage._get_session_dir(recording_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        transcript_dir = session_dir / "transcript"
        transcript_dir.mkdir(parents=True, exist_ok=True)

        existing_transcript = {
            "recording_id": recording_id,
            "schema_version": 1,
            "language": "uz",
            "duration_ms": 45000,
            "provider": "mock",
            "model": "mock-multilingual-v1",
            "created_at": "2026-09-26T00:00:00Z",
            "text": "Existing transcript text.",
            "segments": [
                {
                    "id": "seg_0000",
                    "start_ms": 0,
                    "end_ms": 5000,
                    "text": "Existing transcript text.",
                    "language": "uz",
                    "confidence": 0.95,
                    "chunk_id": "chunk_0000",
                    "words": [],
                }
            ],
        }
        with open(transcript_dir / "transcript.json", "w", encoding="utf-8") as f:
            json.dump(existing_transcript, f)

        meta = {
            "recording_id": recording_id,
            "status": RecordingStatus.TRANSCRIBED.value,
            "duration_seconds": 45.0,
        }
        with open(session_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

        mock_provider = MockSTTProvider()
        pipeline = TranscriptionPipeline(storage=temp_storage, provider=mock_provider)

        result = await pipeline.transcribe_recording(recording_id)
        assert result.text == "Existing transcript text."
        # Provider was never called due to idempotency!
        assert mock_provider.call_count == 0

    asyncio.run(_run())


def test_pipeline_cost_control_incremental_cache(temp_storage, temp_dir):
    """When a failure occurs on chunk 2, chunk 1 is cached and NOT re-transcribed on retry."""
    async def _run():
        recording_id = "rec_cost_control"
        session_dir = temp_storage._get_session_dir(recording_id)
        session_dir.mkdir(parents=True, exist_ok=True)

        audio_path = session_dir / "audio.wav"
        generate_wav(audio_path, duration_s=40.0)

        chunks_dir = session_dir / "processed" / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)
        c0 = chunks_dir / "chunk_0000.wav"
        c1 = chunks_dir / "chunk_0001.wav"
        generate_wav(c0)
        generate_wav(c1)

        chunks_meta = [
            {"chunk_id": "chunk_0000", "recording_id": recording_id, "index": 0, "start_ms": 0, "end_ms": 20000, "duration_ms": 20000, "file_path": str(c0.as_posix()), "file_size_bytes": 100},
            {"chunk_id": "chunk_0001", "recording_id": recording_id, "index": 1, "start_ms": 18000, "end_ms": 38000, "duration_ms": 20000, "file_path": str(c1.as_posix()), "file_size_bytes": 100},
        ]
        with open(session_dir / "processed" / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks_meta, f)

        meta = {"recording_id": recording_id, "status": RecordingStatus.PROCESSED.value}
        with open(session_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

        # Provider fails on chunk_0001 once
        provider1 = MockSTTProvider(fail_chunks=["chunk_0001"], fail_times_per_chunk=1)
        pipeline1 = TranscriptionPipeline(storage=temp_storage, provider=provider1)

        # Attempt 1: should transcribe chunk_0000, then fail on chunk_0001
        with pytest.raises(STTError):
            await pipeline1.transcribe_recording(recording_id)

        assert provider1.call_count == 2
        # Check that chunk_0000 was cached in chunk_transcripts.json
        cache_path = session_dir / "transcript" / "chunk_transcripts.json"
        assert cache_path.exists()
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
            assert "chunk_0000" in cached
            assert "chunk_0001" not in cached

        # Attempt 2: retry with provider that succeeds. chunk_0000 should be read from cache!
        provider2 = MockSTTProvider()
        pipeline2 = TranscriptionPipeline(storage=temp_storage, provider=provider2)
        # Restore status to processed
        meta["status"] = RecordingStatus.PROCESSED.value
        await temp_storage.update_recording_metadata(recording_id, meta)

        res = await pipeline2.transcribe_recording(recording_id)
        assert res is not None
        # In attempt 2, provider2 should only be called ONCE (for chunk_0001), saving cost!
        assert provider2.call_count == 1

    asyncio.run(_run())


def test_pipeline_unprocessed_recording_fails(temp_storage, temp_dir):
    async def _run():
        recording_id = "rec_not_proc"
        session_dir = temp_storage._get_session_dir(recording_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        # Status is still 'uploaded', not 'processed'
        meta = {"recording_id": recording_id, "status": RecordingStatus.UPLOADED.value}
        with open(session_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

        pipeline = TranscriptionPipeline(storage=temp_storage, provider=MockSTTProvider())
        with pytest.raises(STTError) as exc:
            await pipeline.transcribe_recording(recording_id)
        assert exc.value.error_code == "RECORDING_NOT_PROCESSED"

    asyncio.run(_run())


# =========================================================================
# 6. API ENDPOINTS TESTS
# =========================================================================

def test_api_transcribe_and_get_transcript(temp_storage, temp_dir):
    recording_id = "rec_api_stt_test"
    wav_path = temp_dir / "test.wav"
    generate_wav(wav_path, duration_s=10.0)

    # 1. Upload
    res_up = client.post(
        "/api/v1/recordings",
        data={"recording_id": recording_id, "duration_seconds": "10.0"},
        files={"file": ("test.wav", wav_path.read_bytes(), "audio/wav")},
    )
    assert res_up.status_code == 201

    # 2. Process
    res_proc = client.post(f"/api/v1/recordings/{recording_id}/process")
    assert res_proc.status_code == 200

    # 3. GET /transcript before transcription returns 404
    res_pre_trans = client.get(f"/api/v1/recordings/{recording_id}/transcript")
    assert res_pre_trans.status_code == 404

    # 4. Transcribe
    res_trans = client.post(f"/api/v1/recordings/{recording_id}/transcribe")
    assert res_trans.status_code == 200
    data = res_trans.json()
    assert data["recording_id"] == recording_id
    assert data["status"] == "transcribed"
    assert data["segments_count"] >= 1
    assert data["language"] == "uz"

    # 5. GET /transcript returns full Transcript schema
    res_full = client.get(f"/api/v1/recordings/{recording_id}/transcript")
    assert res_full.status_code == 200
    full_data = res_full.json()
    assert full_data["recording_id"] == recording_id
    assert full_data["schema_version"] == 1
    assert len(full_data["segments"]) == data["segments_count"]
    assert "text" in full_data
    assert len(full_data["text"]) > 0

    # 6. GET /recordings/{id} returns updated metadata with transcript summary
    res_get = client.get(f"/api/v1/recordings/{recording_id}")
    assert res_get.status_code == 200
    meta_data = res_get.json()
    assert meta_data["status"] == "transcribed"
    assert meta_data["transcript"] is not None
    assert meta_data["transcript"]["segments_count"] == data["segments_count"]

    # 7. Idempotency test: POST /transcribe a second time returns 200 with idempotent message
    res_idemp = client.post(f"/api/v1/recordings/{recording_id}/transcribe")
    assert res_idemp.status_code == 200
    assert "already transcribed" in res_idemp.json()["message"].lower()


def test_api_transcribe_unprocessed_recording_returns_400(temp_storage, temp_dir):
    recording_id = "rec_api_unproc"
    wav_path = temp_dir / "test.wav"
    generate_wav(wav_path, duration_s=5.0)

    client.post(
        "/api/v1/recordings",
        data={"recording_id": recording_id},
        files={"file": ("test.wav", wav_path.read_bytes(), "audio/wav")},
    )

    res = client.post(f"/api/v1/recordings/{recording_id}/transcribe")
    assert res.status_code == 400
    assert "RECORDING_NOT_PROCESSED" in res.json()["detail"]["error_code"]


def test_api_transcribe_nonexistent_recording_returns_404():
    res = client.post("/api/v1/recordings/rec_non_existent_xyz/transcribe")
    assert res.status_code == 404


def test_api_transcribe_with_json_body_language_hint(temp_storage, temp_dir):
    recording_id = "rec_api_body_hint"
    wav_path = temp_dir / "test_hint.wav"
    generate_wav(wav_path, duration_s=6.0)

    # 1. Upload & Process
    client.post(
        "/api/v1/recordings",
        data={"recording_id": recording_id},
        files={"file": ("test_hint.wav", wav_path.read_bytes(), "audio/wav")},
    )
    client.post(f"/api/v1/recordings/{recording_id}/process")

    # 2. Transcribe with JSON body
    res = client.post(
        f"/api/v1/recordings/{recording_id}/transcribe",
        json={"language_hint": "uz"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["recording_id"] == recording_id
    assert data["status"] == "transcribed"
    assert data["language"] == "uz"
