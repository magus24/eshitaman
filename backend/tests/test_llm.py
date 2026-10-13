import asyncio
import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch
import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings
from app.schemas.recording import RecordingStatus
from app.services.storage.local import LocalStorage
import app.services.storage as storage_module
from app.api.v1 import recordings as recordings_api
from app.services.stt.models import Transcript, TranscriptSegment
from app.services.llm.base import BaseLLMProvider
from app.services.llm.chunker import TranscriptChunker
from app.services.llm.exceptions import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMContextLengthError,
    LLMError,
    LLMInvalidResponseError,
    LLMNetworkError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from app.services.llm.factory import LLMProviderFactory
from app.services.llm.mock_provider import MockLLMProvider
from app.services.llm.models import AIAnalysis, AITerm
from app.services.llm.openai_provider import OpenAILLMProvider
from app.services.llm.parser import LLMResponseParser
from app.services.llm.pipeline import AIUnderstandingPipeline

client = TestClient(app)


def build_sample_transcript(
    recording_id: str = "rec_sample_01",
    language: str = "uz",
    texts: list[str] = None,
) -> Transcript:
    default_texts = [
        "Bugun machine learning modelining accuracy ko'rsatkichi 95 foiz bo'ldi.",
        "Backend qismida FastAPI freymvorki va Python ishlatildi.",
        "Ma'lumotlar bazasi arxitekturasi mikroxizmatlar bilan bog'landi.",
    ]
    lines = texts or default_texts
    segments = []
    for i, t in enumerate(lines):
        segments.append(
            TranscriptSegment(
                id=f"seg_{i:04d}",
                start_ms=i * 5000,
                end_ms=(i + 1) * 5000,
                text=t,
                language=language,
                chunk_id=f"chunk_{i:04d}",
            )
        )
    return Transcript(
        recording_id=recording_id,
        schema_version=1,
        language=language,
        duration_ms=len(segments) * 5000,
        provider="mock",
        model="mock-multilingual-v1",
        text=" ".join(s.text for s in segments),
        segments=segments,
    )


@pytest.fixture
def temp_dir():
    d = Path(tempfile.mkdtemp(prefix="eshitaman_llm_test_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def temp_storage(temp_dir, monkeypatch):
    storage = LocalStorage(base_path=temp_dir / "storage" / "recordings")
    monkeypatch.setattr(storage_module, "storage_service", storage)
    monkeypatch.setattr(recordings_api, "storage_service", storage)
    return storage


# =========================================================================
# 1. PROVIDER & FACTORY TESTS
# =========================================================================

def test_mock_provider_deterministic_multilingual():
    async def _test():
        # Uzbek
        p_uz = MockLLMProvider()
        t_uz = build_sample_transcript(language="uz")
        res_uz = await p_uz.analyze_transcript(t_uz)
        assert res_uz.recording_id == t_uz.recording_id
        assert res_uz.language == "uz"
        assert len(res_uz.key_points) >= 3
        assert len(res_uz.terms) >= 3
        assert len(res_uz.questions) >= 3
        # Ensure technical terms are preserved verbatim
        term_names = [t.term.lower() for t in res_uz.terms]
        assert "machine learning" in term_names
        assert "fastapi" in term_names
        assert res_uz.source_segment_ids

        # Russian
        t_ru = build_sample_transcript(language="ru", texts=[
            "Сегодня мы рассмотрели архитектуру микросервисов.",
            "FastAPI обеспечивает высокую производительность API.",
        ])
        res_ru = await p_uz.analyze_transcript(t_ru)
        assert res_ru.language == "ru"
        assert any("архитектур" in p.lower() for p in res_ru.key_points)

        # English
        t_en = build_sample_transcript(language="en", texts=[
            "We discussed machine learning accuracy and database scalability.",
            "FastAPI was chosen for asynchronous request handling.",
        ])
        res_en = await p_uz.analyze_transcript(t_en)
        assert res_en.language == "en"
        assert any("machine learning" in p.lower() for p in res_en.key_points)

    asyncio.run(_test())


def test_factory_creation():
    mock_p = LLMProviderFactory.create_provider("mock")
    assert isinstance(mock_p, MockLLMProvider)
    assert mock_p.provider_name == "mock"

    openai_p = LLMProviderFactory.create_provider("openai", api_key="sk-test-fake")
    assert isinstance(openai_p, OpenAILLMProvider)
    assert openai_p.provider_name == "openai"

    with pytest.raises(LLMConfigurationError):
        LLMProviderFactory.create_provider("unknown_provider_xyz")


# =========================================================================
# 2. OPENAI PROVIDER INTEGRATION & RETRY TESTS
# =========================================================================

def test_openai_provider_success():
    async def _test():
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "summary": "Leksiyada FastAPI va Machine Learning muhokama qilindi.",
                            "key_points": ["Model aniqligi 95%", "FastAPI tezkor"],
                            "explanation": "Tizim barqarorligi uchun muhim bilimlar berildi.",
                            "terms": [
                                {
                                    "term": "FastAPI",
                                    "explanation": "Python freymvorki",
                                    "source_segment_ids": ["seg_0001"]
                                }
                            ],
                            "questions": ["FastAPI nima uchun kerak?"]
                        })
                    }
                }
            ]
        }

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(
            return_value=httpx.Response(
                status_code=200,
                content=json.dumps(mock_response).encode("utf-8"),
                request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
            )
        )

        provider = OpenAILLMProvider(api_key="sk-test-key", http_client=mock_http)
        transcript = build_sample_transcript()
        res = await provider.analyze_transcript(transcript)

        assert res.summary == "Leksiyada FastAPI va Machine Learning muhokama qilindi."
        assert len(res.key_points) == 2
        assert res.terms[0].term == "FastAPI"
        assert res.terms[0].source_segment_ids == ["seg_0001"]
        assert mock_http.post.call_count == 1

    asyncio.run(_test())


def test_openai_provider_auth_error_no_retry():
    async def _test():
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(
            return_value=httpx.Response(
                status_code=401,
                content=b'{"error": {"message": "Incorrect API key"}}',
                request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
            )
        )

        provider = OpenAILLMProvider(api_key="sk-invalid-key", max_retries=3, http_client=mock_http)
        with pytest.raises(LLMAuthenticationError):
            await provider.analyze_transcript(build_sample_transcript())

        # Must not retry on 401
        assert mock_http.post.call_count == 1

    asyncio.run(_test())


def test_openai_provider_rate_limit_retries_and_recovers():
    async def _test():
        mock_success = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "summary": "Muvaffaqiyatli xulosa",
                            "key_points": ["Fikr 1"],
                            "explanation": "Tushuntirish",
                            "terms": [],
                            "questions": ["Savol?"]
                        })
                    }
                }
            ]
        }

        req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        res_429 = httpx.Response(status_code=429, content=b"Rate limit exceeded", request=req)
        res_200 = httpx.Response(status_code=200, content=json.dumps(mock_success).encode("utf-8"), request=req)

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(side_effect=[res_429, res_200])

        provider = OpenAILLMProvider(api_key="sk-test", max_retries=3, http_client=mock_http)
        res = await provider.analyze_transcript(build_sample_transcript())
        assert res.summary == "Muvaffaqiyatli xulosa"
        # Recovered on attempt 2
        assert mock_http.post.call_count == 2

    asyncio.run(_test())


# =========================================================================
# 3. PARSER & CHUNKER TESTS
# =========================================================================

def test_parser_strips_markdown_code_fences():
    raw_json = """```json
    {
      "summary": "Markdown bilan o'ralgan tahlil.",
      "key_points": ["Nuqta 1"],
      "explanation": "Batafsil ma'lumot",
      "terms": [{"term": "API", "explanation": "Interfeys", "source_segment_ids": ["seg_01"]}],
      "questions": ["API nima?"]
    }
    ```"""

    analysis = LLMResponseParser.parse_analysis_json(
        raw_text=raw_json,
        recording_id="rec_fence_01",
        provider_name="test",
        model_name="test-model",
    )
    assert analysis.summary == "Markdown bilan o'ralgan tahlil."
    assert len(analysis.key_points) == 1
    assert analysis.terms[0].term == "API"
    assert "seg_01" in analysis.source_segment_ids


def test_parser_rejects_corrupted_json():
    with pytest.raises(LLMInvalidResponseError):
        LLMResponseParser.parse_analysis_json(
            raw_text="Bu JSON emas, oddiy xabar",
            recording_id="rec_err_01",
            provider_name="test",
            model_name="test-model",
        )


def test_transcript_chunker_partitioning():
    chunker = TranscriptChunker(max_chars_per_window=100, overlap_segments=1)
    segs = [
        TranscriptSegment(id=f"seg_{i}", start_ms=i * 1000, end_ms=(i + 1) * 1000, text=f"Bu qisqa matn {i} " * 4)
        for i in range(10)
    ]
    t = Transcript(
        recording_id="rec_long_01",
        text=" ".join(s.text for s in segs),
        segments=segs,
    )

    assert chunker.needs_chunking(t) is True
    blocks = chunker.partition_segments(segs)
    assert len(blocks) >= 2
    # Verify overlap exists between consecutive blocks
    assert blocks[0][-1].id == blocks[1][0].id


# =========================================================================
# 4. PIPELINE TESTS (CACHING, PERSISTENCE, IDEMPOTENCY)
# =========================================================================

def test_pipeline_full_analysis_success(temp_storage):
    async def _test():
        recording_id = "rec_pipe_success_01"
        session_dir = temp_storage._get_session_dir(recording_id)
        session_dir.mkdir(parents=True, exist_ok=True)

        # 1. Setup metadata in transcribed status
        meta = {
            "recording_id": recording_id,
            "status": RecordingStatus.TRANSCRIBED.value,
        }
        with open(session_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

        # 2. Setup transcript.json
        t = build_sample_transcript(recording_id=recording_id)
        transcript_dir = session_dir / "transcript"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        with open(transcript_dir / "transcript.json", "w", encoding="utf-8") as f:
            json.dump(t.model_dump(), f)

        mock_p = MockLLMProvider()
        pipeline = AIUnderstandingPipeline(storage=temp_storage, provider=mock_p)

        res = await pipeline.analyze_recording(recording_id)
        assert res.recording_id == recording_id
        assert len(res.key_points) >= 3
        assert len(res.terms) >= 3
        assert mock_p.call_count == 1

        # Check analysis.json was saved
        analysis_path = session_dir / "analysis" / "analysis.json"
        assert analysis_path.exists()
        with open(analysis_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
            assert saved["summary"] == res.summary

        # Check metadata status updated to analyzed
        updated_meta = await temp_storage.get_recording_metadata(recording_id)
        assert updated_meta["status"] == RecordingStatus.ANALYZED.value
        assert updated_meta["analysis"]["key_points_count"] >= 3

        # Idempotency check: second call should read from disk without invoking provider
        res2 = await pipeline.analyze_recording(recording_id)
        assert res2.summary == res.summary
        assert mock_p.call_count == 1  # No additional provider call!

    asyncio.run(_test())


def test_pipeline_untranscribed_recording_fails(temp_storage):
    async def _test():
        recording_id = "rec_untranscribed"
        session_dir = temp_storage._get_session_dir(recording_id)
        session_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "recording_id": recording_id,
            "status": RecordingStatus.UPLOADED.value,  # not transcribed
        }
        with open(session_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

        pipeline = AIUnderstandingPipeline(storage=temp_storage, provider=MockLLMProvider())
        with pytest.raises(LLMError) as exc:
            await pipeline.analyze_recording(recording_id)
        assert exc.value.error_code == "RECORDING_NOT_TRANSCRIBED"

    asyncio.run(_test())


def test_pipeline_missing_transcript_fails(temp_storage):
    async def _test():
        recording_id = "rec_missing_transcript"
        session_dir = temp_storage._get_session_dir(recording_id)
        session_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "recording_id": recording_id,
            "status": RecordingStatus.TRANSCRIBED.value,
        }
        with open(session_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)
        # Transcript file not created

        pipeline = AIUnderstandingPipeline(storage=temp_storage, provider=MockLLMProvider())
        with pytest.raises(LLMError) as exc:
            await pipeline.analyze_recording(recording_id)
        assert exc.value.error_code == "TRANSCRIPT_NOT_FOUND"

    asyncio.run(_test())


# =========================================================================
# 5. API ENDPOINTS TESTS
# =========================================================================

def test_api_analyze_and_get_analysis(temp_storage, temp_dir):
    recording_id = "rec_api_analyze_flow"
    session_dir = temp_storage._get_session_dir(recording_id)
    session_dir.mkdir(parents=True, exist_ok=True)

    # 1. Setup metadata in transcribed status
    meta = {
        "recording_id": recording_id,
        "status": RecordingStatus.TRANSCRIBED.value,
        "duration_seconds": 15.0,
        "file_size": 1000,
        "mime_type": "audio/wav",
        "created_at": "2026-09-26T00:00:00Z",
    }
    with open(session_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)

    # 2. Setup transcript.json
    t = build_sample_transcript(recording_id=recording_id)
    transcript_dir = session_dir / "transcript"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    with open(transcript_dir / "transcript.json", "w", encoding="utf-8") as f:
        json.dump(t.model_dump(), f)

    # 3. GET /analysis before analyze returns 404
    res_pre = client.get(f"/api/v1/recordings/{recording_id}/analysis")
    assert res_pre.status_code == 404

    # 4. POST /analyze triggers AI understanding
    res_analyze = client.post(
        f"/api/v1/recordings/{recording_id}/analyze",
        json={"context_instructions": "Universitet talabalari uchun dars"},
    )
    assert res_analyze.status_code == 200
    data = res_analyze.json()
    assert data["recording_id"] == recording_id
    assert data["status"] == "analyzed"
    assert data["key_points_count"] >= 3
    assert data["terms_count"] >= 3

    # 5. GET /analysis returns full AIAnalysis
    res_full = client.get(f"/api/v1/recordings/{recording_id}/analysis")
    assert res_full.status_code == 200
    full_data = res_full.json()
    assert full_data["recording_id"] == recording_id
    assert full_data["schema_version"] == 1
    assert "summary" in full_data
    assert len(full_data["key_points"]) == data["key_points_count"]
    assert len(full_data["terms"]) == data["terms_count"]

    # 6. GET /recordings/{id} returns updated metadata with analysis block
    res_rec = client.get(f"/api/v1/recordings/{recording_id}")
    assert res_rec.status_code == 200
    rec_data = res_rec.json()
    assert rec_data["status"] == "analyzed"
    assert rec_data["analysis"] is not None
    assert rec_data["analysis"]["summary"] == data["summary"]

    # 7. Idempotent second POST /analyze returns 200 with idempotent message
    res_idemp = client.post(f"/api/v1/recordings/{recording_id}/analyze")
    assert res_idemp.status_code == 200
    assert "already analyzed" in res_idemp.json()["message"].lower()


def test_api_analyze_untranscribed_returns_400(temp_storage):
    recording_id = "rec_api_not_transcribed"
    session_dir = temp_storage._get_session_dir(recording_id)
    session_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "recording_id": recording_id,
        "status": RecordingStatus.PROCESSED.value,
    }
    with open(session_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)

    res = client.post(f"/api/v1/recordings/{recording_id}/analyze")
    assert res.status_code == 400
    assert "RECORDING_NOT_TRANSCRIBED" in res.json()["detail"]["error_code"]


def test_api_analyze_nonexistent_returns_404():
    res = client.post("/api/v1/recordings/rec_nonexistent_xyz/analyze")
    assert res.status_code == 404
