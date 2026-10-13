import asyncio
import json
import os
import pytest
import httpx
from app.core.config import settings
from app.services.stt.models import Transcript, TranscriptSegment
from app.services.llm.base import BaseLLMProvider
from app.services.llm.factory import LLMProviderFactory
from app.services.llm.groq_provider import GroqLLMProvider
from app.services.llm.exceptions import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMRateLimitError,
)


def _make_sample_transcript() -> Transcript:
    return Transcript(
        recording_id="rec_groq_test_1",
        full_text="Sun'iy intellekt va mashinali o'rganish leksiyasi. Bugun biz neyron tarmoqlarini o'rganamiz.",
        language="uz",
        duration_seconds=120.0,
        segment_count=2,
        segments=[
            TranscriptSegment(
                id="seg_0001",
                start_ms=0,
                end_ms=60000,
                text="Sun'iy intellekt va mashinali o'rganish leksiyasi.",
                confidence=0.98,
            ),
            TranscriptSegment(
                id="seg_0002",
                start_ms=60000,
                end_ms=120000,
                text="Bugun biz neyron tarmoqlarini o'rganamiz.",
                confidence=0.97,
            ),
        ],
    )


def test_groq_missing_api_key():
    provider = GroqLLMProvider(api_key="")
    transcript = _make_sample_transcript()
    with pytest.raises(LLMConfigurationError):
        asyncio.run(provider.analyze_transcript(transcript))

    with pytest.raises(LLMConfigurationError):
        asyncio.run(provider.generate_rag_answer("Salom"))


def test_groq_factory_creation():
    provider = LLMProviderFactory.create_provider(
        provider_type="groq",
        api_key="gsk_test_mock_key",
        model="llama-3.3-70b-versatile",
    )
    assert isinstance(provider, GroqLLMProvider)
    assert provider.provider_name == "groq"
    assert provider.model_name == "llama-3.3-70b-versatile"


def test_groq_authentication_failure():
    """Verify 401 returns LLMAuthenticationError immediately without retrying forever."""
    def auth_error_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "Invalid API key"}})

    transport = httpx.MockTransport(auth_error_handler)
    client = httpx.AsyncClient(transport=transport)

    provider = GroqLLMProvider(
        api_key="gsk_invalid_key",
        http_client=client,
        max_retries=2,
    )
    transcript = _make_sample_transcript()

    with pytest.raises(LLMAuthenticationError) as exc_info:
        asyncio.run(provider.analyze_transcript(transcript))
    assert "authentication failed" in str(exc_info.value).lower()


def test_groq_successful_analysis():
    """Verify successful JSON parsing from Groq completions."""
    analysis_payload = {
        "summary": "Sun'iy intellekt asoslari va neyron tarmoqlari haqida umumiy tushunchalar berildi.",
        "key_points": [
            "Sun'iy intellekt axborot texnologiyalarining muhim tarmog'i hisoblanadi.",
            "Neyron tarmoqlari inson miyasi ishlash prinsipiga taqlid qiladi.",
        ],
        "explanation": "Ushbu ma'ruzada talabalar uchun sun'iy intellekt va neyron tarmoqlarining bazaviy tamoyillari tushuntirildi.",
        "terms": [
            {
                "term": "Neyron tarmoqlari",
                "explanation": "Ma'lumotlar ichidagi murakkab bog'liqliklarni o'rganuvchi algoritmlar to'plami.",
                "source_segment_ids": ["seg_0002"],
            }
        ],
        "questions": [
            "Neyron tarmoqlari qanday prinsip asosida ishlaydi?",
        ],
    }

    groq_api_response = {
        "id": "chatcmpl-mock-groq-123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "llama-3.3-70b-versatile",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps(analysis_payload, ensure_ascii=False),
                },
                "finish_reason": "stop",
            }
        ],
    }

    def success_handler(request: httpx.Request) -> httpx.Response:
        assert "/chat/completions" in str(request.url)
        return httpx.Response(200, json=groq_api_response)

    transport = httpx.MockTransport(success_handler)
    client = httpx.AsyncClient(transport=transport)

    provider = GroqLLMProvider(
        api_key="gsk_valid_key",
        model="llama-3.3-70b-versatile",
        http_client=client,
    )
    transcript = _make_sample_transcript()

    result = asyncio.run(provider.analyze_transcript(transcript))
    assert result.summary == analysis_payload["summary"]
    assert len(result.key_points) == 2
    assert len(result.terms) == 1
    assert result.terms[0].term == "Neyron tarmoqlari"
    assert result.provider == "groq"
    assert result.model == "llama-3.3-70b-versatile"


def test_groq_successful_rag_answer():
    """Verify natural language RAG answer generation from Groq."""
    expected_answer = "Ma'ruzada aytilishicha, neyron tarmoqlari inson miyasi ishlashiga taqlid qiladi."
    groq_api_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": expected_answer,
                }
            }
        ]
    }

    def rag_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=groq_api_response)

    transport = httpx.MockTransport(rag_handler)
    client = httpx.AsyncClient(transport=transport)

    provider = GroqLLMProvider(
        api_key="gsk_valid_key",
        http_client=client,
    )

    answer = asyncio.run(provider.generate_rag_answer("Neyron tarmoq nima?"))
    assert answer == expected_answer


def test_groq_live_api_if_key_available():
    """
    Live E2E test against Groq Cloud API.
    Runs only if GROQ_API_KEY is configured in the environment.
    """
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key or groq_key.startswith("your-") or len(groq_key) < 10:
        pytest.skip("GROQ_API_KEY is not set in environment. Skipping live Groq test.")

    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    provider = GroqLLMProvider(
        api_key=groq_key,
        model=model,
    )

    # 1. Test live RAG answer
    rag_prompt = (
        "Quyidagi matn asosida javob bering:\n"
        "[00:00 - 01:00] Toshkent O'zbekistonning poytaxti hisoblanadi.\n\n"
        "Savol: O'zbekistonning poytaxti qaysi shahar?"
    )
    answer = asyncio.run(provider.generate_rag_answer(rag_prompt))
    assert "Toshkent" in answer, f"Expected 'Toshkent' in answer, got: {answer}"

    # 2. Test live transcript analysis
    transcript = _make_sample_transcript()
    analysis = asyncio.run(provider.analyze_transcript(transcript))
    assert analysis.summary is not None and len(analysis.summary) > 10
    assert analysis.provider == "groq"
