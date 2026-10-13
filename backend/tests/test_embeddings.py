import asyncio
import pytest
from app.services.rag.embedding_base import BaseEmbeddingProvider
from app.services.rag.embedding_mock import MockEmbeddingProvider
from app.services.rag.embedding_openai import OpenAIEmbeddingProvider
from app.services.rag.embedding_factory import EmbeddingProviderFactory
from app.services.rag.exceptions import (
    EmbeddingAuthenticationError,
    EmbeddingConfigurationError,
    EmbeddingProviderError,
    EmbeddingRateLimitError,
    EmbeddingTimeoutError,
)
from app.services.rag.vector_store import cosine_similarity


def test_mock_embedding_deterministic_vectors():
    async def _test():
        provider = MockEmbeddingProvider(dimension=512)
        text1 = "Sun'iy intellekt va mashinali o'rganish asoslari"
        text2 = "Sun'iy intellekt va mashinali o'rganish asoslari"

        vec1 = await provider.embed_text(text1)
        vec2 = await provider.embed_text(text2)

        assert len(vec1) == 512
        assert len(vec2) == 512
        assert vec1 == vec2  # Exactly identical across runs!

    asyncio.run(_test())


def test_mock_embedding_similarity_ranking():
    async def _test():
        provider = MockEmbeddingProvider(dimension=512)
        query = "Mashinali o'rganish va gradient descent algoritmi"
        related = "Bugun biz gradient descent algoritmi va machine learning haqida gaplashamiz"
        unrelated = "Oshpazlik retseptlari: mazali palov tayyorlash qoidalari"

        q_vec = await provider.embed_text(query)
        rel_vec = await provider.embed_text(related)
        unrel_vec = await provider.embed_text(unrelated)

        sim_related = cosine_similarity(q_vec, rel_vec)
        sim_unrelated = cosine_similarity(q_vec, unrel_vec)

        assert sim_related > sim_unrelated
        assert sim_related > 0.4
        assert sim_unrelated < 0.25

    asyncio.run(_test())


def test_mock_embedding_batch():
    async def _test():
        provider = MockEmbeddingProvider(dimension=256)
        texts = ["Birinchi matn", "Ikkinchi matn", "Uchinchi matn"]
        vectors = await provider.embed_batch(texts)
        assert len(vectors) == 3
        for v in vectors:
            assert len(v) == 256

    asyncio.run(_test())


def test_mock_embedding_error_injection():
    async def _test():
        # Auth error
        auth_provider = MockEmbeddingProvider(force_error="auth")
        with pytest.raises(EmbeddingAuthenticationError):
            await auth_provider.embed_text("test")

        # Timeout error
        timeout_provider = MockEmbeddingProvider(force_error="timeout")
        with pytest.raises(EmbeddingTimeoutError):
            await timeout_provider.embed_text("test")

        # Rate limit error
        rl_provider = MockEmbeddingProvider(force_error="rate_limit", fail_times=1)
        with pytest.raises(EmbeddingRateLimitError):
            await rl_provider.embed_text("test")

    asyncio.run(_test())


def test_embedding_factory():
    mock_p = EmbeddingProviderFactory.create_provider("mock")
    assert isinstance(mock_p, MockEmbeddingProvider)
    assert mock_p.provider_name == "mock"

    with pytest.raises(EmbeddingConfigurationError):
        EmbeddingProviderFactory.create_provider("unknown_provider_xyz")
