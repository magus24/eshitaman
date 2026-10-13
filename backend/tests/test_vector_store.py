import asyncio
import pytest
from app.services.rag.models import RAGChunk
from app.services.rag.vector_store import InMemoryVectorStore, cosine_similarity


def test_cosine_similarity_basic():
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    v3 = [0.0, 1.0, 0.0]
    assert cosine_similarity(v1, v2) == 1.0
    assert cosine_similarity(v1, v3) == 0.0
    assert cosine_similarity([], []) == 0.0


def test_in_memory_vector_store_upsert_and_count():
    async def _test():
        store = InMemoryVectorStore()
        chunk1 = RAGChunk(
            chunk_id="chunk_1",
            recording_id="rec_1",
            text="Birinchi bo'lim",
            start_ms=0,
            end_ms=1000,
            embedding=[0.5, 0.5],
        )
        chunk2 = RAGChunk(
            chunk_id="chunk_2",
            recording_id="rec_1",
            text="Ikkinchi bo'lim",
            start_ms=1000,
            end_ms=2000,
            embedding=[0.1, 0.9],
        )

        upserted = await store.upsert_chunks("rec_1", [chunk1, chunk2])
        assert upserted == 2
        assert await store.count("rec_1") == 2
        assert await store.count("rec_unknown") == 0

    asyncio.run(_test())


def test_in_memory_vector_store_strict_recording_isolation():
    """
    CRITICAL SECURITY/ISOLATION TEST:
    Verifies that Recording A chunks are NEVER returned when searching Recording B,
    even if Recording A contains an exact 100% vector match to the query.
    """
    async def _test():
        store = InMemoryVectorStore()
        target_vec = [1.0, 0.0, 0.0]

        # Recording A has an exact match
        chunk_a = RAGChunk(
            chunk_id="chunk_a",
            recording_id="rec_A",
            text="Maxfiy ma'lumotlar faqat A yozuvida",
            start_ms=0,
            end_ms=5000,
            embedding=[1.0, 0.0, 0.0],
        )
        await store.upsert_chunks("rec_A", [chunk_a])

        # Recording B has a completely different vector
        chunk_b = RAGChunk(
            chunk_id="chunk_b",
            recording_id="rec_B",
            text="Oddiy ommaviy ma'lumot B yozuvida",
            start_ms=0,
            end_ms=5000,
            embedding=[0.0, 1.0, 0.0],
        )
        await store.upsert_chunks("rec_B", [chunk_b])

        # Query with target_vec for rec_B: MUST NEVER return chunk_a
        results_for_b = await store.search("rec_B", query_vector=target_vec, top_k=5, min_similarity=0.0)
        for chunk, _ in results_for_b:
            assert chunk.recording_id == "rec_B"
            assert chunk.chunk_id != "chunk_a"
            assert "Maxfiy" not in chunk.text

        # Search for rec_A should retrieve chunk_a
        results_for_a = await store.search("rec_A", query_vector=target_vec, top_k=5, min_similarity=0.5)
        assert len(results_for_a) == 1
        assert results_for_a[0][0].chunk_id == "chunk_a"

    asyncio.run(_test())


def test_in_memory_vector_store_top_k_and_min_similarity():
    async def _test():
        store = InMemoryVectorStore()
        q_vec = [1.0, 0.0]

        c1 = RAGChunk(chunk_id="c1", recording_id="rec_k", text="Match high", start_ms=0, end_ms=10, embedding=[0.99, 0.01])
        c2 = RAGChunk(chunk_id="c2", recording_id="rec_k", text="Match medium", start_ms=10, end_ms=20, embedding=[0.7, 0.7])
        c3 = RAGChunk(chunk_id="c3", recording_id="rec_k", text="Match zero", start_ms=20, end_ms=30, embedding=[0.0, 1.0])

        await store.upsert_chunks("rec_k", [c1, c2, c3])

        # Top-1 should return only c1
        top_1 = await store.search("rec_k", query_vector=q_vec, top_k=1, min_similarity=0.1)
        assert len(top_1) == 1
        assert top_1[0][0].chunk_id == "c1"

        # Threshold 0.8 should filter out c2 and c3
        filtered = await store.search("rec_k", query_vector=q_vec, top_k=5, min_similarity=0.8)
        assert len(filtered) == 1
        assert filtered[0][0].chunk_id == "c1"

    asyncio.run(_test())
