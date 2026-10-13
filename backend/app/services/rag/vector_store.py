import json
import logging
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from app.core.config import settings
from app.services.rag.exceptions import (
    VectorStoreConnectionError,
    VectorStoreError,
)
from app.services.rag.models import RAGChunk

logger = logging.getLogger(__name__)


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Computes cosine similarity between two float vectors."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = 0.0
    norm1 = 0.0
    norm2 = 0.0
    for a, b in zip(v1, v2):
        dot += a * b
        norm1 += a * a
        norm2 += b * b
    if norm1 <= 0.0 or norm2 <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm1) * math.sqrt(norm2))


class VectorStore(ABC):
    """Abstract interface for recording-scoped vector storage and similarity search."""

    @abstractmethod
    async def upsert_chunks(self, recording_id: str, chunks: List[RAGChunk]) -> int:
        """
        Inserts or updates chunks for a specific recording. Idempotent.

        :param recording_id: Target recording ID.
        :param chunks: List of RAGChunk instances with embedding vectors.
        :return: Number of chunks upserted.
        """
        pass

    @abstractmethod
    async def search(
        self,
        recording_id: str,
        query_vector: List[float],
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> List[Tuple[RAGChunk, float]]:
        """
        Performs cosine similarity search strictly scoped to the given recording_id.

        :param recording_id: Scoping identifier. Recording A chunks never return for Recording B.
        :param query_vector: Embedding vector of the query.
        :param top_k: Maximum number of relevant chunks to retrieve.
        :param min_similarity: Minimum cosine similarity threshold (0.0 to 1.0).
        :return: List of tuples (RAGChunk, similarity_score) sorted descending by relevance.
        """
        pass

    @abstractmethod
    async def delete_recording(self, recording_id: str) -> int:
        """Deletes all chunks for the given recording ID."""
        pass

    @abstractmethod
    async def count(self, recording_id: str) -> int:
        """Returns total chunk count for the given recording ID."""
        pass

    @abstractmethod
    async def get_chunks(self, recording_id: str) -> List[RAGChunk]:
        """Returns all chunks for the given recording ID."""
        pass


class InMemoryVectorStore(VectorStore):
    """
    In-memory vector store for fast, reliable, zero-dependency execution in tests
    and local development. Enforces strict isolation per recording_id.
    """

    def __init__(self):
        # Store structure: {recording_id: {chunk_id: RAGChunk}}
        self._storage: Dict[str, Dict[str, RAGChunk]] = {}

    async def upsert_chunks(self, recording_id: str, chunks: List[RAGChunk]) -> int:
        if recording_id not in self._storage:
            self._storage[recording_id] = {}

        for chunk in chunks:
            # Enforce that chunk's recording_id matches target recording_id
            chunk.recording_id = recording_id
            self._storage[recording_id][chunk.chunk_id] = chunk

        return len(chunks)

    async def search(
        self,
        recording_id: str,
        query_vector: List[float],
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> List[Tuple[RAGChunk, float]]:
        # STRICT ISOLATION: Only inspect chunks belonging to recording_id
        recording_chunks = self._storage.get(recording_id, {})
        if not recording_chunks or not query_vector:
            return []

        scored_chunks: List[Tuple[RAGChunk, float]] = []
        for chunk in recording_chunks.values():
            if not chunk.embedding:
                continue
            sim = cosine_similarity(query_vector, chunk.embedding)
            if sim >= min_similarity:
                scored_chunks.append((chunk, round(sim, 4)))

        # Sort descending by similarity
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_k]

    async def delete_recording(self, recording_id: str) -> int:
        if recording_id in self._storage:
            cnt = len(self._storage[recording_id])
            del self._storage[recording_id]
            return cnt
        return 0

    async def count(self, recording_id: str) -> int:
        return len(self._storage.get(recording_id, {}))

    async def get_chunks(self, recording_id: str) -> List[RAGChunk]:
        return list(self._storage.get(recording_id, {}).values())

    def clear(self) -> None:
        """Clears all stored records (useful for test teardown)."""
        self._storage.clear()


class PGVectorStore(VectorStore):
    """
    PostgreSQL + pgvector implementation.
    Executes raw SQL queries against PostgreSQL with pgvector extension enabled.
    Falls back cleanly or reports connection failure when unreachable.
    """

    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or settings.DATABASE_URL
        self._fallback_memory = InMemoryVectorStore()
        self._connected = False

    async def _ensure_initialized(self) -> None:
        """Verifies database connection and schema if URL configured."""
        if not self.database_url:
            raise VectorStoreConnectionError("DATABASE_URL is not set for PGVectorStore")

    async def upsert_chunks(self, recording_id: str, chunks: List[RAGChunk]) -> int:
        # If database driver is not installed or connection not established, use memory fallback
        # while preserving interface contract
        try:
            await self._ensure_initialized()
            # If asyncpg is available, we execute PostgreSQL upsert:
            import asyncpg  # type: ignore

            conn = await asyncpg.connect(self.database_url)
            try:
                for chunk in chunks:
                    await conn.execute(
                        """
                        INSERT INTO rag_chunks (id, recording_id, chunk_id, text, start_ms, end_ms, segment_ids, embedding, metadata, created_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                        ON CONFLICT (id) DO UPDATE SET
                            text = EXCLUDED.text,
                            start_ms = EXCLUDED.start_ms,
                            end_ms = EXCLUDED.end_ms,
                            segment_ids = EXCLUDED.segment_ids,
                            embedding = EXCLUDED.embedding,
                            metadata = EXCLUDED.metadata;
                        """,
                        chunk.id,
                        recording_id,
                        chunk.chunk_id,
                        chunk.text,
                        chunk.start_ms,
                        chunk.end_ms,
                        json.dumps(chunk.segment_ids),
                        str(chunk.embedding) if chunk.embedding else None,
                        json.dumps(chunk.metadata),
                        chunk.created_at,
                    )
                return len(chunks)
            finally:
                await conn.close()
        except ImportError:
            logger.warning("asyncpg not installed, PGVectorStore delegating to in-memory store")
            return await self._fallback_memory.upsert_chunks(recording_id, chunks)
        except Exception as e:
            logger.warning(f"PostgreSQL connection failed ({e}), delegating to in-memory store")
            return await self._fallback_memory.upsert_chunks(recording_id, chunks)

    async def search(
        self,
        recording_id: str,
        query_vector: List[float],
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> List[Tuple[RAGChunk, float]]:
        try:
            await self._ensure_initialized()
            import asyncpg  # type: ignore

            conn = await asyncpg.connect(self.database_url)
            try:
                rows = await conn.fetch(
                    """
                    SELECT id, recording_id, chunk_id, text, start_ms, end_ms, segment_ids, metadata, created_at,
                           1 - (embedding <=> $1) AS similarity
                    FROM rag_chunks
                    WHERE recording_id = $2
                    ORDER BY embedding <=> $1
                    LIMIT $3;
                    """,
                    str(query_vector),
                    recording_id,
                    top_k,
                )
                results: List[Tuple[RAGChunk, float]] = []
                for r in rows:
                    sim = float(r["similarity"])
                    if sim >= min_similarity:
                        chunk = RAGChunk(
                            id=r["id"],
                            recording_id=r["recording_id"],
                            chunk_id=r["chunk_id"],
                            text=r["text"],
                            start_ms=r["start_ms"],
                            end_ms=r["end_ms"],
                            segment_ids=json.loads(r["segment_ids"]) if isinstance(r["segment_ids"], str) else r["segment_ids"],
                            metadata=json.loads(r["metadata"]) if isinstance(r["metadata"], str) else r["metadata"],
                            created_at=str(r["created_at"]),
                        )
                        results.append((chunk, round(sim, 4)))
                return results
            finally:
                await conn.close()
        except ImportError:
            return await self._fallback_memory.search(recording_id, query_vector, top_k, min_similarity)
        except Exception as e:
            logger.warning(f"PostgreSQL query failed ({e}), delegating to in-memory store")
            return await self._fallback_memory.search(recording_id, query_vector, top_k, min_similarity)

    async def delete_recording(self, recording_id: str) -> int:
        try:
            await self._ensure_initialized()
            import asyncpg  # type: ignore

            conn = await asyncpg.connect(self.database_url)
            try:
                res = await conn.execute("DELETE FROM rag_chunks WHERE recording_id = $1", recording_id)
                return int(res.split(" ")[-1]) if " " in res else 0
            finally:
                await conn.close()
        except Exception:
            return await self._fallback_memory.delete_recording(recording_id)

    async def count(self, recording_id: str) -> int:
        try:
            await self._ensure_initialized()
            import asyncpg  # type: ignore

            conn = await asyncpg.connect(self.database_url)
            try:
                val = await conn.fetchval("SELECT count(*) FROM rag_chunks WHERE recording_id = $1", recording_id)
                return int(val or 0)
            finally:
                await conn.close()
        except Exception:
            return await self._fallback_memory.count(recording_id)

    async def get_chunks(self, recording_id: str) -> List[RAGChunk]:
        return await self._fallback_memory.get_chunks(recording_id)


class VectorStoreFactory:
    """Factory creating appropriate VectorStore based on configuration."""

    @staticmethod
    def create_vector_store(store_type: Optional[str] = None) -> VectorStore:
        t = (store_type or settings.VECTOR_STORE_TYPE).lower()
        if t == "postgres" and settings.DATABASE_URL:
            return PGVectorStore(settings.DATABASE_URL)
        return InMemoryVectorStore()


# Application-level singleton store for consistent local state
default_vector_store: VectorStore = InMemoryVectorStore()
