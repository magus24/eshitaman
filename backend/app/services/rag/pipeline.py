import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from app.core.config import settings
from app.schemas.recording import RecordingStatus
from app.services.storage.base import BaseStorage
from app.services.stt.models import Transcript
from app.services.rag.chunker import SemanticChunker
from app.services.rag.context_builder import ContextBuilder
from app.services.rag.embedding_base import BaseEmbeddingProvider
from app.services.rag.embedding_factory import EmbeddingProviderFactory
from app.services.rag.exceptions import (
    EmbeddingError,
    NoRelevantContextError,
    RAGError,
    VectorStoreError,
)
from app.services.rag.models import (
    IndexingResponse,
    RAGAnswer,
    RAGChunk,
    RAGSource,
)
from app.services.rag.prompts import RAG_SYSTEM_PROMPT, build_rag_user_prompt
from app.services.rag.vector_store import (
    VectorStore,
    default_vector_store,
)
from app.services.llm.base import BaseLLMProvider
from app.services.llm.factory import LLMProviderFactory

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Orchestrates end-to-end Retrieval-Augmented Generation:
    1. Semantic chunking of STT transcripts.
    2. Vector embedding generation with provider abstraction.
    3. Recording-scoped vector storage in pgvector / in-memory.
    4. Deterministic similarity search with top-k retrieval and threshold gating.
    5. Grounded prompt construction with traceable source timestamps.
    6. LLM synthesis without hallucinations.
    """

    def __init__(
        self,
        storage: BaseStorage,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        vector_store: Optional[VectorStore] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
        chunker: Optional[SemanticChunker] = None,
        context_builder: Optional[ContextBuilder] = None,
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None,
    ):
        self.storage = storage
        self.embedding_provider = embedding_provider or EmbeddingProviderFactory.create_provider()
        self.vector_store = vector_store or default_vector_store
        self.llm_provider = llm_provider or LLMProviderFactory.create_provider()
        self.chunker = chunker or SemanticChunker()
        self.context_builder = context_builder or ContextBuilder()
        self.top_k = top_k or settings.RAG_TOP_K
        self.min_similarity = min_similarity if min_similarity is not None else settings.RAG_MIN_SIMILARITY

    async def index_recording(
        self,
        recording_id: str,
        force_reindex: bool = False,
    ) -> IndexingResponse:
        """
        Indexes a transcribed recording for vector retrieval.
        Idempotent: Reuses existing index if already processed and not forced.
        """
        # 1. Validate recording metadata
        meta = await self.storage.get_recording_metadata(recording_id)
        if not meta:
            raise RAGError(
                f"Recording '{recording_id}' not found",
                error_code="RECORDING_NOT_FOUND",
                status_code=404,
            )

        audio_path = await self.storage.get_audio_path(recording_id)
        session_dir = (
            self.storage._get_session_dir(recording_id)
            if hasattr(self.storage, "_get_session_dir")
            else (audio_path.parent if audio_path else None)
        )
        if not session_dir or not session_dir.exists():
            raise RAGError(
                f"Storage directory for recording '{recording_id}' not found",
                error_code="RECORDING_NOT_FOUND",
                status_code=404,
            )

        rag_dir = session_dir / "rag"
        chunks_file = rag_dir / "chunks.json"

        # 2. Idempotency Check: if already indexed and chunks exist
        existing_count = await self.vector_store.count(recording_id)
        if not force_reindex and (meta.get("status") == RecordingStatus.INDEXED.value or chunks_file.exists()):
            if existing_count > 0:
                return IndexingResponse(
                    recording_id=recording_id,
                    status=RecordingStatus.INDEXED.value,
                    chunks_count=existing_count,
                    embedding_dimension=self.embedding_provider.dimension,
                    message="Recording already indexed",
                )
            elif chunks_file.exists():
                # Reload persisted chunks into vector store
                try:
                    with open(chunks_file, "r", encoding="utf-8") as f:
                        cached_raw = json.load(f)
                    cached_chunks = [RAGChunk(**c) for c in cached_raw]
                    await self.vector_store.upsert_chunks(recording_id, cached_chunks)
                    return IndexingResponse(
                        recording_id=recording_id,
                        status=RecordingStatus.INDEXED.value,
                        chunks_count=len(cached_chunks),
                        embedding_dimension=self.embedding_provider.dimension,
                        message="Recording index restored from storage",
                    )
                except Exception as e:
                    logger.warning(f"Could not load cached chunks.json for '{recording_id}': {e}. Re-indexing.")

        # 3. Check transcript availability
        transcript_path = session_dir / "transcript" / "transcript.json"
        if not transcript_path.exists():
            raise RAGError(
                f"Transcript for recording '{recording_id}' not found. Transcribe the recording first.",
                error_code="TRANSCRIPT_NOT_FOUND",
                status_code=400,
            )

        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript_data = json.load(f)
            transcript = Transcript(**transcript_data)
        except Exception as e:
            raise RAGError(
                f"Failed to parse transcript for recording '{recording_id}': {e}",
                error_code="TRANSCRIPT_PARSE_ERROR",
                status_code=422,
                cause=e,
            )

        if not transcript.segments and not transcript.text:
            raise RAGError(
                f"Transcript for recording '{recording_id}' is empty",
                error_code="TRANSCRIPT_EMPTY",
                status_code=422,
            )

        # 4. Mark status as indexing
        meta["status"] = RecordingStatus.INDEXING.value
        await self.storage.update_recording_metadata(recording_id, meta)

        # 5. Semantic Chunking
        chunks = self.chunker.chunk_transcript(transcript, recording_id=recording_id)
        if not chunks:
            raise RAGError(
                f"No chunks generated from transcript for recording '{recording_id}'",
                error_code="CHUNKS_EMPTY",
                status_code=422,
            )

        # 6. Generate Embeddings in batch
        texts = [c.text for c in chunks]
        embeddings = await self.embedding_provider.embed_batch(texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        # 7. Upsert to Vector Store
        await self.vector_store.upsert_chunks(recording_id, chunks)

        # 8. Persist chunks to disk
        rag_dir.mkdir(parents=True, exist_ok=True)
        chunks_dicts = [c.model_dump() for c in chunks]
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunks_dicts, f, indent=2, ensure_ascii=False)

        # 9. Update Recording Metadata
        indexed_at = datetime.now(timezone.utc).isoformat()
        meta["status"] = RecordingStatus.INDEXED.value
        meta["rag"] = {
            "status": "indexed",
            "chunks_count": len(chunks),
            "embedding_dimension": self.embedding_provider.dimension,
            "indexed_at": indexed_at,
            "provider": self.embedding_provider.provider_name,
            "model": self.embedding_provider.model_name,
        }
        await self.storage.update_recording_metadata(recording_id, meta)

        return IndexingResponse(
            recording_id=recording_id,
            status=RecordingStatus.INDEXED.value,
            chunks_count=len(chunks),
            embedding_dimension=self.embedding_provider.dimension,
            message="Transcript indexed successfully",
        )

    async def answer_question(
        self,
        recording_id: str,
        question: str,
        top_k: Optional[int] = None,
    ) -> RAGAnswer:
        """
        Executes grounded retrieval-augmented question answering:
        - Embeds the user question.
        - Retrieves top-k semantically relevant chunks for recording_id.
        - Gates on relevance threshold to prevent hallucination.
        - Generates response strictly grounded in retrieved chunks.
        """
        if not question or not question.strip():
            raise RAGError("Question cannot be empty", error_code="INVALID_QUESTION", status_code=422)

        # 1. Validate recording metadata
        meta = await self.storage.get_recording_metadata(recording_id)
        if not meta:
            raise RAGError(
                f"Recording '{recording_id}' not found",
                error_code="RECORDING_NOT_FOUND",
                status_code=404,
            )

        # 2. Check if recording is indexed; auto-index if transcript exists
        cnt = await self.vector_store.count(recording_id)
        if cnt == 0:
            # Check if chunks file exists on disk
            audio_path = await self.storage.get_audio_path(recording_id)
            session_dir = (
                self.storage._get_session_dir(recording_id)
                if hasattr(self.storage, "_get_session_dir")
                else (audio_path.parent if audio_path else None)
            )
            chunks_file = (session_dir / "rag" / "chunks.json") if session_dir else None

            if chunks_file and chunks_file.exists():
                try:
                    with open(chunks_file, "r", encoding="utf-8") as f:
                        cached_raw = json.load(f)
                    cached_chunks = [RAGChunk(**c) for c in cached_raw]
                    await self.vector_store.upsert_chunks(recording_id, cached_chunks)
                    cnt = len(cached_chunks)
                except Exception:
                    pass

            if cnt == 0:
                # Attempt automatic on-demand indexing if transcript exists
                transcript_path = (session_dir / "transcript" / "transcript.json") if session_dir else None
                if transcript_path and transcript_path.exists():
                    index_res = await self.index_recording(recording_id)
                    cnt = index_res.chunks_count
                else:
                    raise RAGError(
                        f"Recording '{recording_id}' has not been transcribed or indexed yet.",
                        error_code="RECORDING_NOT_INDEXED",
                        status_code=400,
                    )

        # 3. Generate Question Embedding
        effective_k = top_k or self.top_k
        query_vector = await self.embedding_provider.embed_text(question.strip())

        # 4. Recording-isolated similarity search
        search_results = await self.vector_store.search(
            recording_id=recording_id,
            query_vector=query_vector,
            top_k=effective_k,
            min_similarity=self.min_similarity,
        )

        # 5. Gating: If no chunks pass the similarity threshold, do NOT call LLM
        if not search_results:
            logger.info(f"No relevant chunks found for question '{question}' on recording '{recording_id}'")
            return RAGAnswer(
                recording_id=recording_id,
                answer="Taqdim etilgan darslik yozuvida ushbu savol bo'yicha yetarli ma'lumot topilmadi.",
                sources=[],
                language="uz",
                provider=self.llm_provider.provider_name,
                model=self.llm_provider.model_name,
            )

        # 6. Build Grounded Context
        retrieved_chunks = [chunk for chunk, _ in search_results]
        context_str = self.context_builder.build_context(retrieved_chunks)

        # 7. Construct Grounded Prompt
        prompt = build_rag_user_prompt(question=question, context=context_str)

        # 8. Invoke LLM for Grounded Answer
        answer_text = await self.llm_provider.generate_rag_answer(
            prompt=prompt,
            system_prompt=RAG_SYSTEM_PROMPT,
        )

        # 9. Format Traceable Source References
        sources = [
            RAGSource(
                chunk_id=chunk.chunk_id,
                start_time=chunk.start_seconds,
                end_time=chunk.end_seconds,
                text=chunk.text,
                relevance=score,
            )
            for chunk, score in search_results
        ]

        return RAGAnswer(
            recording_id=recording_id,
            answer=answer_text,
            sources=sources,
            language=meta.get("transcript", {}).get("language", "uz"),
            provider=self.llm_provider.provider_name,
            model=self.llm_provider.model_name,
        )
