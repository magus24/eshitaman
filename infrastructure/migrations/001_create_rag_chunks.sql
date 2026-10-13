-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Table for storing RAG transcript chunks and high-dimensional embeddings
CREATE TABLE IF NOT EXISTS rag_chunks (
    id VARCHAR(128) PRIMARY KEY,
    recording_id VARCHAR(64) NOT NULL,
    chunk_id VARCHAR(64) NOT NULL,
    text TEXT NOT NULL,
    start_ms BIGINT NOT NULL,
    end_ms BIGINT NOT NULL,
    segment_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    embedding vector(1536),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for strict recording_id scoping and filtering
CREATE INDEX IF NOT EXISTS idx_rag_chunks_recording_id ON rag_chunks (recording_id);

-- Cosine distance IVFFLAT index for fast approximate nearest neighbor search within a recording
-- Created when dataset reaches appropriate size, or HNSW index on pgvector 0.5+
CREATE INDEX IF NOT EXISTS idx_rag_chunks_embedding ON rag_chunks USING hnsw (embedding vector_cosine_ops);
