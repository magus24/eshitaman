from datetime import datetime, timezone
from typing import List, Optional
from pydantic import BaseModel, Field


class RAGChunk(BaseModel):
    """A semantically coherent chunk of transcript text for embedding and retrieval."""
    id: Optional[str] = None
    chunk_id: str
    recording_id: str
    text: str
    start_ms: int
    end_ms: int
    segment_ids: List[str] = Field(default_factory=list)
    embedding: Optional[List[float]] = None
    metadata: dict = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def model_post_init(self, __context) -> None:
        if not self.id:
            self.id = f"{self.recording_id}_{self.chunk_id}"

    @property
    def start_seconds(self) -> float:
        return self.start_ms / 1000.0

    @property
    def end_seconds(self) -> float:
        return self.end_ms / 1000.0

    @property
    def timestamp_display(self) -> str:
        """Human-readable MM:SS – MM:SS timestamp range."""
        s_min, s_sec = divmod(int(self.start_ms / 1000), 60)
        e_min, e_sec = divmod(int(self.end_ms / 1000), 60)
        return f"{s_min:02d}:{s_sec:02d} – {e_min:02d}:{e_sec:02d}"


class RAGSource(BaseModel):
    """A source reference for a RAG answer, linking back to the lecture timestamp."""
    chunk_id: str
    start_time: float
    end_time: float
    text: str
    relevance: float = 0.0


class RAGAnswer(BaseModel):
    """Structured response from the RAG pipeline with grounded answer and source references."""
    recording_id: str
    answer: str
    sources: List[RAGSource] = Field(default_factory=list)
    language: Optional[str] = None
    provider: str = "mock"
    model: str = "mock-model"


class IndexingResponse(BaseModel):
    """API response for POST /api/v1/recordings/{id}/index."""
    recording_id: str
    status: str
    chunks_count: int = 0
    embedding_dimension: int = 0
    message: str = "Transcript indexed successfully"
    error_code: Optional[str] = None
    error_message: Optional[str] = None


class ChatRequest(BaseModel):
    """Request payload for POST /api/v1/recordings/{id}/chat."""
    question: str
    top_k: Optional[int] = None
    conversation_history: Optional[List[dict]] = None  # Reserved for future multi-turn


class ChatResponse(BaseModel):
    """API response for POST /api/v1/recordings/{id}/chat."""
    recording_id: str
    answer: str
    sources: List[RAGSource] = Field(default_factory=list)
    language: Optional[str] = None
    message: str = "Question answered successfully"
