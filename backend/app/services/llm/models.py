from datetime import datetime, timezone
from typing import List, Optional
from pydantic import BaseModel, Field


class AITerm(BaseModel):
    """Domain or technical term with an explanation and linked transcript source segments."""
    term: str
    explanation: str
    source_segment_ids: List[str] = Field(default_factory=list)


class AIKeyPoint(BaseModel):
    """Granular key point with linked source segment IDs."""
    point: str
    source_segment_ids: List[str] = Field(default_factory=list)


class AIQuestion(BaseModel):
    """Study question based strictly on transcript content."""
    question: str
    source_segment_ids: List[str] = Field(default_factory=list)


class AIAnalysis(BaseModel):
    """
    Complete structured AI understanding output of a transcript.
    Maintains segment-level source tracking for future RAG / AI Chat capabilities.
    """
    recording_id: str
    schema_version: int = 1
    language: str = "uz"
    summary: str
    key_points: List[str] = Field(default_factory=list)
    explanation: str
    terms: List[AITerm] = Field(default_factory=list)
    questions: List[str] = Field(default_factory=list)
    source_segment_ids: List[str] = Field(default_factory=list)
    provider: str = "mock"
    model: str = "mock-model"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class AnalysisResponse(BaseModel):
    """API response model for POST /api/v1/recordings/{id}/analyze."""
    recording_id: str
    status: str
    language: Optional[str] = None
    summary: str = ""
    key_points_count: int = 0
    terms_count: int = 0
    questions_count: int = 0
    message: str = "Analysis generated successfully"
    error_code: Optional[str] = None
    error_message: Optional[str] = None
