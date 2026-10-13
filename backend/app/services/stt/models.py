from datetime import datetime, timezone
from typing import List, Optional
from pydantic import BaseModel, Field


class TranscriptWord(BaseModel):
    """Optional word-level timestamp information."""
    word: str
    start_ms: int
    end_ms: int
    confidence: Optional[float] = None


class TranscriptSegment(BaseModel):
    """Segment-level transcription unit with global millisecond timestamps."""
    id: str
    start_ms: int
    end_ms: int
    text: str
    language: Optional[str] = None
    confidence: Optional[float] = None
    chunk_id: Optional[str] = None
    words: List[TranscriptWord] = Field(default_factory=list)


class Transcript(BaseModel):
    """Full merged recording transcript with metadata and versioning."""
    recording_id: str
    schema_version: int = 1
    language: Optional[str] = None
    duration_ms: int = 0
    provider: str = "mock"
    model: str = "mock-model"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    text: str = ""
    segments: List[TranscriptSegment] = Field(default_factory=list)


class ChunkTranscriptionResult(BaseModel):
    """Transcription output for an individual audio chunk with chunk-local timestamps."""
    chunk_id: str
    text: str
    language: Optional[str] = None
    duration_ms: int = 0
    segments: List[TranscriptSegment] = Field(default_factory=list)
    words: List[TranscriptWord] = Field(default_factory=list)


class TranscribeRequest(BaseModel):
    """Optional request payload for POST /api/v1/recordings/{id}/transcribe."""
    language_hint: Optional[str] = None


class TranscriptionResponse(BaseModel):
    """API response model for POST /api/v1/recordings/{id}/transcribe."""
    recording_id: str
    status: str
    language: Optional[str] = None
    segments_count: int = 0
    duration_ms: int = 0
    message: str = "Audio transcribed successfully"
    error_code: Optional[str] = None
    error_message: Optional[str] = None
