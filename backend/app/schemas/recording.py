from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class RecordingStatus(str, Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    TRANSCRIBING = "transcribing"
    TRANSCRIBED = "transcribed"
    ANALYZING = "analyzing"
    ANALYZED = "analyzed"
    INDEXING = "indexing"
    INDEXED = "indexed"
    COMPLETED = "completed"
    FAILED = "failed"
 
 
class RecordingMetadata(BaseModel):
    recording_id: str
    filename: str
    file_size_bytes: int
    mime_type: str
    duration_seconds: Optional[float] = None
    duration_millis: Optional[int] = None
    codec: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    sample_rate: Optional[int] = None
    bit_rate: Optional[int] = None
    channel_count: Optional[int] = None
    status: RecordingStatus = RecordingStatus.UPLOADED
    error_code: Optional[str] = None
    error_message: Optional[str] = None
 
 
class RecordingUploadResponse(BaseModel):
    recording_id: str
    status: RecordingStatus = RecordingStatus.UPLOADED
    file_size: int
    duration: Optional[float] = None
    mime_type: str
    created_at: datetime
    message: str = "Audio recording uploaded successfully"
    audio: Optional[dict] = None
    processing: Optional[dict] = None
    transcript: Optional[dict] = None
    analysis: Optional[dict] = None
    rag: Optional[dict] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None

