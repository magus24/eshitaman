from typing import Optional, List
from pydantic import BaseModel, Field


class AudioInfo(BaseModel):
    duration_seconds: float
    sample_rate: int
    channels: int
    codec: str
    container: str
    bitrate: Optional[int] = None
    file_size_bytes: int


class SpeechSegment(BaseModel):
    start_ms: int
    end_ms: int
    duration_ms: int


class AudioChunk(BaseModel):
    chunk_id: str
    recording_id: str
    index: int
    start_ms: int
    end_ms: int
    duration_ms: int
    file_path: str
    file_size_bytes: int


class ProcessingResult(BaseModel):
    recording_id: str
    status: str = "processed"
    audio_info: AudioInfo
    processed_audio_path: str
    speech_segments: List[SpeechSegment] = Field(default_factory=list)
    chunks: List[AudioChunk] = Field(default_factory=list)
    chunks_count: int = 0
    speech_segments_count: int = 0
    processing_duration_seconds: float = 0.0
    error_code: Optional[str] = None
    error_message: Optional[str] = None


class ProcessingResponse(BaseModel):
    recording_id: str
    status: str
    duration_seconds: float
    chunks_count: int
    speech_segments_count: int
    message: str = "Audio processed successfully"
    error_code: Optional[str] = None
    error_message: Optional[str] = None
