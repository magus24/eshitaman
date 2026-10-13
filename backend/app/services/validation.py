import re
from pathlib import Path
from typing import Optional, Tuple
from fastapi import HTTPException, status
from app.core.config import settings

SAFE_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-]{3,64}$")


def validate_recording_id(recording_id: Optional[str]) -> str:
    """Validate that recording_id is clean and safe from path traversal."""
    if not recording_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="recording_id is required",
        )
    recording_id = recording_id.strip()
    if not SAFE_ID_REGEX.match(recording_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="recording_id must be 3-64 characters alphanumeric, hyphen or underscore",
        )
    return recording_id


def validate_file_size(size_bytes: int) -> None:
    """Validate that the file is not empty and does not exceed maximum limit."""
    if size_bytes == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio file cannot be empty",
        )

    if size_bytes > settings.max_audio_size_bytes:
        max_mb = settings.MAX_AUDIO_SIZE_MB
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Audio file exceeds maximum allowed size of {max_mb} MB",
        )


def validate_audio_type(filename: Optional[str], content_type: Optional[str]) -> Tuple[str, str]:
    """Validate file extension and MIME type."""
    if not filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required",
        )

    ext = Path(filename).suffix.lower()
    if not ext or ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{ext}'. Allowed extensions: {', '.join(sorted(settings.ALLOWED_EXTENSIONS))}",
        )

    # Normalize mime type
    mime = (content_type or "").lower().split(";")[0].strip()
    full_mime = (content_type or "").lower().strip()

    # Check if mime or base mime is in allowed set
    if content_type and (mime not in settings.ALLOWED_MIME_TYPES and full_mime not in settings.ALLOWED_MIME_TYPES):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported audio MIME type '{content_type}'",
        )

    # Fallback MIME based on extension if not provided
    if not content_type or content_type == "application/octet-stream":
        ext_to_mime = {
            ".m4a": "audio/m4a",
            ".webm": "audio/webm",
            ".aac": "audio/aac",
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".ogg": "audio/ogg",
        }
        resolved_mime = ext_to_mime.get(ext, "audio/m4a")
    else:
        resolved_mime = content_type

    return ext, resolved_mime
