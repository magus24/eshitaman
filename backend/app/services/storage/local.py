import json
import re
from pathlib import Path
from typing import Optional, Dict, Any
from app.core.config import settings
from app.services.storage.base import BaseStorage

SAFE_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-]+$")


class LocalStorage(BaseStorage):
    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or (settings.STORAGE_PATH / "recordings")
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _sanitize_recording_id(self, recording_id: str) -> str:
        if not recording_id or not SAFE_ID_REGEX.match(recording_id):
            raise ValueError(f"Invalid recording ID format: '{recording_id}'")
        return recording_id

    def _get_session_dir(self, recording_id: str) -> Path:
        sanitized = self._sanitize_recording_id(recording_id)
        session_dir = (self.base_path / sanitized).resolve()
        # Prevent path traversal outside base_path
        if not str(session_dir).startswith(str(self.base_path)):
            raise ValueError("Path traversal attempt detected")
        return session_dir

    async def recording_exists(self, recording_id: str) -> bool:
        try:
            session_dir = self._get_session_dir(recording_id)
            return session_dir.exists() and session_dir.is_dir()
        except ValueError:
            return False

    async def save_recording(
        self,
        recording_id: str,
        audio_content: bytes,
        filename: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        session_dir = self._get_session_dir(recording_id)
        if session_dir.exists():
            raise FileExistsError(f"Recording '{recording_id}' already exists")

        session_dir.mkdir(parents=True, exist_ok=False)

        # Determine safe audio filename with extension
        ext = Path(filename).suffix.lower()
        if not ext:
            ext = ".m4a"
        audio_path = session_dir / f"audio{ext}"

        # Write audio data
        with open(audio_path, "wb") as f:
            f.write(audio_content)

        # Write metadata.json
        meta_path = session_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

        return metadata

    async def get_recording_metadata(self, recording_id: str) -> Optional[Dict[str, Any]]:
        try:
            session_dir = self._get_session_dir(recording_id)
            meta_path = session_dir / "metadata.json"
            if not meta_path.exists():
                return None
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (ValueError, FileNotFoundError):
            return None

    async def get_audio_path(self, recording_id: str) -> Optional[Path]:
        try:
            session_dir = self._get_session_dir(recording_id)
            if not session_dir.exists():
                return None
            for file in session_dir.glob("audio.*"):
                if file.is_file():
                    return file
            return None
        except ValueError:
            return None

    async def update_recording_metadata(
        self, recording_id: str, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        session_dir = self._get_session_dir(recording_id)
        if not session_dir.exists():
            raise FileNotFoundError(f"Recording '{recording_id}' not found")
        meta_path = session_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        return metadata

