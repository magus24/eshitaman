from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any


class BaseStorage(ABC):
    @abstractmethod
    async def save_recording(
        self,
        recording_id: str,
        audio_content: bytes,
        filename: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Save audio file and associated metadata."""
        pass

    @abstractmethod
    async def recording_exists(self, recording_id: str) -> bool:
        """Check if a recording with this ID already exists."""
        pass

    @abstractmethod
    async def get_recording_metadata(self, recording_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve stored metadata for a recording."""
        pass

    @abstractmethod
    async def get_audio_path(self, recording_id: str) -> Optional[Path]:
        """Get the path to the stored audio file."""
        pass

    @abstractmethod
    async def update_recording_metadata(
        self, recording_id: str, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update stored metadata for a recording."""
        pass
