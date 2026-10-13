import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional
from app.core.config import settings
from app.schemas.processing import AudioInfo
from app.services.audio.exceptions import (
    FFmpegNotFoundError,
    AudioConversionError,
    AudioInspectionError,
)


class FFmpegService:
    def __init__(
        self,
        ffmpeg_path: Optional[str] = None,
        ffprobe_path: Optional[str] = None,
    ):
        self._ffmpeg_path = ffmpeg_path or settings.FFMPEG_PATH
        self._ffprobe_path = ffprobe_path or settings.FFPROBE_PATH

    @property
    def ffmpeg_bin(self) -> Optional[str]:
        if self._ffmpeg_path:
            return self._ffmpeg_path
        return shutil.which("ffmpeg")

    @property
    def ffprobe_bin(self) -> Optional[str]:
        if self._ffprobe_path:
            return self._ffprobe_path
        return shutil.which("ffprobe")

    def is_available(self) -> bool:
        """Check if FFmpeg executable is present and callable."""
        return self.ffmpeg_bin is not None

    def is_ffprobe_available(self) -> bool:
        """Check if FFprobe executable is present and callable."""
        return self.ffprobe_bin is not None

    def inspect_audio(self, path: Path) -> AudioInfo:
        """Inspect audio file metadata using ffprobe."""
        if not self.is_ffprobe_available():
            raise FFmpegNotFoundError(
                "ffprobe is not installed or not found in system PATH / FFPROBE_PATH."
            )

        if not path.exists():
            raise AudioInspectionError(f"Audio file not found: {path}")

        cmd = [
            self.ffprobe_bin,
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path.resolve()),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            if result.returncode != 0:
                raise AudioInspectionError(
                    f"ffprobe failed with exit code {result.returncode}: {result.stderr}"
                )

            data = json.loads(result.stdout)
            streams = data.get("streams", [])
            audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
            fmt = data.get("format", {})

            if not audio_stream and not fmt:
                raise AudioInspectionError(f"No audio stream found in {path.name}")

            duration_s = float(
                audio_stream.get("duration")
                or fmt.get("duration")
                or 0.0
            )
            sample_rate = int(audio_stream.get("sample_rate", 16000)) if audio_stream else 16000
            channels = int(audio_stream.get("channels", 1)) if audio_stream else 1
            codec = audio_stream.get("codec_name", "unknown") if audio_stream else "unknown"
            container = fmt.get("format_name", path.suffix.replace(".", ""))
            bitrate = int(fmt.get("bit_rate")) if fmt.get("bit_rate") else None
            file_size = int(fmt.get("size", path.stat().st_size if path.exists() else 0))

            return AudioInfo(
                duration_seconds=round(duration_s, 3),
                sample_rate=sample_rate,
                channels=channels,
                codec=codec,
                container=container,
                bitrate=bitrate,
                file_size_bytes=file_size,
            )
        except subprocess.TimeoutExpired as e:
            raise AudioInspectionError("ffprobe execution timed out", cause=e)
        except json.JSONDecodeError as e:
            raise AudioInspectionError(f"Failed to parse ffprobe JSON output: {e}", cause=e)
        except Exception as e:
            if isinstance(e, AudioInspectionError):
                raise
            raise AudioInspectionError(f"Unexpected error inspecting audio with ffprobe: {e}", cause=e)

    def convert_to_canonical_wav(
        self,
        input_path: Path,
        output_path: Path,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> Path:
        """Convert input audio file (.m4a, .webm, etc.) to canonical WAV (16kHz 16-bit mono PCM)."""
        if not self.is_available():
            raise FFmpegNotFoundError(
                "FFmpeg is required to convert compressed audio (.m4a / .webm), but was not found in PATH or FFMPEG_PATH."
            )

        if not input_path.exists():
            raise AudioConversionError(f"Source audio file not found: {input_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.ffmpeg_bin,
            "-y",  # Overwrite output
            "-i",
            str(input_path.resolve()),
            "-vn",  # No video
            "-acodec",
            "pcm_s16le",  # 16-bit signed little endian PCM
            "-ac",
            str(channels),  # 1 channel (mono)
            "-ar",
            str(sample_rate),  # 16000 Hz
            str(output_path.resolve()),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            if result.returncode != 0:
                raise AudioConversionError(
                    f"FFmpeg conversion failed (code {result.returncode}): {result.stderr}"
                )

            if not output_path.exists() or output_path.stat().st_size == 0:
                raise AudioConversionError("FFmpeg produced empty or missing output WAV file")

            return output_path
        except subprocess.TimeoutExpired as e:
            raise AudioConversionError("FFmpeg conversion timed out", cause=e)
        except Exception as e:
            if isinstance(e, AudioConversionError):
                raise
            raise AudioConversionError(f"Unexpected error during FFmpeg conversion: {e}", cause=e)
