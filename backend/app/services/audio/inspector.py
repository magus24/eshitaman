import wave
from pathlib import Path
from typing import Optional
from app.schemas.processing import AudioInfo
from app.services.audio.exceptions import AudioInspectionError
from app.services.audio.ffmpeg import FFmpegService


class AudioInspector:
    """Inspects audio files and extracts audio stream properties."""

    def __init__(self, ffmpeg_service: Optional[FFmpegService] = None):
        self.ffmpeg_service = ffmpeg_service or FFmpegService()

    def inspect(self, audio_path: Path) -> AudioInfo:
        """Inspect the audio file and return AudioInfo."""
        if not audio_path.exists():
            raise AudioInspectionError(f"Audio file not found: {audio_path}")

        file_size = audio_path.stat().st_size
        if file_size == 0:
            raise AudioInspectionError(f"Audio file is empty (0 bytes): {audio_path}")

        suffix = audio_path.suffix.lower()

        # If it's a WAV file, try standard library inspection first
        if suffix == ".wav":
            try:
                return self._inspect_wav(audio_path, file_size)
            except Exception as e:
                # If native wav parser fails, try ffprobe if available
                if self.ffmpeg_service.is_ffprobe_available():
                    try:
                        return self.ffmpeg_service.inspect_audio(audio_path)
                    except Exception:
                        pass
                raise AudioInspectionError(f"Failed to inspect WAV file: {e}", cause=e)

        # For non-WAV formats (M4A, WebM, etc.), use ffprobe
        return self.ffmpeg_service.inspect_audio(audio_path)

    def _inspect_wav(self, path: Path, file_size: int) -> AudioInfo:
        try:
            with wave.open(str(path), "rb") as wf:
                channels = wf.getnchannels()
                sample_rate = wf.getframerate()
                sampwidth = wf.getsampwidth()
                nframes = wf.getnframes()
                duration = round(nframes / float(sample_rate), 3) if sample_rate > 0 else 0.0
                bitrate = sample_rate * channels * sampwidth * 8

                codec = f"pcm_s{sampwidth * 8}le"

                return AudioInfo(
                    duration_seconds=duration,
                    sample_rate=sample_rate,
                    channels=channels,
                    codec=codec,
                    container="wav",
                    bitrate=bitrate,
                    file_size_bytes=file_size,
                )
        except wave.Error as e:
            raise AudioInspectionError(f"Invalid or corrupted WAV file: {e}", cause=e)
