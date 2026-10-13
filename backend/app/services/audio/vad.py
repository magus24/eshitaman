from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple
import wave
import numpy as np
from app.schemas.processing import SpeechSegment
from app.services.audio.exceptions import VADError


class BaseVAD(ABC):
    """Abstract base class for Voice Activity Detection."""

    @abstractmethod
    def detect_speech_segments(self, wav_path: Path) -> List[SpeechSegment]:
        """Detect speech segments with start and end timestamps in milliseconds."""
        pass

    def detect_silence_segments(
        self, wav_path: Path, total_duration_ms: int
    ) -> List[Tuple[int, int]]:
        """
        Derive silence segments (gaps between speech segments).
        Returns list of (start_ms, end_ms).
        """
        speech_segments = self.detect_speech_segments(wav_path)
        silences: List[Tuple[int, int]] = []
        current_pos = 0

        for seg in speech_segments:
            if seg.start_ms > current_pos:
                silences.append((current_pos, seg.start_ms))
            current_pos = max(current_pos, seg.end_ms)

        if current_pos < total_duration_ms:
            silences.append((current_pos, total_duration_ms))

        return silences


class EnergyVAD(BaseVAD):
    """
    Lightweight energy-based VAD using Root Mean Square (RMS) frame energy.
    Can be seamlessly substituted with SileroVAD or other models later without
    affecting pipeline callers.
    """

    def __init__(
        self,
        frame_ms: int = 30,
        energy_threshold: float = 0.015,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 350,
        padding_ms: int = 100,
    ):
        self.frame_ms = frame_ms
        self.energy_threshold = energy_threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.padding_ms = padding_ms

    def detect_speech_segments(self, wav_path: Path) -> List[SpeechSegment]:
        if not wav_path.exists():
            raise VADError(f"Audio file for VAD not found: {wav_path}")

        try:
            with wave.open(str(wav_path), "rb") as wf:
                sample_rate = wf.getframerate()
                sampwidth = wf.getsampwidth()
                nframes = wf.getnframes()
                raw_bytes = wf.readframes(nframes)

            if sampwidth != 2:
                raise VADError(f"Expected 16-bit PCM for VAD, got sampwidth={sampwidth}")

            samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            if len(samples) == 0:
                return []

            total_duration_ms = int(len(samples) * 1000 / sample_rate)
            frame_size = int(sample_rate * self.frame_ms / 1000)
            if frame_size <= 0:
                frame_size = 1

            num_frames = len(samples) // frame_size
            if num_frames == 0:
                # File shorter than one frame
                rms = np.sqrt(np.mean(samples ** 2)) if len(samples) > 0 else 0.0
                if rms >= self.energy_threshold:
                    return [SpeechSegment(start_ms=0, end_ms=total_duration_ms, duration_ms=total_duration_ms)]
                return []

            # Compute RMS per frame
            # Reshape up to num_frames * frame_size
            trimmed_samples = samples[: num_frames * frame_size].reshape((num_frames, frame_size))
            frame_rms = np.sqrt(np.mean(trimmed_samples ** 2, axis=1))

            is_speech_frame = frame_rms >= self.energy_threshold

            # State machine with hangover to bridge short pauses and filter short noise spikes
            segments: List[SpeechSegment] = []
            in_speech = False
            speech_start_ms = 0
            silence_counter_ms = 0

            for i, is_speech in enumerate(is_speech_frame):
                current_time_ms = int(i * self.frame_ms)

                if is_speech:
                    if not in_speech:
                        in_speech = True
                        speech_start_ms = max(0, current_time_ms - self.padding_ms)
                    silence_counter_ms = 0
                else:
                    if in_speech:
                        silence_counter_ms += self.frame_ms
                        if silence_counter_ms >= self.min_silence_duration_ms:
                            # Speech ended
                            speech_end_ms = min(total_duration_ms, current_time_ms - silence_counter_ms + self.padding_ms)
                            duration = speech_end_ms - speech_start_ms
                            if duration >= self.min_speech_duration_ms:
                                segments.append(
                                    SpeechSegment(
                                        start_ms=speech_start_ms,
                                        end_ms=speech_end_ms,
                                        duration_ms=duration,
                                    )
                                )
                            in_speech = False
                            silence_counter_ms = 0

            # Handle if speech continues until the end of the file
            if in_speech:
                speech_end_ms = total_duration_ms
                duration = speech_end_ms - speech_start_ms
                if duration >= self.min_speech_duration_ms:
                    segments.append(
                        SpeechSegment(
                            start_ms=speech_start_ms,
                            end_ms=speech_end_ms,
                            duration_ms=duration,
                        )
                    )

            return segments

        except VADError:
            raise
        except Exception as e:
            raise VADError(f"VAD speech detection failed: {e}", cause=e)
