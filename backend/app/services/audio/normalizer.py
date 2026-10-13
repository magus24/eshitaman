import wave
from pathlib import Path
import numpy as np
from app.services.audio.exceptions import AudioNormalizationError


class AudioNormalizer:
    """Normalizes 16-bit PCM WAV audio to a consistent peak / RMS level without clipping."""

    def __init__(self, target_peak_dbfs: float = -1.0, max_gain_factor: float = 6.0):
        """
        :param target_peak_dbfs: Target peak level in dBFS (e.g. -1.0 dBFS).
        :param max_gain_factor: Maximum allowed amplification factor to prevent boosting silence/noise floor.
        """
        self.target_peak_dbfs = target_peak_dbfs
        self.max_gain_factor = max_gain_factor
        # Target peak amplitude for 16-bit signed PCM (max is 32767)
        self.target_peak = int(32767 * (10 ** (target_peak_dbfs / 20.0)))

    def normalize(self, input_wav_path: Path, output_wav_path: Path) -> Path:
        """
        Reads input 16-bit PCM WAV, scales audio to target peak, and writes to output_wav_path.
        Non-destructive: original input file is untouched.
        """
        if not input_wav_path.exists():
            raise AudioNormalizationError(f"WAV file for normalization not found: {input_wav_path}")

        try:
            with wave.open(str(input_wav_path), "rb") as wf_in:
                nchannels = wf_in.getnchannels()
                sampwidth = wf_in.getsampwidth()
                framerate = wf_in.getframerate()
                nframes = wf_in.getnframes()

                if sampwidth != 2:
                    raise AudioNormalizationError(
                        f"Expected 16-bit PCM (sampwidth=2), got sampwidth={sampwidth}"
                    )

                raw_bytes = wf_in.readframes(nframes)

            # Load into int16 numpy array
            audio_data = np.frombuffer(raw_bytes, dtype=np.int16)

            if len(audio_data) == 0:
                raise AudioNormalizationError("Input audio file contains 0 frames")

            current_peak = int(np.max(np.abs(audio_data)))

            if current_peak == 0:
                # Complete silence, write unchanged
                normalized_data = audio_data
            else:
                gain = self.target_peak / float(current_peak)
                # Cap gain to avoid blowing up background noise in quiet recordings
                gain = min(gain, self.max_gain_factor)

                # Float scaling + clip to int16 boundaries to guarantee no clipping overflow
                scaled = np.clip(
                    np.round(audio_data.astype(np.float64) * gain),
                    -32768,
                    32767,
                ).astype(np.int16)
                normalized_data = scaled

            # Write normalized file
            output_wav_path.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(output_wav_path), "wb") as wf_out:
                wf_out.setnchannels(nchannels)
                wf_out.setsampwidth(sampwidth)
                wf_out.setframerate(framerate)
                wf_out.writeframes(normalized_data.tobytes())

            return output_wav_path
        except AudioNormalizationError:
            raise
        except Exception as e:
            raise AudioNormalizationError(f"Failed to normalize audio: {e}", cause=e)
