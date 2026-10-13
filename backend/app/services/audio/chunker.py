import json
import uuid
import wave
from pathlib import Path
from typing import List, Optional
import numpy as np
from app.core.config import settings
from app.schemas.processing import AudioChunk, SpeechSegment
from app.services.audio.exceptions import ChunkingError


class AudioChunker:
    """Splits canonical WAV audio into manageable STT chunks, aligning to speech boundaries if available."""

    def __init__(
        self,
        chunk_duration_s: float = None,
        overlap_s: float = None,
        boundary_search_window_s: float = 3.0,
    ):
        self.chunk_duration_s = chunk_duration_s or settings.AUDIO_CHUNK_SECONDS
        self.overlap_s = overlap_s or settings.AUDIO_CHUNK_OVERLAP_SECONDS
        self.boundary_search_window_s = boundary_search_window_s

    def chunk(
        self,
        recording_id: str,
        wav_path: Path,
        chunks_dir: Path,
        speech_segments: Optional[List[SpeechSegment]] = None,
    ) -> List[AudioChunk]:
        """
        Splits wav_path into chunks and writes them into chunks_dir.
        Also persists chunks.json in chunks_dir.parent.
        """
        if not wav_path.exists():
            raise ChunkingError(f"WAV file for chunking not found: {wav_path}")

        chunks_dir.mkdir(parents=True, exist_ok=True)

        try:
            with wave.open(str(wav_path), "rb") as wf:
                channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                sample_rate = wf.getframerate()
                nframes = wf.getnframes()
                raw_bytes = wf.readframes(nframes)

            if sampwidth != 2:
                raise ChunkingError(f"Expected 16-bit PCM for chunking, got sampwidth={sampwidth}")

            samples = np.frombuffer(raw_bytes, dtype=np.int16)
            total_duration_ms = int(nframes * 1000 / sample_rate) if sample_rate > 0 else 0

            if total_duration_ms == 0 or len(samples) == 0:
                raise ChunkingError("Audio duration is 0ms, cannot produce chunks")

            target_chunk_ms = int(self.chunk_duration_s * 1000)
            overlap_ms = int(self.overlap_s * 1000)
            search_window_ms = int(self.boundary_search_window_s * 1000)

            # Determine chunk split intervals
            intervals = []
            current_start_ms = 0
            index = 0

            while current_start_ms < total_duration_ms:
                nominal_end_ms = current_start_ms + target_chunk_ms

                if nominal_end_ms >= total_duration_ms:
                    # Final chunk takes the rest
                    intervals.append((current_start_ms, total_duration_ms))
                    break

                # Attempt to find best boundary in silence gap near nominal_end_ms
                actual_end_ms = self._find_best_boundary(
                    nominal_end_ms=nominal_end_ms,
                    min_boundary_ms=current_start_ms + 5000,  # Ensure at least 5s per chunk
                    max_boundary_ms=min(total_duration_ms, nominal_end_ms + search_window_ms),
                    speech_segments=speech_segments or [],
                )

                intervals.append((current_start_ms, actual_end_ms))

                # Step forward with overlap
                next_start = actual_end_ms - overlap_ms
                if next_start <= current_start_ms:
                    next_start = current_start_ms + 1000  # Guarantee forward progress

                current_start_ms = next_start
                index += 1

            # Slice and save chunk files
            chunks: List[AudioChunk] = []
            for idx, (start_ms, end_ms) in enumerate(intervals):
                chunk_id = f"chunk_{recording_id}_{idx:04d}"
                start_frame = int(start_ms * sample_rate / 1000)
                end_frame = int(end_ms * sample_rate / 1000)

                chunk_samples = samples[start_frame:end_frame]
                chunk_filename = f"chunk_{idx:04d}.wav"
                chunk_path = chunks_dir / chunk_filename

                with wave.open(str(chunk_path), "wb") as chunk_wf:
                    chunk_wf.setnchannels(channels)
                    chunk_wf.setsampwidth(sampwidth)
                    chunk_wf.setframerate(sample_rate)
                    chunk_wf.writeframes(chunk_samples.tobytes())

                chunk_size = chunk_path.stat().st_size
                chunk_obj = AudioChunk(
                    chunk_id=chunk_id,
                    recording_id=recording_id,
                    index=idx,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    duration_ms=end_ms - start_ms,
                    file_path=str(chunk_path.as_posix()),
                    file_size_bytes=chunk_size,
                )
                chunks.append(chunk_obj)

            # Save chunks.json metadata in chunks_dir.parent (processed/)
            chunks_meta_path = chunks_dir.parent / "chunks.json"
            with open(chunks_meta_path, "w", encoding="utf-8") as f:
                json.dump([c.model_dump() for c in chunks], f, indent=2)

            return chunks

        except ChunkingError:
            raise
        except Exception as e:
            raise ChunkingError(f"Failed to chunk audio: {e}", cause=e)

    def _find_best_boundary(
        self,
        nominal_end_ms: int,
        min_boundary_ms: int,
        max_boundary_ms: int,
        speech_segments: List[SpeechSegment],
    ) -> int:
        """Find a silence gap between speech segments closest to nominal_end_ms."""
        if not speech_segments:
            return nominal_end_ms

        best_point = nominal_end_ms
        min_dist = float("inf")

        # Check gaps between consecutive speech segments
        for i in range(len(speech_segments) - 1):
            gap_start = speech_segments[i].end_ms
            gap_end = speech_segments[i + 1].start_ms

            if gap_end > gap_start:  # Real silence gap
                # Candidate split is the center of the silence gap
                gap_mid = (gap_start + gap_end) // 2
                if min_boundary_ms <= gap_mid <= max_boundary_ms:
                    dist = abs(gap_mid - nominal_end_ms)
                    if dist < min_dist:
                        min_dist = dist
                        best_point = gap_mid

        # Also check gap before first segment or after last segment if in window
        if speech_segments[0].start_ms > min_boundary_ms:
            first_gap_mid = speech_segments[0].start_ms // 2
            if min_boundary_ms <= first_gap_mid <= max_boundary_ms:
                dist = abs(first_gap_mid - nominal_end_ms)
                if dist < min_dist:
                    min_dist = dist
                    best_point = first_gap_mid

        return best_point
