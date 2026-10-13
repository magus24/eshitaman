# Audio Processing & STT Preparation Architecture

## Overview

The audio processing layer in **Eshitaman** transforms recorded audio (M4A from mobile, WebM from browser, or WAV) into clean, standardized, silence-aligned audio chunks ready for speech-to-text (STT) inference in Stage 7.

```
Uploaded Audio (M4A / WebM / WAV)
             ↓
     Audio Validation
             ↓
     Audio Inspection
             ↓
Canonical WAV Conversion (16kHz, 16-bit PCM, Mono)
             ↓
   Volume Normalization (Peak / Anti-clipping)
             ↓
Voice Activity Detection (Energy-based VAD)
             ↓
  Silence-aligned Audio Chunking (~30s chunks)
             ↓
       Chunks Metadata (chunks.json)
             ↓
     STT-Ready Audio Chunks
```

---

## 1. Pipeline Stages

### Stage 1: Audio Validation
- Uploaded via `POST /api/v1/recordings` multipart endpoint.
- Validates file size (`<= MAX_AUDIO_SIZE_MB`, default 100MB, non-empty), MIME types (`audio/m4a`, `audio/webm`, etc.), and sanitizes recording IDs (`^[a-zA-Z0-9_\-]+$`) against path traversal attacks.

### Stage 2: Audio Inspection (`AudioInspector`)
- Inspects real container properties rather than trusting client headers.
- Native WAV parsing via Python standard library `wave` (duration, sample rate, channels, bit depth, bitrate).
- Compressed formats (`.m4a`, `.webm`, `.mp3`) inspected via `ffprobe` when available.
- Returns typed `AudioInfo` model.

### Stage 3: Canonical WAV Conversion (`FFmpegService`)
- Standardized canonical STT format:
  - **Container**: WAV
  - **Encoding**: PCM 16-bit signed little-endian (`pcm_s16le`)
  - **Sample Rate**: 16,000 Hz (`AUDIO_SAMPLE_RATE`)
  - **Channels**: 1 (mono) (`AUDIO_CHANNELS`)
- Non-destructive: original recorded audio (`audio.m4a` / `audio.webm`) is preserved in its original location. Working canonical WAV is saved to `processed/audio_raw.wav`.

### Stage 4: Volume Normalization (`AudioNormalizer`)
- Performs peak scaling on 16-bit PCM data to a target peak level of `-1.0 dBFS` (~29,204 amplitude out of 32,767 max).
- Caps gain at `6.0x` (+15.5 dB) to avoid amplifying silence or background noise floor.
- Hardware clipping protection via `numpy.clip(-32768, 32767)`.
- Writes normalized WAV to `processed/audio.wav`.

### Stage 5: Voice Activity Detection (`BaseVAD` / `EnergyVAD`)
- Pluggable interface (`BaseVAD`) allowing future replacement with neural models (such as `SileroVAD`) without changing pipeline callers.
- Current implementation (`EnergyVAD`):
  - 30ms window Root Mean Square (RMS) frame energy.
  - Energy thresholding with hangover smoothing (bridges short intra-word silence pauses up to 350ms).
  - Minimum speech duration filter (250ms) to discard clicks and pops.
- Outputs list of `SpeechSegment(start_ms, end_ms, duration_ms)`.

### Stage 6: Silence-aligned Chunking (`AudioChunker`)
- Long lecture/meeting recordings (up to 60+ minutes) are sliced into manageable chunks.
- **Chunk Duration**: 30.0 seconds (`AUDIO_CHUNK_SECONDS`).
- **Overlap**: 2.0 seconds (`AUDIO_CHUNK_OVERLAP_SECONDS`).
- **Boundary Optimization**: Instead of cutting arbitrarily at 30.0s and truncating mid-word, the chunker uses VAD silence gaps within a ±3.0s window around the nominal cut-off to split at natural pauses.
- Chunks saved in `processed/chunks/chunk_0000.wav`, `chunk_0001.wav`, etc.
- Metadata persisted to `processed/chunks.json`.

---

## 2. File Storage Structure

All generated audio files reside securely in the recording session directory:

```text
backend/storage/recordings/<recording_id>/
├── audio.m4a (or audio.webm)          <-- Untouched original file
├── metadata.json                      <-- Session status & metadata
└── processed/
    ├── audio_raw.wav                  <-- Canonical resampled WAV
    ├── audio.wav                      <-- Peak-normalized canonical WAV
    ├── chunks.json                    <-- List of AudioChunk descriptors
    └── chunks/
        ├── chunk_0000.wav             <-- STT-ready slice 0 (0s - 29.8s)
        ├── chunk_0001.wav             <-- STT-ready slice 1 (27.8s - 58.2s)
        └── ...
```

---

## 3. Idempotency & Status Lifecycle

1. `uploaded` -> Recording received and stored.
2. `processing` -> Pipeline started.
3. `processed` -> Chunks and canonical audio ready for STT.
4. `failed` -> Error encountered (with `error_code` and `error_message`).

Calling `POST /api/v1/recordings/{id}/process` on an already processed recording returns the existing result immediately with status `200 OK` and avoids generating duplicate chunks.
