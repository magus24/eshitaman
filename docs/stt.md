# STT / Uzbek Multilingual Transcription Architecture

## 1. Overview

The Speech-to-Text (STT) layer in **Eshitaman** transforms standardized, silence-aligned audio chunks into a unified, timestamped, multilingual transcript.

The primary design principle is **first-class recognition of real-world Uzbek speech**, which naturally incorporates code-switching with Russian and English, especially within technical, academic, and professional domains.

```
Processed Audio Chunks (~30s, 2s overlap)
                   ↓
        STT Provider Factory
  (WhisperSTTProvider / MockSTTProvider)
                   ↓
        Chunk-Level STT Inference
   (Cost Control & Incremental Caching)
                   ↓
      Local → Global Timestamps
 (chunk.start_ms + seg.start_ms = global_ms)
                   ↓
    TranscriptMerger & Deduplication
 (Overlap artifact removal, sorting, merging)
                   ↓
        Full Merged Transcript
     (transcript/transcript.json)
```

---

## 2. Provider Abstraction

The STT layer is designed to be completely provider-agnostic. No business logic or API contracts depend on a specific vendor.

```
                  BaseSTTProvider (ABC)
                   /                 \
                  /                   \
        MockSTTProvider        WhisperSTTProvider
       (CI/Offline/Tests)      (OpenAI / Whisper-compatible)
```

### `BaseSTTProvider` Interface
Every provider implements:
- `provider_name -> str`
- `model_name -> str`
- `transcribe_chunk(audio_path, chunk_id, language_hint=None) -> ChunkTranscriptionResult`

### `STTProviderFactory`
Instantiates providers dynamically from the `STT_PROVIDER` environment variable:
- `STT_PROVIDER=mock` (default for local development and CI testing)
- `STT_PROVIDER=whisper` or `openai` (production Whisper API)

---

## 3. Real Provider: OpenAI Whisper API

### Selection Rationale:
1. **Multilingual Architecture**: Whisper was pre-trained on 680,000 hours of diverse multilingual audio. It explicitly supports Uzbek (`uz`), Russian (`ru`), and English (`en`).
2. **Code-Switching Fidelity**: When technical terms or Russian phrases are used inside Uzbek sentences (e.g., *"Bugun machine learning haqida gaplashamiz"*, *"Сейчас backend arxitekturasini ko'rib chiqamiz"*), Whisper maintains code-switching verbatim without translating or dropping words.
3. **Timestamp Granularity**: Using `response_format="verbose_json"` and `timestamp_granularities=["segment", "word"]`, Whisper provides millisecond-accurate start/end timestamps for both sentences and individual words.
4. **Compatibility**: Any OpenAI-compatible Whisper endpoint (e.g. self-hosted Whisper, Groq Whisper, local vLLM) can be targeted via `STT_API_BASE_URL`.

---

## 4. Language Strategy & Code-Switching

- **No Rigid Single-Language Constraint**: The system does NOT enforce `language="uz"` globally. Language detection is automatic by default.
- **Optional Language Hint**: Callers can supply `language_hint` (e.g. `uz`), but leaving it empty (`None`) allows the model to detect and transcribe multilingual utterances freely.
- **No Automatic Translation**: The STT pipeline strictly transcribes verbatim. Translation into other languages is reserved for downstream processing.

---

## 5. Global Timestamps & Chunk Overlap Merging

Because Stage 6 slices audio into ~30s chunks with 2s overlap:
1. **Global Timestamp Translation**:
   ```
   global_start_ms = chunk.start_ms + segment.start_ms
   global_end_ms   = chunk.start_ms + segment.end_ms
   ```
2. **Boundary Deduplication (`TranscriptMerger`)**:
   - Segments within the overlapping window [chunk[N+1].start_ms, chunk[N].end_ms] are inspected.
   - If consecutive segments from adjacent chunks share normalized text and proximity, the duplicate segment is removed while preserving the one with higher confidence/word timing.
   - Legitimate repetitions (e.g., repeating a word 20 seconds later) are preserved because their timestamps do not fall within the boundary overlap interval.

---

## 6. Resilience, Retries, and Cost Control

### Error Classification & Retries:
- **Permanent Errors (No Retry)**:
  - `401 / 403`: `STTAuthenticationError` — fail immediately, do not loop.
  - `400`: Invalid client payload — fail immediately.
- **Transient Errors (Exponential Backoff Retry up to 3 attempts)**:
  - `429`: `STTRateLimitError`
  - `5xx`: `STTProviderError`
  - Client / Socket Timeout: `STTTimeoutError`
  - Network Failure: `STTNetworkError`

### Cost Control & Incremental Caching:
- Each transcribed chunk is saved incrementally to `processed/chunk_transcripts.json`.
- If a 60-minute lecture transcription fails on chunk 40 out of 120, a retry will NOT re-send chunks 0 through 39. Only pending or failed chunks are transcribed.

---

## 7. Storage Layout

```text
backend/storage/recordings/<recording_id>/
├── audio.m4a (or audio.webm)          <-- Untouched original audio
├── metadata.json                      <-- Session status & summary
├── processed/
│   ├── audio.wav                      <-- Normalized 16kHz WAV
│   ├── chunks.json                    <-- Audio chunk descriptors
│   └── chunks/
│       ├── chunk_0000.wav
│       └── ...
└── transcript/
    ├── chunk_transcripts.json         <-- Incremental cache (cost control)
    └── transcript.json                <-- Final merged transcript (schema v1)
```

---

## 8. Configuration (`.env`)

```bash
STT_PROVIDER=mock                    # 'mock' or 'whisper'
STT_API_KEY=your-api-key             # Required for whisper
STT_API_BASE_URL=https://api.openai.com/v1
STT_MODEL=whisper-1
STT_TIMEOUT_SECONDS=60.0
STT_MAX_RETRIES=3
STT_LANGUAGE_HINT=                   # Optional language hint
```
