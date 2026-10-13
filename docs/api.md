# Eshitaman API Documentation

## Overview

The Eshitaman backend provides HTTP APIs for uploading, storing, and managing audio recordings.

- **Base URL (Local Development)**: `http://localhost:8000` (or `http://10.0.2.2:8000` from Android Emulator)
- **Content Type**: `multipart/form-data` for file uploads, `application/json` for responses.

---

## Endpoints

### 1. Health Check

Checks backend service health and availability.

- **URL**: `GET /health`
- **Authentication**: None
- **Response**: `200 OK`

```json
{
  "status": "ok",
  "app": "Eshitaman",
  "version": "0.1.0"
}
```

---

### 2. Upload Audio Recording

Receives an audio file together with recording metadata, validates constraints, and stores both audio and metadata securely on the server.

- **URL**: `POST /api/v1/recordings`
- **Content-Type**: `multipart/form-data`
- **Authentication**: None (Stage 5 MVP)

#### Request Parameters (Multipart Form)

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | Binary File | **Yes** | Audio file payload (`.m4a`, `.webm`, `.aac`, `.mp3`, `.wav`, `.ogg`). |
| `recording_id` | String | **Yes** | Unique session identifier (3-64 chars, alphanumeric, `-`, `_`). |
| `duration_seconds` | Float | No | Duration of recorded speech in seconds. |
| `duration_millis` | Integer | No | Duration of recorded speech in milliseconds. |
| `mime_type` | String | No | Audio MIME type (e.g. `audio/m4a`, `audio/webm;codecs=opus`). |
| `codec` | String | No | Codec used (e.g. `aac_lc`, `opus`). |
| `sample_rate` | Integer | No | Audio sample rate in Hz (e.g. `44100`, `16000`). |
| `bit_rate` | Integer | No | Audio bit rate in bps (e.g. `128000`). |
| `channel_count` | Integer | No | Number of channels (e.g. `1` for mono, `2` for stereo). |
| `metadata_json` | String | No | Optional raw JSON string representation of client session metadata. |

#### Constraints & Validation

- **File Size**: Maximum `MAX_AUDIO_SIZE_MB` (default `100 MB`). Cannot be 0 bytes.
- **Allowed Extensions**: `.m4a`, `.webm`, `.aac`, `.mp3`, `.wav`, `.ogg`.
- **Allowed MIME Types**: `audio/m4a`, `audio/mp4`, `audio/x-m4a`, `audio/aac`, `audio/webm`, `audio/webm;codecs=opus`, `audio/wav`, `audio/x-wav`, `audio/wave`, `audio/mpeg`, `audio/mp3`, `audio/ogg`, `application/octet-stream`.
- **Security**: Recording ID is sanitized against path traversal (`^[a-zA-Z0-9_\-]+$`). Client filename is never used directly as a file path.

#### Success Response

- **Status Code**: `201 Created`

```json
{
  "recording_id": "rec_1774640000000_1234",
  "status": "uploaded",
  "file_size": 1048576,
  "duration": 45.5,
  "mime_type": "audio/m4a",
  "created_at": "2026-09-26T00:15:00.000Z",
  "message": "Audio recording uploaded successfully"
}
```

#### Error Responses

| Status Code | Reason | Example Response |
|---|---|---|
| `400 Bad Request` | Empty file, unsupported extension, invalid ID format | `{"detail": "Audio file cannot be empty"}` |
| `409 Conflict` | Duplicate `recording_id` already exists | `{"detail": "Recording with ID 'rec_123' already exists"}` |
| `413 Content Too Large` | File exceeds `MAX_AUDIO_SIZE_MB` | `{"detail": "Audio file exceeds maximum allowed size of 100 MB"}` |
| `500 Internal Error` | Storage write error | `{"detail": "Server error while saving recording"}` |

---

### 3. Get Recording Status & Metadata

Queries the status, audio inspection details, and processing information of a recording by its ID.

- **URL**: `GET /api/v1/recordings/{recording_id}`
- **Authentication**: None
- **Response**: `200 OK`

```json
{
  "recording_id": "rec_1774640000000_1234",
  "status": "processed",
  "file_size": 1048576,
  "duration": 45.5,
  "mime_type": "audio/m4a",
  "created_at": "2026-09-26T00:15:00.000Z",
  "message": "Recording found",
  "audio": {
    "duration_seconds": 45.5,
    "sample_rate": 16000,
    "channels": 1,
    "codec": "pcm_s16le",
    "container": "wav",
    "bitrate": 256000,
    "file_size_bytes": 1456000
  },
  "processing": {
    "chunks_count": 2,
    "speech_segments_count": 3,
    "processed_audio_path": "backend/storage/recordings/rec_1774640000000_1234/processed/audio.wav",
    "processing_duration_seconds": 0.42
  },
  "error_code": null,
  "error_message": null
}
```

---

### 4. Trigger STT Audio Processing Pipeline

Executes STT audio preparation: audio inspection, canonical conversion (16kHz 16-bit mono WAV), peak normalization without clipping, energy-based VAD speech detection, and chunking with speech boundary alignment.

- **URL**: `POST /api/v1/recordings/{recording_id}/process`
- **Authentication**: None
- **Idempotency**: Repeated requests on an already processed recording (`status == "processed"`) return the existing result with status `200 OK` without creating duplicate chunks.

#### Success Response

- **Status Code**: `200 OK`

```json
{
  "recording_id": "rec_1774640000000_1234",
  "status": "processed",
  "duration_seconds": 45.5,
  "chunks_count": 2,
  "speech_segments_count": 3,
  "message": "Audio processed successfully",
  "error_code": null,
  "error_message": null
}
```

#### Error Responses

| Status Code | Reason | Example Detail |
|---|---|---|
| `400 Bad Request` | Invalid recording ID format | `{"detail": "Invalid recording ID format"}` |
| `404 Not Found` | Recording does not exist | `{"detail": "Recording with ID 'xyz' not found"}` |
| `422 Unprocessable Content` | Corrupted audio, missing audio payload, or chunking failure | `{"detail": {"error_code": "AUDIO_INSPECTION_ERROR", "message": "Failed to parse audio"}}` |
| `503 Service Unavailable` | Compressed audio (.m4a/.webm) submitted but FFmpeg is not installed | `{"detail": {"error_code": "FFMPEG_NOT_FOUND", "message": "FFmpeg is required..."}}` |

---

### 5. Trigger Multilingual Speech-to-Text Transcription

Transcribes canonical audio chunks using the configured STT provider (e.g. Whisper API or Mock provider in dev). Supports Uzbek (`uz`), Russian (`ru`), and English (`en`) code-switching. Reconstructs global recording timeline timestamps and deduplicates chunk overlap boundaries.

- **URL**: `POST /api/v1/recordings/{recording_id}/transcribe`
- **Authentication**: None
- **Query / Body Parameters**:
  - `language_hint`: (Optional) Language code hint (e.g. `uz`). Default is `null` for automatic multilingual speech recognition.
- **Idempotency**: If the recording is already transcribed (`status == "transcribed"`), returns the existing result with status `200 OK` without re-transcribing chunks.
- **Cost Control**: Failed runs only re-transcribe pending or failed chunks; completed chunk results are cached.

#### Success Response

- **Status Code**: `200 OK`

```json
{
  "recording_id": "rec_1774640000000_1234",
  "status": "transcribed",
  "language": "uz",
  "segments_count": 48,
  "duration_ms": 145000,
  "message": "Audio transcribed successfully",
  "error_code": null,
  "error_message": null
}
```

#### Error Responses

| Status Code | Reason | Example Detail |
|---|---|---|
| `400 Bad Request` | Recording not in `processed` status | `{"detail": {"error_code": "RECORDING_NOT_PROCESSED", "message": "Recording must be in 'processed' status before transcription."}}` |
| `401 Unauthorized` | Invalid STT provider API key | `{"detail": {"error_code": "STT_AUTHENTICATION_ERROR", "message": "STT authentication failed."}}` |
| `404 Not Found` | Recording does not exist | `{"detail": "Recording with ID 'xyz' not found"}` |
| `422 Unprocessable Content` | Chunks missing or audio parsing error | `{"detail": {"error_code": "CHUNKS_NOT_FOUND", "message": "Processed chunks metadata not found."}}` |
| `429 Too Many Requests` | STT provider rate limit exceeded | `{"detail": {"error_code": "STT_RATE_LIMIT", "message": "STT rate limit exceeded."}}` |
| `500 Internal Error` | STT provider permanent failure | `{"detail": {"error_code": "INTERNAL_TRANSCRIPTION_ERROR", "message": "..."}}` |

---

### 6. Get Full Timestamped Transcript

Retrieves the complete structured transcript with global millisecond timestamps and word-level timing (when available from the provider).

- **URL**: `GET /api/v1/recordings/{recording_id}/transcript`
- **Authentication**: None
- **Response**: `200 OK`

```json
{
  "recording_id": "rec_1774640000000_1234",
  "schema_version": 1,
  "language": "uz",
  "duration_ms": 145000,
  "provider": "whisper",
  "model": "whisper-1",
  "created_at": "2026-09-26T00:20:00.000Z",
  "text": "Bugun machine learning haqida gaplashamiz. Сейчас мы рассмотрим database architecture.",
  "segments": [
    {
      "id": "seg_0000",
      "start_ms": 1000,
      "end_ms": 4800,
      "text": "Bugun machine learning haqida gaplashamiz.",
      "language": "uz",
      "confidence": 0.94,
      "chunk_id": "chunk_0000",
      "words": [
        {
          "word": "Bugun",
          "start_ms": 1000,
          "end_ms": 1600,
          "confidence": 0.98
        }
      ]
    },
    {
      "id": "seg_0001",
      "start_ms": 5200,
      "end_ms": 9500,
      "text": "Сейчас мы рассмотрим database architecture.",
      "language": "ru",
      "confidence": 0.92,
      "chunk_id": "chunk_0000",
      "words": []
    }
  ]
}
```

---

### 7. Trigger AI Understanding Analysis

Generates structured educational analysis (summary, key points, explanation, important terms, and study questions) from the timestamped transcript using the configured LLM provider. Preserves technical terminology, handles Uzbek, Russian, and English code-switching, and maintains segment source tracking.

- **URL**: `POST /api/v1/recordings/{recording_id}/analyze`
- **Authentication**: None
- **Request Body (JSON, Optional)**:
  ```json
  {
    "context_instructions": "Universitet talabalari uchun dars xulosasi"
  }
  ```
- **Idempotency**: Repeated requests on an already analyzed recording (`status == "analyzed"`) immediately return the cached result with status `200 OK` without re-invoking the LLM.

#### Success Response

- **Status Code**: `200 OK`

```json
{
  "recording_id": "rec_1774640000000_1234",
  "status": "analyzed",
  "language": "uz",
  "summary": "Ushbu darsda machine learning modellarining aniqligi, FastAPI asosidagi backend arxitekturasi...",
  "key_points_count": 3,
  "terms_count": 3,
  "questions_count": 3,
  "message": "Analysis generated successfully",
  "error_code": null,
  "error_message": null
}
```

#### Error Responses

| Status Code | Reason | Example Detail |
|---|---|---|
| `400 Bad Request` | Recording not yet transcribed | `{"detail": {"error_code": "RECORDING_NOT_TRANSCRIBED", "message": "Recording must be in 'transcribed' status..."}}` |
| `401 Unauthorized` | Invalid LLM API key | `{"detail": {"error_code": "LLM_AUTHENTICATION_ERROR", "message": "LLM authentication failed."}}` |
| `404 Not Found` | Recording or transcript does not exist | `{"detail": {"error_code": "TRANSCRIPT_NOT_FOUND", "message": "Transcript not found."}}` |
| `429 Too Many Requests` | LLM provider rate limit exceeded | `{"detail": {"error_code": "LLM_RATE_LIMIT", "message": "LLM rate limit exceeded."}}` |
| `500 Internal Error` | Provider error or unexpected failure | `{"detail": {"error_code": "INTERNAL_ANALYSIS_ERROR", "message": "..."}}` |

---

### 8. Get Full Structured AI Analysis

Retrieves the complete persisted AI understanding analysis with linked source transcript segments.

- **URL**: `GET /api/v1/recordings/{recording_id}/analysis`
- **Authentication**: None
- **Response**: `200 OK`

```json
{
  "recording_id": "rec_1774640000000_1234",
  "schema_version": 1,
  "language": "uz",
  "summary": "Ushbu darsda machine learning modellarining aniqligi va FastAPI backend arxitekturasi ko'rib chiqildi.",
  "key_points": [
    "Machine learning modelining aniqligi 95 foiz natijani ko'rsatdi.",
    "Backend tizimida yuqori unumdorlik uchun FastAPI ishlatildi.",
    "Ma'lumotlar bazasi arxitekturasi mikroxizmatlar bilan ishonchli bog'landi."
  ],
  "explanation": "Ma'ruzada zamonaviy axborot tizimlarini barpo etishning asosiy tamoyillari tushuntirildi...",
  "terms": [
    {
      "term": "Machine Learning",
      "explanation": "Ma'lumotlar tahlili orqali o'rganuvchi algoritmlar to'plami.",
      "source_segment_ids": ["seg_0000"]
    },
    {
      "term": "FastAPI",
      "explanation": "Python tilida yuqori tezlikda ishlovchi zamonaviy REST API freymvorki.",
      "source_segment_ids": ["seg_0001"]
    }
  ],
  "questions": [
    "Darsda aytilgan machine learning modelining aniqligi necha foiz bo'ldi?",
    "Backend qismida nima sababdan FastAPI freymvorki tanlandi?"
  ],
  "source_segment_ids": ["seg_0000", "seg_0001"],
  "provider": "openai",
  "model": "gpt-4o-mini",
  "created_at": "2026-09-26T00:25:00.000Z"
}
```

---

## Status Model

| Status Value | Meaning |
|---|---|
| `uploaded` | Audio file has been successfully uploaded and stored on the server. |
| `processing` | Audio is actively undergoing inspection, conversion, normalization, VAD, or chunking. |
| `processed` | Canonical WAV and STT chunks have been created and saved in `processed/`. Ready for STT. |
| `transcribing` | Audio chunks are actively being transcribed by the STT provider. |
| `transcribed` | Transcription complete; global timestamped `transcript.json` is ready. |
| `analyzing` | Transcript is actively undergoing LLM educational understanding analysis. |
| `analyzed` | AI analysis complete; structured `analysis.json` is persisted. |
| `completed` | Full end-to-end lifecycle complete. |
| `failed` | Audio processing, transcription, or AI analysis encountered an error (`error_code` and `error_message` populated). |


