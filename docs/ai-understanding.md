# AI Understanding / LLM Integration Architecture

## 1. Overview

The AI Understanding layer of **Eshitaman va Tushuntirib beraman** analyzes timestamped recording transcripts produced by Stage 7 STT to generate a structured, comprehensive educational synthesis.

It produces:
1. **Summary (`summary`)**: A concise, fact-grounded explanation answering *"What was this lecture/meeting mainly about?"*
2. **Key Points (`key_points`)**: 3–7 primary takeaways directly grounded in the transcript.
3. **Detailed Explanation (`explanation`)**: Pedagogically organized, student-friendly explanation clarifying difficult ideas.
4. **Important Terms (`terms`)**: Domain-specific and technical terms with definitions and linked source transcript segments.
5. **Study Questions (`questions`)**: Conceptual check questions formulated strictly from the lecture material.

```text
Timestamped Transcript (transcript/transcript.json)
                         ↓
                TranscriptChunker
   (Direct window or context-window partitioning)
                         ↓
               LLMProviderFactory
   (OpenAILLMProvider / MockLLMProvider)
                         ↓
            Structured LLM Inference
      (JSON Schema enforcement, System Prompt)
                         ↓
               LLMResponseParser
    (Markdown stripping, schema validation)
                         ↓
            AIUnderstandingPipeline
        (Incremental caching & persistence)
                         ↓
        Persisted Analysis (analysis/analysis.json)
```

---

## 2. Provider Abstraction

The architecture is strictly provider-agnostic. No domain logic or API routes are tightly coupled to any single LLM vendor:

```text
                  BaseLLMProvider (ABC)
                    /                 \
                   /                   \
          MockLLMProvider         OpenAILLMProvider
         (CI/Offline/Tests)      (OpenAI / Compatible API)
```

### `BaseLLMProvider` Interface
Every provider implements:
* `provider_name -> str` (e.g., `"mock"`, `"openai"`)
* `model_name -> str` (e.g., `"mock-analysis-v1"`, `"gpt-4o-mini"`)
* `analyze_transcript(transcript, context_instructions=None) -> AIAnalysis`

### Dynamic Factory (`LLMProviderFactory`)
Instantiates the configured provider at runtime via `settings.LLM_PROVIDER`:
* `LLM_PROVIDER=mock`: Deterministic offline provider for CI and testing without API keys.
* `LLM_PROVIDER=openai`: Production provider supporting OpenAI or any OpenAI-compatible API (Groq, DeepSeek, vLLM, Ollama).

---

## 3. Multilingual Strategy & Code-Switching

The target audience speaks Uzbek with natural code-switching into Russian and English, particularly in software engineering, academic, and technical contexts:
* **Transcript-Grounded Language**: The analysis strictly follows the dominant language of the transcript (`transcript.language`). If the speaker delivers the lecture in Uzbek, the output is generated in fluent Uzbek.
* **Terminology Retention**: Technical terms (e.g., *"machine learning"*, *"FastAPI"*, *"database architecture"*, *"microservices"*, *"latency"*, *"accuracy"*) are preserved in their standard industry form without awkward or incorrect translation.
* **No Speculative Invention**: The system strictly enforces factual adherence to the transcript.

---

## 4. Long Transcript Handling Strategy

To prevent context-window overflow on multi-hour recordings:
1. **Window Limit Check**: The `TranscriptChunker` checks whether total transcript characters exceed `LLM_MAX_CHUNK_CHARACTERS` (default `16000` chars, $\approx 4000$ tokens).
2. **Context Window Partitioning**: For recordings exceeding the threshold, segments are partitioned into windowed blocks with an overlap of 1–2 segments to maintain boundary context continuity.
3. **Hierarchical Synthesis**:
   - Each context window generates a partial structured analysis with attributed segment IDs.
   - A synthesis pass unifies the partial analyses into a single, cohesive `AIAnalysis`, deduplicating key points, merging terms, and combining summary points without data loss.

---

## 5. Storage Layout

The analysis is stored in the recording session directory:

```text
backend/storage/recordings/<recording_id>/
├── audio.m4a (or audio.webm, audio.wav)  <-- Original audio
├── metadata.json                         <-- Session status, audio & analysis metadata
├── processed/
│   ├── audio.wav                         <-- Normalized 16kHz WAV
│   └── chunks.json                       <-- Chunks descriptor
├── transcript/
│   ├── chunk_transcripts.json            <-- Cached chunk outputs
│   └── transcript.json                   <-- Timestamped full transcript (Stage 7)
└── analysis/
    └── analysis.json                     <-- Structured AI understanding (Stage 8)
```

---

## 6. Resilience, Retries, and Idempotency

### Transient vs Permanent Errors
* **Permanent (No Retry)**:
  - `401 / 403`: `LLMAuthenticationError` — fails immediately.
  - `400`: `LLMContextLengthError` or client error — fails immediately without infinite retry loops.
  - Malformed transcript or missing recording — fails immediately.
* **Transient (Exponential Backoff up to 3 retries)**:
  - `429`: `LLMRateLimitError`
  - `5xx`: `LLMProviderError`
  - Client / Socket Timeout: `LLMTimeoutError`
  - Network Failure: `LLMNetworkError`

### Idempotency
Repeated `POST /api/v1/recordings/{id}/analyze` calls on an already analyzed recording (`status == "analyzed"`) verify that `analysis/analysis.json` exists and return the cached result immediately with `200 OK`, avoiding duplicate API costs.

---

## 7. Configuration & Environment Variables

Add the following to your backend `.env` file:

```bash
# STT Configuration (Stage 7)
STT_PROVIDER=mock
STT_MODEL=whisper-1

# LLM Configuration (Stage 8)
LLM_PROVIDER=mock                     # 'mock' for local offline dev, 'openai' for production
LLM_API_KEY=your-llm-api-key-here     # Required when LLM_PROVIDER=openai
LLM_API_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini                 # Configurable model name
LLM_TIMEOUT_SECONDS=60.0              # API timeout
LLM_MAX_RETRIES=3                     # Bounded retries on transient errors
LLM_MAX_CHUNK_CHARACTERS=16000        # Context window threshold for long transcripts
```

### Running Locally Without API Keys
Set `LLM_PROVIDER=mock` and `STT_PROVIDER=mock`. All unit tests, API endpoints, and end-to-end flows will execute completely locally with deterministic, high-quality Uzbek multilingual transcripts and structured analyses.
