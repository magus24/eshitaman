# Eshitaman va Tushuntirib beraman

**Eshitaman** — mobile-first платформа, которая записывает лекции, занятия, встречи и длинные
аудиозаписи, распознаёт речь (в первую очередь узбекскую с естественным смешением узбекского,
русского и английского) и объясняет содержание простым языком: конспект, ключевые тезисы,
разбор терминов и вопросы для самопроверки.

**Eshitaman** is a mobile-first platform that records lectures, lessons, meetings and long audio,
transcribes speech (Uzbek first, with natural Uzbek + Russian + English code-switching) and explains
it in plain language: summary, key points, terminology breakdown and self-check questions.

> Основной язык документации — русский, ниже идёт англоязычная версия. /
> Documentation is primarily in Russian, the English version follows below.

---

## О проекте (RU)

### Что делает проект

1. **Запись** — мобильное приложение (Flutter) пишет аудио с микрофона, показывает волновую форму и сохраняет сессию локально.
2. **Обработка аудио** — backend приводит звук к каноническому формату (16 кГц, 16-бит PCM, моно WAV), нормализует громкость по пику (без клиппинга), находит речевые участки (energy-based VAD) и режет длинную запись на чанки ~30 с с перекрытием 2 с, привязываясь к паузам.
3. **Speech-to-Text (STT)** — чанки распознаются через Whisper-совместимый провайдер (OpenAI Whisper, Groq Whisper и т. д.) или Mock-провайдер для офлайн-тестов. Локальные таймстемпы пересчитываются в глобальные по всей записи, дубли на стыках чанков убираются.
4. **AI-анализ (LLM)** — по транскрипту генерируется структурированный разбор: `summary`, `key_points`, `explanation`, `terms` (с привязкой к сегментам-источникам) и `questions`. Модель строго опирается на транскрипт и сохраняет техническую терминологию в исходном виде.
5. **RAG-чат** — транскрипт разбивается на смысловые чанки, для них считаются эмбеддинги, они складываются в векторное хранилище (pgvector / in-memory), и пользователь может задавать вопросы по содержанию записи с указанием использованных источников и таймкодов.

### Как это работает (архитектура)

```text
Flutter (mobile-first UI)
        │  multipart upload
        ▼
FastAPI  ── /process ──▶ Audio pipeline (inspect → convert → normalize → VAD → chunk)
        ├── /transcribe ──▶ STT (Whisper / Mock)  → global timestamps → merge
        ├── /analyze ─────▶ LLM (Groq / OpenAI / Mock) → structured analysis
        └── /index,/chat ─▶ RAG (embedding → vector store → grounded answer)
        │
        ▼
Local storage (backend/storage/recordings/<id>/)  +  PostgreSQL/pgvector (RAG)
```

Ключевой принцип — **провайдер-агностичность**: STT, LLM и эмбеддинги спрятаны за
интерфейсами (`BaseSTTProvider`, `BaseLLMProvider`, `BaseEmbeddingProvider`) и
фабриками, поэтому Mock-провайдеры позволяют прогонять весь пайплайн полностью
офлайн, без API-ключей.

### Технологический стек

| Слой | Технологии |
|---|---|
| Mobile | Flutter, Dart, `record`, `path_provider`, `http` |
| Backend | Python, FastAPI, Uvicorn, Pydantic v2, NumPy, httpx |
| STT | Whisper (`whisper-1`, `whisper-large-v3`) через OpenAI-совместимый API |
| LLM | Groq (`llama-3.3-70b-versatile`, `openai/gpt-oss-120b`), OpenAI-совместимые API |
| Embeddings / RAG | `text-embedding-3-small`, pgvector (HNSW) / in-memory store |
| Инфраструктура | Docker Compose, PostgreSQL + pgvector |
| Тесты | pytest (backend), `flutter_test` (mobile) |

### API (backend)

| Метод | Эндпоинт | Назначение |
|---|---|---|
| `GET` | `/health` | Проверка доступности сервиса |
| `POST` | `/api/v1/recordings` | Загрузка аудио + метаданных |
| `GET` | `/api/v1/recordings/{id}` | Статус и метаданные записи |
| `POST` | `/api/v1/recordings/{id}/process` | Подготовка аудио (VAD, нормализация, чанки) |
| `POST` | `/api/v1/recordings/{id}/transcribe` | Мультиязычная транскрипция |
| `GET` | `/api/v1/recordings/{id}/transcript` | Полный транскрипт с таймстемпами |
| `POST` | `/api/v1/recordings/{id}/analyze` | LLM-анализ (конспект, тезисы, термины, вопросы) |
| `GET` | `/api/v1/recordings/{id}/analysis` | Сохранённый структурированный анализ |
| `POST` | `/api/v1/recordings/{id}/index` | Индексация транскрипта для RAG |
| `POST` | `/api/v1/recordings/{id}/chat` | Вопрос-ответ по записи (RAG) |

Подробные контракты — в [`docs/api.md`](docs/api.md).

### Как запустить

**Backend**

```bash
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -r requirements.txt
copy .env.example .env              # при необходимости заполните ключи
pytest -v                           # запуск тестов (без API-ключей, на Mock)
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

**Mobile**

```bash
cd mobile
flutter pub get
flutter run
```

**Инфраструктура (RAG / pgvector)**

```bash
cd infrastructure
docker compose up -d
```

### Переменные окружения и ключи

Шаблон переменных — [`backend/.env.example`](backend/.env.example). Для офлайн-работы
достаточно `STT_PROVIDER=mock`, `LLM_PROVIDER=mock`, `EMBEDDING_PROVIDER=mock`.

**Важно:** файл `.env` и любые API-ключи не попадают в репозиторий (см. `.gitignore`).
В мобильном приложении ключ Groq передаётся во время сборки, а не хранится в коде:

```bash
flutter build apk --release --dart-define=GROQ_API_KEY=gsk_...
```

### Текущий этап развития

Проект находится в стадии **рабочего MVP** — backend-пайплайн и мобильный клиент
функциональны, покрыты тестами и работают end-to-end на Mock-провайдерах, а также на
реальных Whisper/Groq/OpenAI-совместимых API.

Реализованные этапы:

* **Stage 1–4** — архитектура мобильного приложения, запись аудио, UI/UX.
* **Stage 5** — загрузка, хранение и валидация аудио.
* **Stage 6** — подготовка аудио (16 кГц PCM WAV, нормализация, VAD, чанкинг по 30 с).
* **Stage 7** — мультиязычный STT-пайплайн (Whisper + Mock, глобальные таймстемпы, слияние чанков, сохранение code-switching).
* **Stage 8** — AI-понимание и LLM-интеграция (конспект, тезисы, объяснение, термины, вопросы, кеш и идемпотентность).
* **Stage 9** — RAG (семантический чанкинг, эмбеддинги, векторное хранилище, ответы с источниками).
* **Stage 10** — serverless-вариант на Groq (прямые вызовы STT и LLM из приложения).

Что впереди (roadmap):

- [ ] Диаризация спикеров и определение ролей.
- [ ] Аутентификация и мультипользовательский режим.
- [ ] Фоновая обработка через очередь (Celery/ARQ + Redis) и WebSocket-прогресс.
- [ ] Экспорт конспектов (PDF/DOCX) и шаринг.
- [ ] Улучшение узбекского ASR (собственный fine-tune / NeMo).
- [ ] Полноценный CI/CD и деплой.

### Структура проекта

```text
eshitaman/
├── mobile/            # Flutter-приложение (feature-first архитектура)
├── backend/           # FastAPI-сервис + pytest-тесты
├── ai/                # Модули AI/STT (документация пайплайна)
├── infrastructure/    # Docker Compose, миграции PostgreSQL + pgvector
├── docs/              # Техническая документация
└── README.md
```

Дополнительная документация: [архитектура](docs/architecture.md),
[обработка аудио](docs/audio-processing.md), [STT](docs/stt.md),
[AI-понимание](docs/ai-understanding.md), [API](docs/api.md).

---

## About the Project (EN)

### What it does

1. **Record** — a Flutter mobile app captures microphone audio with a live waveform and stores the session locally.
2. **Audio processing** — the backend converts audio to a canonical format (16 kHz, 16-bit PCM, mono WAV), peak-normalizes volume without clipping, runs energy-based VAD, and slices long recordings into ~30 s chunks with 2 s overlap aligned to natural pauses.
3. **Speech-to-Text (STT)** — chunks are transcribed through a Whisper-compatible provider (OpenAI Whisper, Groq Whisper, etc.) or a Mock provider for offline tests. Per-chunk timestamps are translated into global timeline timestamps and overlap duplicates are removed.
4. **AI analysis (LLM)** — a structured breakdown is generated from the transcript: `summary`, `key_points`, `explanation`, `terms` (linked to source segments) and `questions`. The model stays grounded in the transcript and preserves technical terminology verbatim.
5. **RAG chat** — the transcript is split into semantic chunks, embedded, and stored in a vector store (pgvector / in-memory), enabling question answering over the recording with cited sources and timecodes.

### Key strengths

- **Uzbek-first multilingual**: designed for real Uzbek speech with Uzbek + Russian + English code-switching.
- **Provider-agnostic**: STT, LLM and embeddings sit behind interfaces and factories, so Mock providers run the whole pipeline offline with no API keys.
- **Resilient & idempotent**: transient errors retried with exponential backoff, permanent errors fail fast, chunk results cached incrementally to control API cost.
- **Long-recording ready**: silence-aligned chunking and hierarchical synthesis handle 60+ minute lectures.

### Tech stack

| Layer | Technologies |
|---|---|
| Mobile | Flutter, Dart, `record`, `path_provider`, `http` |
| Backend | Python, FastAPI, Uvicorn, Pydantic v2, NumPy, httpx |
| STT | Whisper (`whisper-1`, `whisper-large-v3`) via OpenAI-compatible API |
| LLM | Groq (`llama-3.3-70b-versatile`, `openai/gpt-oss-120b`), OpenAI-compatible APIs |
| Embeddings / RAG | `text-embedding-3-small`, pgvector (HNSW) / in-memory store |
| Infrastructure | Docker Compose, PostgreSQL + pgvector |
| Testing | pytest (backend), `flutter_test` (mobile) |

### Current status

The project is a **working MVP**: the backend pipeline and the mobile client are
functional, covered by tests, and run end-to-end on Mock providers as well as on real
Whisper / Groq / OpenAI-compatible APIs.

Implemented stages: app architecture & recording (1–4), audio upload & validation (5),
audio preparation (6), multilingual STT (7), AI understanding & LLM integration (8),
RAG question answering (9), and a Groq-based serverless flow (10).

### Quick start

```bash
# Backend
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
pytest -v
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

# Mobile
cd mobile
flutter pub get
flutter run

# Infrastructure (RAG / pgvector)
cd infrastructure
docker compose up -d
```

### Environment & secrets

See [`backend/.env.example`](backend/.env.example). Offline mode only requires
`STT_PROVIDER=mock`, `LLM_PROVIDER=mock`, `EMBEDDING_PROVIDER=mock`.

**Note:** `.env` and all API keys are excluded from the repository (see `.gitignore`).
In the mobile app the Groq key is injected at build time, never stored in source:

```bash
flutter build apk --release --dart-define=GROQ_API_KEY=gsk_...
```

---

## License

Учебный / демонстрационный проект для портфолио. /
Educational / portfolio demonstration project.
