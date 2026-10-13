import os
from pathlib import Path
from typing import Set, Optional
from dotenv import load_dotenv

# Load .env file from backend root if it exists
backend_root = Path(__file__).resolve().parent.parent.parent
env_path = backend_root / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv(backend_root / ".env.example")


class Settings:
    APP_NAME: str = os.getenv("APP_NAME", "Eshitaman API")
    APP_VERSION: str = os.getenv("APP_VERSION", "0.1.0")
    DEBUG: bool = os.getenv("DEBUG", "true").lower() in ("true", "1", "yes")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # Audio upload & storage configuration
    STORAGE_PATH: Path = Path(os.getenv("STORAGE_PATH", str(backend_root / "storage"))).resolve()
    MAX_AUDIO_SIZE_MB: int = int(os.getenv("MAX_AUDIO_SIZE_MB", "100"))

    # FFmpeg paths (defaults to system PATH lookup if None)
    FFMPEG_PATH: Optional[str] = os.getenv("FFMPEG_PATH")
    FFPROBE_PATH: Optional[str] = os.getenv("FFPROBE_PATH")

    # STT Canonical Audio Format Configuration
    AUDIO_SAMPLE_RATE: int = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
    AUDIO_CHANNELS: int = int(os.getenv("AUDIO_CHANNELS", "1"))
    AUDIO_SAMPLE_WIDTH: int = int(os.getenv("AUDIO_SAMPLE_WIDTH", "2"))  # 16-bit = 2 bytes

    # Audio Chunking Configuration
    AUDIO_CHUNK_SECONDS: float = float(os.getenv("AUDIO_CHUNK_SECONDS", "30.0"))
    AUDIO_CHUNK_OVERLAP_SECONDS: float = float(os.getenv("AUDIO_CHUNK_OVERLAP_SECONDS", "2.0"))

    # VAD Configuration
    AUDIO_VAD_ENERGY_THRESHOLD: float = float(os.getenv("AUDIO_VAD_ENERGY_THRESHOLD", "0.015"))
    AUDIO_VAD_FRAME_MS: int = int(os.getenv("AUDIO_VAD_FRAME_MS", "30"))

    # STT Provider Configuration
    STT_PROVIDER: str = os.getenv("STT_PROVIDER", "mock")
    STT_API_KEY: Optional[str] = os.getenv("STT_API_KEY") or os.getenv("OPENAI_API_KEY")
    STT_API_BASE_URL: str = os.getenv("STT_API_BASE_URL", "https://api.openai.com/v1")
    STT_MODEL: str = os.getenv("STT_MODEL", "whisper-1")
    STT_TIMEOUT_SECONDS: float = float(os.getenv("STT_TIMEOUT_SECONDS", "60.0"))
    STT_MAX_RETRIES: int = int(os.getenv("STT_MAX_RETRIES", "3"))
    STT_LANGUAGE_HINT: Optional[str] = os.getenv("STT_LANGUAGE_HINT")

    # LLM / AI Understanding Configuration
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "mock")
    LLM_API_KEY: Optional[str] = os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    LLM_API_BASE_URL: str = os.getenv("LLM_API_BASE_URL", "https://api.openai.com/v1")
    LLM_MODEL: str = os.getenv("LLM_MODEL") or os.getenv("GROQ_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
    LLM_TIMEOUT_SECONDS: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "60.0"))
    LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "3"))
    LLM_MAX_CHUNK_CHARACTERS: int = int(os.getenv("LLM_MAX_CHUNK_CHARACTERS", "16000"))

    # Groq Configuration (Stage 10 Competition MVP)
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY") or os.getenv("LLM_API_KEY")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL") or os.getenv("LLM_MODEL") or "llama-3.3-70b-versatile"
    GROQ_BASE_URL: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    GROQ_TIMEOUT_SECONDS: float = float(os.getenv("GROQ_TIMEOUT_SECONDS", "60.0"))
    GROQ_MAX_RETRIES: int = int(os.getenv("GROQ_MAX_RETRIES", "3"))

    # RAG & Embedding Configuration
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "mock")
    EMBEDDING_API_KEY: Optional[str] = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
    EMBEDDING_API_BASE_URL: str = os.getenv("EMBEDDING_API_BASE_URL", "https://api.openai.com/v1")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
    EMBEDDING_TIMEOUT_SECONDS: float = float(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "30.0"))
    EMBEDDING_MAX_RETRIES: int = int(os.getenv("EMBEDDING_MAX_RETRIES", "3"))

    # Vector Store & Database Configuration
    VECTOR_STORE_TYPE: str = os.getenv("VECTOR_STORE_TYPE", "memory")
    DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL")

    # RAG Retrieval & Chunking Configuration
    RAG_CHUNK_MAX_TOKENS: int = int(os.getenv("RAG_CHUNK_MAX_TOKENS", "250"))
    RAG_CHUNK_OVERLAP_TOKENS: int = int(os.getenv("RAG_CHUNK_OVERLAP_TOKENS", "40"))
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "5"))
    RAG_MIN_SIMILARITY: float = float(os.getenv("RAG_MIN_SIMILARITY", "0.20"))

    ALLOWED_EXTENSIONS: Set[str] = {
        ".m4a",
        ".webm",
        ".aac",
        ".mp3",
        ".wav",
        ".ogg",
    }

    ALLOWED_MIME_TYPES: Set[str] = {
        "audio/m4a",
        "audio/mp4",
        "audio/x-m4a",
        "audio/aac",
        "audio/webm",
        "audio/webm;codecs=opus",
        "audio/wav",
        "audio/x-wav",
        "audio/wave",
        "audio/mpeg",
        "audio/mp3",
        "audio/ogg",
        "application/octet-stream",  # Sometimes reported by browsers/clients for raw audio
    }

    @property
    def max_audio_size_bytes(self) -> int:
        return self.MAX_AUDIO_SIZE_MB * 1024 * 1024


settings = Settings()
