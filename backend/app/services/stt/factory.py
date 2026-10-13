from typing import Optional
from app.core.config import settings
from app.services.stt.base import BaseSTTProvider
from app.services.stt.exceptions import STTConfigurationError
from app.services.stt.mock_provider import MockSTTProvider
from app.services.stt.whisper_provider import WhisperSTTProvider


class STTProviderFactory:
    """Factory to instantiate STT providers based on application configuration."""

    @staticmethod
    def create_provider(
        provider_type: Optional[str] = None,
        **kwargs,
    ) -> BaseSTTProvider:
        """
        Creates an STT provider instance.
        :param provider_type: 'mock', 'whisper', 'openai', etc. Defaults to settings.STT_PROVIDER.
        :param kwargs: Optional overrides passed to provider constructor.
        """
        ptype = (provider_type or settings.STT_PROVIDER).strip().lower()

        if ptype == "mock":
            return MockSTTProvider(**kwargs)
        elif ptype in ("whisper", "openai"):
            return WhisperSTTProvider(**kwargs)
        else:
            raise STTConfigurationError(
                f"Unsupported STT provider '{ptype}'. Supported providers: 'mock', 'whisper', 'openai'."
            )
