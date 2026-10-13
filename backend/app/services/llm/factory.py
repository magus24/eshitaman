from typing import Optional
from app.core.config import settings
from app.services.llm.base import BaseLLMProvider
from app.services.llm.exceptions import LLMConfigurationError
from app.services.llm.groq_provider import GroqLLMProvider
from app.services.llm.mock_provider import MockLLMProvider
from app.services.llm.openai_provider import OpenAILLMProvider


class LLMProviderFactory:
    """Factory to instantiate LLM providers based on application configuration."""

    @staticmethod
    def create_provider(
        provider_type: Optional[str] = None,
        **kwargs,
    ) -> BaseLLMProvider:
        """
        Creates an LLM provider instance.
        :param provider_type: 'mock', 'groq', 'openai'. Defaults to settings.LLM_PROVIDER.
        :param kwargs: Optional overrides passed to provider constructor.
        """
        ptype = (provider_type or settings.LLM_PROVIDER).strip().lower()

        if ptype == "mock":
            return MockLLMProvider(**kwargs)
        elif ptype == "groq":
            return GroqLLMProvider(**kwargs)
        elif ptype in ("openai", "openai-compatible"):
            return OpenAILLMProvider(**kwargs)
        else:
            raise LLMConfigurationError(
                f"Unsupported LLM provider '{ptype}'. Supported providers: 'mock', 'groq', 'openai'."
            )
