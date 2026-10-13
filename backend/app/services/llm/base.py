from abc import ABC, abstractmethod
from typing import Optional
from app.services.stt.models import Transcript
from app.services.llm.models import AIAnalysis


class BaseLLMProvider(ABC):
    """Abstract base class for provider-agnostic Large Language Model inference."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Identifier of the LLM provider (e.g. 'mock', 'openai', 'groq')."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier (e.g. 'mock-analysis-v1', 'gpt-4o-mini')."""
        pass

    @abstractmethod
    async def analyze_transcript(
        self,
        transcript: Transcript,
        context_instructions: Optional[str] = None,
    ) -> AIAnalysis:
        """
        Generates structured educational analysis from a transcript.

        :param transcript: The merged timestamped Transcript object.
        :param context_instructions: Optional extra domain instructions or prompt context.
        :return: AIAnalysis containing summary, key points, explanation, terms, and questions.
        """
        pass

    @abstractmethod
    async def generate_rag_answer(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generates a grounded natural language answer given a RAG prompt containing context and question.

        :param prompt: Formatted user prompt with context and user question.
        :param system_prompt: Optional system prompt enforcing grounding.
        :return: Grounded natural language answer string.
        """
        pass
