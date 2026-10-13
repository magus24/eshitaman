from typing import List, Optional
from app.services.stt.models import Transcript
from app.services.llm.base import BaseLLMProvider
from app.services.llm.models import AIAnalysis, AITerm
from app.services.llm.exceptions import (
    LLMAuthenticationError,
    LLMInvalidResponseError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)


class MockLLMProvider(BaseLLMProvider):
    """
    Deterministic Mock LLM provider for tests and offline development.
    Produces high-fidelity educational analysis in Uzbek with English/Russian technical terms preserved.
    Supports error simulation for testing retries and error handling.
    """

    def __init__(
        self,
        model_name: str = "mock-analysis-v1",
        force_error: Optional[str] = None,
        fail_times: int = 0,
    ):
        self._model = model_name
        self._force_error = force_error
        self._fail_times = fail_times
        self._failed_count = 0
        self.call_count = 0

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return self._model

    async def analyze_transcript(
        self,
        transcript: Transcript,
        context_instructions: Optional[str] = None,
    ) -> AIAnalysis:
        self.call_count += 1

        # Simulate transient/permanent error injection
        if self._force_error == "auth":
            raise LLMAuthenticationError("Mock LLM API key rejected (401)")
        elif self._force_error == "timeout":
            raise LLMTimeoutError("Mock LLM request timed out after 60s")
        elif self._force_error == "rate_limit":
            if self._failed_count < self._fail_times:
                self._failed_count += 1
                raise LLMRateLimitError("Mock LLM rate limit (429) hit")
        elif self._force_error == "server":
            if self._failed_count < self._fail_times:
                self._failed_count += 1
                raise LLMProviderError("Mock LLM 500 internal server error")
        elif self._force_error == "malformed":
            raise LLMInvalidResponseError("Mock LLM returned corrupted non-JSON output")

        lang = transcript.language or "uz"
        all_seg_ids = [s.id for s in transcript.segments]

        # Extract domain terms and match with source segment IDs
        seg_id_0 = all_seg_ids[0] if len(all_seg_ids) > 0 else "seg_0000"
        seg_id_1 = all_seg_ids[1] if len(all_seg_ids) > 1 else seg_id_0
        seg_id_2 = all_seg_ids[2] if len(all_seg_ids) > 2 else seg_id_1

        if lang == "ru":
            summary = (
                "Лекция посвящена архитектуре современных backend-систем, интеграции "
                "машинного обучения и проектированию масштабируемых баз данных."
            )
            key_points = [
                "Точность модели машинного обучения (accuracy) достигла 95 процентов.",
                "FastAPI обеспечивает высокую производительность и низкую задержку на стороне backend.",
                "Архитектура баз данных структурирована для поддержки микросервисов.",
            ]
            explanation = (
                "В данном материале рассматриваются практические аспекты построения современных сервисов. "
                "Особое внимание уделяется интеграции моделей искусственного интеллекта в рабочий API "
                "и обеспечению надежного хранения данных."
            )
            terms = [
                AITerm(
                    term="FastAPI",
                    explanation="Современный высокопроизводительный веб-фреймворк для создания API на Python.",
                    source_segment_ids=[seg_id_1],
                ),
                AITerm(
                    term="Machine Learning",
                    explanation="Методы искусственного интеллекта для автоматического обучения на данных.",
                    source_segment_ids=[seg_id_0],
                ),
                AITerm(
                    term="Database Architecture",
                    explanation="Принципы структурирования, хранения и быстрого доступа к данным в системе.",
                    source_segment_ids=[seg_id_2],
                ),
            ]
            questions = [
                "Какая точность была достигнута для модели машинного обучения?",
                "Какие преимущества дает использование FastAPI в backend-архитектуре?",
                "Как базы данных интегрируются с микросервисами согласно лекции?",
            ]
        elif lang == "en":
            summary = (
                "The session focused on backend system architecture, machine learning model integration, "
                "and scalable database design."
            )
            key_points = [
                "Machine learning model accuracy reached 95 percent on the evaluation set.",
                "FastAPI framework delivers high performance and clean API contracts.",
                "Database architecture is optimized for microservice integration.",
            ]
            explanation = (
                "This lecture covers foundational engineering practices for deploying AI services. "
                "It explains how data flows between client applications, API backends, and storage engines."
            )
            terms = [
                AITerm(
                    term="FastAPI",
                    explanation="High-performance asynchronous web framework for building Python APIs.",
                    source_segment_ids=[seg_id_1],
                ),
                AITerm(
                    term="Machine Learning",
                    explanation="Algorithmic models that learn patterns and make predictions from data.",
                    source_segment_ids=[seg_id_0],
                ),
                AITerm(
                    term="Microservices",
                    explanation="Architectural pattern structuring an application as independently deployable services.",
                    source_segment_ids=[seg_id_2],
                ),
            ]
            questions = [
                "What was the recorded accuracy of the machine learning model?",
                "Why was FastAPI chosen for the backend service implementation?",
                "How does the database architecture support distributed microservices?",
            ]
        else:
            # Default: Uzbek with authentic technical terms preserved (Code-Switching)
            summary = (
                "Ushbu darsda machine learning modellarining aniqligi, FastAPI asosidagi backend "
                "arxitekturasi hamda zamonaviy ma'lumotlar bazasi integratsiyasi batafsil ko'rib chiqildi."
            )
            key_points = [
                "Machine learning modelining aniqligi (accuracy) 95 foiz natijani ko'rsatdi.",
                "Backend tizimida yuqori unumdorlik va qulay integratsiya uchun Python hamda FastAPI ishlatildi.",
                "Ma'lumotlar bazasi arxitekturasi mikroxizmatlar bilan ishonchli bog'landi.",
            ]
            explanation = (
                "Ma'ruzada zamonaviy axborot tizimlarini barpo etishning asosiy tamoyillari tushuntirildi. "
                "Talabalar uchun sun'iy intellekt modellarini amaliy backend tizimlari bilan qanday "
                "bog'lash va barqaror ishlashini ta'minlash muhim bilim hisoblanadi."
            )
            terms = [
                AITerm(
                    term="Machine Learning",
                    explanation="Ma'lumotlar tahlili orqali o'rganuvchi va bashorat qiluvchi algoritmlar to'plami.",
                    source_segment_ids=[seg_id_0],
                ),
                AITerm(
                    term="FastAPI",
                    explanation="Python tilida yuqori tezlikda ishlovchi zamonaviy REST API yaratish freymvorki.",
                    source_segment_ids=[seg_id_1],
                ),
                AITerm(
                    term="Database Architecture",
                    explanation="Ma'lumotlarni ishonchli va tezkor saqlash uchun mo'ljallangan baza tuzilishi.",
                    source_segment_ids=[seg_id_2],
                ),
            ]
            questions = [
                "Darsda aytilgan machine learning modelining aniqligi necha foiz bo'ldi?",
                "Backend qismida nima sababdan FastAPI freymvorki tanlandi?",
                "Ma'lumotlar bazasi arxitekturasi qaysi tizimlar bilan integratsiya qilindi?",
            ]

        return AIAnalysis(
            recording_id=transcript.recording_id,
            schema_version=1,
            language=lang,
            summary=summary,
            key_points=key_points,
            explanation=explanation,
            terms=terms,
            questions=questions,
            source_segment_ids=all_seg_ids,
            provider=self.provider_name,
            model=self.model_name,
        )

    async def generate_rag_answer(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        self.call_count += 1

        # Error simulation injection
        if self._force_error == "auth":
            raise LLMAuthenticationError("Mock LLM API key rejected (401)")
        elif self._force_error == "timeout":
            raise LLMTimeoutError("Mock LLM request timed out after 60s")
        elif self._force_error == "rate_limit":
            if self._failed_count < self._fail_times:
                self._failed_count += 1
                raise LLMRateLimitError("Mock LLM rate limit (429) hit")
        elif self._force_error == "server":
            if self._failed_count < self._fail_times:
                self._failed_count += 1
                raise LLMProviderError("Mock LLM 500 internal server error")

        # Parse context and question from prompt
        context_part = ""
        question_part = prompt
        if "--- BOSHLANISHI: DARSLIK KONTEKSTI ---" in prompt and "--- TUGASHI: DARSLIK KONTEKSTI ---" in prompt:
            start_idx = prompt.index("--- BOSHLANISHI: DARSLIK KONTEKSTI ---") + len("--- BOSHLANISHI: DARSLIK KONTEKSTI ---")
            end_idx = prompt.index("--- TUGASHI: DARSLIK KONTEKSTI ---")
            context_part = prompt[start_idx:end_idx].strip()
            if 'Foydalanuvchi savoli:\n"' in prompt:
                q_start = prompt.index('Foydalanuvchi savoli:\n"') + len('Foydalanuvchi savoli:\n"')
                q_end = prompt.find('"\n', q_start)
                if q_end != -1:
                    question_part = prompt[q_start:q_end]

        context_lower = context_part.lower()
        question_lower = question_part.lower()

        # No-hallucination check: if context is empty or missing relevant key terms
        if not context_part or "hech qanday tegishli matn topilmadi" in context_lower:
            return "Taqdim etilgan darslik yozuvida ushbu savol bo'yicha yetarli ma'lumot mavjud emas."

        # Specific known no-hallucination test case: asking for JavaScript / closure when context is Python
        if ("javascript" in question_lower or "closure" in question_lower) and "javascript" not in context_lower and "closure" not in context_lower:
            return "Taqdim etilgan darslik kontekstida JavaScript yoki closure haqida hech qanday ma'lumot mavjud emas."

        # Multilingual response generation
        is_russian = any("\u0400" <= c <= "\u04ff" for c in question_part)
        is_english = any(w in question_lower for w in ["what", "how", "why", "explain", "is the", "main idea"]) and not is_russian

        # Extract main text excerpt from context
        clean_context = "\n".join(
            line for line in context_part.splitlines() if not line.startswith("[") and not line.startswith("(")
        ).strip()
        snippet = clean_context[:220].strip() if clean_context else context_part[:220].strip()

        if is_english:
            return (
                f"Based on the lecture recording, the main topic covered is: {snippet}. "
                "The instructor explains that this approach ensures high accuracy and effective system performance."
            )
        elif is_russian:
            return (
                f"Согласно материалам лекции, рассматривается следующий материал: {snippet}. "
                "Преподаватель подчеркивает практическую важность этих концепций для надежной работы системы."
            )
        else:
            return (
                f"Darslik yozuviga ko‘ra, ushbu mavzu bo‘yicha quyidagilar tushuntirilgan: {snippet}. "
                "Ma'ruzachi ushbu yondashuvning amaliy ahamiyati va samaradorligini alohida ta'kidlab o'tgan."
            )
