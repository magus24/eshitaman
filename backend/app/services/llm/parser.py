import json
import re
from typing import Any, Dict
from app.services.llm.exceptions import LLMInvalidResponseError
from app.services.llm.models import AIAnalysis, AITerm


class LLMResponseParser:
    """Safely extracts, validates, and parses structured LLM output into typed AIAnalysis."""

    @staticmethod
    def parse_analysis_json(
        raw_text: str,
        recording_id: str,
        provider_name: str,
        model_name: str,
        language: str = "uz",
    ) -> AIAnalysis:
        if not raw_text or not raw_text.strip():
            raise LLMInvalidResponseError("LLM returned an empty response string.")

        cleaned = raw_text.strip()

        # 1. Strip markdown code blocks if present (```json ... ``` or ``` ...)
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            cleaned = cleaned.strip()

        # 2. Find outermost JSON object if surrounded by preamble/chatter
        if not (cleaned.startswith("{") and cleaned.endswith("}")):
            match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(1)

        # 3. Parse JSON
        try:
            data: Dict[str, Any] = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise LLMInvalidResponseError(
                f"Failed to decode JSON from LLM output: {e}. Raw content: {raw_text[:200]}...",
                cause=e,
            )

        if not isinstance(data, dict):
            raise LLMInvalidResponseError("Parsed LLM output is not a JSON object.")

        # 4. Extract and validate required components with graceful defaults
        summary = str(data.get("summary", "")).strip()
        explanation = str(data.get("explanation", "")).strip()

        raw_points = data.get("key_points", [])
        key_points = []
        if isinstance(raw_points, list):
            for p in raw_points:
                if isinstance(p, str) and p.strip():
                    key_points.append(p.strip())
                elif isinstance(p, dict) and "point" in p:
                    key_points.append(str(p["point"]).strip())

        raw_terms = data.get("terms", [])
        terms = []
        if isinstance(raw_terms, list):
            for t in raw_terms:
                if isinstance(t, dict) and "term" in t:
                    terms.append(
                        AITerm(
                            term=str(t["term"]).strip(),
                            explanation=str(t.get("explanation", "")).strip(),
                            source_segment_ids=list(t.get("source_segment_ids", [])),
                        )
                    )

        raw_questions = data.get("questions", [])
        questions = []
        if isinstance(raw_questions, list):
            for q in raw_questions:
                if isinstance(q, str) and q.strip():
                    questions.append(q.strip())
                elif isinstance(q, dict) and "question" in q:
                    questions.append(str(q["question"]).strip())

        source_segment_ids = list(data.get("source_segment_ids", []))
        if not source_segment_ids:
            # Aggregate segment IDs linked across individual terms
            for t in terms:
                for sid in t.source_segment_ids:
                    if sid not in source_segment_ids:
                        source_segment_ids.append(sid)

        return AIAnalysis(
            recording_id=recording_id,
            schema_version=1,
            language=data.get("language") or language,
            summary=summary,
            key_points=key_points,
            explanation=explanation,
            terms=terms,
            questions=questions,
            source_segment_ids=source_segment_ids,
            provider=provider_name,
            model=model_name,
        )
