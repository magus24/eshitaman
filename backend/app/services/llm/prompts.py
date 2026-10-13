import json
from typing import List
from app.services.stt.models import Transcript, TranscriptSegment

SYSTEM_PROMPT = """Siz "Eshitaman va Tushuntirib beraman" platformasining ekspert ta'limiy yordamchisisiz.
Sizning vazifangiz — taqdim etilgan audio transkripti asosida talabalar va mutaxassislar uchun yuqori sifatli, tushunarli va tuzilmaviy tahlil yaratish.

MUHIM QOIDALAR:
1. HAQIQAT MANBAI (GROUND TRUTH): Barcha xulosa, asosiy fikrlar, tushuntirish va savollar FAQAT taqdim etilgan transkript matniga asoslanishi shart. Transkriptda aytilmagan ma'lumotlarni to'qimang yoki xayoliy faktlar qo'shmang.
2. TIL STRATEGIYASI:
   - Transkriptning asosiy tiliga qat'iy rioya qiling. Agar transkript o'zbek tilida bo'lsa (yoki o'zbekcha dominant bo'lsa), tahlilni ravon o'zbek tilida yozing.
   - Agar transkript rus tilida bo'lsa — rus tilida, ingliz tilida bo'lsa — ingliz tilida yozing.
   - Agar nutq aralash bo'lsa (Uzbek + Russian + English, code-switching), texnik va dasturlash atamalarini, qisqartmalarni o'z holicha saqlang.
3. TEXNIK ATAMALARNI SAQLASH:
   - Dasturlash, arxitektura va texnik terminlarni (masalan: "machine learning", "FastAPI", "API backend", "database architecture", "microservices", "latency", "accuracy") tarjima qilib buzib yubormang, ularni professional darajada saqlang.
4. MANBA SEGMENTLARINI BOG'LASH:
   - Transkript har bir satrida [seg_XXXX] identifikatori mavjud. Har bir tushuncha (term) yoki fikr qaysi segmentlardan olingan bo'lsa, ularning segment ID larini source_segment_ids ro'yxatiga kiriting.
5. QAT'IY JSON FORMATI:
   - Javobingiz faqat va faqat belgilangan JSON formatida bo'lishi shart. Hech qanday ortiqcha matn, markdown kod bloki yoki kirish so'zlari qo'shmang."""

ANALYSIS_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "Leksiya yoki uchrashuvning qisqacha va lo'nda mazmuni (faqat transkript asosida)."
        },
        "key_points": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Transkriptda ko'tarilgan eng muhim 3-7 ta asosiy fikr yoki xulosalar."
        },
        "explanation": {
            "type": "string",
            "description": "Talabaga mavzuni tushunishni osonlashtiruvchi mantiqiy va batafsil tushuntirish."
        },
        "terms": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "term": {"type": "string"},
                    "explanation": {"type": "string"},
                    "source_segment_ids": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["term", "explanation"]
            },
            "description": "Transkriptda eslatib o'tilgan asosiy texnik va sohaviy atamalar hamda ularning izohlari."
        },
        "questions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Faqatgina ushbu dars materiallari bo'yicha bilimlarni tekshiruvchi 3-5 ta savol."
        }
    },
    "required": ["summary", "key_points", "explanation", "terms", "questions"]
}


def build_user_prompt(transcript: Transcript, segments_slice: List[TranscriptSegment] = None) -> str:
    """Formats transcript segments with their IDs into a prompt payload."""
    segs = segments_slice if segments_slice is not None else transcript.segments
    dominant_lang = transcript.language or "uz"

    lines = []
    for s in segs:
        text = s.text.strip()
        if text:
            lines.append(f"[{s.id}] {text}")

    formatted_transcript = "\n".join(lines)
    if not formatted_transcript.strip():
        formatted_transcript = transcript.text or "(Bo'sh transkript)"

    prompt = (
        f"Dominant til: {dominant_lang}\n"
        f"Transkript segmentlari soni: {len(segs)}\n"
        f"Quyidagi audio transkripti matnini chuqur tahlil qiling va qat'iy JSON formatida qaytaring:\n\n"
        f"--- TRANSKRIPT BOSHLANISHI ---\n"
        f"{formatted_transcript}\n"
        f"--- TRANSKRIPT TUGASHI ---\n\n"
        f"Eslatma: Faqat yuqoridagi transkriptga asoslaning. Qat'iy JSON ob'ektini qaytaring."
    )
    return prompt


def build_hierarchical_synthesis_prompt(partial_summaries: List[str], dominant_lang: str = "uz") -> str:
    """Prompt to synthesize multiple context-window chunks into a single coherent AIAnalysis."""
    joined_summaries = "\n\n".join(
        f"Qism {idx + 1}:\n{summary}" for idx, summary in enumerate(partial_summaries)
    )
    return (
        f"Dominant til: {dominant_lang}\n"
        f"Quyida uzoq davom etgan ma'ruzaning turli qismlaridan olingan oraliq tahlillar keltirilgan.\n"
        f"Ularni yagona, to'liq, takrorlanishlarsiz yaxlit tahlil (yagona summary, key_points, explanation, terms, questions) holiga keltiring.\n\n"
        f"{joined_summaries}\n\n"
        f"Qat'iy JSON formatida qaytaring."
    )
