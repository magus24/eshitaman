# "Eshitaman va Tushuntirib beraman" — Arxitektura hujjati

## 1. Umumiy maqsad
Ushbu mahsulot — darslar, ma'ruzalar, uchrashuvlar va uzun audio yozuvlarni qabul qilib, o'zbek tilidagi va aralash nutqni (Uzbek + Russian + English) aniq tushunib, tahlil qilib, foydalanuvchiga sodda va tushunarli qilib beruvchi mobile-first platformadir.

## 2. Tizim qismlari
- **Mobile (`mobile/`)**: Flutter platformasida yaratilgan zamonaviy, minimalist, qulay foydalanuvchi interfeysi (UI/UX).
- **Backend (`backend/`)**: FastAPI asosidagi asinxron API xizmati (REST, WebSockets, fon jarayonlari).
- **AI Pipeline (`ai/`)**: Nutqni matnga aylantirish (STT) va tushuntirish beruvchi generativ til modellari (LLM).
- **Infratuzilma (`infrastructure/`)**: Docker, Redis navbatlari (Celery/ARQ), audio saqlash xizmatlari.

## 3. Loyiha tuzilmasi
```text
eshitaman/
├── mobile/            # Flutter mobile ilovasi
├── backend/           # FastAPI backend xizmati
├── ai/                # AI & STT pipeline modellari
├── infrastructure/    # Docker, Redis, DB konfiguratsiyalari
├── docs/              # Hujjatlar va arxitektura tavsiflari
└── README.md          # Loyiha bosh hujjati
```
