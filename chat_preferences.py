"""
chat_preferences.py: the "Describe it in a chat" option on the preferences page.

The learner describes the plan in plain words, in English, Spanish or Macedonian.
For each message:
  1. Code reads the values it can find (dates, lengths, weekdays, minutes, topics,
     countries) with word rules for all three languages.
  2. The local model also returns the values it finds (EXTRACT_SCHEMA). The small
     model invents values, so a model value is used only when code found nothing AND
     the message has words about that field, or answers that field's question.
  3. Code asks for ONE missing necessary item, and never asks the same item twice:
     when an answer does not give it, a suggested value fills it (marked "suggested")
     and the learner can change it later. So the learner never gets stuck.
  4. The tutor replies in the language of the learner's message, and every reply can
     also be shown in English.

"Restart" / "start over" / "empezar de nuevo" / "почни одново" clears the preferences.

Necessary items: language, study window (first and last day), practice weekdays,
minutes per session, and at least one topic. Optional: the Spanish country
(default Spain). The model never writes the tutor's words and never decides when
the plan is ready.
"""

from __future__ import annotations

import json
import re
from datetime import date, timedelta

import llm
from llm import LLMError
from planner import GOAL_HINTS, LANGUAGE_HINTS, WEEKDAYS, parse_timeline
from prompts import LANGUAGES
from retriever import DEFAULT_VARIETY, VARIETIES

REQUIRED = ("language", "window", "weekdays", "minutes", "topics")
MAX_MINUTES = 240
TOPICS = ("restaurant", "directions", "shopping", "hotel", "transport", "weather",
          "sightseeing", "emergency")

# Suggested values: the same defaults as the "Choose with buttons" screen.
SUGGEST_DAYS = 14
SUGGESTED = {"language": "spanish", "weekdays": ["Monday", "Wednesday", "Friday"],
             "minutes": 15, "topics": ["restaurant", "directions"]}

# --- Words the tutor says, in the learner's language ---------------------------------
# Macedonian text: please have a native speaker check it.

TEXT = {
    "en": {
        "opening": "Tell me about your trip and how you want to study: the language, your study "
                   "dates, the days and minutes you can practice, and the topics you need. For "
                   "example: I'm going to Mexico City in December and can practice 20 minutes "
                   "on weekends, mostly restaurants and directions.",
        "have": "Here is what I have:", "missing": "Missing", "suggested": "suggested",
        "ready": "That is everything I need. Check the details, then build your plan. "
                 "You can change anything: just tell me.",
        "ready_suggested": "I filled in the rest with a suggested plan (marked “suggested”). "
                           "Build it now, or tell me what to change.",
        "model_down": "The local model did not answer, so I read your message with simple "
                      "word rules. Check the details.",
        "restarted": "Done. I cleared your preferences. Tell me about your trip again.",
        "labels": {"language": "Language", "window": "Study window", "weekdays": "Practice days",
                   "minutes": "Minutes per session", "topics": "Topics (by priority)",
                   "stages": "Lesson types"},
        "stage_names": ["Vocabulary", "Sentences", "Role play"], "all_stages": "All three",
        "questions": {
            "language": "Which language do you want to learn: Spanish or Macedonian?",
            "window": "What are the first and last days of your study window? "
                      "For example: Dec 1 to Jan 15, or three weeks from today.",
            "window_end": "What is the last day of your study window?",
            "weekdays": "Which days of the week can you practice? "
                        "For example: Monday, Wednesday and Friday, or weekends.",
            "minutes": "How many minutes can you practice in one session?",
            "topics": "Which topics do you want to learn? Choose from: {topics}. "
                      "Name the most important one first."},
        "problems": {"past": "The first day can't be in the past.",
                     "order": "The last day must be on or after the first day.",
                     "max_minutes": f"A session can be at most {MAX_MINUTES} minutes."},
        "languages": {"spanish": "Spanish", "macedonian": "Macedonian"},
        "default": "default",
        "window": "{start} to {end} ({days} days)",
        "months": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        "date": "{month} {day:02d}, {year}",
        "weekdays": list(WEEKDAYS),
        "topics": ["Restaurants", "Directions", "Shopping", "Hotels", "Transportation", "Weather",
                   "Sightseeing", "Getting help"],
        "countries": dict(VARIETIES),
    },
    "es": {
        "have": "Esto es lo que tengo:", "missing": "Falta", "suggested": "sugerido",
        "ready": "Ya tengo todo lo que necesito. Revisa los datos y crea tu plan. "
                 "Puedes cambiar cualquier cosa: solo dímelo.",
        "ready_suggested": "Completé el resto con un plan sugerido (marcado “sugerido”). "
                           "Créalo ahora o dime qué quieres cambiar.",
        "model_down": "El modelo local no respondió, así que leí tu mensaje con reglas "
                      "simples. Revisa los datos.",
        "restarted": "Listo. Borré tus preferencias. Cuéntame otra vez sobre tu viaje.",
        "labels": {"language": "Idioma", "window": "Período de estudio", "weekdays": "Días de práctica",
                   "minutes": "Minutos por sesión", "topics": "Temas (por prioridad)",
                   "stages": "Tipos de lección"},
        "stage_names": ["Vocabulario", "Oraciones", "Juego de roles"], "all_stages": "Los tres",
        "questions": {
            "language": "¿Qué idioma quieres aprender: español o macedonio?",
            "window": "¿Cuál es el primer y el último día de tu período de estudio? "
                      "Por ejemplo: del 1 de diciembre al 15 de enero, o tres semanas desde hoy.",
            "window_end": "¿Cuál es el último día de tu período de estudio?",
            "weekdays": "¿Qué días de la semana puedes practicar? "
                        "Por ejemplo: lunes, miércoles y viernes, o los fines de semana.",
            "minutes": "¿Cuántos minutos puedes practicar en cada sesión?",
            "topics": "¿Qué temas quieres aprender? Elige entre: {topics}. "
                      "Nombra primero el más importante."},
        "problems": {"past": "El primer día no puede estar en el pasado.",
                     "order": "El último día debe ser igual o posterior al primero.",
                     "max_minutes": f"Una sesión puede durar como máximo {MAX_MINUTES} minutos."},
        "languages": {"spanish": "Español", "macedonian": "Macedonio"},
        "default": "predeterminado",
        "window": "del {start} al {end} ({days} días)",
        "months": ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"],
        "date": "{day} {month} {year}",
        "weekdays": ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"],
        "topics": ["Restaurantes", "Direcciones", "Compras", "Hoteles", "Transporte", "Clima",
                   "Turismo", "Pedir ayuda"],
        "countries": {"ES": "España", "419": "Latinoamérica (sin país específico)",
                      "AR": "Argentina", "BO": "Bolivia", "CL": "Chile", "CO": "Colombia",
                      "CR": "Costa Rica", "CU": "Cuba", "DO": "República Dominicana",
                      "EC": "Ecuador", "SV": "El Salvador", "GT": "Guatemala", "HN": "Honduras",
                      "MX": "México", "NI": "Nicaragua", "PA": "Panamá", "PY": "Paraguay",
                      "PE": "Perú", "PR": "Puerto Rico", "UY": "Uruguay", "VE": "Venezuela"},
    },
    "mk": {
        "have": "Еве што имам:", "missing": "Недостасува", "suggested": "предлог",
        "ready": "Го имам сè што ми треба. Провери ги деталите, па направи го својот план. "
                 "Можеш да смениш сè: само кажи ми.",
        "ready_suggested": "Остатокот го пополнив со предложен план (означено „предлог“). "
                           "Направи го сега или кажи ми што да сменам.",
        "model_down": "Локалниот модел не одговори, па пораката ја прочитав со едноставни "
                      "правила. Провери ги деталите.",
        "restarted": "Готово. Ги избришав твоите поставки. Кажи ми повторно за твоето патување.",
        "labels": {"language": "Јазик", "window": "Период на учење", "weekdays": "Денови за вежбање",
                   "minutes": "Минути по вежба", "topics": "Теми (по важност)",
                   "stages": "Видови лекции"},
        "stage_names": ["Вокабулар", "Реченици", "Игра на улоги"], "all_stages": "Сите три",
        "questions": {
            "language": "Кој јазик сакаш да го учиш: шпански или македонски?",
            "window": "Кој е првиот и последниот ден од твојот период на учење? "
                      "На пример: од 1 декември до 15 јануари, или три недели од денес.",
            "window_end": "Кој е последниот ден од твојот период на учење?",
            "weekdays": "Кои денови во неделата можеш да вежбаш? "
                        "На пример: понеделник, среда и петок, или за викенд.",
            "minutes": "Колку минути можеш да вежбаш секој пат?",
            "topics": "Кои теми сакаш да ги учиш? Избери од: {topics}. "
                      "Прво наведи ја најважната."},
        "problems": {"past": "Првиот ден не може да биде во минатото.",
                     "order": "Последниот ден мора да биде ист или по првиот.",
                     "max_minutes": f"Една вежба може да трае најмногу {MAX_MINUTES} минути."},
        "languages": {"spanish": "Шпански", "macedonian": "Македонски"},
        "default": "основно",
        "window": "од {start} до {end} ({days} дена)",
        "months": ["јан", "фев", "мар", "апр", "мај", "јун", "јул", "авг", "сеп", "окт", "ное", "дек"],
        "date": "{day} {month} {year}",
        "weekdays": ["понеделник", "вторник", "среда", "четврток", "петок", "сабота", "недела"],
        "topics": ["Ресторани", "Насоки", "Купување", "Хотели", "Превоз", "Време",
                   "Знаменитости", "Барање помош"],
        "countries": {"ES": "Шпанија", "419": "Латинска Америка (без одредена земја)",
                      "AR": "Аргентина", "BO": "Боливија", "CL": "Чиле", "CO": "Колумбија",
                      "CR": "Костарика", "CU": "Куба", "DO": "Доминиканска Република",
                      "EC": "Еквадор", "SV": "Ел Салвадор", "GT": "Гватемала", "HN": "Хондурас",
                      "MX": "Мексико", "NI": "Никарагва", "PA": "Панама", "PY": "Парагвај",
                      "PE": "Перу", "PR": "Порторико", "UY": "Уругвај", "VE": "Венецуела"},
    },
}
QUESTIONS = TEXT["en"]["questions"]
OPENING = TEXT["en"]["opening"]
TOPIC_NAMES = dict(zip(TOPICS, TEXT["en"]["topics"]))

# What the model is told about the tutor's last question (not the example text, which
# the model copied as dates in a live check).
FIELD_HINT = {"language": "which language to learn", "window": "the first and last study day",
              "window_end": "the last study day", "weekdays": "the practice weekdays",
              "minutes": "the minutes per session", "topics": "the topics to learn"}

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "language": {"type": "string", "enum": ["spanish", "macedonian", "unknown"]},
        "country": {"type": "string"},
        "start_date": {"type": "string"},
        "end_date": {"type": "string"},
        "duration": {"type": "string"},
        "weekdays": {"type": "array", "items": {"type": "string", "enum": list(WEEKDAYS)}},
        "minutes": {"type": "integer"},
        "topics": {"type": "array", "items": {"type": "string", "enum": list(TOPICS)}},
    },
    "required": ["language", "country", "start_date", "end_date", "duration",
                 "weekdays", "minutes", "topics"],
    "additionalProperties": False,
}


def _empty_found() -> dict:
    return {"language": None, "variety": None, "start": None, "end": None,
            "weekdays": [], "minutes": None, "topics": [], "stages": None}


def new_chat() -> dict:
    """Empty chat state, kept in st.session_state["pref_chat"]."""
    return {"found": _empty_found(), "asked": [], "suggested": [], "lang": "en",
            "messages": [{"role": "tutor", "text": OPENING}], "asking": None}


# --- 1. Model extraction ---------------------------------------------------------

def _prompt(text: str, today: date, asking: str | None) -> list[dict]:
    asked = f" The tutor last asked for {FIELD_HINT[asking]}." if asking else ""
    system = (
        "You read a traveler's message about a language study plan and return the values "
        "it states, as JSON. The message may be in English, Spanish or Macedonian. Do not "
        "guess. Use \"unknown\" or \"\" or 0 or [] for anything the message does not state.\n"
        f"Today is {today:%A, %Y-%m-%d}.{asked}\n"
        "- language: the language they want to LEARN: spanish or macedonian. A trip to a "
        "Spanish-speaking country means spanish; a trip to North Macedonia means macedonian. "
        "The language the message is written in does not count.\n"
        "- country: the Spanish-speaking country they travel to, in English, or \"\". A city "
        "counts: Buenos Aires = Argentina, Lima = Peru, Bogotá = Colombia.\n"
        "- start_date / end_date: the first and last study day as YYYY-MM-DD. If no year is "
        "given, use the next such date on or after today. A trip date is the end of study.\n"
        "- duration: a study length given instead of dates, as written (e.g. \"three weeks\").\n"
        "- weekdays: practice days. Weekends = Saturday, Sunday. Weekdays = Monday to Friday. "
        "Every day = all seven.\n"
        "- minutes: minutes per practice session (an hour = 60).\n"
        "- topics: what they want to learn, most important first. restaurant = food and "
        "drinks; directions = finding the way; transport = trains, buses, taxis; hotel = "
        "where they stay; weather = small talk and chatting with people, family or friends; "
        "emergency = getting help.")
    return [{"role": "system", "content": system}, {"role": "user", "content": text}]


def extract(text: str, today: date, asking: str | None = None) -> dict:
    """Ask the model for the values in one message. Raises LLMError when it cannot answer."""
    messages = _prompt(text, today, asking)
    for _ in range(2):
        reply = llm.generate_response(messages, {"temperature": 0, "format": EXTRACT_SCHEMA,
                                                 "num_predict": 300})
        try:
            data = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", reply.strip()))
            if isinstance(data, dict):
                return data
        except (ValueError, TypeError):
            pass
        messages = messages + [{"role": "user", "content": "Return only JSON matching the schema."}]
    raise LLMError("The model did not return usable JSON.")


# --- 2. Word rules (English, Spanish, Macedonian) ----------------------------------

def _has(low: str, word: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(word)}(?:s|es)?(?!\w)", low) is not None


def _alt(words) -> str:
    return "|".join(sorted(map(re.escape, words), key=len, reverse=True))


# The language a message is written in (for the tutor's reply).
_ES_WORDS = {"quiero", "voy", "ir", "de", "el", "la", "los", "las", "y", "para", "hasta",
             "con", "mis", "mi", "que", "en", "un", "una", "practicar", "hablar", "días", "dias",
             "semana", "sí", "hoy", "cada", "minutos", "hora", "tengo", "puedo", "quisiera",
             "gustaría", "viaje", "fin", "año", "desde", "todos", "por", "favor", "sobre",
             "todo", "aprender", "español", "gracias", "sé", "también", "más", "cuando",
             "dónde", "como", "cómo", "estudiar", "empezar", "lunes", "martes", "jueves",
             "viernes", "sábado", "domingo", "pero", "solo", "del", "al", "nada", "yo"}
_EN_WORDS = {"i", "i'm", "im", "the", "to", "and", "want", "for", "my", "on", "in", "of",
             "can", "would", "like", "go", "going", "practice", "learn", "days", "week",
             "minutes", "hour", "from", "until", "with", "please", "also", "about", "trip",
             "study", "start", "end", "yes", "every", "weekends", "it", "is", "do", "how",
             "what", "which", "but", "only", "just", "be", "will", "at", "after", "before"}


def detect_language(text: str, previous: str = "en") -> str:
    """'en', 'es' or 'mk' for the tutor's reply. Unclear messages ("30") keep the last one."""
    if re.search(r"[Ѐ-ӿ]", text):
        return "mk"
    words = re.findall(r"[a-záéíóúñü']+", text.lower())
    es = sum(w in _ES_WORDS for w in words) + len(re.findall(r"[ñáéíóú¿¡]", text.lower()))
    en = sum(w in _EN_WORDS for w in words)
    return "es" if es > en else "en" if en > es else previous


_COUNTRY_NAMES = {}
for _lang in ("en", "es", "mk"):
    for _code, _name in TEXT[_lang]["countries"].items():
        if _code != "419":
            _COUNTRY_NAMES[_name.lower()] = _code
_COUNTRY_NAMES.update({"mexico": "MX", "peru": "PE", "panama": "PA", "spain": "ES"})


def _country(name: str) -> str | None:
    low = (name or "").lower().strip()
    if not low:
        return None
    if "latin america" in low or "latinoam" in low:
        return "419"
    return _COUNTRY_NAMES.get(low)


def _country_in_text(low: str) -> str | None:
    if "latin america" in low or "latinoamérica" in low or "latinoamerica" in low:
        return "419"
    found = [(m.start(), code) for name, code in _COUNTRY_NAMES.items()
             for m in re.finditer(rf"(?<!\w){re.escape(name)}(?!\w)", low)]
    return min(found)[1] if found else None


_LANGUAGE_WORDS = {
    "spanish": list(LANGUAGE_HINTS["spanish"]) + ["español", "espanol", "castellano", "шпански"],
    "macedonian": list(LANGUAGE_HINTS["macedonian"]) + [
        "macedonio", "македонски", "македонија", "скопје", "охрид", "битола", "струга"],
}


def _language_in_text(low: str) -> str | None:
    scores = {lang: sum(_has(low, h) for h in hints) for lang, hints in _LANGUAGE_WORDS.items()}
    if _country_in_text(low) not in (None, "ES"):
        scores["spanish"] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] and list(scores.values()).count(scores[best]) == 1 else None


_DAY_WORDS = {
    "Monday": ["monday", "mon", "lunes", "понеделник"],
    "Tuesday": ["tuesday", "tue", "tues", "martes", "вторник"],
    "Wednesday": ["wednesday", "wed", "miércoles", "miercoles", "среда"],
    "Thursday": ["thursday", "thu", "thur", "thurs", "jueves", "четврток"],
    "Friday": ["friday", "fri", "viernes", "петок"],
    "Saturday": ["saturday", "sat", "sábado", "sabado", "сабота"],
    "Sunday": ["sunday", "sun", "domingo"],   # "недела" is also "week": handled below
}


def _weekdays_in_text(low: str) -> list[str]:
    if re.search(r"\b(every ?day|daily|each day|all week|todos los días|todos los dias|"
                 r"cada día|cada dia|diario|diariamente|секој ден|секојдневно)\b", low):
        return list(WEEKDAYS)
    days = {d for d, words in _DAY_WORDS.items() if any(_has(low, w) for w in words)}
    if re.search(r"(?<!\w)недела(?!\w)", low) and not re.search(r"\d\s*недел|недели", low):
        days.add("Sunday")
    if re.search(r"\b(weekends?|fines? de semana|викенд\w*)\b", low):
        days |= {"Saturday", "Sunday"}
    if re.search(r"\b(weekdays|entre semana|días de semana|dias de semana|días laborables|"
                 r"работни денови|работен ден)\b", low):
        days |= set(WEEKDAYS[:5])
    return [d for d in WEEKDAYS if d in days]


def _minutes_in_text(low: str, asking: str | None) -> int | None:
    if m := re.search(r"(\d+)\s*(?:-\s*)?(?:min|mins|minute|minutes|minutos|мин|минути|минута)(?!\w)", low):
        return int(m.group(1))
    if re.search(r"\b(half an hour|media hora|половина час|пола час)\b", low):
        return 30
    if re.search(r"\b(quarter of an hour|quarter hour|cuarto de hora)\b", low):
        return 15
    if m := re.search(r"(\d+(?:[.,]\d+)?)\s*(?:hour|hours|hr|hrs|hora|horas|час|часа|часови)(?!\w)", low):
        return round(float(m.group(1).replace(",", ".")) * 60)
    if re.search(r"\b(an hour|one hour|una hora|еден час)\b", low):
        return 60
    if asking == "minutes" and (m := re.fullmatch(r"\s*(\d+)\s*", low)):
        return int(m.group(1))
    return None


_TOPIC_EXTRA = {
    "restaurant": ["restaurante", "comida", "comer", "cenar", "almorzar", "menú", "mesero",
                   "camarero", "bebida", "ресторан", "храна", "јадење", "кафе", "мени", "келнер"],
    "directions": ["direcciones", "cómo llegar", "perdido", "perdida", "calle", "orientarme",
                   "насоки", "правец", "улица", "изгубен"],
    "shopping": ["compras", "comprar", "tienda", "mercado", "precio", "dinero", "farmacia",
                 "купување", "купам", "продавница", "пазар", "цена", "пари"],
    "hotel": ["habitación", "alojamiento", "hostal", "reserva", "хотел", "соба", "сместување"],
    "transport": ["transporte", "tren", "autobús", "autobus", "billete", "boleto", "estación",
                  "превоз", "воз", "автобус", "такси", "билет", "станица"],
    "weather": ["clima", "charla", "charlar", "conversar", "conversación", "hablar con",
                "familia", "abuela", "abuelo", "tíos", "tios", "tías", "tias", "primos",
                "amigos", "family", "grandmother", "grandma", "grandfather", "relatives",
                "aunt", "uncle", "cousin", "friends", "small talk",
                "разговор", "муабет", "семејство", "баба", "дедо", "роднини", "пријатели"],
    "sightseeing": ["turismo", "museo", "monumento", "iglesia", "playa", "знаменитости",
                    "музеј", "црква", "езеро"],
    "emergency": ["emergencia", "médico", "medico", "policía", "ayuda", "enfermo",
                  "итно", "лекар", "болница", "полиција", "помош", "болен"],
}


def _topics_in_text(low: str) -> list[str]:
    hits = []
    for topic in TOPICS:
        words = ([TOPIC_NAMES[topic].lower(), topic] + GOAL_HINTS.get(topic, [])
                 + _TOPIC_EXTRA[topic] + [t.lower() for t in (TEXT["es"]["topics"][TOPICS.index(topic)],
                                                              TEXT["mk"]["topics"][TOPICS.index(topic)])])
        places = [m.start() for w in words
                  for m in re.finditer(rf"(?<!\w){re.escape(w)}(?:s|es)?(?!\w)", low)]
        if places:
            hits.append((min(places), topic))
    return [t for _, t in sorted(hits)]


_MONTH_NUMBERS = {}
for _i, _names in enumerate([
        ["january", "jan", "enero", "ene", "јануари"], ["february", "feb", "febrero", "февруари"],
        ["march", "mar", "marzo", "март"], ["april", "apr", "abril", "abr", "април"],
        ["may", "mayo", "мај"], ["june", "jun", "junio", "јуни"], ["july", "jul", "julio", "јули"],
        ["august", "aug", "agosto", "ago", "август"],
        ["september", "sept", "sep", "septiembre", "setiembre", "септември"],
        ["october", "oct", "octubre", "октомври"], ["november", "nov", "noviembre", "ноември"],
        ["december", "dec", "diciembre", "dic", "декември"]], 1):
    for _name in _names:
        _MONTH_NUMBERS[_name] = _i
_MONTH = f"({_alt(_MONTH_NUMBERS)})"
_ORD = r"(?:st|nd|rd|th|º|°|-ти|-ви|-ри)?"
_DATE = re.compile(
    r"(?<!\w)(\d{4})-(\d{1,2})-(\d{1,2})(?!\w)"                                        # 2026-12-01
    rf"|(?<!\w){_MONTH}\.?\s+(\d{{1,2}}){_ORD}(?!\w)(?:,?\s+(\d{{4}}))?"                # Dec 1, 2026
    rf"|(?<!\w)(\d{{1,2}}){_ORD}\s+(?:of\s+|de\s+)?{_MONTH}(?!\w)(?:,?\s+(?:de\s+)?(\d{{4}}))?")  # 31 de diciembre
_YEAR_END = re.compile(r"\b(end of (?:the )?year|year's end|new year'?s eve|fin de año|fin del año|"
                       r"fin de ano|fin del ano|nochevieja|крај на годината|крајот на годината)\b")


def _dates_in_text(low: str, today: date) -> list[date]:
    found = []
    for m in _DATE.finditer(low):
        try:
            if m.group(1):
                found.append((m.start(), date(int(m.group(1)), int(m.group(2)), int(m.group(3)))))
                continue
            month, day, year = ((m.group(4), m.group(5), m.group(6)) if m.group(4)
                                else (m.group(8), m.group(7), m.group(9)))
            d = date(int(year) if year else today.year, _MONTH_NUMBERS[month], int(day))
            if not year and d < today:
                d = d.replace(year=d.year + 1)
            found.append((m.start(), d))
        except ValueError:
            continue
    if m := _YEAR_END.search(low):
        found.append((m.start(), date(today.year, 12, 31)))
    return [d for _, d in sorted(found)]


_NUMBER_WORDS = {"un": 1, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
                 "seis": 6, "siete": 7, "ocho": 8, "diez": 10, "еден": 1, "една": 1, "два": 2,
                 "две": 2, "три": 3, "четири": 4, "пет": 5, "шест": 6, "седум": 7, "осум": 8,
                 "десет": 10}
_UNITS = {"día": 1, "dia": 1, "días": 1, "dias": 1, "semana": 7, "semanas": 7, "mes": 30,
          "meses": 30, "ден": 1, "дена": 1, "денови": 1, "недела": 7, "недели": 7,
          "месец": 30, "месеци": 30}
_LENGTH = re.compile(rf"(?<!\w)(\d+|{_alt(_NUMBER_WORDS)})\s+({_alt(_UNITS)})(?!\w)")


_PER_WEEK = re.compile(r"(?<!\w)\w+\s+(?:days?|times?|días?|dias?|veces|vez|дена|денови|ден|пати)\s+"
                       r"(?:a|per|each|every|a la|por|al|во|по|секоја)\s+"
                       r"(?:week|semana|недела|неделата)\w*")


def _length_in_text(text: str, low: str) -> int | None:
    """A study length: '6 weeks', 'tres semanas', 'три недели'. '3 days a week' is not one."""
    low = _PER_WEEK.sub(" ", low)
    text = _PER_WEEK.sub(" ", text.lower())
    if re.search(r"\b(day|week|fortnight|month)s?\b", low) and (n := parse_timeline(text)):
        return n
    if m := _LENGTH.search(low):
        n = int(m.group(1)) if m.group(1).isdigit() else _NUMBER_WORDS[m.group(1)]
        return n * _UNITS[m.group(2)] if n > 0 else None
    return None


_STAGE_WORDS = {
    1: ["vocabulary", "vocab", "word lists", "vocabulario", "palabras", "вокабулар", "зборови"],
    2: ["sentences", "sentence building", "sentence practice", "oraciones", "frases", "реченици"],
    3: ["role play", "role-play", "roleplay", "role plays", "role-plays", "roleplays", "juego de roles",
        "juegos de rol", "juego de rol", "simulaciones", "игра на улоги", "игри на улоги"],
}
_ONLY = re.compile(r"\b(only|just|solo|sólo|solamente|únicamente|nada más|само)\b")
_NOT = re.compile(r"\b(no|not|without|skip|except|sin|excepto|menos|без|не|освен)\b")


def _stages_in_text(low: str) -> list[int] | None:
    """Lesson types the message chooses, or None. One type counts only with "only" / "solo"
    / "само" ("learn vocabulary for restaurants" keeps all three); "without sentences" /
    "sin oraciones" removes a type."""
    yes, no = set(), set()
    for stage, words in _STAGE_WORDS.items():
        for m in re.finditer(rf"(?<!\w)({_alt(words)})(?!\w)", low):
            before = " ".join(low[:m.start()].split()[-3:])
            (no if _NOT.search(before) else yes).add(stage)
    yes -= no
    if yes and (len(yes) >= 2 or _ONLY.search(low)):
        return sorted(yes)
    if no and not yes and len(no) < 3:
        return sorted({1, 2, 3} - no)
    return None


_START_WORDS = re.compile(r"\b(start|starting|begin|beginning|from|desde|empiezo|empezar|"
                          r"empezando|comienzo|comenzar|од|почнувам|почнам)\b")
_DATE_CUES = re.compile(rf"(?<!\w){_MONTH}(?!\w)|\b(today|tomorrow|christmas|new year|easter|"
                        r"hoy|mañana|navidad|año nuevo|semana santa|денес|утре|божиќ)\b"
                        r"|\b(?:next|this) (?:week|month|weekend|year|monday|tuesday|wednesday|"
                        r"thursday|friday|saturday|sunday)\b"
                        r"|\b(?:el próximo|este) (?:mes|año|lunes|martes|miércoles|jueves|viernes|"
                        r"sábado|domingo)\b|\b(?:la próxima|esta) semana\b"
                        r"|(?<!\w)(?:следниот|овој) месец(?!\w)|(?<!\w)(?:следната|оваа) недела(?!\w)"
                        r"|\b\d{1,2}(?:st|nd|rd|th)\b")
_RESTART = re.compile(r"\b(restart|start over|start again|begin again|reset|clear (?:everything|"
                      r"it all|my preferences|all)|from scratch|reiniciar|reinicia|empezar de nuevo|"
                      r"empezar otra vez|volver a empezar|comenzar de nuevo|borrar todo|desde cero|"
                      r"почни одново|почнам одново|од почеток|избриши сè|избриши се)\b")


def wants_restart(text: str) -> bool:
    return _RESTART.search(text.lower()) is not None


def _iso(value) -> date | None:
    try:
        return date.fromisoformat(str(value).strip()) if value else None
    except ValueError:
        return None


# --- 3. One message in, one reply out ----------------------------------------------

def restart(chat: dict, lang: str | None = None) -> dict:
    """Clear the preferences but keep the conversation on screen."""
    lang = lang or chat.get("lang", "en")
    chat.update(found=_empty_found(), asked=[], suggested=[], asking=None, lang=lang)
    chat["messages"].append({"role": "tutor", "lang": lang, "restarted": True})
    return chat


def read_message(chat: dict, text: str, today: date | None = None, record: bool = True) -> dict:
    """Read the learner's message and add the tutor's reply to `chat` (changed in place).
    record=False when the app already put the learner's message on screen."""
    today = today or date.today()
    asking = chat.get("asking")
    if record:
        chat["messages"].append({"role": "you", "text": text})
    lang = chat["lang"] = detect_language(text, chat.get("lang", "en"))
    if wants_restart(text):
        return restart(chat, lang)
    try:
        data, model_ok = extract(text, today, asking), True
    except LLMError:
        data, model_ok = {}, False
    low = " " + text.lower() + " "
    found, problems, given = chat["found"], [], set()

    # Every field: code reads the text first. A model value is used only when code
    # found nothing AND the text has words about that field (or answers its question).

    # Language and Spanish country. The model knows cities (Lima = Peru).
    country = _country_in_text(low) or _country(data.get("country", ""))
    language = _language_in_text(low)
    if not language and data.get("language") in LANGUAGES and (
            asking == "language" or country or re.search(
                r"\b(trip|travel|going|heading|fly|visit|moving|viaje|viajo|voy|visitar|"
                r"патувам|одам|летам)\b", low)):
        language = data["language"]
    if country and country != "ES" and not language:
        language = "spanish"
    if language:
        if language != found["language"]:
            found["variety"] = None
        found["language"] = language
        given.add("language")
    if found["language"] == "spanish" and country:
        found["variety"] = country
    if found["language"] and not found["variety"]:
        found["variety"] = DEFAULT_VARIETY[found["language"]]

    # Study window: dates in the text, else a length from today; model dates only
    # when the text has date words that code cannot read ("mid January").
    start = end = length = None
    dates = _dates_in_text(low, today)
    if len(dates) >= 2:
        start, end = dates[0], dates[1]
    elif len(dates) == 1:
        # One date is the last study day (a trip date, "until ..."), unless the
        # message says it is the start ("from", "desde", "од") and it is not an answer
        # to the last-day question.
        if asking == "window_end" or not _START_WORDS.search(low):
            end = dates[0]
        else:
            start = dates[0]
            length = _length_in_text(text, low)
    elif n := _length_in_text(text, low):
        length = n
    elif _DATE_CUES.search(low):
        start, end = _iso(data.get("start_date")), _iso(data.get("end_date"))
        length = None if (start or end) else parse_timeline(data.get("duration") or "")
    if length and not end:
        start = start or found["start"] or today
        end = start + timedelta(days=length - 1)
    if end and not start:
        start = found["start"] if found["start"] and "window" not in chat["suggested"] else today
    if start:
        found["start"] = start
    if end:
        found["end"] = end
        given.add("window")
    if found["start"] and found["start"] < today:
        problems.append("past")
        found["start"] = None
        given.discard("window")
    if found["start"] and found["end"] and found["end"] < found["start"]:
        problems.append("order")
        found["end"] = None
        given.discard("window")

    # Weekdays, minutes, topics.
    weekdays = _weekdays_in_text(low)
    if not weekdays and (asking == "weekdays" or re.search(
            r"\b(work|workday|weekday|week|morning|evening|night|lunch|trabajo|mañanas|tardes|"
            r"noches|semana|работа|утро|вечер)\w*\b", low)):
        weekdays = [d for d in WEEKDAYS if d in (data.get("weekdays") or [])]
    if weekdays:
        found["weekdays"] = weekdays
        given.add("weekdays")
    minutes = _minutes_in_text(low, asking)
    model_minutes = data.get("minutes")
    if not minutes and isinstance(model_minutes, int) and model_minutes > 0 and (
            asking == "minutes" or re.search(
                r"\d|\b(hour|minute|min|half|quarter|ten|fifteen|twenty|thirty|forty|fifty|"
                r"hora|minuto|media|cuarto|diez|quince|veinte|treinta|cuarenta|"
                r"час|минут|половина|десет|петнаесет|дваесет|триесет)", low)):
        minutes = model_minutes
    if minutes:
        if minutes > MAX_MINUTES:
            problems.append("max_minutes")
        else:
            found["minutes"] = minutes
            given.add("minutes")
    # Topic word rules are loose ("stay", "help"); the model reads topics better but
    # also invents them. Use the model's list (its order) only when the words agree
    # that the message is about topics, or when it answers the topic question.
    word_topics = _topics_in_text(low)
    model_topics = [t for t in (data.get("topics") or []) if t in TOPICS]
    if model_topics and ((set(model_topics) & set(word_topics)) or (asking == "topics" and not word_topics)):
        topics = model_topics
    elif word_topics and (not model_ok or not found["topics"] or asking == "topics"
                          or "topics" in chat["suggested"]):
        topics = word_topics
    else:
        topics = []
    topics = list(dict.fromkeys(topics))
    if topics:
        if (found["topics"] and "topics" not in chat["suggested"]
                and re.search(r"\b(also|add|as well|too|también|además|añade|исто така|додај)\b", low)):
            topics = found["topics"] + [t for t in topics if t not in found["topics"]]
        found["topics"] = topics
        given.add("topics")

    # Lesson types are optional (default: all three) and never asked.
    stages = _stages_in_text(low)
    if stages:
        found["stages"] = stages

    # Values the learner gave are no longer suggestions.
    chat["suggested"] = [k for k in chat["suggested"] if k not in given]

    # Never ask the same item twice: an item that was asked and is still missing
    # gets a suggested value. Then ask the first item not yet asked, if any.
    for gap in missing(chat):
        if gap in chat["asked"] or (gap == "window_end" and "window" in chat["asked"]):
            _suggest(chat, gap, today)
    gaps = missing(chat)
    if gaps:
        chat["asking"] = gaps[0]
        chat["asked"].append(gaps[0])
    else:
        chat["asking"] = None
    chat["messages"].append({
        "role": "tutor", "lang": lang, "found": dict(found),
        "suggested": list(chat["suggested"]), "problems": problems,
        "question": gaps[0] if gaps else None, "model_ok": model_ok})
    return chat


def _suggest(chat: dict, gap: str, today: date) -> None:
    f = chat["found"]
    if gap == "language":
        f["language"], f["variety"] = SUGGESTED["language"], DEFAULT_VARIETY[SUGGESTED["language"]]
    elif gap in ("window", "window_end"):
        f["start"] = f["start"] or today
        f["end"] = f["start"] + timedelta(days=SUGGEST_DAYS - 1)
        gap = "window"
    else:
        f[gap] = list(SUGGESTED[gap]) if isinstance(SUGGESTED[gap], list) else SUGGESTED[gap]
    if gap not in chat["suggested"]:
        chat["suggested"].append(gap)


def missing(chat: dict) -> list[str]:
    """Necessary items still missing, in the order the tutor asks for them."""
    f = chat["found"]
    gaps = []
    if not f["language"]:
        gaps.append("language")
    if not f["end"]:
        gaps.append("window_end" if f["start"] else "window")
    if not f["weekdays"]:
        gaps.append("weekdays")
    if not f["minutes"]:
        gaps.append("minutes")
    if not f["topics"]:
        gaps.append("topics")
    return gaps


# --- 4. The tutor's words, from a stored reply ----------------------------------------

def _date(d: date, lang: str) -> str:
    t = TEXT[lang]
    return t["date"].format(month=t["months"][d.month - 1], day=d.day, year=d.year)


def summary_rows(found: dict, suggested: list[str], lang: str = "en") -> list[tuple[str, str | None, bool]]:
    """(label, value or None when missing, is_suggested) rows for one reply."""
    t = TEXT[lang]
    language = None
    if found["language"]:
        language = t["languages"][found["language"]]
        if found["language"] == "spanish":
            v = found["variety"]
            language += " · " + (f"{t['countries']['ES']} ({t['default']})" if v == "ES"
                                 else t["countries"].get(v, VARIETIES.get(v, v)))
    window = None
    if found["start"] and found["end"]:
        window = t["window"].format(start=_date(found["start"], lang), end=_date(found["end"], lang),
                                    days=(found["end"] - found["start"]).days + 1)
    days = ", ".join(t["weekdays"][WEEKDAYS.index(d)] for d in found["weekdays"]) or None
    topics = ", ".join(f"{i} {t['topics'][TOPICS.index(x)]}"
                       for i, x in enumerate(found["topics"], 1)) or None
    values = {"language": language, "window": window, "weekdays": days,
              "minutes": str(found["minutes"]) if found["minutes"] else None, "topics": topics}
    stages = found.get("stages")
    stage_text = (", ".join(t["stage_names"][s - 1] for s in stages) if stages
                  else f"{t['all_stages']} ({t['default']})")
    return ([(t["labels"][k], values[k], k in suggested) for k in REQUIRED]
            + [(t["labels"]["stages"], stage_text, False)])


def reply_parts(msg: dict, lang: str | None = None) -> dict:
    """The words of one stored tutor reply, in `lang` (default: the reply's own language)."""
    lang = lang or msg.get("lang", "en")
    t = TEXT[lang]
    if "text" in msg:
        return {"text": msg["text"]}
    if msg.get("restarted"):
        return {"text": t["restarted"]}
    question = None
    if msg["question"]:
        question = t["questions"][msg["question"]].format(topics=", ".join(t["topics"]))
    return {"have": t["have"], "missing": t["missing"], "suggested": t["suggested"],
            "rows": summary_rows(msg["found"], msg["suggested"], lang),
            "problems": [t["problems"][p] for p in msg["problems"]],
            "question": question,
            "closing": None if question else t["ready_suggested" if msg["suggested"] else "ready"],
            "model_note": None if msg.get("model_ok", True) else t["model_down"]}


def plan_inputs(chat: dict) -> dict | None:
    """The values for build_schedule, or None while a necessary item is missing."""
    f = chat["found"]
    if missing(chat) or f.get("stages") == []:     # [] = every lesson type turned off
        return None
    return {"language": f["language"], "variety": f["variety"], "goals": list(f["topics"]),
            "days": (f["end"] - f["start"]).days + 1, "weekdays": list(f["weekdays"]),
            "minutes": f["minutes"], "start_date": f["start"], "end_date": f["end"],
            "stages": list(f["stages"]) if f.get("stages") else [1, 2, 3]}


READING = {"en": "Reading your message…", "es": "Leyendo tu mensaje…",
           "mk": "Ја читам твојата порака…"}
