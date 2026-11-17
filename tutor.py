"""
tutor.py: the pipeline. Settings + conversation in, tutor reply out.

    learner message
        -> retriever.py   (rag mode only: find relevant notes)
        -> prompts.py     (build the chat messages for the chosen mode)
        -> llm.py         (run Qwen3-4B locally through Ollama)
        -> reply + sources + timings

Two paths:
  * tutor_reply(): free-text replies. Used by the "baseline" and "prompt"
    modes, so they stay plain model + prompt for the comparison.
  * make_lesson() / roleplay_turn(): the full system ("rag" mode). The model
    returns JSON that picks phrases by number from the curated notes, the
    code renders those phrases exactly, and every roleplay line is checked
    (script, wrong-language forms, echoing, unsupported times) and
    regenerated if a check fails.

app.py and evaluate.py both call these functions, so the app and the
evaluation use exactly the same pipeline.
"""

import json
import re

import numpy as np

from llm import LLMError, generate_response_with_stats
from prompts import (LANGUAGES, LESSON_PICK_TEMPLATE, LESSON_SCHEMA,
                     ROLEPLAY_JSON_TEMPLATE, ROLEPLAY_PARTNER, ROLEPLAY_SCHEMA, SCENARIOS,
                     SCRIPT, TutorSettings, build_messages, language_rules, opening_request)
from retriever import accepted_labels, normalize, parse_phrases

TOP_K = 4  # passages added to the prompt in rag mode

# Tutor modes ("prompt", "rag") run cooler than llm.py's default 0.7 so the
# small model invents less. "baseline" keeps the raw model defaults.
TUTOR_TEMPERATURE = 0.5

# Non-Macedonian words the model produced in testing -> correct Macedonian.
# Only clear-cut cases (these words don't exist in standard Macedonian).
# Applied in the full ("rag") pipeline only, so "baseline" and "prompt" stay raw.
FIXES = {
    "macedonian": [
        (r"добро\s+дене", "добар ден"),
        (r"добар\s+дан", "добар ден"),
        (r"аутобус\w*", "автобус"),  # \w* = also аутобуската, аутобуски ...
        (r"веднагу", "веднаш"),
        (r"потом", "потоа"),
        (r"где", "каде"),
        (r"хвала", "фала"),
        (r"спасибо", "благодарам"),
        (r"извините", "извинете"),
        (r"хлеб", "леб"),
        (r"что", "што"),
        (r"је", "е"),
        (r"питање", "прашање"),
        (r"али", "но"),
        (r"кључ\w*", "клуч"),   # Serbian "key": кључот -> клучот (ending kept)
    ],
}


def start_session(s: TutorSettings) -> list[dict]:
    """A new conversation: just the opening request."""
    return [{"role": "user", "content": opening_request(s)}]


def tutor_reply(s: TutorSettings, history, retriever=None, model_config=None) -> dict:
    """
    history: conversation so far; its last item must be the learner's message.
    retriever: a retriever.Retriever (needed only when s.mode == "rag").

    Returns {"text", "passages", "stats"} — stats are llm.py's timing numbers.
    Raises llm.LLMError if the model call fails.
    """
    if not history or history[-1]["role"] != "user":
        raise ValueError("history must end with the learner's (user) message.")

    passages = []
    if s.mode == "rag":
        if retriever is None:
            raise ValueError("rag mode needs a retriever.")
        # Two searches: the learner's newest message (what they actually asked)
        # first, then the scenario for background. Searching both together let
        # the scenario text drown out the question (e.g. train times got lost
        # under bus/taxi notes).
        by_message = retriever.search(history[-1]["content"], language=s.language, k=TOP_K,
                                      country=s.variety)
        by_scenario = retriever.search(SCENARIOS[s.scenario], language=s.language, k=TOP_K,
                                       country=s.variety)
        passages = by_message[: TOP_K - 1]
        for p in by_scenario + by_message:
            if len(passages) == TOP_K:
                break
            if p["text"] not in {q["text"] for q in passages}:
                passages.append(p)

    messages = build_messages(s, history, passages)
    if s.mode != "baseline":
        model_config = {"temperature": TUTOR_TEMPERATURE, **(model_config or {})}
    stats = generate_response_with_stats(messages, model_cfg(s, model_config))
    text = stats.pop("text")
    if s.mode == "rag":
        text = fix_common_errors(text, s.language)
    if s.task == "roleplay" and len(history) == 1:
        # The small model often adds a "Tip:" before the learner has said
        # anything; drop it on the opening turn.
        text = "\n".join(
            line for line in text.splitlines()
            if not line.strip().lower().startswith("tip:")
        ).strip()
    return {"text": text, "passages": passages, "stats": stats}


def fix_common_errors(text: str, language: str) -> str:
    """Replace known wrong forms (whole words only, keeping capitalisation)."""
    for pattern, right in FIXES.get(language, []):
        stem_len = len(pattern) - 3 if pattern.endswith(r"\w*") else None

        def repl(m, right=right, stem_len=stem_len):
            word = m.group(0)
            out = right + (word[stem_len:] if stem_len else "")  # keep endings
            return out[0].upper() + out[1:] if word[:1].isupper() else out
        # (?<!\w)/(?!\w) = whole words; \w covers Cyrillic letters in Python.
        text = re.sub(rf"(?<!\w){pattern}(?!\w)", repl, text, flags=re.IGNORECASE)
    if language == "spanish":
        # Gemma opens exclamations with "¿" ("¿Bienvenido!"): pair ! with ¡.
        text = re.sub(r"¿([^?¿¡!]*)!", r"¡\1!", text)
        # ...and wraps greetings in question marks ("¿Buenos días?").
        text = re.sub(r"¿((?:Muy\s+)?(?:Buen[oa]s(?:\s+(?:días|tardes|noches))?|Hola|Bienvenid[oa]s?))\?",
                      r"¡\1!", text, flags=re.I)
    if language == "macedonian":
        # "мени" (menu) glossed as "meat": fix the gloss on that line.
        text = "\n".join(
            re.sub(r"\bmeat\b", "menu", line)
            if re.search(r"(?<!\w)мени(?!\w)", line, re.I) and "месо" not in line.lower()
            else line
            for line in text.split("\n")
        )
    return text


# ===========================================================================
# Full system ("rag" mode): structured, grounded, checked
# ===========================================================================

MAX_ATTEMPTS = 3          # first try + 2 regenerations, then a safe fallback
JSON_MAX_TOKENS = 700     # length cap for structured (JSON) replies
RETRY_TEMPERATURES = (TUTOR_TEMPERATURE, 0.8, 1.0)   # one per attempt
_SOFT = ("filler", "not in the phrasebook", "too long", "answers it yourself")
LESSON_MIN, LESSON_MAX = 4, 8
MAX_CANDIDATES = 40       # phrases offered to the model per call

# Wrong-language forms the reply must not contain (checked BEFORE the fix pass,
# so a reply that needed fixing is regenerated rather than silently patched).
FORBIDDEN = {
    "macedonian": [p for p, _ in FIXES["macedonian"]],
    # Role slips seen in testing: staff saying "how can YOU help ME".
    "spanish": [r"c[oó]mo\s+puede\s+ayudarme", r"en\s+qu[eé]\s+puede\s+ayudarme"],
}
_CYR = re.compile(r"[Ѐ-ӿ]")
_LAT = re.compile(r"[A-Za-zÀ-ÿ]")
_TIME = re.compile(r"\b(\d{1,2})[.:](\d{2})\b")


# --- Times written in words ---------------------------------------------------
# Scenes where times are facts (timetables, opening hours), not small talk.
WORD_TIME_SCENES = {"transport", "hotel", "sightseeing"}
YES = {"macedonian": r"^да(?!\w)", "spanish": r"^s[ií](?!\w)"}
WITHOUT = {"macedonian": "без", "spanish": "sin"}

_MK_NUM = {"нула": 0, "еден": 1, "една": 1, "два": 2, "две": 2, "три": 3, "четири": 4, "пет": 5,
           "шест": 6, "седум": 7, "осум": 8, "девет": 9, "десет": 10, "единаесет": 11,
           "дванаесет": 12, "тринаесет": 13, "четиринаесет": 14, "петнаесет": 15,
           "шеснаесет": 16, "седумнаесет": 17, "осумнаесет": 18, "деветнаесет": 19,
           "половина": 30}
_MK_TENS = {"дваес": 20, "триес": 30, "четириес": 40, "педес": 50}   # prefixes: Gemma's "дваесесет" too
_ES_NUM = {"cero": 0, "una": 1, "uno": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6,
           "siete": 7, "ocho": 8, "nueve": 9, "diez": 10, "once": 11, "doce": 12, "trece": 13,
           "catorce": 14, "quince": 15, "dieciséis": 16, "diecisiete": 17, "dieciocho": 18,
           "diecinueve": 19, "veinte": 20, "treinta": 30, "cuarenta": 40, "cincuenta": 50,
           "media": 30, "cuarto": 15, "veintiuno": 21, "veintidós": 22, "veintitrés": 23,
           "veinticuatro": 24, "veinticinco": 25, "veintiséis": 26, "veintisiete": 27,
           "veintiocho": 28, "veintinueve": 29}


def _num_word(w, language):
    if language == "macedonian":
        if w in _MK_NUM:
            return _MK_NUM[w]
        return next((v for k, v in _MK_TENS.items() if w.startswith(k)), None)
    return _ES_NUM.get(w)


def _read_number(words, i, language):
    """Read '<tens> [и/y <units>]' or '<units>' starting at words[i] -> (value, next index)."""
    v = _num_word(words[i], language) if i < len(words) else None
    if v is None:
        return None, i
    i += 1
    joiner = "и" if language == "macedonian" else "y"
    if v >= 20 and i + 1 < len(words) and words[i] == joiner:
        u = _num_word(words[i + 1], language)
        if u is not None and 0 < u < 10:
            return v + u, i + 2
    return v, i


def word_times(text, language):
    """Clock times written in words -> [(hour, minute)].
    Macedonian: "во осум и дваесет", "во еден час и триесет и пет", "во седум часот".
    Spanish:    "a las diez y media", "a la una y cuarto", "a las ocho menos cuarto"."""
    words = normalize(text).split()
    out = []
    for i, w in enumerate(words):
        if language == "macedonian" and w == "во":
            j = i + 1
        elif language == "spanish" and w in ("las", "la") and i > 0 and words[i - 1] == "a":
            j = i + 1
        else:
            continue
        hour, j = _read_number(words, j, language)
        if hour is None or not 0 <= hour <= 24 or hour == 30:   # "во половина" is not a time
            continue
        has_hour_word = j < len(words) and words[j] in ("час", "часот", "часa")
        if has_hour_word:
            j += 1
        # "во" is also "in" ("Во еден правец" = one-way): in Macedonian only count
        # it as a time with "час(от)" or with minutes ("во осум и дваесет").
        if language == "macedonian" and not has_hour_word and not (
                j + 1 < len(words) and words[j] == "и" and _num_word(words[j + 1], language) is not None):
            continue
        minute = 0
        if j + 1 < len(words) and words[j] in ("и", "y"):
            m, _ = _read_number(words, j + 1, language)
            if m is not None and m < 60:
                minute = m
        elif language == "spanish" and j + 1 < len(words) and words[j] == "menos":
            m, _ = _read_number(words, j + 1, language)
            if m is not None and m < 60:
                hour, minute = (hour - 1) % 24, 60 - m
        out.append((hour, minute))
    return out


def _dedupe(phrases):
    seen, out = set(), []
    for p in phrases:
        key = normalize(p["text"])
        if key and key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _numbered(phrases, partner=None):
    """Numbered phrase list. With partner, each line is tagged with who says it,
    so the model doesn't take the traveler's lines ("A table for two, please")."""
    def tag(p):
        if partner is None:
            return ""
        return f"[{partner} says] " if is_partner_phrase(p) else "[TRAVELER says] "
    return "\n".join(f"{i}. {tag(p)}{p['text']} = {p['meaning']}" for i, p in enumerate(phrases, 1))


LORA_MODEL = "tutor-lora"   # built by lora/ (see lora/Modelfile)


def model_cfg(s, model_config=None):
    """Add the fine-tuned model to the call when the settings ask for LoRA."""
    return {**({"model": LORA_MODEL} if getattr(s, "lora", False) else {}), **(model_config or {})}


def _call_json(messages, schema, model_config=None):
    # num_predict caps the reply length: with forced JSON, small models can
    # occasionally loop on whitespace until the context is full, which takes
    # minutes and blocks Ollama for everyone. A lesson/turn needs < 400 tokens.
    cfg = {"temperature": TUTOR_TEMPERATURE, "num_predict": JSON_MAX_TOKENS,
           **(model_config or {}), "format": schema}
    try:
        stats = generate_response_with_stats(messages, cfg)
    except LLMError as e:
        if "structured output is unavailable" not in str(e):
            raise
        # Imported models (e.g. tutor-lora) can't force JSON in Ollama: ask for it
        # in words and pull the first {...} block out of the reply.
        keys = ", ".join(f'"{k}"' for k in schema["properties"])
        ask = (f"\n\nAnswer with ONLY a JSON object with the keys {keys}. No other text.")
        msgs = [dict(m) for m in messages]
        msgs[-1]["content"] += ask
        stats = generate_response_with_stats(msgs, {k: v for k, v in cfg.items() if k != "format"})
        m = re.search(r"\{.*\}", stats["text"], re.S)
        stats["text"] = m.group(0) if m else stats["text"]
    try:
        data = json.loads(stats.pop("text"))
    except json.JSONDecodeError as e:
        raise LLMError(f"Model returned invalid JSON: {e}") from e
    return data, stats


def _known_words(language, retriever):
    """Every normalized word that appears in this language's notes."""
    cache = getattr(retriever, "_known_words", {})
    if language not in cache:
        words = set()
        for c in retriever.chunks:
            if c["language"] == language:
                words.update(normalize(c["text"]).split())
        cache[language] = words
        retriever._known_words = cache
    return cache[language]


def _clean_english(text, language, retriever):
    """
    Keep English helper text (intro, tip) honest: drop any sentence that
    contains target-language words which are not in our notes (so the model
    can't sneak an invented phrase into an English sentence).
    """
    if language != "macedonian":   # Cyrillic makes Macedonian words detectable
        return text.strip()
    known = _known_words(language, retriever)
    keep = []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        cyr_words = [w for w in normalize(sentence).split() if _CYR.search(w)]
        if all(w in known for w in cyr_words):
            keep.append(sentence)
    return " ".join(keep).strip()


def _quiz(phrases):
    """Two deterministic questions from today's phrases (no model needed)."""
    qs = []
    if phrases:
        p = phrases[min(1, len(phrases) - 1)]
        qs.append({"q": f"How do you say \"{p['meaning']}\"?", "a": p["text"]})
    if len(phrases) > 2:
        p = phrases[-1]
        qs.append({"q": f"What does \"{p['text']}\" mean?", "a": p["meaning"]})
    return qs


# Notes lines that talk about our sources are for the team, not the learner.
_SOURCE_WORDS = ("peace corps", "fsi", "dli", "wikivoyage", "kramer", "jankovska", "libro libre",
                 "source", "dialogue says", "the book", "ocr", "transcription")
_TIP_WORDS = ("must", "usually", "polite", "rude", "custom", "tip", "note", "don't",
              "do not", "common", "often", "in spain", "in macedonia", "locals", "manners")


def notes_tip(session, s, retriever):
    """
    A practical/cultural tip copied from the curated notes: an explanatory line
    (no "phrase = meaning") from today's units, or from the scenario's other
    notes. Returns "" if the notes have none, rather than letting the model
    invent one.
    """
    def candidates(chunks):
        for c in chunks:
            for line in c["text"].splitlines():
                line = line.strip().lstrip("-").strip()
                low = line.lower()
                if (40 <= len(line) <= 260 and " = " not in line and not line.startswith(("#", "Source"))
                        and any(w in low for w in _TIP_WORDS)
                        and not any(src in low for src in _SOURCE_WORDS)):
                    yield line

    own = list(session["units"])
    files = {u["source"] for u in own}          # same files as today's units only
    allowed = accepted_labels(s.language, s.variety)   # and the same variety
    rest = [c for c in retriever.chunks if c["source"] in files and c not in own
            and c.get("country") in allowed]
    return next(candidates(own), None) or next(candidates(rest), "")


def make_lesson(s: TutorSettings, session: dict, trip: str, retriever, model_config=None) -> dict:
    """
    Today's lesson for one plan session. The model only chooses phrase numbers
    and writes an English intro and tip; phrases, pronunciations and meanings
    are copied from the notes.
    Returns {"intro", "phrases", "tip", "quiz", "sources", "stats", "fallback"}.
    """
    candidates = _dedupe([p for u in session["units"] for p in u["phrases"]])[:MAX_CANDIDATES]
    notes = "\n\n".join(u["text"] for u in session["units"])[:4000]
    n_max = min(LESSON_MAX, len(candidates))
    n_min = min(LESSON_MIN, n_max)
    prompt = LESSON_PICK_TEMPLATE.format(
        language=LANGUAGES[s.language], trip=trip, topic=session["title"],
        n_min=n_min, n_max=n_max, numbered=_numbered(candidates), notes=notes,
    )
    data, stats, fallback = {}, {}, False
    try:
        # Phrase selection needs JSON, which the fine-tuned model no longer writes
        # reliably, so lessons always use the base model (phrases come from the notes anyway).
        data, stats = _call_json([{"role": "user", "content": prompt}], LESSON_SCHEMA, model_config)
    except LLMError as e:
        if "invalid JSON" not in str(e):
            raise
        fallback = True

    ids = []
    for i in data.get("phrase_ids", []):
        if isinstance(i, int) and 1 <= i <= len(candidates) and i not in ids:
            ids.append(i)
    if len(ids) < n_min:                       # model picked too few: top up in order
        fallback = fallback or not ids
        ids += [i for i in range(1, len(candidates) + 1) if i not in ids][: n_min - len(ids)]
    phrases = [candidates[i - 1] for i in ids[:n_max]]

    intro = _clean_english(data.get("intro", ""), s.language, retriever) or \
        f"Today: {session['title']}. These phrases will help you on your trip."
    tip = notes_tip(session, s, retriever)   # from the notes, never model-written
    return {
        "intro": intro, "phrases": phrases, "tip": tip, "quiz": _quiz(phrases),
        "sources": sorted({u["source"] for u in session["units"]}),
        "stats": stats, "fallback": fallback,
    }


# --- Roleplay ----------------------------------------------------------------

def _english_of(m, language, retriever):
    """English for one history line: the partner's stored translation, or the
    phrasebook meaning of a traveler line ("" if the learner typed freely)."""
    if m["role"] == "assistant":
        return m.get("english", "")
    return _meaning_map(language, retriever).get(normalize(m["content"]), "")


def _search_query(s, history, retriever):
    """What this turn is about: the learner's last line plus the partner's line
    it answers, with English, so retrieval follows the conversation instead of
    one isolated sentence (the embeddings are multilingual)."""
    recent = history[-3:]
    parts = []
    for m in recent:
        parts.append(m["content"])
        en = _english_of(m, s.language, retriever)
        if en:
            parts.append(en)
    return " ".join(parts) or SCENARIOS[s.scenario]


def _context(s, history, session, retriever):
    """Notes and numbered phrasebook for this roleplay turn."""
    from planner import SCENARIO_FILES
    last_learner = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
    own = {f"{s.language}/{f}" for f in SCENARIO_FILES[s.scenario]}
    # Search wide, then keep this scenario's notes first: a restaurant scene
    # should not pull receptionist or taxi lines just because a word matches.
    found = retriever.search(_search_query(s, history, retriever), language=s.language,
                             k=TOP_K * 2, country=s.variety)
    found = sorted(found, key=lambda c: c["source"] not in own)[:TOP_K]   # stable: keeps rank
    if last_learner:
        found += retriever.search(SCENARIOS[s.scenario], language=s.language, k=2, country=s.variety)
    chunks, seen = [], set()
    for c in list(session["units"]) + found:
        if c["text"] not in seen:
            seen.add(c["text"])
            chunks.append(c)
    phrases = _dedupe([p for c in chunks for p in (c.get("phrases") or parse_phrases(c))])
    return chunks, phrases[:MAX_CANDIDATES], last_learner


MAX_UNKNOWN_SHARE = 0.34  # share of reply words never seen in the notes


def unknown_share(reply, language, retriever):
    """Share of target-language words in reply that never appear in our notes
    (prefix match, so endings like мени/менито count as known)."""
    if language != "macedonian":
        return 0.0
    known = _known_words(language, retriever)
    prefixes = {w[:4] for w in known if len(w) >= 4}
    words = [w for w in normalize(reply).split() if _CYR.search(w)]
    if not words:
        return 0.0
    unknown = [w for w in words if w not in known and (len(w) < 4 or w[:4] not in prefixes)]
    return len(unknown) / len(words)


_ENGLISH_WORDS = {"the", "a", "an", "is", "are", "i", "you", "it", "to", "of", "and", "what",
                  "where", "how", "can", "do", "please", "more", "want", "practice", "again",
                  "my", "me", "this", "that", "in", "for", "with", "help", "hello", "thanks",
                  "thank", "yes", "no", "next", "start", "would", "like", "much", "time"}


def wrong_language(text, language):
    """True if the learner's roleplay line is clearly not in the target language
    (e.g. English "practice more"), so it should not be sent to the partner."""
    letters = _CYR.findall(text) + _LAT.findall(text)
    if not letters:
        return False
    if language == "macedonian":
        return len(_CYR.findall(text)) / len(letters) < 0.5
    if _CYR.search(text):
        return True
    words = normalize(text).split()
    if re.search(r"[áéíóúñ¿¡]", text.lower()):
        return False
    return bool(words) and sum(w in _ENGLISH_WORDS for w in words) / len(words) >= 0.5


_NEUTRAL_STARTS = ("it's", "it is", "it was", "yes", "no,", "of course", "ok", "very ",
                   "that's good", "great", "fine",
                   # greetings, thanks and goodbyes: either side says them
                   "see you", "goodbye", "bye", "thank", "thanks", "you're welcome",
                   "have a", "good day", "good morning", "good afternoon", "good evening",
                   "hello", "hi", "welcome", "until")

# "Where is X?" is the traveler's question; a helper character never asks it.
# ("Where are you going?" = каде одите / adónde va is fine and not matched.)
WHERE_IS = {
    "macedonian": r"(?<!\w)каде\s+(?:е|се|има|се\s+наоѓа)(?!\w)",
    "spanish": r"(?<!\w)d[oó]nde\s+(?:est[aá]n?|queda|quedan|hay)(?!\w)",
}


# In these scenes the staff never need to ask "where" at all (the model used
# каде/dónde for "what": "Каде сакате да јадете?" = where do you want to eat).
NO_WHERE_SCENES = {"restaurant", "shopping", "hotel", "weather"}
ANY_WHERE = {"macedonian": r"(?<!\w)(?:каде|од\s+каде)(?!\w)",
             "spanish": r"(?<!\w)a?d[oó]nde(?!\w)"}
# "I want / I'd like / I need" at the start of a sentence is the customer's voice;
# staff in these scenes never say it ("Сакам храна ...", "Quisiera el plato del día").
CUSTOMER_SCENES = {"restaurant", "shopping", "hotel", "transport"}
I_WANT = {"macedonian": r"(?<!\w)(?:сакам|би\s+сакал[a]?|ми\s+треба|ќе\s+(?:го\s+|ја\s+)?земам|за\s+мене|немам\s+пари)(?!\w)",
          "spanish": r"(?<!\w)(?:quisiera|quiero|me\s+gustar[ií]a|necesito|tr[aá]igame|para\s+m[ií]|me\s+l[oa]s?\s+llevo)(?!\w)"}
# Requests that only the customer makes ("Може ли една чаша вода?" = May I have
# a glass of water?, "¿Me trae la cuenta?").
CUSTOMER_REQUEST = {"macedonian": r"(?<!\w)може\s+ли\s+(?:да\s+добијам|да\s+ми|една|еден|едно|два|две)(?!\w)",
                    "spanish": r"(?<!\w)(?:me\s+trae|me\s+puede\s+traer|p[oó]ngame|me\s+da)(?!\w)"}


def is_customer_request(sentence: str, language: str) -> bool:
    """True if a sentence sounds like the customer ordering or asking for something.
    Macedonian also: a terse "..., ве молам" order with no imperative verb
    ("Ве молам, маса за двајца."); staff use ве молам with a verb
    ("Седнете, ве молам.", "Повелете, ве молам.")."""
    low = normalize(sentence)
    if re.search(CUSTOMER_REQUEST[language], low):
        return True
    # Statements only: staff ask politely too ("Може ли пасошот, ве молам?").
    if language == "macedonian" and not sentence.strip().endswith("?") \
            and re.search(r"(?<!\w)ве\s+молам(?!\w)", low):
        words = [w for w in low.split() if w not in ("ве", "молам")]
        imperative = any(w.endswith(("ете", "јте", "ајте")) or w == "повели" for w in words)
        return bool(words) and not imperative
    return False


# Broken "where" combinations in any scene ("Каде ви помагам?", "Каде ја ...").
# (Not "Каде ја изгубивте?" = where did you lose it, a fair police question.)
BAD_WHERE = {"macedonian": r"(?<!\w)каде\s+(?:ви\s+помаг\w*|желате|можам)(?!\w)", "spanish": r"$^"}
# Customers ask the price; staff state it ("Колку е по ноќ?" from a receptionist).
PRICE_QUESTION = {"macedonian": r"^колку\s+(?:чини|е|кошта|ќе\s+чини)(?!\w)",
                  "spanish": r"^cu[aá]nto\s+(?:cuesta|es|vale|sale|salen|cuestan)(?!\w)"}
# The traveler's trouble, said by the helper ("Изгубен сум." = I'm lost, from a police officer).
TRAVELER_TROUBLE = {
    "macedonian": r"(?<!\w)(?:имам\s+проблем|изгубен\s+сум|изгубена\s+сум|(?:ја|го|ги)\s+изгубив|"
                  r"ме\s+ограбија|ми\s+треба\s+помош)(?!\w)",
    "spanish": r"(?<!\w)(?:tengo\s+un\s+problema|estoy\s+perdid[oa]|perd[ií]|me\s+robaron|"
               r"necesito\s+ayuda)(?!\w)",
}
PLEASE = {"macedonian": r"(?<!\w)ве\s+молам(?!\w)", "spanish": r"(?<!\w)por\s+favor(?!\w)"}
# Food talk belongs to the restaurant ("Што сакате за јадење?" from a shop assistant).
FOOD_TALK = {"macedonian": r"(?<!\w)за\s+(?:јадење|пиење)(?!\w)|шопска",
             "spanish": r"(?<!\w)(?:para|de)\s+(?:comer|beber)(?!\w)|van?\s+a\s+comer|"
                        r"qu[eé]\s+van?\s+a\s+tomar\s*\?"}
# Timetable questions are the traveler's; the ticket seller answers them.
TIMETABLE_QUESTION = {"macedonian": r"^(?:во\s+колку\s+часот\s+|кога\s+)(?:тргнува|пристигнува|заминува|поаѓа)",
                      "spanish": r"^(?:a\s+qu[eé]\s+hora|cu[aá]ndo)\s+(?:sale|llega|parte)"}


def check_reply(reply, s, last_learner, prev_reply, notes_text, retriever=None,
                traveler_questions=()):
    """Return a list of problems with a roleplay reply (empty = OK)."""
    problems = []
    questions = [x for x in re.split(r"(?<=[?!.])\s+", reply) if x.strip().endswith("?")]
    lowq = [q.lower() for q in questions]
    if any(re.search(WHERE_IS[s.language], q) for q in lowq):
        problems.append("asks where a place is; that is the TRAVELER's question. "
                        "As the local/staff, answer or ask how you can help instead")
    elif s.scenario in NO_WHERE_SCENES and any(re.search(ANY_WHERE[s.language], q) for q in lowq):
        problems.append("uses 'where' (каде/dónde) in your question; here you want to ask WHAT "
                        "(Што / Qué) the traveler would like, or how you can help")
    elif any(re.search(BAD_WHERE[s.language], q) for q in lowq):
        problems.append("uses 'where' (каде) wrongly; to offer help say Повелете or "
                        "Како можам да ви помогнам?")
    if questions and not reply.strip().endswith("?") and len(re.split(r"(?<=[?!.])\s+", reply.strip())) >= 2 \
            and reply.strip().index("?") < len(reply.strip()) - 1:
        # A question followed by more text that answers it: the character is talking to itself.
        after = reply.strip()[reply.strip().index("?") + 1:].strip()
        if after and not last_learner:
            problems.append("asks a question and then answers it yourself; ask once, then wait")
    letters = _CYR.findall(reply) + _LAT.findall(reply)
    if not reply.strip():
        return ["empty reply"]
    if len(reply) > 300:
        problems.append("too long (keep it to one or two short sentences)")
    if letters:
        target = _CYR if s.language == "macedonian" else _LAT
        if len(target.findall(reply)) / len(letters) < (0.85 if s.language == "macedonian" else 0.99):
            problems.append(f"not written in {LANGUAGES[s.language]} script only")
    if last_learner and normalize(reply) in GENERIC_LINES.get(s.language, set()):
        problems.append("is only a filler line; move the scene forward with the next step "
                        "(see 'Your next move')")
    if s.language == "macedonian":
        # Letters that exist in Russian/Serbian/Bulgarian but not in Macedonian
        # (e.g. LoRA's "Популярно" instead of "Популарно").
        foreign = re.findall(r"\w*[яыэёщъьйюћђ]\w*", reply.lower())
        if foreign:
            problems.append(f"\"{foreign[0]}\" is not Macedonian (Russian/Serbian letters); "
                            "use Macedonian spelling")
    mixed = [w for w in re.findall(r"\w+", reply) if _CYR.search(w) and re.search(r"[A-Za-z]", w)]
    if mixed:
        problems.append(f"mixes Latin and Cyrillic letters in \"{mixed[0]}\"; write it in one alphabet")
    elif s.language == "macedonian":
        # A whole English word inside a Macedonian line ("за desert").
        latin = [w for w in re.findall(r"[A-Za-z]{3,}", reply)]
        if latin:
            problems.append(f"\"{latin[0]}\" is not Macedonian; write it in Macedonian Cyrillic")
    low = reply.lower()
    for pat in FORBIDDEN[s.language]:
        if re.search(rf"(?<!\w){pat}(?!\w)", low):
            found = re.search(rf"(?<!\w){pat}(?!\w)", low).group(0)
            problems.append(f"uses a wrong form \"{found}\" (see the language rules)")
    n_learner, n_reply = normalize(last_learner), normalize(reply)
    sentences = [x for x in re.split(r"(?<=[?!.])\s+", reply.strip()) if x.strip()]
    # (Very short lines like greetings may be answered in kind: "Добар ден!" -> "Добар ден!")
    def mostly(copy):   # the reply is basically the copied line, with little added
        return copy in n_reply and len(n_reply) <= 1.4 * len(copy)

    # Only a reply that is basically the traveler's line counts as an echo;
    # restating an order inside an acknowledgement ("Одлично, ќебапчиња со
    # кромид. Што сакате за пиење?") is exactly what a waiter should do.
    if len(n_learner) > 8 and len(n_learner.split()) > 2 and mostly(n_learner):
        problems.append("repeats the traveler's words. Acknowledge what they said in YOUR own words "
                        "(e.g. 'Muy bien.' / 'Одлично.' plus what you will do) and continue the scene; "
                        "do not ignore what they ordered or asked")
    elif len(n_learner.split()) >= 3 and any(normalize(x) == n_learner for x in sentences):
        problems.append("starts by repeating the traveler's sentence word for word; acknowledge "
                        "it in your own words instead (e.g. 'Muy bien.' / 'Одлично.')")
    if prev_reply and n_reply == normalize(prev_reply):
        problems.append("repeats your previous line")
    if s.scenario in CUSTOMER_SCENES and any(
            re.search(I_WANT[s.language], normalize(x)) or is_customer_request(x, s.language)
            for x in sentences):
        problems.append("speaks as the customer ('I want / I'd like'); you are the staff: "
                        "offer, answer or ask what the traveler would like")
    if s.scenario in CUSTOMER_SCENES and any(
            re.search(PRICE_QUESTION[s.language], normalize(q)) for q in questions):
        problems.append("asks the price; that is the customer's question. As the staff, SAY the price")
    if s.scenario == "transport" and any(
            re.search(TIMETABLE_QUESTION[s.language], normalize(q)) for q in questions):
        problems.append("asks when a train/bus leaves or arrives; that is the traveler's question. "
                        "As the ticket seller, answer it (times only from the notes)")
    if s.scenario not in ("restaurant", "weather", "sightseeing") and re.search(FOOD_TALK[s.language], low):
        problems.append("talks about food or drinks, which is not part of this scene; stay in your own job")
    if s.scenario in ("emergency", "directions") and re.search(TRAVELER_TROUBLE[s.language], low):
        problems.append("talks as if YOU had the problem (lost, robbed, need help); the traveler has "
                        "it. As the helper, ask about it or say what to do")
    # The traveler's own request handed back with "please" ("Un billete a Sevilla, por favor.").
    for x in sentences:
        if not x.strip().endswith("?") and re.search(PLEASE[s.language], normalize(x)) \
                and last_learner and _overlap(x, last_learner, content_only=True) >= 0.5:
            problems.append("repeats the traveler's request with 'please'; confirm it in your own words "
                            "(e.g. 'Muy bien, un billete ...' / 'Одлично, ...') and move on")
            break
    content = [w for w in normalize(reply).split() if len(w) > 3]
    if any(content.count(w) >= 3 for w in set(content)):
        problems.append("repeats the same word several times; write one clear sentence")
    if traveler_questions:
        # No sentence of the reply may be a (near-)copy of a traveler line from the
        # notes, whether a question ("Дали имате мени?") or a statement
        # ("Quisiera el plato del día.").
        for q in traveler_questions:
            # Exact copies only: fuzzy matching flagged good staff lines ("Traigo un
            # vaso de agua" vs the customer's "Traiga ...", or a correct timetable answer).
            if len(q.split()) >= 3 and (mostly(q) or any(normalize(x) == q for x in sentences)):
                problems.append("says a line the TRAVELER would say; answer as the staff/local instead")
                break
    if retriever is not None and unknown_share(reply, s.language, retriever) > MAX_UNKNOWN_SHARE:
        problems.append("uses too many words that are not in the phrasebook; reuse its wording")
    notes_times = {f"{int(h)}.{m}" for h, m in _TIME.findall(notes_text)}
    # Times the notes give in words count too ("во десет и триесет" = 10.30).
    notes_times |= {f"{h}.{m:02d}" for h, m in word_times(notes_text, s.language)}
    for h, m in _TIME.findall(reply):
        if f"{int(h)}.{m}" not in notes_times:
            problems.append(f"time {h}.{m} is not in the timetable notes")
    if s.scenario in WORD_TIME_SCENES:
        # Times written in words ("во осум и дваесет", "a las diez y media"):
        # compare on a 12-hour clock, since speech rarely says 20 for 8 pm.
        notes_12 = {(int(h) % 12, int(m)) for h, m in _TIME.findall(notes_text)} | \
                   {(h % 12, m) for h, m in word_times(notes_text, s.language)}
        for h, m in word_times(reply, s.language):
            if (h % 12, m) not in notes_12:
                problems.append(f"time {h}.{m:02d} (written in words) is not in the timetable notes; "
                                "use only times from the notes, written as digits")
                break
    # A yes/no answer that contradicts itself: "Is there meat in it?" ->
    # "Yes, it's without meat."
    asked = [w for w in normalize(last_learner).split() if len(w) > 3] if last_learner.strip().endswith("?") else []
    n_low = normalize(reply)
    if asked and re.match(YES[s.language], n_low) and any(
            re.search(rf"(?<!\w){WITHOUT[s.language]}\s+{re.escape(w[:4])}", n_low) for w in asked):
        problems.append("says 'yes' but then says it is WITHOUT the thing they asked about; "
                        "answer 'no' if it isn't there (e.g. 'Не, без месо е.' / 'No, no lleva carne.')")
    return problems


# Fallback when the traveler ASKED something and every model reply failed the
# checks (usually an invented time): "Of course." would not answer a question.
# Fixed, simple lines (not from the notes); the Macedonian one is for Marija to check.
QUESTION_FALLBACK = {
    "spanish": ("Lo siento, no lo sé. Mire el horario, por favor.",
                "Sorry, I don't know. Please look at the timetable."),
    "macedonian": ("Извинете, не знам. Погледнете го возниот ред.",
                   "Sorry, I don't know. Look at the timetable."),
}


# Same, for scenes without a timetable.
NOT_SURE_FALLBACK = {
    "spanish": ("Lo siento, no estoy seguro.", "Sorry, I'm not sure."),
    "macedonian": ("Извинете, не сум сигурен.", "Sorry, I'm not sure."),
}


def _fallback_line(language, retriever, first_turn, last_learner="", variety=None, scenario=None):
    """A safe in-character line taken from the notes (of the learner's variety),
    preferring this scenario's own notes (no receptionist lines in a restaurant)."""
    if last_learner.strip().endswith("?"):
        return QUESTION_FALLBACK[language] if scenario == "transport" else NOT_SURE_FALLBACK[language]
    # Mid-conversation, a neutral acknowledgement keeps the scene going better
    # than "I don't understand".
    wanted = ["may i help", "here you are", "welcome"] if first_turn else \
        ["of course", "right away", "let me see", "ok, agreed", "very well", "don't understand"]
    allowed = accepted_labels(language, variety)
    from planner import SCENARIO_FILES
    own = {f"{language}/{f}" for f in SCENARIO_FILES.get(scenario, [])}
    chunks = [c for c in retriever.chunks if c["language"] == language and c.get("country") in allowed]
    chunks.sort(key=lambda c: c["source"] not in own)          # stable: own notes first
    phrases = [p for c in chunks for p in parse_phrases(c)]
    # Staff lines from this scenario, then staff lines from anywhere that aren't
    # tied to another job (the speaker label in the meaning names it).
    staff_first = [p for p in phrases if is_partner_phrase(p)] + [p for p in phrases if not is_partner_phrase(p)]
    for w in wanted:
        for p in staff_first:
            if w in p["meaning"].lower() and "change" not in p["meaning"].lower():
                return p["text"], _drop_speaker(p["meaning"])
    return (phrases[0]["text"], _drop_speaker(phrases[0]["meaning"])) if phrases else ("...", "...")


def _drop_speaker(meaning):
    """'Receptionist: Of course.' -> 'Of course.'"""
    return re.sub(r"^(?:waiter|waitress|receptionist|taxi driver|driver|traveller|traveler|clerk|seller|"
                  r"shop assistant|local|police officer|officer|pharmacist|doctor|passenger|guest|customer|"
                  r"ticket seller|cashier|conductor|bartender):\s+", "", meaning, flags=re.I)


_PARTNER_MARKERS = ("waiter", "келнер", "camarero", "(clerk)", "(receptionist)",
                    "(local)", "(seller)", "you will hear", "may i help")


def is_partner_phrase(p):
    """Lines the waiter/clerk says, not the traveler (by notes labels)."""
    if p.get("speaker"):
        return True
    blob = f"{p['section']} {p['text']} {p['meaning']}".lower()
    return any(m in blob for m in _PARTNER_MARKERS)


def _overlap(a, b, content_only=False):
    """Share of a's words that also appear in b (0..1), on normalized text.
    content_only ignores short function words (el, es, del, за, е, ...)."""
    wa, wb = set(normalize(a).split()), set(normalize(b).split())
    if content_only:
        wa, wb = {w for w in wa if len(w) > 3}, {w for w in wb if len(w) > 3}
    return len(wa & wb) / len(wa) if wa else 0.0


# Scene scripts: ordered stages per scenario, recognised from the English
# meaning of phrases (works for every language). The conversation only moves
# forward: suggestions come from the current or next stage, never an earlier
# one ("how much is it?" after the waiter already gave the total).
SCENE_STAGES = {
    "restaurant": [
        ("arrive", ["table", "menu", "reserve", "how many", "for two", "for one", "people"]),
        # (no generic "i'd like"/"i want"/"for me": "I'd like to reserve a table" is not ordering food)
        ("order food", ["dish", "recommend", "special", "eat", "food", "menu of the day", "i'll have",
                        "salad", "soup", "paella", "meat", "fish", "chicken", "starter", "main course",
                        "appetizer", "cocido", "steak", "hake", "tapas", "bar snack", "olives", "ham",
                        "shrimp", "prawn", "squid", "mussel", "clam", "sausage", "potato", "cheese",
                        "bread", "beans", "tavche", "vegetarian", "lamb", "pork", "beef", "omelette",
                        "rare", "medium, ", "well done", "village meat"]),
        ("order drinks", ["drink", "water", "beer", "wine", "coffee", "tea", "juice", "glass", "round",
                          "sangría", "sangria", "pitcher", "rosé", "milk", "rakija", "brandy", "soda",
                          "lemonade", "caña"]),
        # (no "spicy": "meat, not spicy" is part of an order, not a comment on the meal)
        ("during the meal", ["delicious", "tasty", "another", "spoon", "fork", "salt", "dessert"]),
        ("bill", ["bill", "check", "how much", "cost", "price", "pay", "card", "cash", "denar", "euro"]),
        ("leave", ["keep the change", "thank", "goodbye", "see you", "here you are"]),
    ],
    "shopping": [
        ("browse", ["do you have", "looking for", "size", "colour", "color", "try on", "show me"]),
        ("price", ["how much", "cost", "price", "expensive", "cheap", "discount", "denar", "euro"]),
        ("decide", ["i'll take", "take it", "not interested", "too expensive", "something else"]),
        ("pay", ["pay", "card", "cash", "bag", "receipt", "change"]),
        ("leave", ["thank", "goodbye"]),
    ],
    "hotel": [
        ("arrive", ["room", "available", "reservation", "booking", "night", "single", "double"]),
        ("details", ["breakfast", "bathroom", "price", "how much", "per night", "wifi", "internet", "floor"]),
        ("check in", ["i'll take", "take it", "key", "passport", "sign", "stay"]),
        ("check out", ["check out", "bill", "pay", "card"]),
        ("leave", ["thank", "goodbye"]),
    ],
    "transport": [
        # ("where is/does", not bare "where": "Where are you from?" is not a ticket-desk line)
        ("ask", ["which bus", "which train", "when does", "leave", "depart", "platform", "station",
                 "where is", "where does", "where do"]),
        ("ticket", ["ticket", "one-way", "return", "first", "second", "class", "how much", "cost"]),
        ("pay", ["pay", "card", "cash", "here you are", "change"]),
        ("travel", ["get off", "arrive", "stop", "tell me when", "let me off", "luggage"]),
        ("leave", ["thank", "goodbye"]),
    ],
    "directions": [
        ("ask", ["where is", "how do i get", "how can i get", "excuse me", "looking for", "lead to"]),
        ("understand", ["left", "right", "straight", "far", "near", "close", "corner", "lights",
                        "again", "slowly", "don't understand"]),
        ("leave", ["thank", "goodbye"]),
    ],
    "emergency": [
        ("report", ["lost", "stolen", "thief", "help", "emergency", "sick", "not feeling", "problem", "hurt"]),
        ("details", ["wallet", "bag", "passport", "phone", "where", "when", "what happened", "colour", "color"]),
        ("action", ["police", "ambulance", "doctor", "hospital", "number", "report", "call", "pharmacy"]),
        ("leave", ["thank", "goodbye"]),
    ],
    "weather": [
        ("ask", ["weather", "hot", "cold", "warm", "rain", "sunny", "cloudy", "snow", "wind"]),
        ("talk", ["season", "spring", "summer", "autumn", "winter", "umbrella", "coat", "like best"]),
        ("leave", ["thank", "goodbye", "see you"]),
    ],
    "sightseeing": [
        ("ask", ["recommend", "visit", "see", "famous", "worth", "suggest", "where can i"]),
        ("details", ["how do i get", "how long", "far", "boat", "bus", "trip", "tour", "ticket", "price", "open"]),
        ("leave", ["thank", "goodbye"]),
    ],
}


# What the partner should do at each stage (fed to the model every turn).
STAFF_MOVES = {
    # Index = the step the TRAVELER just took; the move is the waiter's response to it.
    "restaurant": ["answer them (seat them / hand them the menu) and ask what they would like to eat",
                   "confirm the dish they ordered and ask what they would like to drink "
                   "(do NOT order food yourself; only recommend a dish if they asked for one)",
                   "confirm their drink and say you will bring it right away",
                   "check that everything is fine and offer dessert or coffee",
                   "tell them the total of the bill (a made-up price is fine)",
                   "thank them and say goodbye"],
    "shopping": ["ask what they are looking for and show them something",
                 "tell them the price (a made-up price is fine)",
                 "react to their decision; if they hesitate offer something else",
                 "take the payment and offer a bag",
                 "thank them and say goodbye"],
    "hotel": ["say whether a room is free and what kind",
              "answer their question about breakfast, bathroom or price",
              "ask for their passport and give them the key and room number",
              "tell them the total and take the payment",
              "thank them and say goodbye"],
    "transport": ["answer which train/bus and when it leaves (times only from the notes) and from which platform",
                  "sell the ticket: ask one-way or return, first or second class, and say the price",
                  "take the payment and give the ticket",
                  "tell them where to get off or when they arrive",
                  "wish them a good trip"],
    "directions": ["tell them the way using left, right, straight, corner, traffic lights",
                   "answer their follow-up (is it far? again, slowly?)",
                   "say goodbye and wish them a good day"],
    "emergency": ["ask calmly what happened",
                  "ask for details: what was lost or what hurts, where and when",
                  "tell them what to do next (fill in a report, call 192 police / 194 ambulance, see a doctor)",
                  "reassure them and say goodbye"],
    "weather": ["comment on today's weather using words from the notes",
                "talk about the seasons or say what to wear (umbrella, coat)",
                "say goodbye in a friendly way"],
    "sightseeing": ["recommend one place from the notes and say what is special about it",
                    "explain how to get there or how long it takes (no invented timetable times)",
                    "wish them a nice trip and say goodbye"],
}

# Lines that only fill time; after the opening they mean the scene is stuck.
GENERIC_LINES = {
    "macedonian": {"повелете", "се разбира", "да видам", "добар ден", "здраво", "фала", "благодарам",
                   "како можам да ви помогнам", "добар ден повелете", "повелете како можам да ви помогнам"},
    "spanish": {"en qué puedo ayudarle", "dígame", "sí dígame", "desde luego", "claro", "aquí tiene",
                "muy bien", "buenos días", "hola", "gracias", "a ver"},
}


def stage_hits(meaning, scenario):
    """Indexes of every scene stage a line touches (see stage_of)."""
    low = meaning.lower()
    return [i for i, (_, words) in enumerate(SCENE_STAGES.get(scenario, []))
            if any(re.search(rf"(?<!\w){re.escape(w)}", low) for w in words)]


def stage_of(meaning, scenario):
    """Index of the furthest scene stage a line touches, or None
    ("It was delicious. The bill, please." -> bill, not "during the meal")."""
    low = meaning.lower()
    # Keywords match at the start of a word ("drink" -> "drinks"), never inside
    # one ("eat" must not match "great", "tea" not "steak").
    hits = [i for i, (_, words) in enumerate(SCENE_STAGES.get(scenario, []))
            if any(re.search(rf"(?<!\w){re.escape(w)}", low) for w in words)]
    return max(hits) if hits else None


def _meaning_map(language, retriever):
    """normalized phrase -> English meaning, over the whole phrasebook (cached)."""
    cache = getattr(retriever, "_meaning_maps", {})
    if language not in cache:
        cache[language] = {normalize(p["text"]): p["meaning"] for c in retriever.chunks
                           if c["language"] == language for p in parse_phrases(c)}
        retriever._meaning_maps = cache
    return cache[language]


def scene_stage(history_en, scenario):
    """Furthest stage reached so far, from the English of every line said."""
    reached = [stage_of(x, scenario) for x in history_en]
    reached = [i for i in reached if i is not None]
    return max(reached) if reached else 0


_COMMANDS = ("go ", "turn ", "take ", "cross ", "continue", "get on", "get off", "keep ", "follow ",
             "wait", "come ", "drive ", "walk ", "pass ", "look ", "read ", "please go", "please wait",
             "please clean", "sign ", "fill ", "pay at", "let me off", "don't")

_ANSWER_STARTS = ("without", "with ", "yes", "no,", "no.", "sweet", "medium", "bitter",
                  "both", "neither", "either")

_CLOSING_WORDS = ("bill", "check", "receipt", "cheers", "another round", "one more",
                  "thank", "delicious", "finished", "goodbye", "that's all")


# Which notes sections a suggestion may come from, per scene (section titles are
# English in both languages). The transport partner sells bus tickets, so no
# driving/parking lines; the emergency notes file also holds small talk.
SECTION_EXCLUDE = {
    "transport": ("car", "driv", "road", "petrol", "gas station", "breakdown", "repair", "sign",
                  "airport"),
}
SECTION_INCLUDE = {
    "emergency": ("emergenc", "problem", "ill", "doctor", "pharmacy", "help", "goodbye", "feeling"),
}

CONTEXT_WEIGHT = 0.6   # how much the conversation so far counts vs. the partner's last line


def rank_answers(partner_line_en, phrases, retriever, context_en=""):
    """
    Traveler phrases ordered by how well they fit as a reply to the partner's
    line, comparing the English line with each phrase's English meaning
    ("What would you like to eat?" -> ordering phrases first).
    context_en: the conversation so far (and the scene), so an open line like
    "How can I help you?" still gets a reply that continues the same thread
    (size -> try it on, not "how much is a kilo").
    """
    if not phrases:
        return []
    vecs = retriever._encode([partner_line_en, context_en or partner_line_en]
                             + [p["meaning"] for p in phrases])
    scores = vecs[2:] @ vecs[0] + CONTEXT_WEIGHT * (vecs[2:] @ vecs[1])
    order = np.argsort(-scores)
    return [phrases[i] for i in order]


def has_blank(text):
    """Phrasebook templates with a gap to fill: "Сакам ...", "Una mesa para __"."""
    return bool(re.search(r"\.\.\.|…|__", text))


def _nearest(text, phrases, retriever, exclude=()):
    """Phrase most similar to text (by the retriever's embeddings)."""
    pool = [p for p in phrases if normalize(p["text"]) not in exclude] or phrases
    vecs = retriever._encode([text] + [p["text"] + " = " + p["meaning"] for p in pool])
    return pool[int(np.argmax(vecs[1:] @ vecs[0]))]


HELP_OFFERS = {
    "macedonian": {"повелете", "како можам да ви помогнам", "може ли да ви помогнам", "можам ли да ви помогнам",
                   "што можам да направам за вас", "добар ден повелете"},
    "spanish": {"en qué puedo ayudarle", "puedo ayudarle", "dígame", "sí dígame", "qué desea"},
}
_EN_OFFERS = ("may i help", "how can i help", "can i help", "what can i do for you", "please, come in")


def _drop_filler(reply, reply_en, language):
    """After the opening, remove 'May I help you?'-type sentences that the model
    tacks on in front of a real answer ("Може ли да ви помогнам? Одете право...")."""
    parts = [x for x in re.split(r"(?<=[?!.])\s+", reply.strip()) if x.strip()]
    keep = [x for x in parts if normalize(x) not in HELP_OFFERS.get(language, set())]
    if not keep or len(keep) == len(parts):
        return reply, reply_en
    en_parts = [x for x in re.split(r"(?<=[?!.])\s+", reply_en.strip()) if x.strip()]
    en_keep = [x for x in en_parts if not x.lower().startswith(_EN_OFFERS)] or en_parts
    return " ".join(keep), " ".join(en_keep)


ACKNOWLEDGE = {"spanish": ("Muy bien.", "Very good."), "macedonian": ("Одлично.", "Great.")}


def _sentences(text):
    return [x for x in re.split(r"(?<=[?!.])\s+", text.strip()) if x.strip()]


def _repair(data, s, last_learner, prev_reply, notes_text, retriever, traveler_questions):
    """Drop the sentences of a rejected reply that break a rule; keep the rest if
    it passes on its own. Returns a new data dict, or None.
    The English is trimmed the same way, so only when it has as many sentences."""
    parts, en_parts = _sentences(data.get("reply", "")), _sentences(data.get("reply_english", ""))
    if len(parts) < 2 or len(parts) != len(en_parts):
        return None

    def ok(text):
        found = check_reply(text, s, last_learner, prev_reply, notes_text, retriever, traveler_questions)
        return all(any(k in pr for k in _SOFT) for pr in found)

    keep = [i for i, x in enumerate(parts) if ok(x)]
    if not keep or len(keep) == len(parts):
        return None
    reply = " ".join(parts[i] for i in keep)
    reply_en = " ".join(en_parts[i] for i in keep)
    # The dropped sentence was usually the acknowledgement of an order; add a short one back.
    if last_learner and not last_learner.strip().endswith("?") and keep[0] != 0:
        ack, ack_en = ACKNOWLEDGE[s.language]
        reply, reply_en = f"{ack} {reply}", f"{ack_en} {reply_en}"
    if not ok(reply):
        return None
    return {**data, "reply": reply, "reply_english": reply_en}


MAX_RECAP_LINES = 12


def _recap(history, s, retriever, partner):
    """The conversation so far, with English, so the small model keeps track of
    what has been asked, ordered and answered (it reads English far better than
    Macedonian)."""
    lines = []
    for m in history[-MAX_RECAP_LINES:]:
        who = "Traveler" if m["role"] == "user" else f"You ({partner})"
        en = _english_of(m, s.language, retriever)
        lines.append(f"- {who}: {m['content']}" + (f'  = "{en}"' if en else ""))
    return "Conversation so far:\n" + "\n".join(lines)


def _lora_line(s, convo, history, partner, cfg, model_config):
    """Partner line from the fine-tuned model (plain text) + an English translation
    from the base model. Returns (data, stats) shaped like the JSON path."""
    sys_msg = (f"You are {partner} in a {LANGUAGES[s.language]} roleplay with a traveler. "
               f"Reply only in {LANGUAGES[s.language]}, one or two short sentences, as {partner}.")
    msgs = [{"role": "system", "content": sys_msg}]
    if not history:
        msgs.append({"role": "user", "content": "(The traveler arrives. Greet them and offer help.)"})
    for m in history:
        msgs.append({"role": m["role"], "content": m["content"]})
    # Keep any correction feedback from earlier attempts (last messages of convo).
    feedback = [m for m in convo[2:] if m["role"] == "user" and m["content"].startswith("That reply broke")]
    if feedback:
        msgs[-1] = {**msgs[-1], "content": msgs[-1]["content"] + "\n(" + feedback[-1]["content"] + ")"}
    stats = generate_response_with_stats(msgs, {**cfg, "num_predict": 120})
    reply = stats.pop("text").strip().split("\n")[0].strip()
    tr = generate_response_with_stats(
        [{"role": "user", "content": f"Translate this {LANGUAGES[s.language]} sentence into natural English. "
                                     f"Reply with the translation only.\n\n{reply}"}],
        {**(model_config or {}), "temperature": 0, "num_predict": 120})
    return {"reply": reply, "reply_english": tr["text"].strip(), "vocab_ids": [], "suggestion_id": 0,
            "tip": ""}, stats


def roleplay_turn(s: TutorSettings, history: list[dict], session: dict, trip: str,
                  retriever, model_config=None) -> dict:
    """
    One roleplay turn as the scenario partner (waiter, ticket seller ...).
    history: roleplay so far, oldest first: {"role": "assistant", "content":
    <partner line>} and {"role": "user", "content": <learner line>}. Empty
    history = the partner opens the conversation.
    Returns {"reply", "reply_english", "vocab", "suggestion", "tip",
             "attempts", "problems", "fallback", "stats"}.
    """
    chunks, phrases, last_learner = _context(s, history, session, retriever)
    notes_text = "\n\n".join(c["text"] for c in chunks)
    partner = ROLEPLAY_PARTNER[s.scenario]
    system = ROLEPLAY_JSON_TEMPLATE.format(
        partner=partner, language=LANGUAGES[s.language], scenario=SCENARIOS[s.scenario],
        trip=trip, script=" (" + SCRIPT[s.language] + ")",
        language_rules=language_rules(s), numbered=_numbered(phrases, partner),
        notes=notes_text[:4000],
    )
    convo = [{"role": "system", "content": system}]
    for m in history:
        if m["role"] == "user":
            convo.append({"role": "user", "content": f"(The traveler says:) {m['content']}"})
        else:
            convo.append({"role": "assistant", "content": m["content"]})
    stage_note = ""
    if s.scenario in SCENE_STAGES and history:
        # Tell the partner which step of the scene the traveler just took and the
        # matching staff move, so its line connects to the traveler's.
        meaning = _meaning_map(s.language, retriever).get(normalize(last_learner), last_learner)
        names = [n for n, _ in SCENE_STAGES[s.scenario]]
        hist_en = [_meaning_map(s.language, retriever).get(normalize(m["content"]), m["content"])
                   if m["role"] == "user" else m.get("english", "") for m in history]
        step = stage_of(meaning, s.scenario)
        if step is None:
            step = scene_stage(hist_en, s.scenario)
        stage_note = (f"\nThe traveler's last line means: \"{meaning}\". If it fits, a natural next "
                      f"move for you is to {STAFF_MOVES[s.scenario][step]}; but always respond first "
                      f"to what they actually said.")
    if not history:
        convo.append({"role": "user", "content": f"(The traveler arrives. As {partner}, greet them and offer help the way {partner} would, in one short line.)"})
    else:
        convo[-1]["content"] += (f"\n\n{_recap(history, s, retriever, partner)}"
                                 f"\n\n(Reply as {partner}, like a real person in this conversation: "
                                 f"answer what they just said, keeping everything above in mind. Never "
                                 f"ask again for something they already told you; refer back to it "
                                 f"when it helps (\"the salad\", \"your coffee\"). One or two short "
                                 f"sentences; no filler like 'here you are' or 'how can I help'. Do not "
                                 f"repeat their words.){stage_note}")

    prev_reply = next((m["content"] for m in reversed(history) if m["role"] == "assistant"), "")
    # Lines the traveler says (from the notes); the partner must not say them.
    # Only sentence-like lines from the notes' traveler-facing sections are used.
    traveler_questions = {normalize(p["text"]) for c in retriever.chunks if c["language"] == s.language
                          for p in parse_phrases(c)
                          if not is_partner_phrase(p) and p["text"].rstrip()[-1:] in "?.!"
                          and len(p["text"].split()) >= 3
                          # neutral replies either side could say ("Está muy rica." = It's
                          # very tasty) are not traveler-only lines
                          and not p["meaning"].lower().startswith(_NEUTRAL_STARTS)}
    data, stats, problems = {}, {}, []
    rejected = []     # replies that failed the checks (for evaluation/debugging)
    attempts = 0
    for attempts in range(1, MAX_ATTEMPTS + 1):
        # Retries run hotter: at the same temperature the model often returns the
        # exact same rejected reply again.
        cfg = {**model_cfg(s, model_config), "temperature": RETRY_TEMPERATURES[attempts - 1]}
        try:
            if s.lora:
                # Fine-tuned model: it writes ONLY the partner's line, as plain text
                # (how it was trained); the base model translates it. Vocabulary,
                # suggestion and tip are chosen by code below.
                data, stats = _lora_line(s, convo, history, partner, cfg, model_config)
            else:
                data, stats = _call_json(convo, ROLEPLAY_SCHEMA, cfg)
        except LLMError as e:
            if "invalid JSON" not in str(e):
                raise
            problems = ["invalid JSON"]
            continue
        problems = check_reply(data.get("reply", ""), s, last_learner, prev_reply, notes_text,
                               retriever, traveler_questions)
        if not problems:
            break
        rejected.append({"reply": data.get("reply", ""), "problems": problems})
        convo = convo + [
            {"role": "assistant", "content": json.dumps(data, ensure_ascii=False)},
            {"role": "user", "content": "That reply broke these rules: " + "; ".join(problems)
             + f". Write a new reply as {partner} that follows all the rules."},
        ]

    # Soft problems alone (filler, unfamiliar wording, a bit long) don't justify a
    # generic fallback line, which would break the flow of the scene.
    if problems and data.get("reply") and all(any(k in pr for k in _SOFT) for pr in problems):
        problems = []
    if problems and data.get("reply"):
        # Usually one sentence is bad and the rest is a good, on-topic answer
        # ("Muy bien, ... cocido para mí. ¿Qué va a tomar?"): keep the good part
        # rather than replace the whole turn with a generic "Of course."
        fixed = _repair(data, s, last_learner, prev_reply, notes_text, retriever, traveler_questions)
        if fixed:
            data, problems = fixed, []
    fallback = bool(problems)
    if fallback:
        reply, reply_en = _fallback_line(s.language, retriever, first_turn=not history,
                                         last_learner=last_learner, variety=s.variety,
                                         scenario=s.scenario)
    else:
        reply = fix_common_errors(data["reply"].strip(), s.language)
        reply_en = data.get("reply_english", "").strip()
        if history:
            reply, reply_en = _drop_filler(reply, reply_en, s.language)

    def pick(i):
        return phrases[i - 1] if isinstance(i, int) and 1 <= i <= len(phrases) else None

    # Vocabulary should teach something new: not what the traveler just said.
    said_now = {normalize(m["content"]) for m in history if m["role"] == "user"}
    fresh = [p for p in phrases if normalize(p["text"]) not in said_now
             and not any(_overlap(p["text"], x) >= 0.8 for x in said_now)] or phrases
    # Vocabulary help = phrasebook entries that actually appear in the partner's
    # line (so the chips explain what was just said), never unrelated phrases.
    n_rep = " " + normalize(reply) + " "

    def in_reply(p):   # the whole phrase literally occurs in the line
        key = normalize(p["text"])
        words = key.split()
        meaningful = len(words) >= 2 or (len(words) == 1 and len(key) >= 5)
        return meaningful and len(words) <= 6 and f" {key} " in n_rep
    all_lang = [p for c in retriever.chunks if c["language"] == s.language
                for p in (c.get("phrases") or parse_phrases(c))]
    pool = _dedupe([p for p in phrases + all_lang if in_reply(p)])
    # Chips show the plain meaning: "Taxi driver: Of course." -> "Of course."
    vocab = [{**p, "meaning": _drop_speaker(p["meaning"])}
             for p in sorted(pool, key=lambda p: -len(p["text"]))[:3]]
    said = {normalize(m["content"]) for m in history if m["role"] == "user"}
    # Suggestions come from this scenario's own notes and are never (near-)
    # repeats of what the traveler already said.
    from planner import SCENARIO_FILES
    own = {f"{s.language}/{f}" for f in SCENARIO_FILES[s.scenario]} | {u["source"] for u in session["units"]}
    # The whole scenario phrasebook, not just this turn's retrieved notes (that
    # pool was sometimes too small, leaving no good sentence to suggest).
    # Only the learner's variety: a Spain learner must not be offered
    # Venezuelan or Puerto Rican lines.
    allowed = accepted_labels(s.language, s.variety)
    scen_phrases = _dedupe([p for c in retriever.chunks if c["source"] in own
                            and c.get("country") in allowed
                            for p in (c.get("phrases") or parse_phrases(c))]) or phrases
    phrases_for_suggestion = [p for p in scen_phrases
                              if not any(_overlap(p["text"], x) >= 0.7 for x in said)] or scen_phrases
    asked_question = reply.strip().endswith("?")

    def sentence_like(p):   # something to SAY: complete, not a word, list or template
        txt = p["text"].strip()
        words = txt.split()
        greeting = ("greet" in p["section"].lower() or p["meaning"].lower().startswith(
            ("hello", "hi", "good day", "good morning", "good afternoon", "good evening",
             "good night", "goodbye", "bye", "welcome")))
        sign = txt.upper() == txt and any(ch.isalpha() for ch in txt)   # "DESPACHO DE BILLETES"
        # Instructions ("Go past the traffic lights", "Turn left") are what the
        # local/staff says, not the traveler.
        command = p["meaning"].lower().lstrip("(").startswith(_COMMANDS)
        # Answers to a specific partner question ("Without sugar. ..." answers "With
        # sugar or without?") only make sense right after that question.
        # (the partner's question must be about the same thing: "sugar" in both).
        first = re.split(r"(?<=[.!?])\s+", p["meaning"])[0]
        answer = p["meaning"].lower().startswith(_ANSWER_STARTS) and not (
            asked_question and _overlap(first, reply_en or "", content_only=True) > 0)
        return (" / " not in txt and not has_blank(txt) and not greeting and not sign and not command
                and not answer
                and "(" not in txt and txt[-1:] in "?.!" and len(words) >= 2)

    # Only complete traveler sentences; never fall back to word lists or greetings.
    # Taxi lines ("Déjeme ahí") make no sense at a ticket desk, and vice versa.
    def fits_partner(p):
        sec = p["section"].lower()
        if any(k in sec for k in SECTION_EXCLUDE.get(s.scenario, ())):
            return False
        if s.scenario in SECTION_INCLUDE and not any(k in sec for k in SECTION_INCLUDE[s.scenario]):
            return False
        taxi = "taxi" in sec or "такси" in sec
        return taxi == ("taxi" in partner.lower()) if s.scenario == "transport" else True
    traveler_phrases = [p for p in phrases_for_suggestion if not is_partner_phrase(p)
                        and sentence_like(p) and fits_partner(p)]
    # Closing lines (bill, cheers, thanks, another round) only after the
    # traveler has said a couple of things, so the scene moves in order.
    n_said = sum(m["role"] == "user" for m in history)
    if n_said < 2:
        opening_ok = [p for p in traveler_phrases
                      if not any(w in p["meaning"].lower() for w in _CLOSING_WORDS)]
        traveler_phrases = opening_ok or traveler_phrases
    excluded = said | {normalize(reply)}
    scene_done = False
    if s.scenario in SCENE_STAGES:
        # Scene script: the suggestion is the traveler's NEXT step. If the partner
        # just asked about a later step ("What would you like to drink?"), answer
        # that; otherwise move one step on from the traveler's last line.
        meaning_of = _meaning_map(s.language, retriever)
        last_step = stage_of(meaning_of.get(normalize(last_learner), last_learner), s.scenario) \
            if last_learner else -1
        last_step = -1 if last_step is None else last_step
        n_steps = len(SCENE_STAGES[s.scenario])
        # The partner moves the scene ahead only by ASKING about a later step
        # ("What would you like to drink?"); a statement like "Aquí tiene"
        # (here you are) must not jump to the goodbye.
        partner_step = stage_of(reply_en or "", s.scenario) if reply.strip().endswith("?") else None
        target = partner_step if partner_step is not None and partner_step > last_step else last_step + 1
        # A thank-you/goodbye ends the scene only once the scene is nearly over
        # (bill paid, directions given); "Gracias" for a table does not.
        earlier = [stage_of(meaning_of.get(normalize(m["content"]), m["content"]), s.scenario)
                   for m in history[:-1] if m["role"] == "user"]
        earlier_max = max([x for x in earlier if x is not None], default=-1)
        scene_done = last_step == n_steps - 1 and earlier_max >= n_steps - 2
        if last_step == n_steps - 1 and not scene_done:
            last_step, target = earlier_max, earlier_max + 1   # a polite "thanks" mid-scene: carry on
        target = min(max(target, 0), n_steps - 1)
        # Steps the traveler has already finished: no suggestion may go back to
        # one ("one village meat and a salad" after food AND drinks are ordered),
        # even if it also mentions a later step.
        user_steps = [stage_of(meaning_of.get(normalize(m["content"]), m["content"]), s.scenario) or 0
                      for m in history if m["role"] == "user"]
        if not scene_done:   # a polite mid-scene "thanks" doesn't finish the scene
            user_steps = [x for x in user_steps if x != n_steps - 1]
        reached = max([last_step] + user_steps)
        by_step = {}
        for p in traveler_phrases:
            hits = stage_hits(p["meaning"], s.scenario)
            if hits and min(hits) < reached:
                continue
            by_step.setdefault(stage_of(p["meaning"], s.scenario), []).append(p)
        # Short thank-yous/goodbyes ("Благодарам.", "Пријатно!") are fine for the last step.
        by_step.setdefault(n_steps - 1, []).extend(
            p for p in scen_phrases if not is_partner_phrase(p) and p["text"].rstrip()[-1:] in ".!"
            and stage_of(p["meaning"], s.scenario) == n_steps - 1 and p not in by_step[n_steps - 1]
            and normalize(p["text"]) not in said)
        # Last step: a plain thank-you/goodbye, not "I won't have anything, thanks".
        last = by_step.get(n_steps - 1, [])
        plain = [p for p in last if p["meaning"].lower().startswith(
            ("thank", "goodbye", "bye", "see you", "keep the change", "have a nice", "all the best"))]
        if plain:
            by_step[n_steps - 1] = plain
        # Never go back to an earlier step: if nothing fits from here on, the scene ends.
        # The nearest step from here that has lines, in order: mixing all later
        # steps let "Thank you" win when the waiter had just offered the menu.
        traveler_phrases = next((by_step[k] for k in range(target, n_steps) if by_step.get(k)), [])
        if not traveler_phrases:
            scene_done = True
    # The thread so far: the scene plus the last few lines in English. At the
    # start this is just the scene ("checking in at a hotel"), which favours
    # opening lines like "Do you have a room?" over "The AC doesn't work".
    thread = [SCENARIOS[s.scenario]] + [x for x in (_english_of(m, s.language, retriever)
                                                    for m in history[-4:]) if x]
    ranked = rank_answers(reply_en or reply, [p for p in traveler_phrases
                                              if normalize(p["text"]) not in excluded], retriever,
                          context_en=". ".join(thread))
    suggestion = pick(data.get("suggestion_id"))
    # Keep the model's pick only if it is among the best answers to this line.
    if fallback or not suggestion or suggestion not in ranked[:5]:
        suggestion = ranked[0] if ranked else None
    if s.scenario in SCENE_STAGES and scene_done:
        suggestion = None                           # the scene has ended

    tip = ""
    if last_learner:
        exact = next((p for c in retriever.chunks if c["language"] == s.language
                      for p in parse_phrases(c) if normalize(p["text"]) == normalize(last_learner)), None)
        if exact and has_blank(exact["text"]):
            example = suggestion["text"] if suggestion else ""
            tip = ("Finish the sentence: say what you want after it"
                   + (f", for example \"{example}\"." if example else "."))
        elif has_blank(last_learner):
            tip = "Your sentence isn't finished yet: replace the ... with what you want."
        elif exact:
            tip = f"Exactly right: \"{exact['text']}\" is straight from the phrasebook ({exact['meaning']})."
        else:
            tip = _clean_english(data.get("tip", ""), s.language, retriever) or \
                "Good try. Compare your sentence with the suggested phrase below."
    return {"reply": reply, "reply_english": reply_en, "vocab": vocab, "suggestion": suggestion,
            "tip": tip, "attempts": attempts, "problems": problems, "fallback": fallback,
            "rejected": rejected,
            "scene_done": bool(s.scenario in SCENE_STAGES and scene_done), "stats": stats}


# --- Chat roleplay -----------------------------------------------------------
# A normal chat: the model sees the whole conversation as alternating messages
# and writes the partner's line AND the traveler's suggested replies in one go,
# so both sides follow the same thread. The phrasebook is a reference (offered
# in the prompt, and matching suggestions are marked "from the phrasebook"),
# not a cage. Only the checks that matter are kept: script and wrong-language
# forms, role slips (the partner talking like the customer), echoing, and
# timetable times that are not in the notes.

from prompts import CHAT_SCHEMA, CHAT_TEMPLATE, PRACTICE_FACTS, SCENE_GOALS   # noqa: E402

CHAT_MAX_SUGGESTIONS = 3
CHAT_PHRASES = 40          # phrasebook lines offered in the prompt
CHAT_NOTES_CHARS = 4000
# Staff offers that must never be suggested to the learner ("¿Qué desea?").
STAFF_OFFER = {"spanish": r"qu[eé]\s+(?:desea|desean|va\s+a\s+tomar|van\s+a\s+tomar|quiere|le\s+pongo)|en\s+qu[eé]\s+puedo",
               "macedonian": r"што\s+(?:сакате|ќе\s+пиете|ќе\s+јадете|ќе\s+порачате)|како\s+можам\s+да\s+ви"}

# Suggestion repeats: question openings that are answered once ("how much ...").
REPEAT_QUESTIONS = ("how much", "what time", "when does", "when is", "where is", "where are",
                    "how far", "how long", "how do i get", "how can i get", "what is the price")
SNAP_TO_PHRASEBOOK = 0.85  # meaning similarity at which a suggestion becomes the phrasebook phrase
SAME_MEANING = 0.9         # embedding similarity above which a suggestion repeats a past line
MIN_SCENE_TURNS = 3        # learner turns before the model may end a scene on its own
WRAP_UP_AFTER = 6          # traveler turns before the partner is nudged to wrap the scene up
_GOODBYE_EN = re.compile(r"\b(goodbye|bye|see you|have a (?:nice|good|great|lovely)|enjoy your|take care)\b", re.I)
_BYE_EN = re.compile(r"\b(goodbye|bye|see you)\b", re.I)     # the traveler is leaving
_THANKS_EN = re.compile(r"\b(thank|thanks|goodbye|bye|see you)\b", re.I)


def _scene_notes(s, session, retriever):
    """The notes for the whole scene, retrieved once from the scenario (not from
    the latest line), so the background stays the same every turn."""
    found = retriever.search(SCENARIOS[s.scenario], language=s.language, k=TOP_K, country=s.variety)
    chunks, seen = [], set()
    for c in list(session["units"]) + found:
        if c["text"] not in seen:
            seen.add(c["text"])
            chunks.append(c)
    phrases = _dedupe([p for c in chunks for p in (c.get("phrases") or parse_phrases(c))])
    return chunks, phrases[:CHAT_PHRASES]


CHAT_TRAVELER_PHRASES = 120   # checked traveler lines the model can choose suggestions from


def _traveler_phrases(s, retriever):
    """Complete sentences the TRAVELER says, from this scene's own notes (learner's
    variety): offered to the model to copy exactly as suggestions."""
    from planner import SCENARIO_FILES
    own = {f"{s.language}/{f}" for f in SCENARIO_FILES[s.scenario]}
    allowed = accepted_labels(s.language, s.variety)
    out = []
    for c in retriever.chunks:
        if c["source"] in own and c.get("country") in allowed:
            for p in parse_phrases(c):
                t = p["text"].strip()
                if (not is_partner_phrase(p) and not has_blank(t) and " / " not in t and "(" not in t
                        and t[-1:] in ".?!" and len(t.split()) >= 2):
                    out.append(p)
    return _dedupe(out)[:CHAT_TRAVELER_PHRASES]


def _phrasebook_index(language, retriever):
    """normalized phrase text -> phrase, over the whole phrasebook (cached)."""
    cache = getattr(retriever, "_pb_index", {})
    if language not in cache:
        cache[language] = {normalize(p["text"]): p for c in retriever.chunks
                           if c["language"] == language for p in parse_phrases(c)
                           if not has_blank(p["text"])}
        retriever._pb_index = cache
    return cache[language]


def phrasebook_match(text, language, retriever):
    """The phrasebook entry this sentence is (almost) exactly, or None.
    Almost = at least 85% of the words shared both ways (punctuation and case ignored)."""
    index = _phrasebook_index(language, retriever)
    key = normalize(text)
    if key in index:
        return index[key]
    words = set(key.split())
    if len(words) < 2:
        return None
    for k, p in index.items():
        kw = set(k.split())
        if kw and len(words & kw) / len(words) >= 0.85 and len(words & kw) / len(kw) >= 0.85:
            return p
    return None


def sentence_problems(text, language):
    """Language checks for a suggested traveler line (no role rules)."""
    t = text.strip()
    if not t or has_blank(t) or len(t) > 140:
        return ["empty, unfinished or too long"]
    letters = _CYR.findall(t) + _LAT.findall(t)
    target = _CYR if language == "macedonian" else _LAT
    if letters and len(target.findall(t)) / len(letters) < (0.95 if language == "macedonian" else 0.99):
        return ["wrong script"]
    low = t.lower()
    if language == "macedonian":
        if re.search(r"\w*[яыэёщъьйюћђ]\w*", low) or re.search(r"[A-Za-z]{2,}", t):
            return ["not Macedonian letters"]
    if any(re.search(rf"(?<!\w){pat}(?!\w)", low) for pat in FORBIDDEN[language]):
        return ["wrong-language form"]
    return []


def _chat_history(history):
    """History as a normal chat. The partner's past turns are shown as the same
    JSON the model writes now (reply + translation), so the format is consistent."""
    msgs = [{"role": "user", "content": "(The traveler walks up to you.)"}]
    for m in history:
        if m["role"] == "user":
            msgs.append({"role": "user", "content": m["content"]})
        else:
            msgs.append({"role": "assistant", "content": json.dumps(
                {"reply": m["content"], "reply_english": m.get("english", "")}, ensure_ascii=False)})
    return msgs


def _needs_translation(english, text, language):
    """The model sometimes copies the sentence into its "english" field
    ("Quisiera el cocido por favor. · Quisiera el cocido por favor.")."""
    e = (english or "").strip()
    if not e or normalize(e) == normalize(text):
        return True
    if language == "macedonian" and _CYR.search(e):
        return True
    return _overlap(e, text) >= 0.6      # mostly the same words = not translated


def _translate(texts, language, model_config=None):
    """English translations for a few target-language sentences, in one call."""
    if not texts:
        return []
    numbered = "\n".join(f"{i}. {t}" for i, t in enumerate(texts, 1))
    msgs = [{"role": "user", "content":
             f"Translate each {LANGUAGES[language]} sentence into short, natural English. "
             f"Return JSON {{\"translations\": [...]}} with one translation per sentence, in order.\n\n{numbered}"}]
    schema = {"type": "object", "properties": {"translations": {"type": "array", "items": {"type": "string"}}},
              "required": ["translations"]}
    try:
        data, _ = _call_json(msgs, schema, {**(model_config or {}), "temperature": 0})
        out = [str(x).strip() for x in data.get("translations", [])]
    except LLMError:
        out = []
    return out + [""] * (len(texts) - len(out))


def _learner_english(text, s, retriever, model_config=None):
    """English for a traveler line: the phrasebook meaning (exact, or a shortened
    phrasebook entry such as "Колку е соба за еден?" for "...за еден/двајца?"),
    else a one-off model translation, cached. Needed so repeats of typed or
    shortened lines can be recognised."""
    en = _english_of({"role": "user", "content": text}, s.language, retriever)
    if en:
        return en
    words = set(normalize(text).split())
    if len(words) >= 2:
        for key, p in _phrasebook_index(s.language, retriever).items():
            kw = set(key.split())
            if words <= kw and len(kw - words) <= 2:
                return re.sub(r"\s*\([^)]*\)", "", _drop_speaker(p["meaning"])).strip()
    cache = getattr(retriever, "_learner_en", {})
    if text not in cache:
        cache[text] = (_translate([text], s.language, model_config) or [""])[0]
        retriever._learner_en = cache
    return cache[text]


def chat_turn(s: TutorSettings, history: list[dict], session: dict, trip: str,
              retriever, model_config=None) -> dict:
    """
    One turn of the chat roleplay. Same input and output shape as
    roleplay_turn(), plus "suggestions": up to 3 dicts
    {"text", "meaning", "phrasebook": bool}; "suggestion" is the first one.
    """
    chunks, phrases = _scene_notes(s, session, retriever)
    # Made-up practice timetable for scenes whose notes have no times (labelled
    # as practice data in the prompt); the time check accepts these times too.
    facts = PRACTICE_FACTS.get((s.language, s.scenario), "")
    notes_text = "\n\n".join(([facts] if facts else []) + [c["text"] for c in chunks])
    partner = ROLEPLAY_PARTNER[s.scenario]
    traveler = _traveler_phrases(s, retriever)
    staff = [p for p in phrases if is_partner_phrase(p)][:20]
    phrase_block = ("Traveler phrases (checked; use their exact wording when one says what you mean):\n"
                    + "\n".join(f"- {p['text']} = {p['meaning']}" for p in traveler)
                    + (f"\n\n{partner.capitalize()} phrases (checked wording for your own lines):\n"
                       + "\n".join(f"- {p['text']} = {_drop_speaker(p['meaning'])}" for p in staff)
                       if staff else ""))
    system = CHAT_TEMPLATE.format(
        partner=partner, language=LANGUAGES[s.language], scenario=SCENARIOS[s.scenario],
        goal=SCENE_GOALS[s.scenario],
        script=" (" + SCRIPT[s.language] + ")", language_rules=language_rules(s),
        phrases=phrase_block, notes=notes_text[:CHAT_NOTES_CHARS])
    convo = [{"role": "system", "content": system}] + _chat_history(history)
    n_user = sum(m["role"] == "user" for m in history)
    if n_user >= WRAP_UP_AFTER:
        # Long scenes drift: nudge toward a natural ending once the goal is likely done.
        convo[-1] = {**convo[-1], "content": convo[-1]["content"]
                     + "\n(Note for you: the scene has gone on a while. If the traveler's goal is reached, "
                       "start wrapping up, and include a thank-you/goodbye among the suggestions.)"}
    last_learner = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
    prev_reply = next((m["content"] for m in reversed(history) if m["role"] == "assistant"), "")
    said = {normalize(m["content"]) for m in history if m["role"] == "user"}
    said_en = [_learner_english(m["content"], s, retriever, model_cfg(s, model_config))
               for m in history if m["role"] == "user"]

    def suggestion(p):
        # The phrasebook meaning, without speaker labels or usage notes in brackets.
        meaning = re.sub(r"\s*\([^)]*\)", "", _drop_speaker(p["meaning"])).strip() or p["meaning"]
        return {"text": p["text"], "meaning": meaning, "phrasebook": True, "section": "", "speaker": ""}

    said_vecs = retriever._encode([e for e in said_en if e]) if any(said_en) else None

    def usable(p, out):
        key = normalize(p["text"])
        if key in said or any(key == normalize(o["text"]) for o in out):
            return False
        meaning = p["meaning"].lower()
        # The same kind of question again ("How much is it per night?" after
        # "How much is a room for one?"): it has just been answered.
        if any(meaning.startswith(q) and any(e.lower().startswith(q) for e in said_en if e)
               for q in REPEAT_QUESTIONS):
            return False
        # Something the traveler already did ("Can I have the menu?" after
        # "Do you have a menu?"): same content words, or nearly the same meaning.
        if any(_overlap(p["meaning"], e, content_only=True) >= 0.75 for e in said_en if e):
            return False
        if said_vecs is not None and len(said_vecs):
            return float((said_vecs @ retriever._encode([p["meaning"]])[0]).max()) < SAME_MEANING
        return True

    trav_vecs = retriever._encode([p["meaning"] for p in traveler]) if traveler else None

    def clean_suggestions(items, reply):
        """For each line the model wrote: the phrasebook phrase with the same meaning
        if there is one (marked ✓), else the model's sentence if it passes the
        checks (own_suggestion), else nothing."""
        out = []
        for it in items or []:
            text = str(it.get("text", "")).strip()
            english = str(it.get("english", "")).strip()
            # Swapped fields: English in "text", the target language in "english".
            if wrong_language(text, s.language) and english and not wrong_language(english, s.language):
                text, english = english, text
            match = phrasebook_match(text, s.language, retriever) if text else None
            pick = match if match and not is_partner_phrase(match) else None
            if pick is None and trav_vecs is not None and len(trav_vecs) and english:
                sims = trav_vecs @ retriever._encode([english])[0]
                best = int(sims.argmax())
                if float(sims[best]) >= SNAP_TO_PHRASEBOOK:
                    pick = traveler[best]
            if match and is_partner_phrase(match):
                continue                       # a waiter/clerk line, never the traveler's
            if text and _overlap(text, reply) >= 0.8:
                continue                       # the partner's own line copied back
            if pick is not None:
                item = suggestion(pick)
                if usable(item, out):
                    out.append(item)
            else:
                own = own_suggestion(text, english, out)
                if own:
                    out.append(own)
            if len(out) == CHAT_MAX_SUGGESTIONS:
                break
        return out

    def own_suggestion(text, english, out):
        """The model's own sentence, used only if it passes the language and role
        checks; shown without the phrasebook mark."""
        text = fix_common_errors(str(text or "").strip(), s.language)
        if s.language == "spanish" and text.startswith("¿") and "?" not in text:
            text = text.rstrip(".!") + "?"     # "¿Quisiera una habitación." -> a real question
        english = str(english or "").strip()
        if not text or sentence_problems(text, s.language) or wrong_language(text, s.language):
            return None
        if re.search(STAFF_OFFER[s.language], normalize(text)):
            return None                              # a waiter/clerk question, not the traveler's
        if _needs_translation(english, text, s.language):
            english = (_translate([text], s.language, model_cfg(s, model_config)) or [""])[0]
        if not english:
            return None
        item = {"text": text, "meaning": english, "phrasebook": False, "section": "", "speaker": ""}
        return item if usable(item, out) else None

    def farewell_suggestions():
        """Checked thank-you/goodbye lines for closing the scene: from this scene's
        notes first, then from the whole phrasebook (basics hold most goodbyes)."""
        out = []
        allowed = accepted_labels(s.language, s.variety)
        everywhere = [p for c in retriever.chunks if c["language"] == s.language
                      and c.get("country") in allowed for p in parse_phrases(c)
                      if not is_partner_phrase(p) and not has_blank(p["text"]) and " / " not in p["text"]]
        for p in traveler + everywhere:
            if re.match(r"(thank|thanks|goodbye|bye|see you|have a nice)", p["meaning"].lower()):
                item = suggestion(p)
                if normalize(item["text"]) not in said and all(item["text"] != o["text"] for o in out):
                    out.append(item)
            if len(out) == CHAT_MAX_SUGGESTIONS:
                break
        return out

    def ranked_suggestions(reply_en):
        """If the model picked nothing usable: the phrases that fit its line best."""
        thread = ". ".join([SCENARIOS[s.scenario]] + [e for e in said_en[-3:] if e])
        out = []
        for p in rank_answers(reply_en, traveler, retriever, context_en=thread):
            if usable(p, out):
                out.append(suggestion(p))
            if len(out) == CHAT_MAX_SUGGESTIONS:
                break
        return out

    # Exact traveler lines from the notes ("Задржете го кусурот." = keep the change):
    # the partner must not say them (cheap, exact-copy check only).
    traveler_lines = {normalize(p["text"]) for c in retriever.chunks if c["language"] == s.language
                      for p in parse_phrases(c)
                      if not is_partner_phrase(p) and p["text"].rstrip()[-1:] in "?.!"
                      and len(p["text"].split()) >= 3
                      and not p["meaning"].lower().startswith(_NEUTRAL_STARTS)}
    schema = CHAT_SCHEMA
    data, stats, problems, rejected, suggestions = {}, {}, [], [], []
    attempts = 0
    for attempts in range(1, MAX_ATTEMPTS + 1):
        cfg = {**model_cfg(s, model_config), "temperature": RETRY_TEMPERATURES[attempts - 1]}
        try:
            data, stats = _call_json(convo, schema, cfg)
        except LLMError as e:
            if "invalid JSON" not in str(e):
                raise
            problems = ["invalid JSON"]
            continue
        reply = data.get("reply", "")
        problems = check_reply(reply, s, last_learner, prev_reply, notes_text, None, traveler_lines)
        problems = [p for p in problems if not any(k in p for k in _SOFT)]
        if "(" in reply or "[" in reply:
            problems.append("contains a note or stage direction in brackets; write only what you say")
        if not problems:
            break
        rejected.append({"reply": reply, "problems": problems})
        convo = convo + [
            {"role": "assistant", "content": json.dumps(data, ensure_ascii=False)},
            {"role": "user", "content": "That answer broke these rules: " + "; ".join(problems)
             + f". Write it again as {partner}, following all the rules."},
        ]

    reply_problems = list(problems)
    if reply_problems and data.get("reply"):
        fixed = _repair(data, s, last_learner, prev_reply, notes_text, retriever, traveler_lines)
        if fixed:
            data, reply_problems = fixed, []
    fallback = bool(reply_problems) or not data.get("reply")
    if fallback:
        reply, reply_en = _fallback_line(s.language, retriever, first_turn=not history,
                                         last_learner=last_learner, variety=s.variety,
                                         scenario=s.scenario)
    else:
        reply = fix_common_errors(data["reply"].strip(), s.language)
        reply_en = data.get("reply_english", "").strip()

    learner_en = (_learner_english(last_learner, s, retriever, model_cfg(s, model_config))
                  if last_learner else "")
    partner_bye = bool(_GOODBYE_EN.search(reply_en))
    learner_bye = bool(_BYE_EN.search(learner_en))
    # The scene ends when both have said goodbye/thanks, or once the traveler has
    # said goodbye (the partner's reply is the last line).
    farewell = (partner_bye and bool(_THANKS_EN.search(learner_en))) or learner_bye
    # The model's "scene over" only counts after a few exchanges (small models end
    # scenes after one line); a real goodbye exchange always ends it.
    scene_done = bool(history) and not fallback and (
        (bool(data.get("scene_over")) and n_user >= MIN_SCENE_TURNS) or farewell)
    if scene_done:
        suggestions = []
    elif partner_bye:
        # The partner said goodbye: only a thank-you/goodbye fits now
        # (never "Do you take credit cards?" after "Have a nice day!").
        suggestions = farewell_suggestions()
        scene_done = not suggestions            # nothing left to say: the scene is over
    else:
        # The model's picks only (fewer good suggestions beat padding with weak
        # ones), plus its own sentence when no listed phrase fits (e.g. ordering
        # for the rest of a group); code-ranked phrases only if nothing is usable.
        suggestions = ([] if fallback else clean_suggestions(data.get("suggestions"), reply)) \
            or ranked_suggestions(reply_en or reply)

    # Fill in translations the model left out or copied from the sentence.
    todo = [sg for sg in suggestions if _needs_translation(sg["meaning"], sg["text"], s.language)]
    reply_todo = not fallback and _needs_translation(reply_en, reply, s.language)
    if todo or reply_todo:
        texts = ([reply] if reply_todo else []) + [sg["text"] for sg in todo]
        tr = _translate(texts, s.language, model_cfg(s, model_config))
        if reply_todo:
            reply_en, tr = tr[0] or reply_en, tr[1:]
        for sg, t in zip(todo, tr):
            sg["meaning"] = t or sg["meaning"]

    # Vocabulary chips: phrasebook entries that literally occur in the partner's line.
    n_rep = " " + normalize(reply) + " "

    def in_reply(p):
        key = normalize(p["text"])
        words = key.split()
        return (len(words) >= 2 or len(key) >= 5) and len(words) <= 6 and f" {key} " in n_rep
    pool = _dedupe([p for p in _phrasebook_index(s.language, retriever).values() if in_reply(p)])
    # Chips show the plain meaning: "Taxi driver: Of course." -> "Of course."
    vocab = [{**p, "meaning": _drop_speaker(p["meaning"])}
             for p in sorted(pool, key=lambda p: -len(p["text"]))[:3]]

    tip = ""
    if last_learner:
        tip = _clean_english(data.get("tip", ""), s.language, retriever)
        if not tip and phrasebook_match(last_learner, s.language, retriever):
            tip = "Good: that's a phrasebook sentence."
    return {"reply": reply, "reply_english": reply_en, "vocab": vocab,
            "suggestions": suggestions, "suggestion": suggestions[0] if suggestions else None,
            "tip": tip, "attempts": attempts, "problems": problems, "fallback": fallback,
            "rejected": rejected, "scene_done": scene_done, "stats": stats}
