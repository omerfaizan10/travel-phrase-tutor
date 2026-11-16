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


def _dedupe(phrases):
    seen, out = set(), []
    for p in phrases:
        key = normalize(p["text"])
        if key and key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _numbered(phrases):
    return "\n".join(f"{i}. {p['text']} = {p['meaning']}" for i, p in enumerate(phrases, 1))


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

def _context(s, history, session, retriever):
    """Notes and numbered phrasebook for this roleplay turn."""
    last_learner = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
    query = last_learner or SCENARIOS[s.scenario]
    found = retriever.search(query, language=s.language, k=TOP_K, country=s.variety)
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
                   "that's good", "great", "fine")

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
# Broken "where" combinations in any scene ("Каде ви помагам?", "Каде ја ...").
BAD_WHERE = {"macedonian": r"(?<!\w)каде\s+(?:ви|ја|го|ти|желате|можам)(?!\w)", "spanish": r"$^"}


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
            re.search(I_WANT[s.language], normalize(x)) for x in sentences):
        problems.append("speaks as the customer ('I want / I'd like'); you are the staff: "
                        "offer, answer or ask what the traveler would like")
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
    for h, m in _TIME.findall(reply):
        if f"{int(h)}.{m}" not in notes_times:
            problems.append(f"time {h}.{m} is not in the timetable notes")
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


def _fallback_line(language, retriever, first_turn, last_learner="", variety=None):
    """A safe in-character line taken from the notes (of the learner's variety)."""
    if last_learner.strip().endswith("?"):
        return QUESTION_FALLBACK[language]
    # Mid-conversation, a neutral acknowledgement keeps the scene going better
    # than "I don't understand".
    wanted = ["may i help", "here you are", "welcome"] if first_turn else \
        ["of course", "right away", "let me see", "ok, agreed", "very well", "don't understand"]
    allowed = accepted_labels(language, variety)
    phrases = [p for c in retriever.chunks if c["language"] == language
               and c.get("country") in allowed for p in parse_phrases(c)]
    staff_first = [p for p in phrases if is_partner_phrase(p)] + [p for p in phrases if not is_partner_phrase(p)]
    for w in wanted:
        for p in staff_first:
            if w in p["meaning"].lower() and "change" not in p["meaning"].lower():
                return p["text"], p["meaning"]
    return (phrases[0]["text"], phrases[0]["meaning"]) if phrases else ("...", "...")


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
        ("order food", ["i'd like", "i would like", "i want", "order", "dish", "recommend", "special",
                        "for me", "eat", "salad", "soup", "paella", "meat", "fish", "chicken",
                        "starter", "main course"]),
        ("order drinks", ["drink", "water", "beer", "wine", "coffee", "tea", "juice", "glass", "round",
                          "rakija", "brandy", "lemonade", "soda"]),
        ("during the meal", ["delicious", "tasty", "another", "spoon", "fork", "spicy", "salt", "dessert"]),
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
        ("ask", ["which bus", "which train", "when does", "leave", "depart", "platform", "station", "where"]),
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
    "restaurant": ["greet them, offer a table and give them the menu",
                   "if they asked for a recommendation, recommend one dish from the notes; if they "
                   "ordered, repeat their order back as the waiter and ask what they would like to "
                   "drink, like 'Great, one <their dish>. And to drink?' (do not ask what they "
                   "want to eat again, and do not order anything yourself)",
                   "repeat their order back and say it will come soon (do not ask about food or "
                   "drinks again)",
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


def stage_of(meaning, scenario):
    """Index of the furthest scene stage a line touches, or None
    ("It was delicious. The bill, please." -> bill, not "during the meal")."""
    low = meaning.lower()
    hits = [i for i, (_, words) in enumerate(SCENE_STAGES.get(scenario, []))
            if any(w in low for w in words)]
    return max(hits) if hits else None


def _meaning_map(language, retriever):
    """normalized phrase -> English meaning, over the whole phrasebook (cached)."""
    cache = getattr(retriever, "_meaning_maps", {})
    if language not in cache:
        cache[language] = {normalize(p["text"]): p["meaning"] for c in retriever.chunks
                           if c["language"] == language for p in parse_phrases(c)}
        retriever._meaning_maps = cache
    return cache[language]


def guess_meaning(text, language, retriever):
    """English meaning of a learner line, for scene tracking.
    Exact phrasebook match first; for a line the learner wrote themselves, join the
    meanings of the notes phrases it contains ("шопска салата") and of the closest
    full phrase ("Би сакал тавче гравче и една шопска салата" for "Би сакал шопска
    салата и ракија"), so the scene step is still recognised."""
    meanings = _meaning_map(language, retriever)
    key = normalize(text)
    if key in meanings:
        return meanings[key]
    padded = f" {key} "
    parts = [m for p, m in meanings.items()
             if p and (len(p.split()) >= 2 or len(p) >= 5) and f" {p} " in padded]
    content = {w for w in key.split() if len(w) > 3}
    best = max(meanings, key=lambda p: _overlap(text, p, content_only=True), default="")
    shared = content & {w for w in best.split() if len(w) > 3}
    if len(shared) >= 2 and _overlap(text, best, content_only=True) >= 0.5:
        parts.append(meanings[best])
    return " / ".join(parts) or text


def scene_stage(history_en, scenario):
    """Furthest stage reached so far, from the English of every line said."""
    reached = [stage_of(x, scenario) for x in history_en]
    reached = [i for i in reached if i is not None]
    return max(reached) if reached else 0


_COMMANDS = ("go ", "turn ", "take ", "cross ", "continue", "get on", "get off", "keep ", "follow ",
             "wait", "come ", "drive ", "walk ", "pass ", "look ", "read ", "please go", "please wait",
             "please clean", "sign ", "fill ", "pay at", "let me off", "don't")

_CLOSING_WORDS = ("bill", "check", "receipt", "cheers", "another round", "one more",
                  "thank", "delicious", "finished", "goodbye", "that's all")


def rank_answers(partner_line_en, phrases, retriever):
    """
    Traveler phrases ordered by how well they fit as a reply to the partner's
    line, comparing the English line with each phrase's English meaning
    ("What would you like to eat?" -> ordering phrases first).
    """
    if not phrases:
        return []
    vecs = retriever._encode([partner_line_en] + [p["meaning"] for p in phrases])
    order = np.argsort(-(vecs[1:] @ vecs[0]))
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
        language_rules=language_rules(s), numbered=_numbered(phrases),
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
        exact = _meaning_map(s.language, retriever).get(normalize(last_learner))
        guessed = guess_meaning(last_learner, s.language, retriever)
        names = [n for n, _ in SCENE_STAGES[s.scenario]]
        hist_en = [guess_meaning(m["content"], s.language, retriever)
                   if m["role"] == "user" else m.get("english", "") for m in history]
        # The scene only moves forward: never fall back to an earlier step than the
        # conversation already reached (e.g. offering the menu after the order).
        reached = scene_stage(hist_en[:-1], s.scenario)
        step = stage_of(guessed, s.scenario)
        step = reached if step is None else max(step, reached)
        # Only a phrasebook meaning is passed on as a translation; a guessed one is
        # just for tracking, and the model reads the Macedonian line itself.
        meaning = exact or last_learner
        stage_note = (f"\n(The traveler's last line means: \"{meaning}\" - step: {names[step]}. "
                      f"Respond to exactly that. Then, as {partner}: {STAFF_MOVES[s.scenario][step]}. "
                      f"Keep it to one or two short sentences. Do not just say a filler like "
                      f"'here you are' or 'how can I help'.)")
    if not history:
        convo.append({"role": "user", "content": f"(The traveler arrives. As {partner}, greet them and ask how you can help, e.g. what they would like.)"})
    else:
        convo[-1]["content"] += f"\n(Reply as {partner}. Answer it; do not repeat it.){stage_note}"

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
        convo = convo + [
            {"role": "assistant", "content": json.dumps(data, ensure_ascii=False)},
            {"role": "user", "content": "That reply broke these rules: " + "; ".join(problems)
             + f". Write a new reply as {partner} that follows all the rules."},
        ]

    # Soft problems alone (filler, unfamiliar wording, a bit long) don't justify a
    # generic fallback line, which would break the flow of the scene.
    if problems and data.get("reply") and all(any(k in pr for k in _SOFT) for pr in problems):
        problems = []
    fallback = bool(problems)
    if fallback:
        reply, reply_en = _fallback_line(s.language, retriever, first_turn=not history,
                                         last_learner=last_learner, variety=s.variety)
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
    vocab = sorted(pool, key=lambda p: -len(p["text"]))[:3]
    said = {normalize(m["content"]) for m in history if m["role"] == "user"}
    # Suggestions come from this scenario's own notes and are never (near-)
    # repeats of what the traveler already said.
    from planner import SCENARIO_FILES
    own = {f"{s.language}/{f}" for f in SCENARIO_FILES[s.scenario]} | {u["source"] for u in session["units"]}
    # The whole scenario phrasebook, not just this turn's retrieved notes (that
    # pool was sometimes too small, leaving no good sentence to suggest).
    scen_phrases = _dedupe([p for c in retriever.chunks if c["source"] in own
                            for p in (c.get("phrases") or parse_phrases(c))]) or phrases
    phrases_for_suggestion = [p for p in scen_phrases
                              if not any(_overlap(p["text"], x) >= 0.7 for x in said)] or scen_phrases
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
        return (" / " not in txt and not has_blank(txt) and not greeting and not sign and not command
                and "(" not in txt and txt[-1:] in "?.!" and len(words) >= 2)

    # Only complete traveler sentences; never fall back to word lists or greetings.
    # Taxi lines ("Déjeme ahí") make no sense at a ticket desk, and vice versa.
    def fits_partner(p):
        taxi = "taxi" in p["section"].lower() or "такси" in p["section"].lower()
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
        last_step = stage_of(guess_meaning(last_learner, s.language, retriever), s.scenario) \
            if last_learner else -1
        last_step = -1 if last_step is None else last_step
        partner_step = stage_of(reply_en or "", s.scenario)
        n_steps = len(SCENE_STAGES[s.scenario])
        target = partner_step if partner_step is not None and partner_step > last_step else last_step + 1
        scene_done = last_step == n_steps - 1      # traveler already said goodbye/thanks
        target = min(max(target, 0), n_steps - 1)
        by_step = {}
        for p in traveler_phrases:
            by_step.setdefault(stage_of(p["meaning"], s.scenario), []).append(p)
        # Short thank-yous/goodbyes ("Благодарам.", "Пријатно!") are fine for the last step.
        by_step.setdefault(n_steps - 1, []).extend(
            p for p in scen_phrases if not is_partner_phrase(p) and p["text"].rstrip()[-1:] in ".!"
            and stage_of(p["meaning"], s.scenario) == n_steps - 1 and p not in by_step[n_steps - 1]
            and normalize(p["text"]) not in said)
        # Never go back to an earlier step: if nothing fits from here on, the scene ends.
        traveler_phrases = (by_step.get(target) or by_step.get(min(target + 1, n_steps - 1))
                            or [p for k, v in by_step.items() if k is not None and k >= target for p in v])
        if not traveler_phrases:
            scene_done = True
    ranked = rank_answers(reply_en or reply, [p for p in traveler_phrases
                                              if normalize(p["text"]) not in excluded], retriever)
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
            "scene_done": bool(s.scenario in SCENE_STAGES and scene_done), "stats": stats}
