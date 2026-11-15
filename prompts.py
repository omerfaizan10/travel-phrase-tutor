"""
prompts.py — turns the learner's settings + conversation into chat messages.

Three modes, so we can compare them in the evaluation:
    baseline  no system prompt: the raw model
    prompt    tutor system prompt (prompt engineering only)
    rag       tutor system prompt + passages from retriever.py

Every mode starts from the same opening request (opening_request()), so
differences in output come from the mode, not from the question.
"""

from dataclasses import dataclass

from retriever import DEFAULT_VARIETY, LATIN_AMERICA, VARIETIES

LANGUAGES = {"spanish": "Spanish", "macedonian": "Macedonian"}
SCENARIOS = {
    "restaurant": "ordering food and drinks at a restaurant",
    "directions": "asking for and understanding directions",
    "shopping": "shopping, asking prices and paying (including changing money)",
    "hotel": "checking in and asking questions at a hotel",
    "transport": "taking buses, trains and taxis and buying tickets",
    "weather": "small talk about the weather and seasons",
    "sightseeing": "asking for suggestions, planning trips and visiting famous places",
    "emergency": "getting help in an emergency (lost, sick, robbed)",
}
ROLEPLAY_PARTNER = {
    "restaurant": "the waiter",
    "directions": "a local passer-by",
    "shopping": "a shop assistant",
    "hotel": "the hotel receptionist",
    "transport": "a ticket seller at the bus station",
    "weather": "a friendly local making small talk",
    "sightseeing": "a local who recommends places to visit",
    "emergency": "a police officer",
}
MODES = ("baseline", "prompt", "rag")
TASKS = ("lesson", "roleplay")


@dataclass
class TutorSettings:
    language: str = "spanish"     # key of LANGUAGES
    scenario: str = "restaurant"  # key of SCENARIOS
    task: str = "lesson"          # "lesson" or "roleplay"
    mode: str = "rag"             # "baseline", "prompt" or "rag"
    goal: str = "a vacation"
    timeline: str = "2 weeks"
    lora: bool = False            # use the fine-tuned model (tutor-lora) instead of the base model
    # Variety of the language for THIS scenario ("ES", "PR", "419", ...), i.e. the
    # "used" value from planner.resolve_variety(); "" = the language's default.
    country: str = ""

    def __post_init__(self):
        for name, value, allowed in [
            ("language", self.language, LANGUAGES),
            ("scenario", self.scenario, SCENARIOS),
            ("task", self.task, TASKS),
            ("mode", self.mode, MODES),
        ]:
            if value not in allowed:
                raise ValueError(f"{name}={value!r}; use one of {list(allowed)}")
        allowed = VARIETIES if self.language == "spanish" else {DEFAULT_VARIETY[self.language]}
        if self.country and self.country not in allowed:
            raise ValueError(f"country={self.country!r}; use one of {sorted(allowed)}")

    @property
    def variety(self) -> str:
        """The variety in use: the chosen one, or the language's default."""
        return self.country or DEFAULT_VARIETY[self.language]


SYSTEM_TEMPLATE = """You are a patient, encouraging {language} tutor for an English-speaking traveler.

Traveler's goal: {goal}
Time until the trip: {timeline}
Scenario: {scenario}

Rules:
- Teach only what is useful for this scenario. If the trip is soon, teach fewer, more essential phrases.
- Write every {language} phrase {script}, then a simple English pronunciation guide in brackets, then the English meaning.
- Use natural, everyday {language} that locals actually say, not textbook-formal language.
- Keep replies short and easy to scan.
- Write all introductions, explanations and tips in English. The only {language} in your reply should be the phrases you are teaching (and, in a roleplay, your in-character lines).
- Never invent real-world facts such as train or bus times, prices, opening hours or what is at a real address; if the notes don't say, say you're not sure. In a roleplay you may make up simple, harmless details for the scene (e.g. 'go straight, then left', 'this shirt is 500 denars').
{language_rules}
{task_rules}"""

# Mistakes the model actually made in testing, so we warn it explicitly.
LANGUAGE_RULES = {
    "spanish": (
        "- Use Spain Spanish and polite usted with customers and strangers: "
        "'¿En qué puedo ayudarle?' (never '¿Cómo puede ayudarme?', which means 'how can you help me'), "
        "'¿Qué desea?', '¿Qué va a tomar?'.\n"
        "- Spain words: la carta (menu), el billete (ticket), el móvil (phone), la cuenta (bill), camarero (waiter).\n"
        "- dónde means WHERE and is only for places; to ask what someone wants use qué: "
        "'¿Qué va a tomar?', '¿Qué desea comer?'.\n"
    ),
    "macedonian": (
        "- Macedonian is NOT Serbian, Bulgarian or Russian. Use these Macedonian forms: "
        "Добар ден (never 'Добро дене' or 'Добар дан'), автобус (not аутобус), "
        "веднаш (not веднагу), потоа (not потом), каде (not где), е (not је), "
        "но (not али), благодарам or фала (not хвала or спасибо), леб (not хлеб), "
        "извинете (not извините), што (not что), прашање (not питање).\n"
        "- Watch similar words: мени = menu, месо = meat; сметка = bill.\n"
        "- каде means WHERE and is only for places. To ask what someone wants use Што: "
        "'Што сакате за јадење?', 'Што сакате за пиење?', 'Што ќе порачате?'. To offer help say "
        "'Повелете' or 'Како можам да ви помогнам?' (never 'Каде ви помагам?').\n"
    ),
}

# Spanish rules for the variety in use (LANGUAGE_RULES["spanish"] is Spain's).
# Words listed for a country all appear in that country's sections of the notes.
LATIN_AMERICA_RULES = (
    "- Use Latin American Spanish: ustedes for 'you all' (never vosotros), and polite usted "
    "with customers and strangers: '¿En qué puedo ayudarle?', '¿Qué desea?'.\n"
    "- Latin American words: el menú (menu), el boleto or el pasaje (ticket), el celular (phone), "
    "el carro or el auto (car), mesero (waiter), la cuadra (block). For buses and taxis say tomar, "
    "never coger.\n"
    "- dónde means WHERE and is only for places; to ask what someone wants use qué.\n"
)
COUNTRY_RULES = {
    "AR": "- Argentina: vos for informal 'you' (vos tenés, vos querés), usted with staff. "
          "Words: el colectivo (city bus), el subte (metro), el mozo (waiter), la heladera (fridge).\n",
    "UY": "- Uruguay: vos for informal 'you' (vos tenés), usted with staff. "
          "Words: el ómnibus (bus), el mozo (waiter), el boleto (ticket).\n",
    "PY": "- Paraguay: vos for informal 'you' (vos tenés), usted with staff. "
          "Words: el micro (bus), el mozo (waiter), el vuelto (change).\n",
    "CL": "- Chile: la micro (city bus), el auto (car).\n",
    "MX": "- Mexico: el camión (city bus), el platillo (dish), platicar (to chat).\n",
    "PR": "- Puerto Rico: la guagua (bus), el carro público (shared van between towns), "
          "A la orden (you're welcome).\n",
    "CU": "- Cuba: el ciclón (hurricane), el guajiro (country person).\n",
    "DO": "- Dominican Republic: el sector (neighbourhood), el malecón (seafront).\n",
    "CO": "- Colombia: la plata (money), chévere (great), tengo gripa (I have a cold).\n",
    "PE": "- Peru: la combi (minibus).\n",
    "VE": "- Venezuela: la cotufa (popcorn).\n",
}


def language_rules(s: "TutorSettings") -> str:
    """Language rules for the variety in use, so a Spain fallback gets Spain rules."""
    if s.language == "spanish" and (s.variety == "419" or s.variety in LATIN_AMERICA):
        return LATIN_AMERICA_RULES + COUNTRY_RULES.get(s.variety, "")
    return LANGUAGE_RULES[s.language]


SCRIPT = {
    "spanish": "in Spanish",
    "macedonian": "in Macedonian Cyrillic (never Serbian or Bulgarian forms)",
}

TASK_RULES = {
    "lesson": (
        "Task: give a short lesson with 5-8 key phrases for the scenario, one "
        "cultural tip, and end with a 2-question practice quiz. Do not reveal "
        "the quiz answers until the learner replies; then check their answers."
    ),
    "roleplay": (
        "Task: run a roleplay. You play {partner}; the learner plays the "
        "traveler. You speak ONLY as {partner}, never as the traveler: you "
        "answer the traveler's questions and serve them, you do not ask the "
        "traveler's questions yourself. Every reply must ANSWER or RESPOND to "
        "what the learner just said (if they ask when a train leaves, tell them "
        "the time; if they ask for a room, say whether you have one). Never "
        "repeat or copy the learner's message. Your first turn is one short greeting "
        "in character (as {partner} would greet a customer or stranger). "
        "Stay in character and speak only {language}, one short turn "
        "at a time, then wait for the learner. After each turn add one line "
        "starting with 'Tip:' in English about the learner's last message. "
        "If it was correct, just praise it. Only suggest a correction if you "
        "are sure, and only using a phrase from the reference notes. Skip the "
        "Tip on your first turn, before the learner has replied."
    ),
}

RAG_TEMPLATE = {
    "lesson": """

Reference notes from our curated material. Teach ONLY {language} phrases that appear in these notes, copied exactly. If the notes have fewer phrases than the task asks for, teach fewer. If the learner asks for something the notes don't cover, say you don't have a checked phrase for it yet.

{passages}""",
    "roleplay": """

Reference notes from our curated material. Use them for correct {language} wording and for facts (times, platforms, prices, dishes, places). Many notes are phrases a traveler would say: those are the LEARNER's lines, so do not say them yourself. You are {partner}: answer the learner's last message the way {partner} would, using facts from these notes; never repeat the learner's words back.

{passages}""",
}


def opening_request(s: TutorSettings) -> str:
    """The first user message of a session (same in every mode)."""
    language = LANGUAGES[s.language]
    scenario = SCENARIOS[s.scenario]
    if s.task == "lesson":
        return (
            f"About my trip: {s.goal.rstrip('.')}. I have {s.timeline} to prepare. "
            f"Teach me the {language} I need for {scenario}."
        )
    return (
        f"About my trip: {s.goal.rstrip('.')}. I have {s.timeline} to prepare. "
        f"Let's practice {scenario} in {language} with a roleplay. "
        f"I am the traveler; you are {ROLEPLAY_PARTNER[s.scenario]}. "
        f"Greet me in one short line as {ROLEPLAY_PARTNER[s.scenario]}, then I will speak."
    )


def system_prompt(s: TutorSettings, passages=None) -> str:
    language = LANGUAGES[s.language]
    text = SYSTEM_TEMPLATE.format(
        language=language,
        goal=s.goal,
        timeline=s.timeline,
        scenario=SCENARIOS[s.scenario],
        script=SCRIPT[s.language],
        language_rules=language_rules(s),
        task_rules=TASK_RULES[s.task].format(
            partner=ROLEPLAY_PARTNER[s.scenario], language=language
        ),
    )
    if passages:
        numbered = "\n\n".join(
            f"[{i}] ({p['source']}" + (f", {p['country']}" if p.get("country") else "") + f")\n{p['text']}"
            for i, p in enumerate(passages, 1)
        )
        text += RAG_TEMPLATE[s.task].format(
            language=language, partner=ROLEPLAY_PARTNER[s.scenario], passages=numbered
        )
    return text


def build_messages(s: TutorSettings, history, passages=None) -> list[dict]:
    """
    history: the conversation so far, oldest first, starting with
             opening_request(); the last item is the learner's newest message.
    Returns the message list to pass to llm.generate_response().
    """
    if s.mode == "baseline":
        return list(history)
    history = list(history)
    if s.task == "roleplay" and len(history) > 1:
        # Small models follow the most recent instruction best, so label the
        # learner's newest line and restate the role. Only the copy sent to the
        # model changes; the app still shows what the learner typed.
        partner = ROLEPLAY_PARTNER[s.scenario]
        history[-1] = {
            "role": "user",
            "content": (
                f'(The traveler says:) {history[-1]["content"]}\n'
                f"(Reply as {partner}: answer this in {LANGUAGES[s.language]}, "
                "do not repeat it. Then the Tip line.)"
            ),
        }
    return [
        {"role": "system", "content": system_prompt(s, passages if s.mode == "rag" else None)},
        *history,
    ]


# ---------------------------------------------------------------------------
# Full ("rag") pipeline: structured prompts. The model returns JSON that picks
# phrases by number from the curated notes; the code renders the phrases, so
# they are always copied exactly from the notes.
# ---------------------------------------------------------------------------

LESSON_PICK_TEMPLATE = """You are a {language} tutor planning today's lesson for an English-speaking traveler.

Traveler's trip: {trip}
Today's topic: {topic}

Below are numbered {language} phrases from our checked phrasebook. Choose the {n_min}-{n_max} phrases that are MOST useful for this traveler's goal today, in the order you would teach them (simplest and most useful first). Use only these numbers.

Also write:
- "intro": one or two friendly sentences in English saying what today's lesson covers and why it helps with the trip.
- "tip": one short practical or cultural tip in English, based only on the phrases and notes below. Do not write any {language} words except ones that appear below.

Phrases:
{numbered}

Background notes:
{notes}"""

ROLEPLAY_JSON_TEMPLATE = """You are {partner} in a {language} roleplay with an English-speaking traveler who is learning {language}. Scenario: {scenario}. Traveler's trip: {trip}.

Rules for "reply" (your line, in character):
- Write ONLY in {language}{script}. One or two short, simple sentences a real {partner} would say.
- You are {partner}, never the traveler. Answer or respond to what the traveler just said. Never repeat the traveler's words back.
- Never ask where a place is (the traveler asks YOU that), and never ask a question and then answer it yourself: say your line, then wait for the traveler.
- If the traveler asks for a recommendation, a specialty, a price or a direction, ANSWER it concretely (e.g. name a dish, place or item from the notes). Do not dodge with another question.
- Typical moves for staff: if asked for the bill, say the total (a made-up price is fine, e.g. "Тоа е 600 денари." / "Son 25 euros."); if thanked, reply "you're welcome" ("Нема зошто." / "De nada."); if greeted, greet back and offer help.
- When the traveler orders or answers you, first acknowledge it in your own words (e.g. "Muy bien, una paella." / "Одлично, една шопска салата."), then move the scene on (drinks, dessert, the bill, the next step). Never ignore what they just said.
- Prefer wording from the phrasebook and notes below.
- Never invent real-world facts such as train or bus times, opening hours or what is at a real address: use only times from the notes. If the traveler asks for a time that is NOT in the notes, stay in character and tell them to check the departures board or the timetable (in {language}). Small made-up details for the scene are fine (a price in a shop, "go straight, then left").
- If the traveler asks about something outside your job (e.g. a plane or a train at a bus-station ticket desk, a hotel room from a waiter), politely say in {language} that you can't help with that here, then steer back to your own job.
{language_rules}
Also return:
- "reply_english": an English translation of your reply.
- "vocab_ids": 2-3 numbers of phrasebook entries that help the traveler understand your reply or answer it.
- "suggestion_id": the number of ONE phrasebook entry the TRAVELER could say next that directly ANSWERS your reply (if you ask what they want to eat, a phrase that orders food; if you ask how many people, a phrase with a number). It must be a complete sentence, not a template with "..." or "__".
- "tip": one short sentence in English about the traveler's last message: praise it if it was right; if it had a mistake, point them to a phrasebook phrase. Empty string if the traveler has not spoken yet.

Phrasebook (numbered):
{numbered}

Notes:
{notes}"""

LESSON_SCHEMA = {
    "type": "object",
    "properties": {
        "intro": {"type": "string"},
        "phrase_ids": {"type": "array", "items": {"type": "integer"}},
        "tip": {"type": "string"},
    },
    "required": ["intro", "phrase_ids", "tip"],
}

ROLEPLAY_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
        "reply_english": {"type": "string"},
        "vocab_ids": {"type": "array", "items": {"type": "integer"}},
        "suggestion_id": {"type": "integer"},
        "tip": {"type": "string"},
    },
    "required": ["reply", "reply_english", "vocab_ids", "suggestion_id", "tip"],
}
