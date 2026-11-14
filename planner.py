"""
planner.py: from "describe your trip" to a day-by-day lesson plan.

    parse_trip("I'm going to Skopje in 10 days and want to order food")
        -> {"language": "macedonian", "days": 10, "goals": ["restaurant"], ...}
    build_plan(language, goals, days, retriever)
        -> [{"day": 1, "title": "...", "units": [chunk, ...], "scenario": "..."}, ...]

Both steps are plain code (no model call), so the plan is predictable and
always built from the curated notes. The learner can correct anything that
was detected wrongly in the app before the plan is built.
"""

import re

from retriever import parse_phrases

# --- Trip parsing -----------------------------------------------------------

LANGUAGE_HINTS = {
    "spanish": ["spain", "spanish", "madrid", "barcelona", "seville", "sevilla",
                "valencia", "granada", "malaga", "málaga", "bilbao", "ibiza",
                "mallorca", "canary", "tenerife", "castellano"],
    "macedonian": ["macedonia", "macedonian", "skopje", "ohrid", "bitola",
                   "prilep", "struga", "kumanovo", "tetovo", "gevgelija",
                   "veles", "matka", "balkan"],
}

# Order matters only for display; a description can match several goals.
GOAL_HINTS = {
    "restaurant": ["restaurant", "food", "eat", "eating", "eat out", "order", "menu", "dinner",
                   "lunch", "breakfast", "cafe", "café", "coffee", "drink",
                   "bar", "tapas", "dish", "waiter"],
    "directions": ["direction", "way around", "get around", "lost", "find my way",
                   "where is", "navigate", "street", "walk"],
    "shopping": ["shop", "buy", "market", "souvenir", "store", "price",
                 "money", "exchange", "currency", "pharmacy", "bargain"],
    "hotel": ["hotel", "room", "check in", "check-in", "stay", "hostel",
              "accommodation", "airbnb", "reception"],
    "transport": ["train", "bus", "taxi", "metro", "ticket", "station",
                  "travel between", "get to", "railway", "rail", "car"],
    "weather": ["weather", "small talk", "chat", "talk to locals", "season"],
    "sightseeing": ["sightseeing", "museum", "visit", "tour", "famous",
                    "see the", "recommend", "suggestion", "places", "sights",
                    "church", "monastery", "lake"],
    "emergency": ["emergency", "doctor", "hospital", "police", "sick",
                  "help", "safety", "robbed"],
}
DEFAULT_GOALS = ["restaurant", "directions"]  # the proposal's core scenarios

NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "a few": 3,
    "few": 3, "a couple of": 2, "couple of": 2, "several": 4,
}
UNIT_DAYS = {"day": 1, "week": 7, "fortnight": 14, "month": 30}
DEFAULT_DAYS = 14
MAX_DAYS = 60


def parse_trip(text: str) -> dict:
    """
    Read an English trip description. Returns
        {"language": str | None, "days": int, "goals": [scenario, ...],
         "timeline": str, "found": {"language": bool, "days": bool, "goals": bool}}
    "found" tells the app which values were detected and which are defaults,
    so it can ask the learner to confirm them.
    """
    low = " " + text.lower() + " "

    def has(hint):  # whole words only ("bar" must not match "Barcelona"), plurals ok
        return re.search(rf"(?<!\w){re.escape(hint)}(?:s|es)?(?!\w)", low) is not None

    scores = {lang: sum(has(h) for h in hints) for lang, hints in LANGUAGE_HINTS.items()}
    best = max(scores, key=scores.get)
    language = best if scores[best] and list(scores.values()).count(scores[best]) == 1 else None

    goals = [g for g, hints in GOAL_HINTS.items() if any(has(h) for h in hints)]

    days, timeline = _parse_days(low)
    return {
        "language": language,
        "days": days or DEFAULT_DAYS,
        "goals": goals or list(DEFAULT_GOALS),
        "timeline": timeline or f"{DEFAULT_DAYS} days",
        "found": {"language": language is not None, "days": days is not None,
                  "goals": bool(goals)},
    }


def _parse_days(low: str):
    if re.search(r"\b(tomorrow|tonight)\b", low):
        return 1, "tomorrow"
    if "next week" in low:
        return 7, "next week"
    if "next month" in low:
        return 30, "next month"
    if re.search(r"\b(this|next) weekend\b", low):
        return 3, "this weekend"
    num = r"(\d+|" + "|".join(sorted(map(re.escape, NUMBER_WORDS), key=len, reverse=True)) + r")"
    m = re.search(num + r"[\s-]+(day|week|fortnight|month)s?\b", low)
    if m:
        n = int(m.group(1)) if m.group(1).isdigit() else NUMBER_WORDS[m.group(1)]
        days = min(max(n * UNIT_DAYS[m.group(2)], 1), MAX_DAYS)
        return days, m.group(0).strip()
    return None, None


# --- Lesson plan -------------------------------------------------------------

# Which note files teach each scenario (same names for every language).
SCENARIO_FILES = {
    "restaurant": ["restaurant.md"],
    "directions": ["directions.md"],
    "shopping": ["shopping.md"],
    "hotel": ["hotel_and_time.md"],
    "transport": ["transport.md", "trains.md"],
    "weather": ["weather.md"],
    "sightseeing": ["sightseeing.md"],
    "emergency": ["everyday_conversation.md"],
}
BASICS_FILE = "basics.md"
BASICS_PRIORITY = ["greet", "polite", "courtes"]   # day 1 = greetings + polite words

# Teaching order inside each scenario: units whose title contains an earlier
# keyword come first; anything unmatched keeps file order at the end. Titles
# are English in every language's notes, so this works for both languages.
UNIT_PRIORITY = {
    "restaurant": ["table", "conversation step", "waiter ask", "ordering food", "ordering drink", "paying",
                   "bill", "attention", "dietary", "dialogue", "dishes", "asking about"],
    "directions": ["asking where", "understanding the answer", "don't understand",
                   "more ways", "direction", "taxi", "dialogue", "position", "places"],
    "shopping": ["asking about", "deciding", "money", "paying", "shops"],
    "hotel": ["finding", "reception", "time", "month"],
    "transport": ["train", "ticket", "station", "timetable", "bus", "metro", "taxi", "car", "sign"],
    "weather": ["hace", "asking", "describing", "forecast", "season"],
    "sightseeing": ["planning", "useful sightseeing", "tour", "famous"],
    "emergency": ["emergenc", "asking for help", "question"],
}
MIN_PHRASES = 2           # a unit must have at least this many phrases to teach
MAX_UNITS_PER_DAY = 3


def teachable_units(retriever, language: str, filename: str) -> list[dict]:
    """Chunks of one notes file that contain enough phrases, in file order."""
    src = f"{language}/{filename}"
    units = []
    for c in retriever.chunks:
        if c["source"] == src:
            phrases = parse_phrases(c)
            if len(phrases) >= MIN_PHRASES:
                heading = next((l[3:] for l in c["text"].splitlines() if l.startswith("## ")), filename)
                # Learner-facing title: drop source tags like "(Kramer)" or "(Libro Libre)".
                title = re.sub(r"\s*\([^)]*\)", "", heading).strip() or heading
                units.append({**c, "phrases": phrases, "title": title})
    return units


def _ordered(units: list[dict], keywords: list[str]) -> list[dict]:
    def rank(u):
        title = u["title"].lower()
        return next((i for i, k in enumerate(keywords) if k in title), len(keywords))
    return sorted(units, key=rank)  # sorted() is stable: ties keep file order


def build_plan(language: str, goals: list[str], days: int, retriever) -> list[dict]:
    """
    Spread the units for the chosen goals over the available days:
      day 1 starts with basics; goals follow in the order given; the last
      session is a review + roleplay rehearsal. With few days, the most
      useful units (the first ones in each goal's files) are kept.
    Returns sessions: {"day", "title", "units", "scenario"}.
    """
    n_basics = 1 if days <= 2 else 2
    basics = _ordered(teachable_units(retriever, language, BASICS_FILE), BASICS_PRIORITY)[:n_basics]
    per_goal = {g: _ordered([u for f in SCENARIO_FILES[g] for u in teachable_units(retriever, language, f)],
                            UNIT_PRIORITY.get(g, []))
                for g in goals}
    if not any(per_goal.values()):
        return []

    study_days = max(days - 1, 1) if days > 1 else 1   # keep the last day for review
    # Interleave goals so a short plan still touches every goal early.
    queue, i = [], 0
    while any(i < len(v) for v in per_goal.values()):
        for g, v in per_goal.items():
            if i < len(v):
                queue.append((g, v[i]))
        i += 1
    capacity = study_days * MAX_UNITS_PER_DAY
    queue = queue[: max(capacity - len(basics), 2 * len(goals))]

    items = [("basics", u) for u in basics] + queue
    n_sessions = min(study_days, len(items))
    sessions = []
    for k in range(n_sessions):
        chunk = items[k * len(items) // n_sessions:(k + 1) * len(items) // n_sessions]
        scenario = next((g for g, _ in chunk if g != "basics"), goals[0])
        day = 1 + (k * study_days) // n_sessions
        sessions.append({
            "day": day,
            "title": " + ".join(u["title"] for _, u in chunk),
            "units": [u for _, u in chunk],
            "scenario": scenario,
        })
    if days > 1:
        sessions.append({
            "day": days,
            "title": "Review and roleplay rehearsal: " + ", ".join(goals),
            "units": [u for s in sessions for u in s["units"]],
            "scenario": goals[0],
            "review": True,
        })
    return sessions
