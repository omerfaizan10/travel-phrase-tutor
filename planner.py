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

from __future__ import annotations

import re

from retriever import DEFAULT_VARIETY, LATIN_AMERICA, VARIETIES, accepted_labels, parse_phrases

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


def parse_timeline(text: str) -> int | None:
    """Parse a questionnaire answer without guessing or silently capping it.

    Return None when the learner needs to clarify. The caller confirms the
    resulting number of days with the learner before building a plan.
    """
    low = text.lower().strip()
    if low.isdigit():
        return int(low) if int(low) > 0 else None
    # Reject zero/negative durations rather than turning them into one day.
    if re.search(r"(?<!\w)-\s*\d|\b(?:0|zero)\s+(?:days?|weeks?|months?)\b", low):
        return None
    if re.search(r"\b(?:this|next) weekend\b", low):
        return None  # actual days depend on today's date; ask for a duration
    # Reuse the existing vocabulary, but do not cap long answers at 60.
    number = r"(\d+|" + "|".join(sorted(map(re.escape, NUMBER_WORDS), key=len, reverse=True)) + r")"
    match = re.search(r"(?<!\w)" + number + r"[\s-]+(day|week|fortnight|month)s?\b", low)
    if match:
        value = int(match.group(1)) if match.group(1).isdigit() else NUMBER_WORDS[match.group(1)]
        return value * UNIT_DAYS[match.group(2)] if value > 0 else None
    days, _ = _parse_days(low)
    return days


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


# --- Which variety of the language each topic is taught in -------------------
# Rule: a topic uses the learner's chosen variety when the notes have at least
# one teachable section labeled with exactly that variety for the topic;
# otherwise it falls back to the default variety (Spain for Spanish), which has
# notes for every topic. The learner can switch a topic to any variety that has
# notes for it.

def topic_files(topic: str) -> list[str]:
    return [BASICS_FILE] if topic == "basics" else SCENARIO_FILES[topic]


def varieties_for_topic(retriever, language: str, topic: str) -> list[str]:
    """Varieties with their own teachable notes for this topic: default first,
    then Latin America in general, then countries by name."""
    coverage = retriever.coverage(language, MIN_PHRASES)
    have = {code for f in topic_files(topic) for code, n in coverage.get(f, {}).items() if n}
    default = DEFAULT_VARIETY.get(language)
    order = [default, "419"] + sorted(LATIN_AMERICA, key=lambda c: VARIETIES[c])
    return [c for c in order if c in have] + sorted(have - set(order) - {"ALL"})


def resolve_variety(retriever, language: str, requested: str | None, topic: str) -> dict:
    """
    {"requested": code, "used": code, "fallback": bool, "alternatives": [codes]}.
    fallback=True means the requested variety has no notes for this topic, so
    the default variety is used instead.
    """
    default = DEFAULT_VARIETY.get(language)
    requested = requested or default
    available = varieties_for_topic(retriever, language, topic)
    used = requested if requested in available else default
    return {"requested": requested, "used": used,
            "fallback": used != requested, "alternatives": available}


def plan_varieties(retriever, language: str, requested: str | None, topics: list[str]) -> dict:
    """resolve_variety() for every topic: {topic: {...}}."""
    return {t: resolve_variety(retriever, language, requested, t) for t in topics}


def teachable_units(retriever, language: str, filename: str, variety: str | None = None) -> list[dict]:
    """
    Chunks of one notes file that contain enough phrases. With a variety, only
    sections labeled for it (accepted_labels), the variety's own sections first;
    file order otherwise.
    """
    src = f"{language}/{filename}"
    allowed = accepted_labels(language, variety) if variety else None
    units = []
    for c in retriever.chunks:
        if c["source"] == src and (allowed is None or c.get("country") in allowed):
            phrases = parse_phrases(c)
            if len(phrases) >= MIN_PHRASES:
                heading = next((l[3:] for l in c["text"].splitlines() if l.startswith("## ")), filename)
                # Learner-facing title: drop source tags like "(Kramer)" or "(Libro Libre)".
                title = re.sub(r"\s*\([^)]*\)", "", heading).strip() or heading
                units.append({**c, "phrases": phrases, "title": title})
    if allowed:
        units.sort(key=lambda u: allowed.index(u.get("country")))   # stable: keeps file order
    return units


def _ordered(units: list[dict], keywords: list[str]) -> list[dict]:
    def rank(u):
        title = u["title"].lower()
        return next((i for i, k in enumerate(keywords) if k in title), len(keywords))
    return sorted(units, key=rank)  # sorted() is stable: ties keep file order


def build_plan(language: str, goals: list[str], days: int, retriever,
               variety: str | None = None) -> list[dict]:
    """
    Spread the units for the chosen goals over the available days:
      day 1 starts with basics; goals follow in the order given; the last
      session is a review + roleplay rehearsal. With few days, the most
      useful units (the first ones in each goal's files) are kept.
    variety: the learner's variety ("ES", "PR", ...); each topic falls back to
      the default variety when it has no notes for it (resolve_variety).
    Returns sessions: {"day", "title", "units", "scenario", "variety"}.
    """
    used = {t: v["used"] for t, v in plan_varieties(retriever, language, variety, ["basics"] + goals).items()}
    n_basics = 1 if days <= 2 else 2
    basics = _ordered(teachable_units(retriever, language, BASICS_FILE, used["basics"]),
                      BASICS_PRIORITY)[:n_basics]
    per_goal = {g: _ordered([u for f in SCENARIO_FILES[g] for u in teachable_units(retriever, language, f, used[g])],
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
            "variety": used[scenario],
        })
    if days > 1:
        sessions.append({
            "day": days,
            "title": "Review and roleplay rehearsal: " + ", ".join(goals),
            "units": [u for s in sessions for u in s["units"]],
            "scenario": goals[0],
            "variety": used[goals[0]],
            "review": True,
        })
    return sessions


# --- Availability-based calendar (no language-model calls) -------------------
from datetime import date, timedelta
from math import floor

WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
STAGE_NAMES = {1: "Essential words and expressions", 2: "Supported sentence building",
               3: "Scenario practice and feedback"}


def _allocate_minutes(total, weights, minimum=0):
    """Round weighted shares to whole minutes; preserve the exact total.

    A minimum protects topic coverage. When a quota is below one minute,
    coverage takes precedence over exact proportions.
    """
    if total < minimum * len(weights):
        raise ValueError("Not enough practice time to cover all selected topics in each stage.")
    quotas = [total * weight / sum(weights) for weight in weights]
    result = [max(minimum, floor(q)) for q in quotas]
    while sum(result) < total:
        i = max(range(len(result)), key=lambda j: (quotas[j] - result[j], -j))
        result[i] += 1
    while sum(result) > total:
        candidates = [j for j in range(len(result)) if result[j] > minimum]
        i = max(candidates, key=lambda j: (result[j] - quotas[j], j))
        result[i] -= 1
    return result


def build_schedule(language, goals, days, weekdays, minutes, start_date,
                   variety=None, topic_variety=None):
    """Create dated sessions before departure, split into stage/topic blocks.

    Include today if selected. Departure day is excluded. Stage percentages
    apply to practice minutes, so one session may cross a stage boundary.
    Every selected topic has at least one minute in every stage. Return only
    plain dictionaries/lists so the UI can store the schedule safely.

    variety: the learner's chosen variety (default: the language's default).
    topic_variety: {topic: resolve_variety() result}, from plan_varieties();
    stored in the schedule so each topic's practice uses its notes.
    """
    if language not in LANGUAGE_HINTS:
        raise ValueError("Choose Spanish or Macedonian.")
    variety = variety or DEFAULT_VARIETY[language]
    if language == "spanish" and variety not in VARIETIES:
        raise ValueError(f"Unknown Spanish variety {variety!r}.")
    topic_variety = topic_variety or {g: {"requested": variety, "used": variety, "fallback": False,
                                          "alternatives": [variety]} for g in goals}
    if not goals or len(set(goals)) != len(goals) or any(g not in SCENARIO_FILES for g in goals):
        raise ValueError("Select topics and give each one a unique priority.")
    if not weekdays or any(d not in WEEKDAYS for d in weekdays):
        raise ValueError("Select at least one practice weekday.")
    if not isinstance(days, int) or not 1 <= days <= MAX_DAYS:
        raise ValueError(f"Choose a study window between 1 and {MAX_DAYS} days.")
    if not isinstance(minutes, int) or minutes < 1:
        raise ValueError("Choose at least one minute per session.")
    if not isinstance(start_date, date):
        raise ValueError("Choose a valid start date.")
    dates = [start_date + timedelta(days=i) for i in range(days)
             if WEEKDAYS[(start_date + timedelta(days=i)).weekday()] in weekdays]
    if not dates:
        raise ValueError("None of your practice weekdays fall before departure. Change your days or study window.")
    total = len(dates) * minutes
    stage_minutes = _allocate_minutes(total, [3, 3, 4])
    if min(stage_minutes) < len(goals):
        raise ValueError("This schedule is too short to include every topic in all three stages. "
                         "Choose fewer topics, more practice days, or longer sessions.")
    weights = list(range(len(goals), 0, -1))
    remaining_capacity = minutes
    sessions = [{"day": (d - start_date).days + 1, "date": d.isoformat(),
                 "minutes": minutes, "blocks": []} for d in dates]
    session_index = 0
    for stage, budget in enumerate(stage_minutes, 1):
        shares = _allocate_minutes(budget, weights, minimum=1)
        # Short repeated topic blocks spread priorities throughout a stage.
        # Block length is up to five minutes, not an estimate of mastery.
        left = shares[:]
        while any(left):
            for i, topic in enumerate(goals):
                amount = min(5, left[i])
                left[i] -= amount
                while amount:
                    take = min(amount, remaining_capacity)
                    sessions[session_index]["blocks"].append({
                        "stage": stage, "stage_name": STAGE_NAMES[stage],
                        "topic": topic, "minutes": take,
                    })
                    amount -= take
                    remaining_capacity -= take
                    if remaining_capacity == 0:
                        session_index += 1
                        remaining_capacity = minutes
    return {"language": language, "start_date": start_date.isoformat(),
            "departure_date": (start_date + timedelta(days=days)).isoformat(),
            "total_minutes": total, "stage_minutes": stage_minutes,
            "topic_weights": {g: weights[i] / sum(weights) for i, g in enumerate(goals)},
            "variety": variety,
            "topic_variety": {g: dict(topic_variety[g]) for g in goals},
            "sessions": sessions}
