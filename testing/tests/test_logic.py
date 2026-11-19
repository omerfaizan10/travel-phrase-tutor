"""Unit set: planner, Stage 1 practice and sentence-round validation. No SLM.

Each test's docstring says What it tests, Why, and what it means If it fails.
The report shows these texts in its test catalog.
"""
from datetime import date, timedelta

import pytest

from planner import (_allocate_minutes, build_schedule, parse_timeline, parse_trip,
                     topic_files)
from practice import (bank_size, spanish_answer, spanish_feedback, spanish_typing_activity,
                      vocabulary_activity)
from retriever import load_chunks, normalize, parse_phrases
from sentence_practice import _validate_round
from conftest import ROOT, make_round

pytestmark = pytest.mark.unit


# --- planner ----------------------------------------------------------------

@pytest.mark.parametrize("text, days", [
    ("14", 14), ("10 days", 10), ("two weeks", 14), ("a fortnight", 14), ("3 weeks", 21),
    ("next month", 30), ("tomorrow", 1), ("0 days", None), ("-3 days", None),
    ("this weekend", None), ("whenever", None), ("90 days", 90),
])
def test_parse_timeline(text, days):
    """What: A trip length in words ("three weeks", "10 days") changes to a number of days.
    Why: The chat preferences screen uses this to set the study window from the learner's message.
    If it fails: The chat sets a study window that is too long or too short, or asks again for a length that the learner already gave."""
    assert parse_timeline(text) == days


def test_parse_trip_detects_language_goals_and_days():
    """What: A free-text trip description gives the correct language, topics and days.
    Why: The older trip screen (previous_app.py) uses this function.
    If it fails: The learner must correct the language, topics or days by hand."""
    trip = parse_trip("I'm going to Skopje in two weeks and want to order at restaurants")
    assert trip["language"] == "macedonian"
    assert trip["days"] == 14
    assert "restaurant" in trip["goals"]
    assert all(trip["found"].values())


def test_parse_trip_falls_back_to_defaults():
    """What: A trip description with no details gives the default values and says so.
    Why: The app must ask the learner to confirm values that it did not find.
    If it fails: The app can show a guess as if the learner had said it."""
    trip = parse_trip("a holiday")
    assert trip["language"] is None
    assert trip["found"] == {"language": False, "days": False, "goals": False}


@pytest.mark.parametrize("total, weights, minimum", [
    (100, [3, 3, 4], 0), (7, [3, 2, 1], 1), (10, [1, 1, 1], 1), (61, [5, 4, 3, 2, 1], 1),
])
def test_allocate_minutes_keeps_total_and_minimum(total, weights, minimum):
    """What: Practice minutes are divided between stages and topics without loss.
    Why: The calendar must use exactly the time that the learner has.
    If it fails: The calendar shows too much or too little practice time, or a topic gets no time."""
    result = _allocate_minutes(total, weights, minimum)
    assert sum(result) == total
    assert min(result) >= minimum


def test_allocate_minutes_rejects_too_little_time():
    """What: The app refuses a plan with too few minutes for all topics.
    Why: The learner must get a clear message, not a plan with empty topics.
    If it fails: The app can make a plan that skips a topic without a warning."""
    with pytest.raises(ValueError):
        _allocate_minutes(2, [1, 1, 1], minimum=1)


@pytest.mark.parametrize("language", ["spanish", "macedonian"])
def test_schedule_covers_every_topic_in_every_stage(language):
    """What: Every selected topic is in every chosen lesson type (here all three: Vocabulary, Sentences, Role play), and no session is too long (Spanish and Macedonian).
    Why: The learning method needs vocabulary, then sentences, then role play for each topic.
    If it fails: The learner cannot practice a topic in a lesson type, or a session is longer than they said."""
    goals = ["restaurant", "directions", "transport"]
    sched = build_schedule(language, goals, 14, ["Monday", "Wednesday", "Friday"], 20,
                           date(2026, 11, 16))
    blocks = [b for s in sched["sessions"] for b in s["blocks"]]
    for stage in (1, 2, 3):
        assert {b["topic"] for b in blocks if b["stage"] == stage} == set(goals)
    assert sum(b["minutes"] for b in blocks) == sched["total_minutes"]
    for s in sched["sessions"]:
        assert sum(b["minutes"] for b in s["blocks"]) <= s["minutes"]
        assert date.fromisoformat(s["date"]) < date.fromisoformat(sched["departure_date"])


@pytest.mark.parametrize("kwargs", [
    {"goals": []}, {"goals": ["restaurant", "restaurant"]}, {"weekdays": []},
    {"days": 0}, {"days": -5}, {"minutes": 0}, {"language": "french"},
])
def test_schedule_rejects_bad_input(kwargs):
    """What: The calendar builder refuses impossible settings (no topics, a repeated topic, no weekdays, 0 or fewer days, 0 minutes, a language we do not teach).
    Why: The learner must get a clear message instead of a broken calendar.
    If it fails: The app can crash or show an empty calendar."""
    args = {"language": "spanish", "goals": ["restaurant"], "days": 7, "weekdays": ["Monday"],
            "minutes": 15, "start_date": date(2026, 11, 16), **kwargs}
    with pytest.raises(ValueError):
        build_schedule(**args)


@pytest.mark.parametrize("days", [61, 120, 365])
def test_schedule_accepts_long_study_windows(days):
    """What: The calendar builder accepts study windows longer than 60 days.
    Why: The learner picks any first and last day on the calendar; there is no upper limit.
    If it fails: A long trip preparation gets an error instead of a calendar."""
    start = date(2026, 11, 16)
    sched = build_schedule("spanish", ["restaurant", "directions"], days, ["Monday"], 15, start)
    assert sched["departure_date"] == (start + timedelta(days=days)).isoformat()
    assert all(date.fromisoformat(s["date"]) < date.fromisoformat(sched["departure_date"])
               for s in sched["sessions"])
    assert len(sched["sessions"]) >= days // 7


@pytest.mark.parametrize("language", ["spanish", "macedonian"])
def test_every_topic_has_note_files(language):
    """What: Every topic in the app has notes files in both languages.
    Why: All practice comes from these checked notes.
    If it fails: A topic opens with no content."""
    for topic in ("restaurant", "directions", "shopping", "hotel", "transport", "weather",
                  "sightseeing"):
        for name in topic_files(topic):
            assert (ROOT / "knowledge_base" / language / name).exists()


# --- knowledge base -----------------------------------------------------------

def test_every_chunk_has_phrases_with_meanings():
    """What: Every phrase in the notes (both languages) has an English meaning.
    Why: The study cards and the questions show this meaning.
    If it fails: The learner sees a card with no meaning, or a question with no answer."""
    for chunk in load_chunks(ROOT / "knowledge_base"):
        for p in parse_phrases(chunk):
            assert p["text"].strip(), chunk["source"]
            assert p["meaning"].strip(), f"{chunk['source']}: {p['text']} has no meaning"


def test_normalize_ignores_case_and_punctuation():
    """What: Text comparison ignores capital letters and punctuation.
    Why: Many checks compare a learner's line with the notes.
    If it fails: Correct learner answers can be marked wrong."""
    assert normalize("¿Dónde está el BAÑO?") == normalize("dónde está el baño")


# --- Stage 1 practice ------------------------------------------------------------

@pytest.mark.parametrize("answer, expected, verdict", [
    ("La cuenta, por favor.", "la cuenta por favor", "correct"),
    ("  la   CUENTA por favor  ", "La cuenta, por favor.", "correct"),
    ("cafe", "café", "accent"),
    ("te", "té", "accent"),
    ("agua", "café", "wrong"),
])
def test_spanish_feedback(answer, expected, verdict):
    """What: Vocabulary (Stage 1) marks a typed Spanish answer as correct, accent missing, or wrong.
    Why: The learner must get a fair result and a useful message about accents.
    If it fails: Correct answers are rejected, or a missing accent is called a wrong word."""
    assert spanish_feedback(answer, expected) == verdict


def test_spanish_answer_keeps_spelling():
    """What: Answer clean-up removes punctuation and extra spaces but keeps the letters.
    Why: Spelling must still count, but punctuation must not.
    If it fails: Wrong spellings are accepted, or punctuation makes a correct answer wrong."""
    assert spanish_answer("¡Hola, Señor!") == "hola señor"


@pytest.mark.parametrize("share, expected", [(1.0, (10, 5)), (0.5, (5, 3)), (0.1, (3, 2))])
def test_bank_size_follows_topic_priority(share, expected):
    """What: A higher-priority topic gets more cards and questions (10 cards and 5 questions for the top topic).
    Why: The learner ranked the topics, and practice must follow that ranking.
    If it fails: A low-priority topic gets as much practice as the top topic, or a topic gets too little."""
    assert bank_size(share) == expected


@pytest.mark.parametrize("topic", ["restaurant", "directions", "transport", "hotel", "shopping"])
@pytest.mark.parametrize("variety", ["ES", "MX", "AR"])
def test_spanish_typing_activity(retriever, topic, variety):
    """What: Each Spanish topic and variety gives 1–10 cards, and every question has an answer on a card.
    Why: Vocabulary (Stage 1) asks only about items that the learner studied.
    If it fails: The learner gets a question that no card prepared them for, or an empty activity."""
    activity = spanish_typing_activity(retriever, topic, variety)
    assert 1 <= len(activity["phrases"]) <= 10
    texts = {p["text"] for p in activity["phrases"]}
    for q in activity["questions"]:
        assert q["answer"] in texts and set(q["accepted"]) <= texts
    assert len({q["meaning"].casefold() for q in activity["questions"]}) == len(activity["questions"])


@pytest.mark.parametrize("topic", ["restaurant", "directions", "transport"])
def test_macedonian_vocabulary_activity(retriever, topic):
    """What: The older Macedonian multiple-choice activity has the right answer in each option list.
    Why: Other screens can still use this function.
    If it fails: A question can have no correct option."""
    activity = vocabulary_activity(retriever, "macedonian", topic, "MK", 15)
    assert activity["phrases"]
    for q in activity["questions"]:
        assert q["answer"] in q["options"]
        assert len(set(q["options"])) == len(q["options"])


def test_activity_is_stable_between_reruns(retriever):
    """What: The same activity gives the same cards each time that the screen reloads.
    Why: Streamlit reloads the page on every click.
    If it fails: Cards change while the learner works."""
    a = spanish_typing_activity(retriever, "restaurant", "ES", offset=3)
    b = spanish_typing_activity(retriever, "restaurant", "ES", offset=3)
    assert a == b


# --- Sentence round validation (guards what the SLM returns) -----------------------

def test_valid_round_passes():
    """What: A correct Spanish sentence round passes the round check.
    Why: The check must not reject good SLM output.
    If it fails: Sentences (Stage 2) can show "not usable" errors for good rounds."""
    assert _validate_round(make_round(10), list(range(10)))


@pytest.mark.parametrize("break_it, reason", [
    (lambda r: r["tasks"].pop(), "only four tasks"),
    (lambda r: r["examples"].append(r["examples"][0]), "four examples"),
    (lambda r: r["tasks"][0].update(spanish="Agua."), "one-word answer"),
    (lambda r: r["tasks"][0].update(spanish=" ".join(["palabra"] * 23)), "too long"),
    (lambda r: r["tasks"][0].update(english=""), "empty English"),
    (lambda r: r["tasks"][0].update(vocabulary_ids=[]), "no vocabulary ids"),
    (lambda r: r["tasks"][0].update(vocabulary_ids=[99]), "id out of range"),
    (lambda r: r["tasks"][0].update(vocabulary_ids=["1"]), "id is text"),
    (lambda r: r["tasks"][1].update(r["tasks"][0]), "duplicate task"),
    (lambda r: r["tasks"][0].update(r["examples"][0]), "task copies an example"),
    (lambda r: r["tasks"][0].update(spanish="Quiero agua fría. Quiero pan también."), "two sentences for one"),
])
def test_bad_round_is_rejected(break_it, reason):
    """What: The round check rejects bad SLM output (missing tasks, one-word answers, repeated tasks, wrong word numbers).
    Why: The SLM sometimes returns broken rounds. The app must ask again, not show them.
    If it fails: The learner can get duplicate, empty or impossible tasks in Sentences (Stage 2)."""
    data = make_round(10)
    break_it(data)
    with pytest.raises((ValueError, TypeError, KeyError)):
        _validate_round(data, list(range(10)))
