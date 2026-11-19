"""
Tests for country / variety labels and the fallback rules.

Run:  .venv/bin/python -m pytest testing/tests/test_variety.py -v

Rules under test:
  1. No variety chosen -> Spain ("ES") for Spanish, "MK" for Macedonian.
  2. A chosen variety with no notes for a topic -> that topic falls back to
     Spain, and the learner can switch it to any variety that has notes.
Most tests read the notes only (no embedding model). The last one runs a real
search and is skipped if the embedding model is not available.
"""

import re
from datetime import date
from pathlib import Path

import pytest

from planner import (SCENARIO_FILES, build_plan, build_schedule, plan_varieties,
                     resolve_variety, teachable_units)
from retriever import VARIETIES, Retriever, accepted_labels, load_chunks

KB = Path(__file__).resolve().parents[2] / "knowledge_base"
LABELS = set(VARIETIES) | {"ALL"}


@pytest.fixture(scope="module")
def r():
    return Retriever()   # loads the notes; the embedding model loads only on search()


# --- Labels in the notes -----------------------------------------------------

def test_every_spanish_section_has_a_valid_label():
    """What: Every section of the Spanish notes has a valid country label (ES, MX, AR, ...).
    Why: The app picks notes for the learner's Spanish variety by this label.
    If it fails: A section is never shown, or it is shown for the wrong country."""
    for path in (KB / "spanish").glob("*.md"):
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if line.startswith("## "):
                m = re.match(r"Country: (\S+) \(", lines[i + 1])
                assert m and m.group(1) in LABELS, f"{path.name}: {line}"


def test_labels_reach_every_chunk_and_leave_the_text(r):
    """What: Every loaded note piece has its label, and the label line is not in the text (Spanish and Macedonian).
    Why: The label line must not appear in lessons or SLM prompts.
    If it fails: The learner can see "Country: ..." lines, or notes go to the wrong variety."""
    for c in r.chunks:
        assert "\nCountry:" not in c["text"], c["source"]
        if c["language"] == "spanish":
            assert c["country"] in LABELS
        else:
            assert c["country"] == "MK"


def test_split_sections_keep_their_label():
    """What: When a long section is split into pieces, all pieces keep the same label.
    Why: Part of a section must not change country.
    If it fails: Half of a section can appear for the wrong Spanish variety."""
    by_heading = {}
    for c in load_chunks():
        key = (c["source"], c["text"].splitlines()[1])
        by_heading.setdefault(key, set()).add(c["country"])
    assert all(len(v) == 1 for v in by_heading.values())


def test_accepted_labels():
    """What: Each variety accepts the correct labels, best first (Puerto Rico: PR, then Latin America, then all).
    Why: This order decides which notes the learner sees first.
    If it fails: The learner sees another country's words before their own."""
    assert accepted_labels("spanish", None) == ["ES", "ALL"]
    assert accepted_labels("spanish", "PR") == ["PR", "419", "ALL"]
    assert accepted_labels("spanish", "419") == ["419", "ALL"]
    assert accepted_labels("macedonian", None) == ["MK"]


# --- Fallback rules ------------------------------------------------------------

def test_no_choice_defaults_to_spain(r):
    """What: With no Spanish variety chosen, the app uses Spain.
    Why: Spain is the documented default.
    If it fails: The learner without a choice gets mixed or empty notes."""
    v = resolve_variety(r, "spanish", None, "restaurant")
    assert v["used"] == "ES" and not v["fallback"]


def test_country_with_notes_uses_them(r):
    """What: A chosen country with notes for a topic gets its own notes first.
    Why: The learner chose the country because they travel there.
    If it fails: The learner learns words from another country."""
    v = resolve_variety(r, "spanish", "PR", "restaurant")
    assert v["used"] == "PR" and not v["fallback"]
    units = teachable_units(r, "spanish", "restaurant.md", v["used"])
    assert units and {u["country"] for u in units} <= {"PR", "419", "ALL"}
    assert units[0]["country"] == "PR"


def test_country_without_notes_falls_back_to_spain(r):
    """What: A country with no notes for a topic uses Spain for that topic and says so.
    Why: The learner must get content, and must know that it is not their country's Spanish.
    If it fails: The topic is empty, or the learner thinks Spain words are local words."""
    v = resolve_variety(r, "spanish", "MX", "hotel")
    assert v["used"] == "ES" and v["fallback"] and v["requested"] == "MX"
    assert v["alternatives"][0] == "ES"
    assert "MX" not in v["alternatives"]


def test_country_with_no_notes_at_all_falls_back_everywhere(r):
    """What: A country with no notes at all uses Spain for every topic.
    Why: The app lists countries that have no notes yet.
    If it fails: The learner gets empty topics."""
    for topic in SCENARIO_FILES:
        assert resolve_variety(r, "spanish", "NI", topic)["used"] == "ES"


def test_alternatives_only_list_varieties_with_notes(r):
    """What: The variety menu for a topic lists only varieties that have notes for it.
    Why: The learner must not pick a variety that gives an empty topic.
    If it fails: The learner picks a variety and gets no content."""
    for topic in SCENARIO_FILES:
        for code in resolve_variety(r, "spanish", None, topic)["alternatives"]:
            assert teachable_units(r, "spanish", SCENARIO_FILES[topic][0], code) or \
                any(teachable_units(r, "spanish", f, code) for f in SCENARIO_FILES[topic])


def test_switching_a_topic_to_latin_america(r):
    """What: A topic switched to general Latin America uses only Latin American or shared notes.
    Why: The learner asked for Latin American words.
    If it fails: Spain-only words (vosotros, coger) appear for a Latin America learner."""
    v = resolve_variety(r, "spanish", "419", "shopping")
    assert v["used"] == "419"
    units = teachable_units(r, "spanish", "shopping.md", "419")
    assert {u["country"] for u in units} <= {"419", "ALL"}


def test_macedonian_ignores_varieties(r):
    """What: Macedonian always uses its one variety (MK) and never falls back.
    Why: Macedonian has one standard variety in the notes.
    If it fails: Macedonian topics can be empty or marked as a fallback."""
    v = resolve_variety(r, "macedonian", None, "restaurant")
    assert v["used"] == "MK" and not v["fallback"]


# --- Plans and schedules ---------------------------------------------------------

def test_default_plan_is_spain_only(r):
    """What: A Spanish plan with no country uses only Spain and shared notes.
    Why: Spain is the default variety.
    If it fails: The default learner gets a mix of countries."""
    plan = build_plan("spanish", ["restaurant", "directions"], 5, r)
    assert plan and all(u["country"] in {"ES", "ALL"} for s in plan for u in s["units"])


def test_plan_for_a_country_mixes_in_fallback_topics(r):
    """What: A plan for Mexico uses Mexico notes where they exist and Spain for the other topics.
    Why: Each topic must have content, in the best available variety.
    If it fails: Topics without Mexico notes are empty, or Mexico notes are not used."""
    plan = build_plan("spanish", ["restaurant", "hotel"], 6, r, variety="MX")
    by_topic = {s["scenario"]: s["variety"] for s in plan if not s.get("review")}
    assert by_topic.get("restaurant") == "MX"
    assert by_topic.get("hotel", "ES") == "ES"
    for s in plan:
        assert all(u["language"] == "spanish" for u in s["units"])


def test_schedule_stores_topic_varieties(r):
    """What: The calendar stores the variety that each topic uses.
    Why: Each practice activity must open with the correct notes.
    If it fails: An activity opens with the wrong country's notes."""
    goals = ["restaurant", "hotel"]
    tv = plan_varieties(r, "spanish", "MX", goals)
    sched = build_schedule("spanish", goals, 14, ["Monday", "Wednesday"], 20, date(2026, 11, 16),
                           variety="MX", topic_variety=tv)
    assert sched["variety"] == "MX"
    assert sched["topic_variety"]["restaurant"]["used"] == "MX"
    assert sched["topic_variety"]["hotel"]["fallback"] is True


def test_schedule_without_variety_defaults_to_spain():
    """What: A calendar built with no variety uses Spain for every topic.
    Why: Spain is the default variety.
    If it fails: The calendar has no variety and activities can open empty."""
    sched = build_schedule("spanish", ["restaurant"], 7, ["Monday"], 15, date(2026, 11, 16))
    assert sched["variety"] == "ES" and sched["topic_variety"]["restaurant"]["used"] == "ES"


def test_schedule_rejects_unknown_variety():
    """What: The calendar builder refuses an unknown variety code.
    Why: A wrong code must give a clear error.
    If it fails: The app can build a calendar with no content."""
    with pytest.raises(ValueError):
        build_schedule("spanish", ["restaurant"], 7, ["Monday"], 15, date(2026, 11, 16), variety="XX")


# --- Search filter (needs the embedding model) -----------------------------------

def test_search_respects_the_variety(r):
    """What: Note search returns only notes in the right language and variety (Spanish and Macedonian).
    Why: The SLM prompts get these notes as reference.
    If it fails: The SLM gets another country's or language's notes and uses the wrong words."""
    try:
        r._encode(["hola"])
    except Exception as e:  # model not cached and no internet
        pytest.skip(f"embedding model unavailable: {e}")
    for c in r.search("¿Dónde para el autobús?", language="spanish", k=8, country="PR"):
        assert c["country"] in {"PR", "419", "ALL"}
    for c in r.search("¿Dónde para el autobús?", language="spanish", k=8):
        assert c["language"] == "spanish"
    assert all(c["country"] == "MK" for c in r.search("автобус", language="macedonian", k=5, country=None))
