"""
UI update 3: the learner chooses the lesson types (vocabulary, sentences, role play).

unit: build_schedule with chosen types (planner.py), and the chat's lesson-type words
ui:   the lesson type cards, the summary page, the calendar, and the word list that a
      plan without vocabulary shows before its first sentence or role-play activity
"""
from datetime import date

import pytest
from streamlit.testing.v1 import AppTest

import chat_preferences as cp
from llm import LLMError
from planner import build_schedule
from conftest import ROOT

START = date(2026, 11, 16)


def schedule(stages, days=28, minutes=20, goals=("restaurant", "directions")):
    return build_schedule("spanish", list(goals), days, ["Monday", "Wednesday", "Friday"],
                          minutes, START, stages=stages)


# --- unit: planner ----------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("stages", [[1], [2], [3], [1, 3], [2, 3], [1, 2, 3]])
def test_plan_has_only_the_chosen_lesson_types(stages):
    """What: A plan contains blocks of the chosen lesson types only, and all practice time goes to them.
    Why: The learner's choice must be reflected in the lesson plan.
    If it fails: The calendar shows lessons the learner turned off, or loses practice time."""
    plan = schedule(stages)
    used = {b["stage"] for s in plan["sessions"] for b in s["blocks"]}
    assert used == set(stages)
    assert plan["stages"] == stages
    assert sum(plan["stage_minutes"]) == plan["total_minutes"]
    assert all(plan["stage_minutes"][s - 1] == 0 for s in (1, 2, 3) if s not in stages)


@pytest.mark.unit
def test_chosen_types_keep_the_3_3_4_proportion():
    """What: Vocabulary + role play divides the time 3:4, as in the full 3:3:4 plan.
    Why: The proportion between lesson types stays the same when some are turned off.
    If it fails: One lesson type gets far more or less time than planned."""
    plan = schedule([1, 3], days=70)
    vocab, roleplay = plan["stage_minutes"][0], plan["stage_minutes"][2]
    assert abs(vocab / (vocab + roleplay) - 3 / 7) < 0.02


@pytest.mark.unit
def test_default_is_all_three_types():
    """What: Without the stages argument, the plan is the same as before update 3.
    Why: Teammates' code that calls build_schedule must not change behavior.
    If it fails: Existing plans change without anyone choosing it."""
    old_style = build_schedule("spanish", ["restaurant"], 14, ["Monday"], 30, START)
    assert old_style == build_schedule("spanish", ["restaurant"], 14, ["Monday"], 30, START,
                                       stages=[3, 1, 2])
    assert old_style["stages"] == [1, 2, 3]


@pytest.mark.unit
@pytest.mark.parametrize("stages", [[], [4], None])
def test_no_lesson_type_is_refused(stages):
    """What: An empty or unknown lesson-type list gives a clear error.
    Why: A plan needs at least one lesson type.
    If it fails: The app builds an empty calendar."""
    with pytest.raises(ValueError, match="lesson type"):
        schedule(stages)


# --- unit: chat words -------------------------------------------------------------------

@pytest.fixture
def model_down(monkeypatch):
    def down(*args, **kwargs):
        raise LLMError("down")
    monkeypatch.setattr(cp.llm, "generate_response", down)


@pytest.mark.unit
@pytest.mark.parametrize("text, stages", [
    ("Only vocabulary and role play.", [1, 3]),
    ("just role play please", [3]),
    ("vocabulary and sentences, no role play", [1, 2]),
    ("everything but no sentences", [1, 3]),
    ("solo vocabulario y juego de roles", [1, 3]),
    ("sin oraciones", [1, 3]),
    ("само вокабулар", [1]),
    ("I want to learn vocabulary for restaurants", None),
])
def test_chat_reads_lesson_types(model_down, text, stages):
    """What: The chat reads lesson types in English, Spanish and Macedonian; one type alone needs "only".
    Why: The learner can choose lesson types in natural language too.
    If it fails: The plan has the wrong lesson types, or "vocabulary for restaurants" drops role play."""
    chat = cp.new_chat()
    cp.read_message(chat, text, date(2026, 11, 18))
    assert chat["found"]["stages"] == stages


@pytest.mark.unit
def test_chat_lesson_types_reach_the_plan(model_down):
    """What: Lesson types from the chat are in the plan values, and default to all three.
    Why: The chat and the buttons must build the same kind of plan.
    If it fails: The chat ignores the learner's lesson types."""
    chat = cp.new_chat()
    cp.read_message(chat, "Spanish, Dec 1 to Dec 20, weekends, 20 minutes, restaurants", date(2026, 11, 18))
    assert cp.plan_inputs(chat)["stages"] == [1, 2, 3]
    cp.read_message(chat, "only role play", date(2026, 11, 18))
    assert cp.plan_inputs(chat)["stages"] == [3]
    rows = cp.reply_parts(chat["messages"][-1])["rows"]
    assert ("Lesson types", "Role play", False) in rows


# --- ui: preferences, summary, calendar, word list ----------------------------------------

def run_app():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
    at.run()
    return at


def ok(at):
    assert not at.exception, [e.value for e in at.exception]


def page(at):
    return "\n".join(str(m.value) for m in list(at.markdown) + list(at.caption) + list(at.warning)
                     + list(at.info))


def click(at, label):
    next(b for b in at.button if b.label == label).click().run()
    ok(at)


def plan_with(at, off, language="spanish"):
    """Turn off the given lesson types, build the plan, and open the summary."""
    if language != "spanish":
        at.selectbox(key="q_language").set_value(language).run()
    for stage in off:
        at.checkbox(key=f"q_type_{stage}").uncheck().run()
    click(at, "Build my plan")
    return at


@pytest.mark.ui
def test_lesson_type_cards_on_the_preferences_page():
    """What: The preferences page has three lesson-type checkboxes, all on by default; none on gives a warning.
    Why: The learner chooses the lesson types with buttons.
    If it fails: The learner cannot choose, or can build a plan with no lessons."""
    at = run_app()
    assert [at.checkbox(key=f"q_type_{s}").value for s in (1, 2, 3)] == [True, True, True]
    for s in (1, 2, 3):
        at.checkbox(key=f"q_type_{s}").uncheck()
    at.run()
    assert "Choose at least one lesson type." in page(at)
    click(at, "Build my plan")
    assert "Your suggested learning plan" not in page(at)


@pytest.mark.ui
def test_summary_and_calendar_show_only_chosen_types():
    """What: With sentences off, the summary lists Vocabulary and Role play, marks Sentences "not in this plan", and the calendar has no sentence lessons.
    Why: The learner's choice must show in the plan they see.
    If it fails: The learner sees lessons they turned off."""
    at = plan_with(run_app(), off=[2])
    text = page(at)
    assert "1. Vocabulary — about 43%" in text and "2. Role play — about 57%" in text
    assert "Sentences · not in this plan" in text
    click(at, "Yes, this sounds good — show my calendar")
    labels = [b.label for b in at.button]
    assert "Start sentence practice" not in labels
    assert "Start vocabulary practice" in labels


@pytest.mark.ui
def test_plan_without_vocabulary_shows_the_word_list_first():
    """What: In a plan without vocabulary, the first role play of a topic shows the word list; after "I've read the words" the words are saved for the activity.
    Why: Role play uses a topic's words, and the learner chose not to have vocabulary lessons.
    If it fails: The activity says "complete Stage 1 first", a lesson the plan does not have."""
    at = plan_with(run_app(), off=[1, 2])
    assert "Without vocabulary lessons" in page(at)
    click(at, "Yes, this sounds good — show my calendar")
    click(at, "Start roleplay")
    assert "Words for this topic" in page(at)
    assert "Complete a Stage 1 activity" not in page(at)
    banks = at.session_state["studied_banks"] if "studied_banks" in at.session_state else {}
    assert not banks
    next(b for b in at.button if b.label == "I've read the words — start").click()
    try:
        at.run()
    except Exception:
        pass        # the role play then calls the model, which these tests block
    assert any(k.startswith("wordlist_") for k in at.session_state["studied_banks"])


# --- ui: lesson types in the chat, and keeping choices until the plan is approved ---------

def to_chat(at):
    at.segmented_control(key="q_mode").set_value("Describe it in a chat").run()
    ok(at)
    return at


def send(at, text):
    at.text_input(key="q_chat_text").input(text)
    click(at, "Send")
    at.run()                      # the reply is read on the next run
    ok(at)


def card_values(at, prefix):
    return {int(c.key.rsplit("_", 1)[1]): c.value for c in at.checkbox if c.key.startswith(prefix)}


@pytest.mark.ui
def test_chat_screen_has_lesson_type_cards_in_step_with_the_chat(model_down):
    """What: The chat screen shows the lesson type cards; "only role play" in the chat unticks the other cards, and the plan has role play only.
    Why: On the chat screen the learner can say the lesson types or tick them (reviewer feedback).
    If it fails: Lesson types work on the button screen only."""
    at = to_chat(run_app())
    assert card_values(at, "chat_type_") == {1: True, 2: True, 3: True}
    send(at, "Spanish, Dec 1 to Dec 20, weekends, 20 minutes, restaurants. Only role play.")
    assert card_values(at, "chat_type_") == {1: False, 2: False, 3: True}
    click(at, "Build my plan")
    assert "1. Role play — about 100%" in page(at)
    assert at.session_state["preferences"]["stages"] == [3]


@pytest.mark.ui
def test_ticking_cards_on_the_chat_screen_changes_the_plan(model_down):
    """What: Unticking Sentences on the chat screen removes sentences from the chat's plan.
    Why: The cards and the chat are two ways to make the same choice.
    If it fails: The cards on the chat screen do nothing."""
    at = to_chat(run_app())
    send(at, "Spanish, Dec 1 to Dec 20, weekends, 20 minutes, restaurants")
    next(c for c in at.checkbox if c.key.endswith("_2")).uncheck().run()
    ok(at)
    assert at.session_state["pref_chat"]["found"]["stages"] == [1, 3]
    click(at, "Build my plan")
    assert "Sentences · not in this plan" in page(at)


@pytest.mark.ui
def test_button_choices_are_kept_after_change_my_preferences():
    """What: After the summary page, "Change my preferences" shows the same button choices (days, minutes, lesson types).
    Why: The learner may need to edit one thing without starting again (reviewer feedback).
    If it fails: Every choice goes back to its default."""
    at = run_app()
    at.multiselect(key="q_weekdays").set_value(["Tuesday", "Thursday"])
    at.number_input(key="q_minutes").set_value(25)
    at.checkbox(key="q_type_2").uncheck()
    at.run()
    click(at, "Build my plan")
    click(at, "Change my preferences")
    assert at.multiselect(key="q_weekdays").value == ["Tuesday", "Thursday"]
    assert at.number_input(key="q_minutes").value == 25
    assert card_values(at, "q_type_") == {1: True, 2: False, 3: True}


@pytest.mark.ui
def test_chat_and_history_are_kept_after_change_my_preferences(model_down):
    """What: A plan built in the chat returns to the chat screen with the whole chat history.
    Why: The learner who used the chat must not lose it when they go back to edit.
    If it fails: The learner lands on the button screen and the chat looks lost."""
    at = to_chat(run_app())
    send(at, "Spanish, Dec 1 to Dec 20, weekends, 20 minutes, restaurants")
    click(at, "Build my plan")
    click(at, "Change my preferences")
    assert at.segmented_control(key="q_mode").value == "Describe it in a chat"
    assert "Spanish, Dec 1 to Dec 20, weekends, 20 minutes, restaurants" in page(at)
    assert "Here is what I have:" in page(at)


@pytest.mark.ui
def test_approving_the_plan_locks_it_and_clears_the_preferences(model_down):
    """What: After "show my calendar", the preference choices and chat are cleared; a new plan starts from defaults.
    Why: An approved plan is locked; the kept choices are only for editing before approval.
    If it fails: Old choices leak into the next plan."""
    at = run_app()
    at.number_input(key="q_minutes").set_value(25).run()
    click(at, "Build my plan")
    click(at, "Yes, this sounds good — show my calendar")
    assert "pref_kept" not in at.session_state and "pref_chat" not in at.session_state
    click(at, "Start a new plan")
    assert at.number_input(key="q_minutes").value == 15


@pytest.mark.ui
@pytest.mark.parametrize("language", ["spanish", "macedonian"])
def test_word_list_before_sentences_in_both_languages(language):
    """What: In a plan without vocabulary, the first sentence practice shows the word list (Macedonian cards with the "say:" guide); after "I've read the words" the sentence practice starts.
    Why: The word list replaces vocabulary lessons in both languages, and Macedonian learners need the pronunciation guide.
    If it fails: The activity asks for a vocabulary lesson the plan does not have, or Macedonian cards lose their guide."""
    at = plan_with(run_app(), off=[1], language=language)
    click(at, "Yes, this sounds good — show my calendar")
    click(at, "Start sentence practice")
    assert "Words for this topic" in page(at)
    if language == "macedonian":
        assert "say:" in page(at)
    click(at, "I've read the words — start")
    assert any(k.startswith("wordlist_") for k in at.session_state["studied_banks"])
    assert any(b.label == "Prepare my five sentences" for b in at.button)
