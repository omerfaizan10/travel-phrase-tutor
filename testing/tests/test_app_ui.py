"""
UI set: runs app.py headless with Streamlit's AppTest and clicks through the
real screens, in Spanish and Macedonian. The SLM is the FakeLLM, so these runs
are fast and repeatable.
"""
import json

from datetime import date, timedelta

import pytest
from streamlit.testing.v1 import AppTest

from llm import LLMError
from practice import spanish_typing_activity
from pronounce import transliterate
from conftest import ROOT, make_round

pytestmark = pytest.mark.ui

APP = str(ROOT / "app.py")
NAME = {"spanish": "Spanish", "macedonian": "Macedonian"}
VARIETY = {"spanish": "ES", "macedonian": "MK"}


def find(widgets, label):
    for w in widgets:
        if w.label == label:
            return w
    raise AssertionError(f"No widget labelled {label!r}. Found: {[w.label for w in widgets]}")


def texts(at):
    parts = [m.value for m in at.markdown] + [m.value for m in at.caption]
    parts += [e.value for e in at.error] + [e.value for e in at.info] + [e.value for e in at.warning]
    parts += [e.value for e in at.success]
    return "\n".join(str(p) for p in parts)


def no_crash(at):
    assert not at.exception, [e.value for e in at.exception]


def open_calendar(language="spanish", topics=("restaurant", "directions")):
    at = AppTest.from_file(APP, default_timeout=90)
    at.run()
    no_crash(at)
    at.selectbox(key="q_language").set_value(language).run()
    at.multiselect(key="q_topics").set_value(list(topics)).run()
    find(at.button, "Build my plan").click().run()
    no_crash(at)
    assert "Your suggested learning plan" in texts(at)
    find(at.button, "Yes, this sounds good — show my calendar").click().run()
    no_crash(at)
    assert "Your learning calendar" in texts(at)
    return at


def start(at, label):
    next(b for b in at.button if b.label == label).click().run()
    no_crash(at)


def give_studied_bank(at, retriever, language="spanish", topic="restaurant"):
    bank = spanish_typing_activity(retriever, topic, VARIETY[language], language=language)["phrases"]
    at.session_state["studied_banks"] = {"test": {"topic": topic, "variety": VARIETY[language],
                                                  "phrases": bank}}
    return bank


# --- Questionnaire and plan ---------------------------------------------------------

def test_app_opens_on_the_questionnaire():
    """What: The app starts with no error and shows the preferences questionnaire.
    Why: This is the first screen that every learner sees.
    If it fails: No learner can use the app."""
    at = AppTest.from_file(APP, default_timeout=90)
    at.run()
    no_crash(at)
    assert "Your learning preferences" in texts(at)


def test_old_trip_length_fields_are_gone():
    """What: The preferences page has no trip-length text box and no days box.
    Why: One date-range calendar replaces the two redundant questions.
    If it fails: The learner sees two ways to give the same information."""
    at = AppTest.from_file(APP, default_timeout=90)
    at.run()
    labels = [w.label for w in list(at.text_input) + list(at.number_input)]
    assert "How long until your trip?" not in labels
    assert "Confirm days until departure / study window" not in labels


def test_half_chosen_window_asks_for_the_last_day():
    """What: When only the first day is chosen, the app asks for the last day.
    Why: A study window needs both days.
    If it fails: The app can crash or build a plan with no end."""
    at = AppTest.from_file(APP, default_timeout=90)
    at.run()
    at.date_input(key="q_window").set_value((date.today(),)).run()
    no_crash(at)
    assert "Choose the last day of your study window." in texts(at)


def test_long_window_builds_a_plan():
    """What: A study window longer than 60 days (the old limit) builds a plan, and the summary and calendar show the window.
    Why: The study window has no upper limit.
    If it fails: A long trip preparation gets an error instead of a plan."""
    at = AppTest.from_file(APP, default_timeout=90)
    at.run()
    start, end = date.today(), date.today() + timedelta(days=119)
    at.date_input(key="q_window").set_value((start, end)).run()
    assert "120 days" in texts(at)
    find(at.button, "Build my plan").click().run()
    no_crash(at)
    assert "Your suggested learning plan" in texts(at)
    assert end.strftime("%b %d, %Y") in texts(at)
    find(at.button, "Yes, this sounds good — show my calendar").click().run()
    no_crash(at)
    assert f"Study window: {start:%b %d, %Y} to {end:%b %d, %Y}" in texts(at)


def test_duplicate_ranks_are_refused():
    """What: The same topic in two priority places gives a warning.
    Why: Each topic needs one clear rank to share the practice time.
    If it fails: The plan can give the time to the wrong topics."""
    at = AppTest.from_file(APP, default_timeout=90)
    at.run()
    ranks = [s for s in at.selectbox if s.label.startswith("Priority")]
    ranks[1].set_value(ranks[0].value).run()
    assert "Each topic must appear exactly once" in texts(at)


@pytest.mark.parametrize("language", ["spanish", "macedonian"])
def test_plan_and_calendar(language):
    """What: A learner can build a plan and open the calendar (Spanish and Macedonian).
    Why: All practice starts from the calendar.
    If it fails: The learner cannot reach any practice."""
    open_calendar(language)


@pytest.mark.parametrize("language", ["spanish", "macedonian"])
def test_every_stage_has_a_start_button(language):
    """What: With all three lesson types (the default), the calendar has start buttons for Vocabulary, Sentences and Role play (Stages 1, 2 and 3), in Spanish and Macedonian.
    Why: Each chosen lesson type must open from the calendar, in both languages.
    If it fails: The learner sees a lesson in the calendar but cannot open it."""
    at = open_calendar(language)
    labels = {b.label for b in at.button}
    assert {"Start vocabulary practice", "Start sentence practice", "Start roleplay"} <= labels


# --- Stage 1 (no SLM) -------------------------------------------------------------

def card_text(at):
    return next(m.value[4:] for m in at.markdown if m.value.startswith("### ")
                and m.value[4:] not in {"Study the vocabulary and expressions"})


@pytest.mark.parametrize("language, typed_as", [
    ("spanish", "same"), ("macedonian", "same"), ("macedonian", "latin")])
def test_stage1_typing_counts_correct_entries(language, typed_as):
    """What: In Vocabulary (Stage 1), typing the card item gives "Correct! Typed 1 out of 3" (Spanish, Macedonian Cyrillic, Macedonian Latin).
    Why: Stage 1 is a typing screen for both languages, and Macedonian accepts Latin letters.
    If it fails: The learner cannot pass Stage 1, so Sentences and Role play (Stages 2 and 3) never open."""
    at = open_calendar(language)
    start(at, "Start vocabulary practice")
    item = card_text(at)
    answer = transliterate(item) if typed_as == "latin" else item
    find(at.text_input, f"Type the {NAME[language]} word or expression shown above").set_value(answer).run()
    find(at.button, "Check typing").click().run()
    no_crash(at)
    assert "Correct! Typed 1 out of 3" in texts(at), texts(at)[-300:]


def test_stage1_wrong_answer_says_try_again():
    """What: A wrong Vocabulary (Stage 1) answer gives "Not quite" and the counter does not move (Macedonian).
    Why: The learner must type the item correctly 3 times.
    If it fails: Wrong answers count as practice, or the learner gets no message."""
    at = open_calendar("macedonian")
    start(at, "Start vocabulary practice")
    find(at.text_input, "Type the Macedonian word or expression shown above").set_value("погрешно").run()
    find(at.button, "Check typing").click().run()
    assert "Not quite" in texts(at)


def test_macedonian_card_shows_pronunciation():
    """What: The Macedonian Vocabulary (Stage 1) card shows the Latin spelling and a "say:" pronunciation guide.
    Why: Beginners cannot read Cyrillic yet.
    If it fails: The learner cannot read or pronounce the item."""
    at = open_calendar("macedonian")
    start(at, "Start vocabulary practice")
    assert "say:" in texts(at)


# --- Stage 2: sentence practice (fake SLM) ---------------------------------------------

@pytest.mark.parametrize("language", ["spanish", "macedonian"])
def test_stage2_needs_stage1_first(language):
    """What: When the plan includes Vocabulary but the learner has not studied a topic yet, Sentences (Stage 2) asks them to finish a Vocabulary activity first (Spanish and Macedonian). A plan without Vocabulary shows a word list instead (see test_lesson_types).
    Why: Sentences use only the words that the learner studied.
    If it fails: Sentences start with no vocabulary, or the screen crashes."""
    at = open_calendar(language)
    start(at, "Start sentence practice")
    assert "Complete a Stage 1 activity" in texts(at)


@pytest.mark.parametrize("language", ["spanish", "macedonian"])
def test_stage2_round_and_feedback(fake_llm, retriever, language):
    """What: A learner prepares a sentence round, answers, and sees the feedback (Spanish and Macedonian, fake SLM).
    Why: This is the full Sentences (Stage 2) path that a learner uses.
    If it fails: Learners cannot practice sentences in this language."""
    at = open_calendar(language)
    bank = give_studied_bank(at, retriever, language)
    start(at, "Start sentence practice")
    fake_llm.queue(make_round(len(bank), language=language))
    find(at.button, "Prepare my five sentences").click().run()
    no_crash(at)
    assert "Sentence 1 of 5" in texts(at)
    fake_llm.queue({"meaning_ok": False, "grammar_ok": True,
                    "meaning_feedback": "You asked for something else.",
                    "grammar_feedback": "Structure is fine.", "hint": "Check the noun."})
    answer = "Quisiera vino" if language == "spanish" else "Sakam vino"
    find(at.text_input, f"Your full {NAME[language]} sentence").set_value(answer).run()
    find(at.button, "Check my sentence").click().run()
    no_crash(at)
    assert "You asked for something else." in texts(at)
    if language == "macedonian":
        assert "Read as: **Сакам вино**" in texts(at)


def test_stage2_shows_slm_errors_without_crashing(fake_llm, retriever):
    """What: When Ollama is down, Sentences (Stage 2) shows the error message and does not crash.
    Why: The local SLM can stop or be missing.
    If it fails: The learner sees a crash screen instead of a clear message."""
    at = open_calendar("spanish")
    give_studied_bank(at, retriever)
    start(at, "Start sentence practice")
    fake_llm.queue(LLMError("Can't reach Ollama. Is it running?"))
    find(at.button, "Prepare my five sentences").click().run()
    no_crash(at)
    assert "Can't reach Ollama" in texts(at)


# --- Stage 3: role-play (fake SLM) -----------------------------------------------------

def roleplay_fake(language):
    """Answers every Stage 3 call by its schema: a partner turn, or the end feedback."""
    reply = {"spanish": "¡Hola! ¿Qué desea?", "macedonian": "Добар ден! Повелете."}[language]

    def answer(messages, config):
        props = config.get("format", {}).get("properties", {})
        if "goal_met" in props:
            out = {"meaning_feedback": "Clear requests.", "grammar_feedback": "Good.",
                   "goal_met": True, "corrections": []}
            if "reply" in props:
                out.update(reply="Gracias, adiós.", english="Thanks, bye.")
            return out
        return {"reply": reply, "english": "Hello, what would you like?", "hint": "Order a drink."}
    return answer


@pytest.mark.parametrize("language", ["spanish", "macedonian"])
def test_stage3_full_conversation(fake_llm, retriever, language):
    """What: A learner starts a role-play, sends 3 lines, and sees the end feedback (Spanish and Macedonian, fake SLM).
    Why: This is the full Role play (Stage 3) path that a learner uses.
    If it fails: Learners cannot finish a role-play or never see their feedback."""
    at = open_calendar(language)
    give_studied_bank(at, retriever, language)
    start(at, "Start roleplay")
    fake_llm.respond(roleplay_fake(language))
    find(at.button, "Start this scenario").click().run()
    no_crash(at)
    assert "Your responses: 0 of 3" in texts(at)
    lines = {"spanish": ["Un café, por favor.", "No, gracias.", "La cuenta, por favor."],
             "macedonian": ["Edno kafe, ve molam.", "Не, благодарам.", "Може ли сметката?"]}[language]
    for line in lines:
        find(at.text_input, f"Your {NAME[language]} response").set_value(line).run()
        find(at.button, "Send response").click().run()
        no_crash(at)
    assert "Your conversation feedback" in texts(at)
    assert "Clear requests." in texts(at)
    if language == "macedonian":
        sent = json.loads(fake_llm.calls[-1]["messages"][1]["content"])
        assert "Едно кафе, ве молам." in sent["learner_responses"], "Latin input must reach the SLM as Cyrillic"


def test_stage3_failed_turn_keeps_the_conversation(fake_llm, retriever):
    """What: When a partner turn fails, the learner sees an error and keeps the conversation and their turn count.
    Why: A retry must not lose or skip a learner turn.
    If it fails: The learner loses the conversation or a turn after one SLM error."""
    at = open_calendar("spanish")
    give_studied_bank(at, retriever)
    start(at, "Start roleplay")
    fake_llm.queue({"reply": "¡Hola!", "english": "Hi!", "hint": "Greet back."})
    find(at.button, "Start this scenario").click().run()
    fake_llm.queue('{"broken', '{"broken')
    find(at.text_input, "Your Spanish response").set_value("Hola").run()
    find(at.button, "Send response").click().run()
    no_crash(at)
    assert "usable response" in texts(at)
    assert "Your responses: 0 of 3" in texts(at)
