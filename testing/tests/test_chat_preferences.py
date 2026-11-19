"""
UI update 2: the "Describe it in a chat" option on the preferences page.

unit:         the word rules, ask-once rule, restart and reply language, with the model down
slm_contract: the code around the model (JSON reading, retry, checks), FakeLLM replies
ui:           the chat on the real preferences page, through to the summary page
"""
from datetime import date, timedelta
from html import escape

import pytest
from streamlit.testing.v1 import AppTest

import chat_preferences as cp
from llm import LLMError
from conftest import ROOT

TODAY = date(2026, 11, 18)
EMPTY = {"language": "unknown", "country": "", "start_date": "", "end_date": "", "duration": "",
         "weekdays": [], "minutes": 0, "topics": []}


def model(**values):
    """A model reply: EMPTY with some values filled in."""
    return {**EMPTY, **values}


@pytest.fixture
def model_down(monkeypatch):
    def down(*args, **kwargs):
        raise LLMError("Can't reach Ollama. Is it running?")
    monkeypatch.setattr(cp.llm, "generate_response", down)


def chat_with(*messages):
    chat = cp.new_chat()
    for text in messages:
        cp.read_message(chat, text, TODAY)
    return chat


def last(chat):
    return chat["messages"][-1]


# --- unit: word rules and checks (model down) ------------------------------------------

@pytest.mark.unit
def test_full_message_needs_only_the_window(model_down):
    """What: One message with language, country, days, minutes and topics leaves only the study window to ask.
    Why: The chat must ask only for necessary items that are missing.
    If it fails: The learner answers questions they already answered."""
    chat = chat_with("I'm going to Mexico City. I want to practice Spanish on weekends "
                     "for 20 minutes, mostly restaurants and directions.")
    f = chat["found"]
    assert (f["language"], f["variety"], f["weekdays"], f["minutes"], f["topics"]) == (
        "spanish", "MX", ["Saturday", "Sunday"], 20, ["restaurant", "directions"])
    assert cp.missing(chat) == ["window"]
    assert last(chat)["question"] == "window"


@pytest.mark.unit
def test_answering_the_question_completes_the_plan(model_down):
    """What: Answering the window question with two dates makes the plan ready, with nothing suggested.
    Why: When every necessary item is present, the learner can build the plan.
    If it fails: Build my plan stays off after a complete answer."""
    chat = chat_with("Spanish, weekends, 20 minutes, restaurants", "Start Dec 1, end Jan 15")
    inputs = cp.plan_inputs(chat)
    assert (inputs["start_date"], inputs["end_date"], inputs["days"]) == (
        date(2026, 12, 1), date(2027, 1, 15), 46)
    assert last(chat)["question"] is None and chat["suggested"] == []


@pytest.mark.unit
@pytest.mark.parametrize("text, start, end", [
    ("Macedonian for three weeks", TODAY, TODAY + timedelta(days=20)),
    ("I fly to Madrid on March 3", TODAY, date(2027, 3, 3)),
    ("Spanish from 2026-12-01 to 2026-12-31", date(2026, 12, 1), date(2026, 12, 31)),
    ("Heading to Skopje on January 10th", TODAY, date(2027, 1, 10)),
    ("I'm going to Buenos Aires in about 6 weeks", TODAY, TODAY + timedelta(days=41)),
    ("quiero ir a uruguay fin de año", TODAY, date(2026, 12, 31)),
    ("tengo hasta el 31 de diciembre", TODAY, date(2026, 12, 31)),
    ("hoy hasta 31 de diciembre", TODAY, date(2026, 12, 31)),
    ("español durante tres semanas", TODAY, TODAY + timedelta(days=20)),
    ("Сакам македонски од 1 декември до 15 јануари", date(2026, 12, 1), date(2027, 1, 15)),
])
def test_study_window_from_words(model_down, text, start, end):
    """What: Lengths, trip dates and date pairs in English, Spanish and Macedonian give the right study window.
    Why: Learners describe dates in different ways and languages.
    If it fails: The plan has the wrong dates, or the tutor keeps asking."""
    f = chat_with(text)["found"]
    assert (f["start"], f["end"]) == (start, end)


@pytest.mark.unit
@pytest.mark.parametrize("text", ["Spanish, 2 days a week, 15 min", "español, un día a la semana"])
def test_days_per_week_is_not_a_study_window(model_down, text):
    """What: "2 days a week" is not read as a study window of 2 days.
    Why: It describes how often, not how long.
    If it fails: The plan lasts one or two days."""
    f = chat_with(text)["found"]
    assert (f["start"], f["end"]) == (None, None)


@pytest.mark.unit
def test_spanish_message_gets_a_spanish_reply(model_down):
    """What: A message in Spanish fills Uruguay, the window and the small-talk topic, and the reply is in Spanish.
    Why: The tutor answers in the learner's language (a reviewer's test case).
    If it fails: A Spanish speaker gets English replies, or the tutor does not understand them."""
    chat = chat_with("quiero ir a uruguay fin de año y practicar como hablar con mis tios y mi abuela")
    f = chat["found"]
    assert (f["language"], f["variety"], f["end"], f["topics"]) == (
        "spanish", "UY", date(2026, 12, 31), ["weather"])
    parts = cp.reply_parts(last(chat))
    assert last(chat)["lang"] == "es"
    assert parts["have"] == "Esto es lo que tengo:"
    assert parts["question"].startswith("¿Qué días de la semana")
    assert ("Idioma", "Español · Uruguay", False) in parts["rows"]


@pytest.mark.unit
def test_every_reply_can_be_shown_in_english(model_down):
    """What: A Spanish reply has the same content in English.
    Why: The learner may write in Spanish by mistake or not understand the reply.
    If it fails: The "Show in English" switch shows nothing or the wrong content."""
    msg = last(chat_with("español, fines de semana, 20 minutos"))
    es, en = cp.reply_parts(msg), cp.reply_parts(msg, "en")
    assert en["have"] == "Here is what I have:" and en["question"] == cp.QUESTIONS["window"]
    assert [v is None for _, v, _ in es["rows"]] == [v is None for _, v, _ in en["rows"]]
    assert ("Practice days", "Saturday, Sunday", False) in en["rows"]


@pytest.mark.unit
def test_macedonian_message_gets_a_macedonian_reply(model_down):
    """What: A message in Cyrillic is read (weekend, 30 minutes, topics, three weeks) and gets a Macedonian reply.
    Why: The tutor answers in the learner's language.
    If it fails: Macedonian speakers get English replies, or their message is not understood."""
    chat = chat_with("Сакам да учам македонски, викенд, 30 минути, ресторан и превоз, три недели")
    f = chat["found"]
    assert (f["language"], f["weekdays"], f["minutes"], f["topics"]) == (
        "macedonian", ["Saturday", "Sunday"], 30, ["restaurant", "transport"])
    assert last(chat)["lang"] == "mk"
    assert cp.plan_inputs(chat) is not None
    assert cp.reply_parts(last(chat))["closing"] == cp.TEXT["mk"]["ready"]


@pytest.mark.unit
@pytest.mark.parametrize("text, lang", [
    ("I want to learn Spanish", "en"), ("quiero aprender español", "es"),
    ("Сакам македонски", "mk"), ("30", "es")])
def test_reply_language(text, lang):
    """What: The reply language follows the message; a bare number keeps the last language.
    Why: Short answers ("30") have no language of their own.
    If it fails: The tutor switches language at random."""
    assert cp.detect_language(text, previous="es") == lang


@pytest.mark.unit
def test_no_question_is_asked_twice(model_down):
    """What: Unclear answers never repeat a question; each missing item gets a suggested value, and the plan becomes ready.
    Why: A strict tutor that asks again and again leaves the learner stuck (reviewer feedback).
    If it fails: The learner cannot get past the chat."""
    chat = chat_with("hello", "not sure", "whatever", "no idea", "dunno", "ok")
    questions = [m["question"] for m in chat["messages"] if m.get("question")]
    assert len(questions) == len(set(questions)) == 5
    assert cp.plan_inputs(chat) is not None
    assert sorted(chat["suggested"]) == sorted(cp.REQUIRED)
    assert cp.reply_parts(last(chat))["closing"] == cp.TEXT["en"]["ready_suggested"]


@pytest.mark.unit
def test_suggested_value_is_replaced_by_an_answer(model_down):
    """What: A value given later replaces a suggestion and is no longer marked "suggested".
    Why: The suggested plan is only a starting point that the learner can change.
    If it fails: A suggestion stays after the learner gives a real value."""
    chat = chat_with("Spanish, 20 minutes", "not sure")       # window was asked: now suggested
    assert "window" in chat["suggested"]
    cp.read_message(chat, "until Jan 15", TODAY)
    assert chat["found"]["end"] == date(2027, 1, 15) and "window" not in chat["suggested"]


@pytest.mark.unit
@pytest.mark.parametrize("text", ["can i restart my preferences?", "quiero empezar de nuevo",
                                  "почни одново", "start over please"])
def test_restart_clears_the_preferences(model_down, text):
    """What: "Restart" in English, Spanish or Macedonian clears every preference but keeps the chat on screen.
    Why: The learner must be able to restart at any time (reviewer feedback).
    If it fails: The learner is stuck with earlier answers."""
    chat = chat_with("Spanish, weekends, 20 minutes, restaurants", text)
    assert chat["found"] == cp._empty_found()
    assert chat["asked"] == [] and chat["suggested"] == []
    assert last(chat).get("restarted")
    cp.read_message(chat, "Macedonian", TODAY)
    assert chat["found"]["language"] == "macedonian"


@pytest.mark.unit
@pytest.mark.parametrize("text, problem", [
    ("Spanish from 2026-11-01 to 2026-12-01", "past"),
    ("Spanish from 2026-12-10 to 2026-12-01", "order"),
    ("Spanish, 500 minutes a session", "max_minutes"),
])
def test_impossible_values_are_refused(model_down, text, problem):
    """What: A past first day, an end before the start, or too many minutes gives a clear problem line.
    Why: The planner must only get values it can use.
    If it fails: The plan fails later with an unclear error, or has wrong dates."""
    assert problem in last(chat_with(text))["problems"]


@pytest.mark.unit
def test_later_message_corrects_a_value(model_down):
    """What: A later message changes a value given earlier.
    Why: The learner must be able to correct the tutor in the chat.
    If it fails: A wrong value stays and the learner must start again."""
    assert chat_with("Spanish, 20 minutes", "Actually make it 45 minutes")["found"]["minutes"] == 45


@pytest.mark.unit
def test_model_down_is_marked(model_down):
    """What: When the model does not answer, the reply says that word rules were used.
    Why: The learner should check the details more carefully.
    If it fails: The learner trusts a rough reading without warning."""
    assert cp.reply_parts(last(chat_with("Spanish")))["model_note"] == cp.TEXT["en"]["model_down"]


# --- slm_contract: code around the model -----------------------------------------------

@pytest.mark.slm_contract
def test_model_values_are_used(fake_llm):
    """What: Values from the model's JSON fill the plan when the message has words for them, and the schema is sent.
    Why: The model reads phrasing that word rules miss ("mid January", "twenty-five", Bogotá).
    If it fails: The chat ignores the model and asks again."""
    fake_llm.queue(model(language="spanish", country="Colombia", start_date="2026-12-01",
                         end_date="2027-01-15", weekdays=["Tuesday", "Thursday"], minutes=25,
                         topics=["transport", "hotel"]))
    chat = chat_with("Bogotá trip, start early December until mid January, twenty-five "
                     "minutes Tue/Thu after work, buses and my hostel")
    assert cp.plan_inputs(chat) == {
        "language": "spanish", "variety": "CO", "goals": ["transport", "hotel"], "days": 46,
        "weekdays": ["Tuesday", "Thursday"], "minutes": 25,
        "start_date": date(2026, 12, 1), "end_date": date(2027, 1, 15), "stages": [1, 2, 3]}
    config = fake_llm.calls[0]["config"]
    assert config["format"] == cp.EXTRACT_SCHEMA and config["temperature"] == 0
    assert "2026-11-18" in fake_llm.calls[0]["messages"][0]["content"]


@pytest.mark.slm_contract
def test_restart_does_not_call_the_model(fake_llm):
    """What: A restart message is handled by code without a model call.
    Why: Restart must work at once, even when the model is slow or down.
    If it fails: Restart waits for the model, or the model reads "restart" as preferences."""
    chat = cp.new_chat()
    cp.read_message(chat, "restart", TODAY)
    assert fake_llm.calls == [] and last(chat).get("restarted")


@pytest.mark.slm_contract
def test_invented_model_dates_are_ignored(fake_llm):
    """What: Model dates are not used when the message has no date words.
    Why: The small model invents a study window that the learner never gave.
    If it fails: The plan gets dates the learner did not choose, and the tutor skips the question."""
    fake_llm.queue(model(language="spanish", start_date="2026-11-18", end_date="2026-12-15"))
    chat = chat_with("Spanish please, mostly restaurants")
    assert (chat["found"]["start"], chat["found"]["end"]) == (None, None)
    assert "window" in cp.missing(chat)


@pytest.mark.slm_contract
def test_invented_model_values_do_not_replace_answers(fake_llm):
    """What: A short answer ("an hour") keeps the earlier days and topics, even when the model invents new ones.
    Why: In a live check the model changed Monday/Thursday to Monday-Friday and added topics after "an hour".
    If it fails: Earlier answers change without the learner saying so."""
    fake_llm.queue(model(language="spanish", weekdays=["Monday", "Thursday"],
                         topics=["shopping"]),
                   model(minutes=60, weekdays=list(cp.WEEKDAYS[:5]), topics=["restaurant"]))
    f = chat_with("Spanish on mondays and thursdays, shopping", "an hour")["found"]
    assert (f["weekdays"], f["topics"], f["minutes"]) == (["Monday", "Thursday"], ["shopping"], 60)


@pytest.mark.slm_contract
def test_model_topics_must_agree_with_the_words(fake_llm):
    """What: When the model's topics share nothing with the topic words in the message, the words win.
    Why: In a live check the model turned "talk with my aunts, uncles and grandmother" into restaurants and getting help.
    If it fails: The plan teaches topics the learner never asked for."""
    fake_llm.queue(model(language="spanish", country="Uruguay", topics=["restaurant", "emergency"]))
    chat = chat_with("quiero ir a uruguay fin de año y practicar como hablar con mis tios y mi abuela")
    assert chat["found"]["topics"] == ["weather"]


@pytest.mark.slm_contract
def test_code_dates_win_over_model_dates(fake_llm):
    """What: Dates that code can read are used even when the model returns other dates.
    Why: The model got "Dec 1 to Jan 15" wrong in a live check (it gave Dec 15).
    If it fails: The plan ends on the wrong day."""
    fake_llm.queue(model(start_date="2026-12-01", end_date="2026-12-15"))
    f = chat_with("Start Dec 1, end Jan 15")["found"]
    assert (f["start"], f["end"]) == (date(2026, 12, 1), date(2027, 1, 15))


@pytest.mark.slm_contract
def test_broken_json_is_retried_once(fake_llm):
    """What: Broken JSON gets one retry; the second reply is used.
    Why: A small model sometimes breaks the format.
    If it fails: One bad reply loses the learner's message."""
    fake_llm.queue("not json", model(language="macedonian"))
    chat = chat_with("Macedonian please")
    assert chat["found"]["language"] == "macedonian"
    assert last(chat)["model_ok"] is True
    assert len(fake_llm.calls) == 2


@pytest.mark.slm_contract
def test_bad_model_values_are_checked(fake_llm):
    """What: A wrong date format, unknown topic, or zero minutes from the model is not used.
    Why: Code, not the model, decides what reaches the planner.
    If it fails: A bad model reply breaks the plan."""
    fake_llm.queue(model(language="spanish", start_date="Dec 1st", end_date="soon",
                         minutes=0, topics=[]))
    chat = chat_with("Spanish")
    f = chat["found"]
    assert (f["start"], f["end"], f["minutes"], f["topics"]) == (None, None, None, [])
    assert cp.missing(chat) == ["window", "weekdays", "minutes", "topics"]


@pytest.mark.slm_contract
def test_last_question_is_sent_to_the_model_without_examples(fake_llm):
    """What: The model is told which item was asked, but not the example dates in the question.
    Why: "Weekends" alone means practice days only because of the question; the model copied example dates in a live check.
    If it fails: Short answers are not understood, or the example dates become the plan."""
    fake_llm.queue(model(language="spanish"), model(weekdays=["Saturday", "Sunday"]))
    chat = chat_with("Spanish, Dec 1 to Dec 20, 20 minutes, restaurants", "weekends")
    prompt = fake_llm.calls[1]["messages"][0]["content"]
    assert cp.FIELD_HINT["weekdays"] in prompt and "Jan 15" not in prompt
    assert cp.plan_inputs(chat) is not None


# --- ui: the chat on the preferences page ------------------------------------------------

def chat_page():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
    at.run()
    at.segmented_control(key="q_mode").set_value("Describe it in a chat").run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def send(at, text):
    at.text_input(key="q_chat_text").input(text)
    next(b for b in at.button if b.label == "Send").click().run()
    assert not at.exception, [e.value for e in at.exception]
    at.run()          # the reply is read on the next run, below the learner's message
    assert not at.exception, [e.value for e in at.exception]


def page_text(at):
    return "\n".join(str(m.value) for m in list(at.markdown) + list(at.caption) + list(at.warning))


@pytest.mark.ui
def test_chat_option_replaces_the_buttons():
    """What: The chat option hides the preference buttons and shows the opening line, an empty box with no example text, and Start over.
    Why: The learner can skip the buttons; example text in the box flashed after each message (reviewer feedback).
    If it fails: The learner sees both screens, or the example text flashes again."""
    at = chat_page()
    assert escape(cp.OPENING) in page_text(at)
    assert not [s for s in at.selectbox if s.label == "Which language would you like to learn?"]
    assert next(b for b in at.button if b.label == "Build my plan").disabled
    assert not at.text_input(key="q_chat_text").placeholder
    assert any(b.label == "Start over" for b in at.button)


@pytest.mark.ui
def test_spanish_reply_has_an_english_switch(model_down):
    """What: A Spanish reply shows in Spanish, with a "Show in English" switch that shows it in English.
    Why: The learner may not fully understand the reply in Spanish.
    If it fails: The learner cannot read the tutor's reply."""
    at = chat_page()
    send(at, "quiero aprender español los fines de semana")
    assert "Esto es lo que tengo:" in page_text(at)
    at.toggle[0].set_value(True).run()
    assert "Here is what I have:" in page_text(at)


@pytest.mark.ui
def test_start_over_button_clears_the_chat(model_down):
    """What: Start over clears the chat and the preferences.
    Why: The learner can restart at any time.
    If it fails: Earlier answers stay."""
    at = chat_page()
    send(at, "Spanish, weekends")
    next(b for b in at.button if b.label == "Start over").click().run()
    assert "Here is what I have:" not in page_text(at)
    assert at.session_state["pref_chat"]["found"]["language"] is None


@pytest.mark.ui
def test_chat_through_to_the_summary(fake_llm):
    """What: Two chat messages give every necessary item, then Build my plan opens the summary.
    Why: This is the whole chat path from description to plan.
    If it fails: The chat cannot build a plan."""
    start = date.today() + timedelta(days=1)
    end = start + timedelta(days=27)
    fake_llm.queue(
        model(language="spanish", country="Mexico", weekdays=["Saturday", "Sunday"], minutes=20,
              topics=["restaurant", "directions"]),
        model(start_date=start.isoformat(), end_date=end.isoformat()))
    at = chat_page()
    send(at, "Mexico City, Spanish on weekends, 20 minutes, restaurants and directions")
    assert cp.QUESTIONS["window"] in page_text(at)
    assert next(b for b in at.button if b.label == "Build my plan").disabled
    send(at, f"From {start:%b %d} to {end:%b %d}")
    assert "That is everything I need" in page_text(at)
    next(b for b in at.button if b.label == "Build my plan").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert "Your suggested learning plan" in page_text(at)
    assert at.session_state["preferences"]["variety"] == "MX"
