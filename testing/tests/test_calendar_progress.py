"""
UI update 4: calendar progress, the next lesson, and moving between practices.

ui: the calendar's progress line, "Next lesson" card, session states and which session is
    open; the bar at the top of a practice (Calendar, Previous, Next); progress that is
    recorded as soon as an activity has a result.
"""
import pytest
from streamlit.testing.v1 import AppTest

from conftest import ROOT

pytestmark = pytest.mark.ui


def ok(at):
    assert not at.exception, [e.value for e in at.exception]


def text(at):
    return "\n".join(str(m.value) for m in list(at.markdown) + list(at.caption) + list(at.success))


def click(at, label):
    next(b for b in at.button if b.label == label).click().run()
    ok(at)


def calendar(language="spanish"):
    """A plan (restaurants, directions; default 14 days, Mon/Wed/Fri, 15 min)."""
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
    at.run()
    if language != "spanish":
        at.selectbox(key="q_language").set_value(language).run()
    click(at, "Build my plan")
    click(at, "Yes, this sounds good — show my calendar")
    return at


def activities(at):
    """Activity ids in calendar order, read from the calendar's buttons."""
    return [b.key[len("practice_"):] for b in at.button if (b.key or "").startswith("practice_")]


def sessions_of(at):
    """{session number: [activity ids]} from the ids' dates (one date per session)."""
    by_date = {}
    for a in activities(at):
        by_date.setdefault(a.split("_")[0], []).append(a)
    return {n: ids for n, ids in enumerate(by_date.values(), 1)}


def mark_done(at, ids):
    progress = dict(at.session_state["practice_progress"]) if "practice_progress" in at.session_state else {}
    for a in ids:
        date_text, stage, topic, variety = a.split("_")
        progress[a] = {"stage": int(stage), "topic": topic, "variety": variety,
                       "date": date_text, "result": {}}
    at.session_state["practice_progress"] = progress
    at.run()
    ok(at)


def expanders(at):
    return [(e.proto.label, e.proto.expanded) for e in at.expander]


@pytest.mark.parametrize("language", ["spanish", "macedonian"])
def test_new_calendar_shows_progress_and_the_next_lesson(language):
    """What: A new calendar says "0 of N sessions complete", shows the first activity as the next lesson, and opens only Session 1 (Spanish and Macedonian).
    Why: The learner must see at once where they are and what comes next.
    If it fails: The learner does not know where to start."""
    at = calendar(language)
    n = len(sessions_of(at))
    assert f"0 of {n} sessions complete" in text(at)
    assert "Next lesson" in text(at) and "Session 1 · Vocabulary · Restaurants" in text(at)
    opened = [label for label, expanded in expanders(at) if expanded]
    assert len(opened) == 1 and opened[0].startswith("Session 1 ·") and "In progress · 0 of" in opened[0]


def test_completed_session_moves_the_next_lesson_and_the_open_session():
    """What: When every activity of Session 1 is done, Session 1 says "Complete", the next lesson is in Session 2, and Session 2 is the open one.
    Why: Before update 4, Session 1 always stayed open, so it was not clear what was next (reviewer feedback).
    If it fails: The calendar keeps sending the learner back to Session 1."""
    at = calendar()
    sessions = sessions_of(at)
    mark_done(at, sessions[1])
    assert f"1 of {len(sessions)} sessions complete" in text(at)
    labels = dict(expanders(at))
    s1 = next(label for label in labels if label.startswith("Session 1 ·"))
    s2 = next(label for label in labels if label.startswith("Session 2 ·"))
    assert s1.endswith("Complete") and not labels[s1]
    assert labels[s2] and "In progress" in s2
    assert "Next lesson" in text(at) and "Session 2 ·" in text(at)
    assert next(b for b in at.button if b.key == f"practice_{sessions[1][0]}").label == "Practice again"


def test_all_done_says_so():
    """What: With every activity done, the calendar says the plan is complete and shows no next lesson.
    Why: The learner should see that they finished.
    If it fails: The calendar points to a lesson that does not exist."""
    at = calendar()
    mark_done(at, activities(at))
    assert "You completed every session in this plan." in text(at)
    assert not any(b.label == "Continue" for b in at.button)


@pytest.mark.parametrize("language", ["spanish", "macedonian"])
def test_continue_opens_the_next_lesson_and_the_bar_moves_between_activities(language):
    """What: Continue opens the next lesson; Next and Previous move to the neighbor activities (across sessions); Calendar goes back (Spanish and Macedonian).
    Why: Moving between practices and sessions must be easy once you are in one (reviewer feedback).
    If it fails: The learner must go back to the calendar to reach the next practice."""
    at = calendar(language)
    order = activities(at)
    click(at, "Continue")
    assert at.session_state["active_practice"]["id"] == order[0]
    assert next(b for b in at.button if b.label == "‹ Previous").disabled
    assert "**Session 1** · activity 1 of" in text(at)
    click(at, "Next ›")
    assert at.session_state["active_practice"]["id"] == order[1]
    for _ in range(len(sessions_of_ids(order)[1]) - 1):
        click(at, "Next ›")
    assert at.session_state["active_practice"]["id"] == order[len(sessions_of_ids(order)[1])]
    assert "**Session 2** · activity 1 of" in text(at)
    click(at, "‹ Previous")
    assert at.session_state["active_practice"]["id"] == order[len(sessions_of_ids(order)[1]) - 1]
    click(at, "← Calendar")
    assert "active_practice" not in at.session_state
    assert "Your learning calendar" in text(at)


def sessions_of_ids(order):
    by_date = {}
    for a in order:
        by_date.setdefault(a.split("_")[0], []).append(a)
    return {n: ids for n, ids in enumerate(by_date.values(), 1)}


def test_last_activity_has_no_next():
    """What: On the plan's last activity, Next is off.
    Why: There is nothing after the last activity.
    If it fails: Next crashes or loops."""
    at = calendar()
    order = activities(at)
    at.session_state["active_practice"] = {"id": order[-1], "session": len(sessions_of(at)) - 1,
                                           "block": len(sessions_of(at)[len(sessions_of(at))]) - 1}
    at.run()
    ok(at)
    assert next(b for b in at.button if b.label == "Next ›").disabled


def test_finished_activity_counts_without_the_finish_button(fake_llm, retriever):
    """What: A sentence round that has a result counts as done at once, even if the learner leaves with Calendar instead of Finish.
    Why: Before update 4, progress was saved only by the Finish button, so leaving another way lost it (reviewer feedback).
    If it fails: Finished practice is not shown as complete."""
    from conftest import make_round
    from practice import spanish_typing_activity
    at = calendar()
    bank = spanish_typing_activity(retriever, "restaurant", "ES")["phrases"]
    at.session_state["studied_banks"] = {"test": {"topic": "restaurant", "variety": "ES", "phrases": bank}}
    sentence = next(a for a in activities(at) if a.split("_")[1] == "2" and a.split("_")[2] == "restaurant")
    next(b for b in at.button if b.key == f"practice_{sentence}").click().run()
    ok(at)
    fake_llm.queue(make_round(len(bank)))
    click(at, "Prepare my five sentences")
    for n in range(5):
        click(at, "Show an example answer")
        click(at, "Next sentence" if n < 4 else "See my practice recap")
    assert sentence in at.session_state["practice_progress"]
    click(at, "← Calendar")
    assert next(b for b in at.button if b.key == f"practice_{sentence}").label == "Practice again"
