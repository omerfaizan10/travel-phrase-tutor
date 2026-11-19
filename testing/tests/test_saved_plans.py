"""
UI update 5: shelve a plan, keep several plans, and come back to any of them, also after
the app restarts (plans are saved to a file by plan_store.py).

unit: plan_store (save, load, rename, delete, what is saved, a broken file)
ui:   approve -> saved; Shelf this plan -> Your plans; Open -> same progress; several
      plans; a new app session (= restart) starts on Your plans; rename and delete
"""
import json

import pytest
from streamlit.testing.v1 import AppTest

import plan_store as store
from conftest import ROOT


# --- unit: plan_store -----------------------------------------------------------------

@pytest.mark.unit
def test_save_load_rename_delete(isolated_plans_file):
    """What: A saved plan can be loaded, renamed and deleted, and other plans stay as they are.
    Why: The Your plans page relies on these four steps.
    If it fails: Plans are lost, duplicated, or cannot be removed."""
    store.save("a", "Spanish (Spain)", {"preferences": {"x": 1}}, {"done": 0, "total": 3})
    store.save("b", "Macedonian", {"preferences": {"x": 2}}, {"done": 1, "total": 5})
    plans = store.load_all()
    assert set(plans) == {"a", "b"} and plans["a"]["state"] == {"preferences": {"x": 1}}
    store.rename("a", "  Trip to Madrid  ")
    store.save("a", "ignored", {"preferences": {"x": 3}}, {"done": 1, "total": 3})
    assert store.load_all()["a"]["name"] == "Trip to Madrid"           # a save keeps the name
    assert store.load_all()["a"]["state"]["preferences"] == {"x": 3}
    store.delete("a")
    assert set(store.load_all()) == {"b"}
    assert json.loads(isolated_plans_file.read_text())["plans"]["b"]["name"] == "Macedonian"


@pytest.mark.unit
def test_snapshot_keeps_only_the_plan(isolated_plans_file):
    """What: A snapshot holds the plan keys and each activity's state, not widget values or other screens.
    Why: Restoring a plan must not overwrite unrelated parts of the app.
    If it fails: Opening a plan brings back stale widgets or loses practice state."""
    session = {"preferences": {"language": "spanish"}, "schedule_draft": {"sessions": []},
               "practice_progress": {"x": 1}, "es_study_x": {"learned": {"b", "a"}},
               "s3_round_y": {"history": [("a", "b")]}, "q_minutes": 20, "pref_chat": {"m": 1},
               "es_input_study_x_1": "half typed"}
    state = store.snapshot(session)
    assert set(state) == {"preferences", "schedule_draft", "practice_progress", "es_study_x",
                          "s3_round_y"}
    assert state["es_study_x"]["learned"] == ["a", "b"]                # sets become lists
    assert state["s3_round_y"]["history"] == [["a", "b"]]


@pytest.mark.unit
def test_broken_file_means_no_plans(isolated_plans_file):
    """What: A missing or broken plans file gives no plans instead of a crash, and the next save repairs it.
    Why: A half-written or hand-edited file must not stop the app from starting.
    If it fails: The app crashes on start."""
    assert store.load_all() == {}
    isolated_plans_file.write_text("{not json")
    assert store.load_all() == {}
    store.save("a", "Spanish", {}, {"done": 0, "total": 1})
    assert set(store.load_all()) == {"a"}


# --- ui ------------------------------------------------------------------------------------

def ok(at):
    assert not at.exception, [e.value for e in at.exception]


def text(at):
    return "\n".join(str(m.value) for m in list(at.markdown) + list(at.caption) + list(at.info))


def click(at, label, key=None):
    next(b for b in at.button if (b.key == key if key else b.label == label)).click().run()
    ok(at)


def fresh_app():
    """A new browser session. With the same plans file this is what an app restart looks like."""
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
    at.run()
    ok(at)
    return at


def approved_plan(at, minutes=None):
    if any(b.label == "Create a new plan" for b in at.button):
        click(at, "Create a new plan")
    if minutes:
        at.number_input(key="q_minutes").set_value(minutes).run()
    click(at, "Build my plan")
    click(at, "Yes, this sounds good — show my calendar")
    return at


def first_activity_done(at):
    first = next(b.key for b in at.button if (b.key or "").startswith("practice_"))[len("practice_"):]
    date_text, stage, topic, variety = first.split("_")
    at.session_state["practice_progress"] = {first: {"stage": int(stage), "topic": topic,
                                                     "variety": variety, "date": date_text,
                                                     "result": {"done": True}}}
    at.run()
    ok(at)
    return first


@pytest.mark.ui
def test_approving_saves_the_plan():
    """What: Approving a plan saves it to the plans file at once.
    Why: An approved plan must survive closing the browser or restarting the app.
    If it fails: The learner loses the plan when the app stops."""
    at = approved_plan(fresh_app())
    plans = store.load_all()
    assert len(plans) == 1 and next(iter(plans.values()))["name"] == "Spanish (Spain)"


@pytest.mark.ui
def test_shelf_and_open_keeps_the_progress():
    """What: Shelf this plan goes to Your plans (the plan is listed as Shelved); Open brings back the calendar with the same progress.
    Why: The learner can leave a plan and return without losing progress (reviewer request).
    If it fails: The learner is stuck in the calendar, or progress is lost on return."""
    at = approved_plan(fresh_app())
    done = first_activity_done(at)
    click(at, "Shelf this plan")
    assert "Your plans" in text(at) and "Shelved" in text(at)
    assert "schedule_draft" not in at.session_state
    click(at, "Open")
    assert "Your learning calendar" in text(at)
    assert done in at.session_state["practice_progress"]
    assert next(b for b in at.button if b.key == f"practice_{done}").label == "Practice again"


@pytest.mark.ui
def test_several_plans_and_a_restart():
    """What: Two plans are both listed; a new app session (a restart) starts on Your plans and opens either plan with its progress.
    Why: Learners keep several plans and switch between them as they please, also after a restart.
    If it fails: Only one plan is kept, or plans are gone after a restart."""
    at = approved_plan(fresh_app(), minutes=15)
    done = first_activity_done(at)
    click(at, "Start a new plan")
    approved_plan(at, minutes=30)
    click(at, "Shelf this plan")
    assert len(store.load_all()) == 2
    restarted = fresh_app()
    assert "Your plans" in text(restarted)
    assert sum(b.label == "Open" for b in restarted.button) == 2
    fifteen = next(p for p in store.load_all().values()
                   if p["state"]["preferences"]["minutes"] == 15)
    click(restarted, None, key=f"open_{fifteen['id']}")
    assert done in restarted.session_state["practice_progress"]
    assert "15 minutes per session" in text(restarted)


@pytest.mark.ui
def test_rename_and_delete_from_your_plans():
    """What: A plan can be renamed, and deleted only after ticking the confirmation.
    Why: Deleting loses progress for good, so it needs a clear extra step.
    If it fails: A plan is deleted by one stray click, or cannot be renamed."""
    at = approved_plan(fresh_app())
    click(at, "Shelf this plan")
    plan_id = next(iter(store.load_all()))
    at.text_input(key=f"name_{plan_id}").set_value("Trip to Madrid").run()
    click(at, None, key=f"rename_{plan_id}")
    assert store.load_all()[plan_id]["name"] == "Trip to Madrid"
    assert next(b for b in at.button if b.key == f"delete_{plan_id}").disabled
    at.checkbox(key=f"sure_{plan_id}").check().run()
    click(at, None, key=f"delete_{plan_id}")
    assert store.load_all() == {}
    assert "You have no saved plans" in text(at)


@pytest.mark.ui
def test_your_plans_and_new_plan_links():
    """What: With saved plans, the preferences page has "← Your plans", and Your plans has "Create a new plan".
    Why: The learner can move between plans and a new plan at any time.
    If it fails: The learner cannot get back to their plans from a new plan."""
    at = approved_plan(fresh_app())
    click(at, "Shelf this plan")
    click(at, "Create a new plan")
    assert "Your learning preferences" in text(at)
    click(at, "← Your plans")
    assert any(b.label == "Open" for b in at.button)
