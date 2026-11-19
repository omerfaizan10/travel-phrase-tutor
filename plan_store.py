"""
plan_store.py: approved learning plans saved to a file, so the learner can shelve a plan,
keep several plans, and come back to any of them (also after the app restarts).

One JSON file holds every plan:
    {"plans": {plan_id: record, ...}}
A record:
    {"id", "name", "created", "last_opened", "progress": {"done": k, "total": n},
     "state": {session_state key: value, ...}}

"state" is everything the app needs to continue where the learner stopped: the
preferences and schedule, the progress and studied words, and each activity's
in-progress state (STATE_KEYS and ACTIVITY_PREFIXES). Text that was typed into a box
but not sent is not saved.

The file is TUTOR_PLANS_FILE, or ~/.travel_phrase_tutor/plans.json. It is outside the
repository on purpose (learner data). The app runs on one laptop, so one file is shared
by everyone who uses that copy of the app.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

STATE_KEYS = ("preferences", "schedule_draft", "practice_progress", "practice_results",
              "studied_banks")
ACTIVITY_PREFIXES = ("es_study_", "es_recall_", "s2_round_", "s3_round_")


def plans_file() -> Path:
    return Path(os.environ.get("TUTOR_PLANS_FILE")
                or Path.home() / ".travel_phrase_tutor" / "plans.json")


def _jsonable(value):
    """Plain JSON values (practice state is lists and dicts; sets/tuples become lists)."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(v) for v in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def load_all() -> dict:
    """{plan_id: record}. A missing or unreadable file gives no plans (never a crash)."""
    try:
        data = json.loads(plans_file().read_text(encoding="utf-8"))
        plans = data.get("plans", {})
        return plans if isinstance(plans, dict) else {}
    except (OSError, ValueError, AttributeError):
        return {}


def _write_all(plans: dict) -> None:
    """Write the whole file atomically (temp file + rename), so a crash never leaves half a file."""
    path = plans_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".plans-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"plans": plans}, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def new_plan_id() -> str:
    return uuid.uuid4().hex[:10]


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def snapshot(session_state) -> dict:
    """The part of session_state that belongs to the current plan."""
    state = {k: session_state[k] for k in STATE_KEYS if k in session_state}
    for key in list(session_state.keys()):
        if isinstance(key, str) and key.startswith(ACTIVITY_PREFIXES):
            state[key] = session_state[key]
    return _jsonable(state)


def save(plan_id: str, name: str, state: dict, progress: dict, created: str | None = None,
         last_opened: str | None = None) -> dict:
    """Add or update one plan. Other plans in the file are kept as they are."""
    plans = load_all()
    old = plans.get(plan_id, {})
    record = {"id": plan_id, "name": old.get("name") or name,
              "created": old.get("created") or created or now(),
              "last_opened": last_opened or old.get("last_opened") or now(),
              "progress": _jsonable(progress), "state": _jsonable(state)}
    plans[plan_id] = record
    _write_all(plans)
    return record


def rename(plan_id: str, name: str) -> None:
    plans = load_all()
    if plan_id in plans and name.strip():
        plans[plan_id]["name"] = name.strip()[:80]
        _write_all(plans)


def touch(plan_id: str) -> None:
    """Record that the plan was just opened."""
    plans = load_all()
    if plan_id in plans:
        plans[plan_id]["last_opened"] = now()
        _write_all(plans)


def delete(plan_id: str) -> None:
    plans = load_all()
    if plans.pop(plan_id, None) is not None:
        _write_all(plans)


def clear_plan_state(session_state) -> None:
    """Remove the current plan from session_state (it stays in the file)."""
    for key in list(session_state.keys()):
        if isinstance(key, str) and (key in STATE_KEYS or key.startswith(ACTIVITY_PREFIXES)):
            del session_state[key]


def restore(session_state, record: dict) -> None:
    """Put a saved plan back into session_state (after clearing the current one)."""
    clear_plan_state(session_state)
    for key, value in record.get("state", {}).items():
        session_state[key] = value
