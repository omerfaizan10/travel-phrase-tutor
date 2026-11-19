"""
Shared test setup.

Test sets (see pytest.ini):  unit, slm_contract, ui.
The live SLM quality check is not a pytest set: it is testing/tools/collect_transcripts.py
plus the judge step (see testing/skills/test-app/SKILL.md).

The SLM is never called from pytest. Every test gets a guard that fails loudly
if code reaches Ollama; tests that need SLM replies use the `fake_llm` fixture.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]   # the repository root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import llm  # noqa: E402
import sentence_practice  # noqa: E402
import tutor  # noqa: E402
from llm import LLMError  # noqa: E402
from retriever import Retriever  # noqa: E402


SETS = ("unit", "slm_contract", "ui")


def pytest_addoption(parser):
    parser.addoption("--catalog", default=None,
                     help="write every test's What / Why / If it fails text to this JSON file")


def pytest_collection_modifyitems(config, items):
    """Tests without a set marker belong to the unit set. With --catalog, also write the
    test catalog that the report shows (from each test's docstring)."""
    for item in items:
        if not any(item.get_closest_marker(m) for m in SETS):
            item.add_marker(pytest.mark.unit)
    path = config.getoption("--catalog")
    if not path:
        return
    catalog = {}
    for item in items:
        func = getattr(item, "function", None)
        key = f"{Path(str(item.fspath)).stem}::{item.originalname}"
        entry = catalog.setdefault(key, {
            "set": next(m for m in SETS if item.get_closest_marker(m)),
            "file": Path(str(item.fspath)).name, "name": item.originalname,
            "language": _language_of(item), "cases": 0, **_parse_doc(func)})
        entry["cases"] += 1
    Path(path).write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")


def _language_of(item):
    text = (item.nodeid + " " + (item.function.__doc__ or "")).lower()
    has_mk, has_es = "macedonian" in text, "spanish" in text
    return "macedonian" if has_mk and not has_es else "spanish" if has_es and not has_mk else "both"


def _parse_doc(func):
    """Docstring lines 'What:', 'Why:' and 'If it fails:' (each may wrap)."""
    out, field = {"what": "", "why": "", "fails": ""}, None
    for line in (getattr(func, "__doc__", None) or "").splitlines():
        line = line.strip()
        for label, key in (("What:", "what"), ("Why:", "why"), ("If it fails:", "fails")):
            if line.startswith(label):
                field, line = key, line[len(label):].strip()
        if field and line:
            out[field] = (out[field] + " " + line).strip()
    return out


# --- Fake SLM -------------------------------------------------------------------

class FakeLLM:
    """
    Stands in for llm.generate_response.

    Queue replies with .queue(...). Each item is:
      - a dict or list  -> returned as JSON text
      - a str           -> returned as is (use it for broken JSON)
      - an Exception    -> raised (e.g. LLMError("Ollama is down"))
    Every call is saved in .calls as {"messages", "config"}.
    """

    def __init__(self):
        self.replies = []
        self.calls = []
        self.responder = None

    def queue(self, *replies):
        self.replies.extend(replies)
        return self

    def respond(self, fn):
        """Answer every call with fn(messages, config) instead of a queue (long UI flows)."""
        self.responder = fn
        return self

    def __call__(self, messages, model_config=None):
        self.calls.append({"messages": messages, "config": model_config or {}})
        if self.responder:
            reply = self.responder(messages, model_config or {})
        elif not self.replies:
            raise AssertionError("FakeLLM was called more times than replies were queued.")
        else:
            reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, (dict, list)):
            return json.dumps(reply, ensure_ascii=False)
        return reply

    def with_stats(self, messages, model_config=None):
        return {"text": self(messages, model_config), "model": "fake", "total_seconds": 0.0,
                "load_seconds": 0.0, "prompt_tokens": 0, "output_tokens": 0,
                "tokens_per_second": 0.0}


def _blocked(*_args, **_kwargs):
    raise AssertionError("A test reached the real SLM. Use the fake_llm fixture.")


@pytest.fixture(autouse=True)
def no_real_slm(monkeypatch):
    """Every module imports the SLM function by name, so patch each copy."""
    monkeypatch.setattr(llm, "generate_response", _blocked)
    monkeypatch.setattr(llm, "generate_response_with_stats", _blocked)
    monkeypatch.setattr(sentence_practice, "generate_response", _blocked)
    monkeypatch.setattr(tutor, "generate_response_with_stats", _blocked)


@pytest.fixture(autouse=True)
def isolated_plans_file(tmp_path, monkeypatch):
    """Saved plans (plan_store.py, UI update 5) go to a temporary file in every test,
    never to the learner's real ~/.travel_phrase_tutor/plans.json."""
    path = tmp_path / "plans.json"
    monkeypatch.setenv("TUTOR_PLANS_FILE", str(path))
    return path


@pytest.fixture
def fake_llm(monkeypatch):
    """A FakeLLM wired into every place the app calls the SLM."""
    fake = FakeLLM()
    monkeypatch.setattr(llm, "generate_response", fake)
    monkeypatch.setattr(sentence_practice, "generate_response", fake)  # also roleplay_practice
    monkeypatch.setattr(llm, "generate_response_with_stats", fake.with_stats)
    monkeypatch.setattr(tutor, "generate_response_with_stats", fake.with_stats)
    return fake


# --- Shared data ----------------------------------------------------------------

@pytest.fixture(scope="session")
def retriever():
    return Retriever()  # notes only; the embedding model loads on the first search()


@pytest.fixture(scope="session")
def spanish_bank(retriever):
    """A real Stage 1 studied bank: Spain Spanish, restaurant topic."""
    from practice import spanish_typing_activity
    phrases = spanish_typing_activity(retriever, "restaurant", "ES")["phrases"]
    assert len(phrases) >= 5, "the restaurant notes should give a bank of phrases"
    return phrases


@pytest.fixture(scope="session")
def macedonian_bank(retriever):
    """A real Stage 1 studied bank: Macedonian, restaurant topic."""
    from practice import spanish_typing_activity
    phrases = spanish_typing_activity(retriever, "restaurant", "MK", language="macedonian")["phrases"]
    assert len(phrases) >= 5
    return phrases


MK_EXAMPLES = [
    ("Can I have the menu?", "Може ли менито, ве молам?"),
    ("I would like a coffee.", "Би сакал едно кафе."),
    ("The bill, please.", "Сметката, ве молам."),
]
MK_TASKS = [
    ("I would like a shopska salad.", "Би сакал една шопска салата."),
    ("Do you have a menu?", "Дали имате мени?"),
    ("What is the local specialty?", "Кој е локалниот специјалитет?"),
    ("I do not eat meat.", "Јас не јадам месо."),
    ("Thank you very much.", "Ви благодарам многу."),
]


def make_round(n_vocab, examples=None, tasks=None, language="spanish"):
    """A sentence round that passes sentence_practice._validate_round."""
    ids = [1] if n_vocab else []
    if language == "macedonian":
        examples, tasks = examples or MK_EXAMPLES, tasks or MK_TASKS
        pair = lambda e, s: {"english": e, "macedonian": s, "vocabulary_ids": list(ids)}  # noqa: E731
        return {"examples": [pair(*p) for p in examples], "tasks": [pair(*p) for p in tasks]}
    examples = examples or [
        ("I would like a coffee.", "Quisiera un café, por favor."),
        ("The bill, please.", "La cuenta, por favor, señor."),
        ("We want a table.", "Queremos una mesa para dos."),
    ]
    tasks = tasks or [
        ("I would like water.", "Quisiera agua, por favor."),
        ("Is there a menu?", "¿Hay una carta, por favor?"),
        ("A table for one, please.", "Una mesa para uno, por favor."),
        ("I want the soup.", "Quiero la sopa, por favor."),
        ("Thank you very much.", "Muchas gracias por todo, señor."),
    ]
    pair = lambda e, s: {"english": e, "spanish": s, "vocabulary_ids": list(ids)}  # noqa: E731
    return {"examples": [pair(*p) for p in examples], "tasks": [pair(*p) for p in tasks]}


@pytest.fixture
def valid_round(spanish_bank):
    return make_round(len(spanish_bank))
