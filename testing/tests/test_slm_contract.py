"""
SLM-contract set (Spanish and shared code): the code around every SLM call, with
the SLM replaced by FakeLLM. Checks what happens with good, broken and incomplete
replies, and that a failure becomes a friendly LLMError (never a crash).

App SLM paths:
  Stage 2  sentence_practice.build_sentence_round, sentence_feedback
  Stage 3  roleplay_practice.roleplay_response
Macedonian versions: test_slm_contract_mk.py
"""
import json

import pytest

from llm import LLMError
from roleplay_practice import SCENES, _traveler_hint, roleplay_response
from sentence_practice import _json_reply, build_sentence_round, sentence_feedback
from conftest import make_round

pytestmark = pytest.mark.slm_contract

MSG = [{"role": "user", "content": "hi"}]
SCHEMA = {"type": "object"}


# --- _json_reply: the shared JSON wrapper ----------------------------------------

@pytest.mark.parametrize("text", ['{"a": 1}', '```json\n{"a": 1}\n```', '  {"a": 1}  '])
def test_json_reply_accepts_clean_and_fenced_json(fake_llm, text):
    """What: The app reads SLM JSON with or without a ```json fence (both languages).
    Why: Small models often wrap JSON in a code fence.
    If it fails: Good SLM replies are thrown away, and the learner waits for retries or sees an error."""
    fake_llm.queue(text)
    assert _json_reply(MSG, SCHEMA, 100) == {"a": 1}


def test_json_reply_retries_once_after_broken_json(fake_llm):
    """What: After broken JSON, the app asks the SLM once more for valid JSON (both languages).
    Why: One bad reply must not stop the activity.
    If it fails: The learner sees an error after one bad reply that a retry would fix."""
    fake_llm.queue('{"a": ', {"a": 2})
    assert _json_reply(MSG, SCHEMA, 100) == {"a": 2}
    assert "valid JSON" in fake_llm.calls[1]["messages"][-1]["content"]


@pytest.mark.parametrize("bad", ["not json", "[1, 2]", "null", ""])
def test_json_reply_gives_up_with_friendly_error(fake_llm, bad):
    """What: After two unusable replies, the app shows a friendly "please try again" message (both languages).
    Why: The learner must get a clear message, not a crash or a spinner.
    If it fails: The screen can crash or hang when the SLM returns nonsense."""
    fake_llm.queue(bad, bad)
    with pytest.raises(LLMError, match="usable response"):
        _json_reply(MSG, SCHEMA, 100)


def test_json_reply_sends_schema_and_low_temperature(fake_llm):
    """What: Each call sends the JSON schema, the length limit and a low temperature.
    Why: These settings make the SLM output more correct and stop very long replies.
    If it fails: Replies are more random, more often broken, or so long that Ollama blocks."""
    fake_llm.queue({"a": 1})
    _json_reply(MSG, SCHEMA, 123)
    cfg = fake_llm.calls[0]["config"]
    assert cfg["format"] == SCHEMA and cfg["num_predict"] == 123 and cfg["temperature"] <= 0.3


def test_ollama_errors_reach_the_app_as_llmerror(fake_llm):
    """What: An Ollama failure (not running, model missing) reaches the screen as a plain-English message.
    Why: The learner or the team must know what to fix.
    If it fails: The app shows a Python error, or nothing."""
    fake_llm.queue(LLMError("Can't reach Ollama"))
    with pytest.raises(LLMError, match="reach Ollama"):
        _json_reply(MSG, SCHEMA, 100)


# --- Stage 2: build_sentence_round ------------------------------------------------

def test_round_uses_the_studied_bank(fake_llm, spanish_bank, valid_round):
    """What: The Spanish Stage 2 request contains the variety, the topic and exactly the words that the learner studied.
    Why: Stage 2 must practice what the learner learned in Stage 1.
    If it fails: The SLM writes sentences with words that the learner never saw."""
    fake_llm.queue(valid_round)
    assert build_sentence_round(spanish_bank, "Spain", "restaurant") == valid_round
    sent = json.loads(fake_llm.calls[0]["messages"][1]["content"])
    assert sent["variety"] == "Spain" and sent["topic"] == "restaurant"
    assert [v["spanish"] for v in sent["studied_vocabulary"]] == [p["text"] for p in spanish_bank]


def test_round_retries_with_the_reason(fake_llm, spanish_bank, valid_round):
    """What: A bad Spanish round is sent back to the SLM with the reason, and a good second round is used.
    Why: The SLM can fix its own mistake when it knows the reason.
    If it fails: The learner sees an error instead of a fixed round."""
    bad = make_round(len(spanish_bank))
    bad["tasks"].pop()
    fake_llm.queue(bad, valid_round)
    assert build_sentence_round(spanish_bank, "Spain", "restaurant") == valid_round
    assert "Fix:" in fake_llm.calls[1]["messages"][-1]["content"]


def test_round_gives_up_after_two_bad_rounds(fake_llm, spanish_bank):
    """What: After two bad Spanish rounds, the app stops and shows a friendly message.
    Why: The learner must not wait forever.
    If it fails: The app retries without end, or shows a broken round."""
    bad = make_round(len(spanish_bank))
    bad["tasks"][0]["vocabulary_ids"] = [999]
    fake_llm.queue(bad, bad)
    with pytest.raises(LLMError, match="not usable"):
        build_sentence_round(spanish_bank, "Spain", "restaurant")


# --- Stage 2: sentence_feedback ----------------------------------------------------

TASK = {"english": "The bill, please.", "spanish": "La cuenta, por favor.", "vocabulary_ids": [1]}
GOOD_FEEDBACK = {"meaning_ok": False, "grammar_ok": True, "meaning_feedback": "You asked for the menu.",
                 "grammar_feedback": "The structure is fine.", "hint": "Which word means 'bill'?"}


def test_exact_answer_needs_no_slm_call(fake_llm, spanish_bank):
    """What: A Spanish answer that matches the example answer is marked correct without an SLM call.
    Why: This is faster and it can never be judged wrong.
    If it fails: The learner waits for the SLM, and the SLM can reject a perfect answer."""
    result = sentence_feedback(TASK, "la cuenta por favor", [], spanish_bank, "Spain")
    assert result["meaning_ok"] and result["grammar_ok"]
    assert fake_llm.calls == []


def test_feedback_is_returned(fake_llm, spanish_bank):
    """What: Good Spanish SLM feedback goes to the learner unchanged, with the learner's own answer in the request.
    Why: The SLM must judge what the learner actually typed.
    If it fails: The learner gets feedback on a different answer."""
    fake_llm.queue(GOOD_FEEDBACK)
    result = sentence_feedback(TASK, "La carta, por favor", [], spanish_bank, "Spain")
    assert result == GOOD_FEEDBACK
    sent = json.loads(fake_llm.calls[0]["messages"][1]["content"])
    assert sent["learner_answer"] == "La carta, por favor"


@pytest.mark.parametrize("field, value", [
    ("meaning_ok", "yes"), ("grammar_ok", None), ("hint", None), ("meaning_feedback", 3)])
def test_incomplete_feedback_is_an_error(fake_llm, spanish_bank, field, value):
    """What: Feedback with a missing or wrong-type field gives a friendly error, and the answer stays (Spanish).
    Why: The screen cannot show half a result.
    If it fails: The screen can crash or show "None" to the learner."""
    fake_llm.queue({**GOOD_FEEDBACK, field: value})
    with pytest.raises(LLMError, match="incomplete"):
        sentence_feedback(TASK, "La carta", [], spanish_bank, "Spain")


CODE_LIKE = [
    ("meaning_feedback", "Meaning_ok"), ("grammar_feedback", "grammar_ok"),
    ("meaning_feedback", "meaning_not_ok"), ("meaning_feedback", "Meaning_no"),
    ("meaning_feedback", "Meaning_ok Grammar_ok"), ("grammar_feedback", "true"),
    ("meaning_feedback", "Your answer is meaning_ok because you asked for the menu."),
    ("hint", "Fix grammar_ok first."),
]


@pytest.mark.parametrize("field, text", CODE_LIKE)
def test_feedback_never_looks_like_code(fake_llm, spanish_bank, field, text):
    """What: Stage 2 feedback and hints never show code-like words ("meaning_not_ok", "Meaning_no", "grammar_ok"), alone or inside a sentence (Spanish).
    Why: Learners saw these labels in the app. They look like a bug and explain nothing.
    If it fails: The learner sees words with underscores or code labels instead of feedback."""
    fake_llm.queue({**GOOD_FEEDBACK, field: text})
    result = sentence_feedback(TASK, "La carta", [], spanish_bank, "Spain")
    for key in ("meaning_feedback", "grammar_feedback", "hint"):
        assert "_" not in result[key] and result[key].strip().lower() not in {"true", "false", "ok"}, result[key]
    if field != "hint":
        assert len(result[field].split()) >= 4


def test_normal_feedback_is_not_changed(fake_llm, spanish_bank):
    """What: Normal feedback sentences pass the code-word guard unchanged (Spanish).
    Why: The guard must remove only code-like text, not useful feedback.
    If it fails: Learners get generic sentences instead of the SLM's real explanation."""
    fake_llm.queue(GOOD_FEEDBACK)
    assert sentence_feedback(TASK, "La carta", [], spanish_bank, "Spain") == GOOD_FEEDBACK


@pytest.mark.parametrize("field, text", [
    ("meaning_feedback", "goal_met"), ("meaning_feedback", "Meaning_ok"),
    ("grammar_feedback", "The grammar_ok flag is true."), ("explanation", "grammar_not_ok here."),
])
def test_end_feedback_never_looks_like_code(fake_llm, spanish_bank, retriever, field, text):
    """What: Stage 3 end feedback and correction notes never show code-like words such as "goal_met" (Spanish).
    Why: The end feedback has the same field names as Stage 2, so the same problem can happen.
    If it fails: The learner sees code words in the conversation feedback."""
    fix = {"learner_text": "La cuenta por favor.", "example_revision": "La cuenta, por favor, señor.",
           "explanation": "Add a polite word."}
    final = {"reply": "Gracias, adiós.", "english": "Thanks, bye.", "meaning_feedback": "Clear.",
             "grammar_feedback": "Good.", "goal_met": True, "corrections": [fix]}
    if field == "explanation":
        final["corrections"] = [{**fix, "explanation": text}]
    else:
        final[field] = text
    fake_llm.queue(final)
    out = roleplay_response(scene(), HISTORY, spanish_bank, "ES", "Spain", "restaurant", retriever, final=True)
    texts = [out["meaning_feedback"], out["grammar_feedback"]] + [c["explanation"] for c in out["corrections"]]
    assert all("_" not in t and t.strip() for t in texts), texts


def test_feedback_never_reveals_the_answer(fake_llm, spanish_bank):
    """What: If Spanish feedback contains the full example answer, the app hides it.
    Why: The learner must try again first. "Show an example answer" is a separate choice.
    If it fails: The hint gives away the answer, and the learner copies it without learning."""
    fake_llm.queue({**GOOD_FEEDBACK, "hint": "Say: La cuenta, por favor."})
    result = sentence_feedback(TASK, "La carta", [], spanish_bank, "Spain")
    assert "cuenta" not in result["hint"].lower()


# --- Stage 3: roleplay_response -----------------------------------------------------

def scene(topic="restaurant"):
    partner, title, goal = SCENES[topic][0]
    return {"partner": partner, "title": title, "goal": goal, "round_number": 1,
            "previous_openings": []}


TURN = {"reply": "¡Hola! ¿Qué desea tomar?", "english": "Hi! What would you like?",
        "hint": "Order a drink."}
HISTORY = [{"role": "assistant", "text": "¡Hola! ¿Qué desea tomar?"},
           {"role": "user", "text": "Un café, por favor."},
           {"role": "assistant", "text": "Muy bien. ¿Algo más?"},
           {"role": "user", "text": "No, gracias."},
           {"role": "assistant", "text": "Perfecto."},
           {"role": "user", "text": "La cuenta por favor."}]


def test_opening_turn(fake_llm, spanish_bank, retriever):
    """What: The Spanish role-play opening gets the topic notes and an "arrives" instruction.
    Why: The partner must greet the traveler with correct, grounded Spanish.
    If it fails: The partner opens with no notes, or does not know that the scene has started."""
    fake_llm.queue(TURN)
    result = roleplay_response(scene(), [], spanish_bank, "ES", "Spain", "restaurant", retriever)
    assert result == TURN
    data = json.loads(fake_llm.calls[0]["messages"][1]["content"])
    assert data["reference_notes"], "the SLM must get the topic notes"
    assert "arrives" in fake_llm.calls[0]["messages"][-1]["content"]


def test_restaurant_prompt_orders_drinks_first(fake_llm, spanish_bank, retriever):
    """What: The restaurant role-play prompt says: drinks first, then food, and do not ask again (both languages).
    Why: Earlier versions asked about drinks again after the traveler ordered.
    If it fails: The waiter repeats questions, and the scene feels broken."""
    fake_llm.queue(TURN)
    roleplay_response(scene(), [], spanish_bank, "ES", "Spain", "restaurant", retriever)
    assert "drinks first" in fake_llm.calls[0]["messages"][0]["content"]


@pytest.mark.parametrize("field", ["reply", "english", "hint"])
def test_turn_with_an_empty_field_is_an_error(fake_llm, spanish_bank, retriever, field):
    """What: A Spanish turn with an empty reply, translation or hint gives a friendly error, and the conversation stays.
    Why: The screen cannot show an empty partner line.
    If it fails: The learner sees an empty chat bubble or loses the conversation."""
    fake_llm.queue({**TURN, field: "  "})
    with pytest.raises(LLMError, match="incomplete"):
        roleplay_response(scene(), [], spanish_bank, "ES", "Spain", "restaurant", retriever)


@pytest.mark.parametrize("hint, replaced", [
    ("You could ask for the bill.", False),
    ("Ask the customer if they want dessert.", True),
    ("Offer them a table by the window.", True),
])
def test_hint_is_advice_for_the_traveler(hint, replaced):
    """What: A hint that tells the partner what to do is replaced with the learner's scene goal (both languages).
    Why: The hint is for the learner, not for the waiter.
    If it fails: The learner gets advice from the waiter's point of view and does not know what to say."""
    out = _traveler_hint(hint, scene())
    assert (out != hint) == replaced and (not replaced or out.startswith("Remember your goal"))


def test_final_turn_keeps_only_real_corrections(fake_llm, spanish_bank, retriever):
    """What: Spanish end feedback keeps at most 3 corrections, and only ones that quote a real learner line.
    Why: The SLM invents learner mistakes.
    If it fails: The learner is corrected for something that they never wrote."""
    fix = lambda text: {"learner_text": text, "example_revision": "La cuenta, por favor.",  # noqa: E731
                        "explanation": "Add a comma."}
    fake_llm.queue({"reply": "Gracias, adiós.", "english": "Thanks, bye.",
                    "meaning_feedback": "Clear.", "grammar_feedback": "Good.", "goal_met": True,
                    "corrections": [fix("La cuenta por favor."), fix("I never wrote this"),
                                    fix("Un café, por favor."), fix("No, gracias."),
                                    {"learner_text": "No, gracias."}]})
    result = roleplay_response(scene(), HISTORY, spanish_bank, "ES", "Spain", "restaurant",
                               retriever, final=True)
    texts = [c["learner_text"] for c in result["corrections"]]
    assert "I never wrote this" not in texts
    assert len(texts) <= 3 and all(t in {m["text"] for m in HISTORY} for t in texts)


@pytest.mark.parametrize("change", [{"goal_met": "yes"}, {"corrections": None}, {"reply": ""}])
def test_incomplete_final_feedback_is_an_error(fake_llm, spanish_bank, retriever, change):
    """What: Spanish end feedback with a missing or wrong field gives a friendly error.
    Why: The feedback screen needs every field.
    If it fails: The feedback screen can crash or show empty sections."""
    final = {"reply": "Adiós.", "english": "Bye.", "meaning_feedback": "ok",
             "grammar_feedback": "ok", "goal_met": True, "corrections": [], **change}
    fake_llm.queue(final)
    with pytest.raises(LLMError):
        roleplay_response(scene(), HISTORY, spanish_bank, "ES", "Spain", "restaurant",
                          retriever, final=True)


@pytest.mark.parametrize("variety", ["ES", "MX", "AR"])
def test_roleplay_notes_follow_the_variety(fake_llm, spanish_bank, retriever, variety):
    """What: The Spanish role-play request names the learner's variety and contains notes for it.
    Why: A Mexico learner must hear Mexican Spanish.
    If it fails: The partner speaks another country's Spanish."""
    fake_llm.queue(TURN)
    roleplay_response(scene("transport"), [], spanish_bank, variety, variety, "transport", retriever)
    data = json.loads(fake_llm.calls[0]["messages"][1]["content"])
    assert data["variety"] == variety and data["reference_notes"]
