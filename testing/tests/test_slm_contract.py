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
    """What: The Spanish Sentences (Stage 2) request contains the variety, the topic and exactly the words that the learner studied.
    Why: Stage 2 must practice what the learner learned in Vocabulary (Stage 1).
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


def test_round_gives_up_after_three_bad_rounds(fake_llm, spanish_bank):
    """What: After three bad Spanish rounds, the app stops and shows a friendly message.
    Why: The learner must not wait forever. (Three tries, not two: with two, about 1 round in 12 failed in a real-model test.)
    If it fails: The app retries without end, or shows a broken round."""
    bad = make_round(len(spanish_bank))
    bad["tasks"][0]["spanish"] = "Agua."
    fake_llm.queue(bad, bad, bad)
    with pytest.raises(LLMError, match="not usable"):
        build_sentence_round(spanish_bank, "Spain", "restaurant")
    assert len(fake_llm.calls) == 3


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
    """What: Sentences (Stage 2) feedback and hints never show code-like words ("meaning_not_ok", "Meaning_no", "grammar_ok"), alone or inside a sentence (Spanish).
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
    """What: Role play (Stage 3) end feedback and correction notes never show code-like words such as "goal_met" (Spanish).
    Why: The end feedback has the same field names as Sentences (Stage 2), so the same problem can happen.
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
    """What: A Spanish turn with an empty reply, translation or hint is asked for again (naming the empty field); after three empty tries the learner gets a friendly error, and the conversation stays.
    Why: The screen cannot show an empty partner line, and one empty field is often fixed by asking again (as in Macedonian).
    If it fails: The learner sees an empty chat bubble, loses the conversation, or gets an error that one retry would have avoided."""
    fake_llm.queue(*[{**TURN, field: "  "}] * 3)
    with pytest.raises(LLMError, match="incomplete"):
        roleplay_response(scene(), [], spanish_bank, "ES", "Spain", "restaurant", retriever)
    assert field in fake_llm.calls[1]["messages"][-1]["content"]


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


@pytest.mark.parametrize("change", [{"goal_met": "yes"}, {"corrections": None}, {"meaning_feedback": ""}])
def test_incomplete_final_feedback_is_an_error(fake_llm, spanish_bank, retriever, change):
    """What: Spanish end feedback with a missing or wrong field gives a friendly error (an empty feedback text is asked for once more first).
    Why: The feedback screen needs every field. The closing line is fixed, so only the feedback fields come from the SLM.
    If it fails: The feedback screen can crash or show empty sections."""
    final = {"meaning_feedback": "ok", "grammar_feedback": "ok", "goal_met": True, "corrections": [], **change}
    fake_llm.queue(final, final)
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


# --- Fixes from the reviewed run 2026-11-24_184055 ----------------------------------

@pytest.mark.parametrize("verdict", [(True, True), (False, False)])
def test_hidden_answer_text_matches_the_verdict(fake_llm, spanish_bank, verdict):
    """What: When feedback that gives away the example answer is replaced, the new text agrees with the verdict (correct answers are not told to "try again").
    Why: A reviewer saw "Check the sentence structure and try again" under an answer marked correct.
    If it fails: The learner is told both that the answer is right and that it is wrong."""
    meaning_ok, grammar_ok = verdict
    fake_llm.queue({"meaning_ok": meaning_ok, "grammar_ok": grammar_ok, "hint": "",
                    "meaning_feedback": "You wrote La cuenta, por favor.",
                    "grammar_feedback": "Same as La cuenta, por favor.  "},
                   {"wrong_words": "Quisiera", "corrected_words": "Quiero"})
    result = sentence_feedback(TASK, "Quisiera la cuenta", [], spanish_bank, "Spain")
    if meaning_ok:
        assert "try again" not in (result["meaning_feedback"] + result["grammar_feedback"]).lower()
        assert result["hint"] == ""
    else:
        assert "not yet" in result["meaning_feedback"]
    assert "cuenta" not in (result["meaning_feedback"] + result["grammar_feedback"]).lower()


def test_english_answer_is_not_accepted(fake_llm, spanish_bank):
    """What: An answer written in English ("The bill please") is marked not correct, without an SLM call.
    Why: The SLM marked English answers as meaning_ok in 2 of 3 runs.
    If it fails: A learner who types English is told they communicated in Spanish."""
    result = sentence_feedback(TASK, "The bill please", [], spanish_bank, "Spain")
    assert result["meaning_ok"] is False and "Spanish" in result["meaning_feedback"]
    assert fake_llm.calls == []


def test_spacing_slip_counts_as_the_example_answer(fake_llm, spanish_bank):
    """What: "la cuenta porfavor" (a missing space) is accepted like the example answer, without an SLM call.
    Why: The SLM marked this answer wrong, and a reviewer confirmed that it should pass.
    If it fails: A learner with the right sentence and a spacing slip is told the grammar is wrong."""
    result = sentence_feedback(TASK, "la cuenta porfavor", [], spanish_bank, "Spain")
    assert result["meaning_ok"] and result["grammar_ok"] and fake_llm.calls == []


@pytest.mark.parametrize("check, flipped", [
    ({"wrong_words": "", "corrected_words": ""}, True),                         # no error quoted
    ({"wrong_words": "la cuenta", "corrected_words": "la cuenta"}, True),       # nothing changes
    ({"wrong_words": "una mesa", "corrected_words": "un mesa"}, True),          # not in the answer
    ({"wrong_words": "Las cuenta", "corrected_words": "La cuenta"}, False),     # a real error
])
def test_grammar_error_must_be_quoted(fake_llm, spanish_bank, check, flipped):
    """What: When the SLM says a Spanish answer has a grammar error, it must quote wrong words from the answer and a different correction; otherwise grammar is marked correct.
    Why: The SLM marked correct sentences ("Me puede traer agua por favor", "La carta, por favor") as grammar errors.
    If it fails: Learners with correct grammar are told to fix it, or a real error ("Las cuenta") is let through."""
    fake_llm.queue({"meaning_ok": True, "grammar_ok": False, "meaning_feedback": "The request is clear.",
                    "grammar_feedback": "Check the article.", "hint": "Look at the article."}, check)
    result = sentence_feedback(TASK, "Las cuenta por favor", [], spanish_bank, "Spain")
    assert result["grammar_ok"] is flipped
    if flipped:
        assert result["hint"] == "" and "No structure" in result["grammar_feedback"]


def test_customer_voice_turn_is_retried_then_replaced(fake_llm, spanish_bank, retriever):
    """What: A Spanish partner turn that speaks like the customer ("¡Camarero! ¿Qué van a beber?") is asked for again with the problem named; after three such turns a safe staff line from the notes is used.
    Why: Most Spanish role-play turns in the reviewed run were in the customer's voice.
    If it fails: The waiter calls for a waiter, and the learner cannot practice their own role."""
    bad = {"reply": "¡Camarero! ¿Qué van a beber?", "english": "Waiter! What will you drink?",
           "hint": "Order a drink."}
    fake_llm.queue(bad, bad, bad)
    history = [{"role": "assistant", "text": "¡Hola! ¿Qué desea?"},
               {"role": "user", "text": "Un café con leche, por favor."}]
    out = roleplay_response(scene(), history, spanish_bank, "ES", "Spain", "restaurant", retriever)
    assert "customer" in fake_llm.calls[1]["messages"][-1]["content"]
    assert "camarero" not in out["reply"].lower() and out["reply"].strip()


def test_spanish_partner_lines_are_staff_lines_of_the_variety(fake_llm, spanish_bank, retriever):
    """What: The Spanish role-play request contains partner_lines from the notes, and none of them is a customer line such as "¡Camarero!".
    Why: Staff lines from the notes keep the 4B model in its own role (the same fix that worked for Macedonian).
    If it fails: The model copies customer lines or has no examples of what its role says."""
    fake_llm.queue(TURN)
    roleplay_response(scene(), [], spanish_bank, "ES", "Spain", "restaurant", retriever)
    lines = json.loads(fake_llm.calls[0]["messages"][1]["content"])["partner_lines"]
    assert lines and not any("camarero" in x.lower() for x in lines)


def test_spanish_final_uses_a_separate_feedback_call(fake_llm, spanish_bank, retriever):
    """What: Spanish end feedback comes from a separate tutor call with a speaker-labelled transcript, and the closing line is fixed.
    Why: Asked inside the role-play, the SLM restarted the scene and praised the learner for the partner's lines.
    If it fails: The learner gets credit or blame for lines they never said, and the scene does not end."""
    fake_llm.queue({"meaning_feedback": "Clear.", "grammar_feedback": "Good.", "goal_met": True,
                    "corrections": []})
    out = roleplay_response(scene(), HISTORY, spanish_bank, "ES", "Spain", "restaurant", retriever, final=True)
    assert out["reply"] == "¡Muchas gracias, adiós!"
    data = json.loads(fake_llm.calls[0]["messages"][1]["content"])
    assert {t["speaker"] for t in data["transcript"]} == {"partner", "learner"}
    assert data["learner_responses"] == [m["text"] for m in HISTORY if m["role"] == "user"]


@pytest.mark.parametrize("topic, bad", [
    ("transport", "¿Por cuánto me lleva al aeropuerto?"),
    ("directions", "¿A cuántas cuadras está el museo?"),
    ("directions", "¡Hola! ¿Cómo llego al centro?"),
])
def test_traveler_question_is_retried(fake_llm, spanish_bank, retriever, topic, bad):
    """What: A Spanish partner turn that asks the traveler's question (the fare, the way, the distance) is asked for again with the problem named.
    Why: In the re-test, the taxi driver and the local asked the traveler's questions in most turns.
    If it fails: The partner asks the learner for directions or a fare, and the scene makes no sense."""
    good = {"reply": "Queda a seis cuadras de aquí, más o menos.", "english": "It's about six blocks from here.",
            "hint": "You could say thank you."}
    fake_llm.queue({"reply": bad, "english": "A question?", "hint": "Answer."}, good)
    history = [{"role": "assistant", "text": "¡Hola! ¿En qué le puedo ayudar?"},
               {"role": "user", "text": "Perdón, ¿dónde está el museo?"}]
    out = roleplay_response(scene(topic), history, spanish_bank, "AR", "Argentina", topic, retriever)
    assert "TRAVELER would ask" in fake_llm.calls[1]["messages"][-1]["content"]
    assert out["reply"] == good["reply"]


def test_staff_question_is_not_flagged(fake_llm, spanish_bank, retriever):
    """What: A normal staff question ("¿Cuánto tiempo piensa quedarse?") passes the Spanish checks on the first try.
    Why: The traveler-question check must not reject the receptionist's own questions.
    If it fails: Good partner lines are thrown away and replaced by fallback lines."""
    fake_llm.queue({"reply": "Muy bien. ¿Cuánto tiempo piensa quedarse?",
                    "english": "Very good. How long are you planning to stay?", "hint": "Say how many nights."})
    history = [{"role": "assistant", "text": "¡Buenas noches! ¿Tiene reserva?"},
               {"role": "user", "text": "Sí, a nombre de Ana García."}]
    roleplay_response(scene("hotel"), history, spanish_bank, "ES", "Spain", "hotel", retriever)
    assert len(fake_llm.calls) == 1


def test_repeated_question_is_retried(fake_llm, spanish_bank, retriever):
    """What: A Spanish partner turn that asks a question it already asked earlier is asked for again with the problem named.
    Why: The receptionist asked "¿Cuánto tiempo piensa quedarse?" three times, ignoring the answers.
    If it fails: The partner loops on one question and the learner cannot reach the goal."""
    history = [{"role": "assistant", "text": "¡Hola! ¿Cuánto tiempo piensa quedarse?"},
               {"role": "user", "text": "Tres noches."}]
    fake_llm.queue({"reply": "Muy bien, Ana. ¿Cuánto tiempo piensa quedarse?",
                    "english": "Very well, Ana. How long are you planning to stay?", "hint": "x"},
                   {"reply": "Perfecto. Es en el segundo piso, habitación 234.",
                    "english": "It's on the second floor, room 234.", "hint": "Ask about breakfast."})
    roleplay_response(scene("hotel"), history, spanish_bank, "ES", "Spain", "hotel", retriever)
    assert "same question again" in fake_llm.calls[1]["messages"][-1]["content"]


TAXI_OPEN = [{"role": "assistant", "text": "Buenas tardes. ¿A dónde lo llevo?"},
             {"role": "user", "text": "Al aeropuerto, por favor."}]


def test_partner_keeps_its_form_of_you(fake_llm, spanish_bank, retriever):
    """What: A Spanish partner who opened with usted and then switches to tú ("¿Te ayudo con las maletas?") is asked for a new reply that keeps usted.
    Why: The notes teach both tú and usted (formality depends on the situation), so the model sees both; a reviewer asked that one conversation stay consistent.
    If it fails: The taxi driver jumps between formal and informal "you", which teaches the learner that mixing them is normal."""
    fake_llm.queue({"reply": "Muy bien. ¿Te ayudo con las maletas?", "english": "Very well. Can I help you with your bags?",
                    "hint": "Answer the driver."},
                   {"reply": "Muy bien. ¿Le ayudo con las maletas?", "english": "Very well. Can I help you with your bags?",
                    "hint": "Answer the driver."})
    out = roleplay_response(scene("transport"), TAXI_OPEN, spanish_bank, "MX", "Mexico", "transport", retriever)
    assert "keep using usted" in fake_llm.calls[1]["messages"][-1]["content"]
    assert out["reply"] == "Muy bien. ¿Le ayudo con las maletas?"


def test_partner_may_switch_to_tu_when_asked(fake_llm, spanish_bank, retriever):
    """What: If the traveler asks to be called tú ("háblame de tú"), the partner may switch from usted to tú.
    Why: A speaker switches to tú when the other person asks for it; that is correct Spanish.
    If it fails: The partner refuses a natural request and the learner learns the wrong rule."""
    history = TAXI_OPEN + [{"role": "assistant", "text": "Son ... pesos."},
                           {"role": "user", "text": "Está bien. Puedes hablarme de tú."}]
    fake_llm.queue({"reply": "Claro. ¿Te ayudo con las maletas?", "english": "Sure. Can I help you with your bags?",
                    "hint": "Answer the driver."})
    roleplay_response(scene("transport"), history, spanish_bank, "MX", "Mexico", "transport", retriever)
    assert len(fake_llm.calls) == 1


def test_mexico_taxi_notes_teach_both_forms(retriever):
    """What: The Mexico taxi driver's notes lines include both tú and usted forms, and the role-play gets both as partner lines.
    Why: The Spanish reviewer asked to keep both forms, because the choice depends on formality and learners need to see it.
    If it fails: Learners only ever see one form of "you" for a taxi driver."""
    from roleplay_practice import _partner_phrases, _register
    forms = {_register(p["text"]) for p in _partner_phrases(retriever, "transport", "spanish", "MX")}
    assert {"tu", "usted"} <= forms


def test_line_with_a_gap_is_never_shown(fake_llm, spanish_bank, retriever):
    """What: A Spanish partner reply with a gap from the notes ("Son ... por día.") is asked for again, and the safe fallback line never has a gap.
    Why: In the second re-test the receptionist said "Son ... por día." with a literal "..." in four of five runs.
    If it fails: The learner sees "..." where a price or time should be."""
    gap = {"reply": "Son ... por día.", "english": "It's ... per day.", "hint": "Ask about breakfast."}
    fake_llm.queue(gap, gap, gap)
    history = [{"role": "assistant", "text": "Buenas noches. ¿Tiene reserva?"},
               {"role": "user", "text": "Sí, a nombre de Ana García."}]
    out = roleplay_response(scene("hotel"), history, spanish_bank, "ES", "Spain", "hotel", retriever)
    assert "gap" in fake_llm.calls[1]["messages"][-1]["content"]
    assert "..." not in out["reply"] and "…" not in out["reply"]


def test_no_promise_of_revisions_when_none_are_left(fake_llm, spanish_bank, retriever):
    """What: When every correction from the SLM is invalid and removed, the end feedback does not say "See the suggested revisions below" (both languages share this code).
    Why: A judge saw that sentence over an empty list of revisions.
    If it fails: The learner looks for revisions that are not there."""
    fake_llm.queue({"meaning_feedback": "Clear.", "grammar_feedback": "", "goal_met": True,
                    "corrections": [{"learner_text": "I never wrote this", "example_revision": "x", "explanation": "y"}]})
    out = roleplay_response(scene(), HISTORY, spanish_bank, "ES", "Spain", "restaurant", retriever, final=True)
    assert out["corrections"] == [] and "below" not in out["grammar_feedback"]


@pytest.mark.parametrize("answer", ["asdf qwerty", "jkjkjk"])
def test_nonsense_is_rejected_without_the_slm(fake_llm, spanish_bank, answer):
    """What: An answer with no word that appears anywhere in the Spanish notes ("asdf qwerty") is marked not correct in meaning and grammar, without an SLM call.
    Why: In the re-tests the SLM called random letters grammatical, and later also said they asked for the bill.
    If it fails: A learner who types random letters is told the answer is correct."""
    result = sentence_feedback(TASK, answer, [], spanish_bank, "Spain")
    assert result["meaning_ok"] is False and result["grammar_ok"] is False
    assert "Spanish" in result["meaning_feedback"] and fake_llm.calls == []


def test_real_spanish_outside_the_bank_still_goes_to_the_slm(fake_llm, spanish_bank):
    """What: A Spanish answer made of words that are not in the studied bank ("Necesito pagar") is still judged by the SLM.
    Why: The nonsense check must only stop non-Spanish input, not valid alternative answers.
    If it fails: A learner's valid Spanish is rejected as "not Spanish"."""
    fake_llm.queue(GOOD_FEEDBACK)
    sentence_feedback(TASK, "Necesito pagar", [], spanish_bank, "Spain")
    assert len(fake_llm.calls) == 1


def test_invented_price_is_retried(fake_llm, spanish_bank, retriever):
    """What: A Spanish partner reply with a number that is not in the notes ("Son 30 pesos.") is asked for again, and an English line that still has a "..." gap is flagged too.
    Why: In the third re-test the taxi driver filled the notes gap with a made-up price, and the English kept "It's ... pesos."
    If it fails: The learner is given invented prices or times, and an English meaning with a gap in it."""
    fake_llm.queue({"reply": "Son 87654 pesos.", "english": "It's ... pesos.", "hint": "Answer."},
                   {"reply": "Súbase, por favor.", "english": "Get in, please.", "hint": "Say where you are going."})
    out = roleplay_response(scene("transport"), TAXI_OPEN, spanish_bank, "MX", "Mexico", "transport", retriever)
    retry = fake_llm.calls[1]["messages"][-1]["content"]
    assert "invents a number" in retry and "gap" in retry
    assert out["reply"] == "Súbase, por favor."


def test_repeated_greeting_is_retried(fake_llm, spanish_bank, retriever):
    """What: A Spanish partner reply that repeats a sentence it already said (a second "Buenas noches, señora.") is asked for again.
    Why: In the third re-test the receptionist greeted the guest again in every turn and the scene stalled.
    If it fails: The partner repeats itself and the learner cannot move the scene on."""
    history = [{"role": "assistant", "text": "Buenas noches, señora. ¿Tiene reserva?"},
               {"role": "user", "text": "Sí, a nombre de Ana García."}]
    fake_llm.queue({"reply": "Buenas noches, señora. Aquí tiene la llave.", "english": "Good evening. Here is the key.", "hint": "x"},
                   {"reply": "Muy bien. Aquí tiene la llave.", "english": "Very good. Here is the key.", "hint": "Ask about breakfast."})
    roleplay_response(scene("hotel"), history, spanish_bank, "ES", "Spain", "hotel", retriever)
    assert "already said" in fake_llm.calls[1]["messages"][-1]["content"]


def test_goodbye_never_follows_a_question(fake_llm, spanish_bank, retriever):
    """What: When the Spanish role-play ends after a traveler question, the partner's answer keeps only its statements before the goodbye (no "¿me deja su pasaporte? ¡Adiós!").
    Why: In the third re-test the receptionist asked for the passport and said goodbye in the same line.
    If it fails: The last partner line asks something that the learner can never answer."""
    history = [{"role": "assistant", "text": "Buenas noches. ¿Tiene reserva?"},
               {"role": "user", "text": "Sí, a nombre de Ana García."},
               {"role": "assistant", "text": "Aquí tiene la llave."},
               {"role": "user", "text": "Gracias."},
               {"role": "assistant", "text": "¿Necesita algo más?"},
               {"role": "user", "text": "¿A qué hora es el desayuno?"}]
    fake_llm.queue({"reply": "Es en el segundo piso, habitación 234. ¿Me deja su pasaporte, por favor?",
                    "english": "It's on the second floor, room 234. May I have your passport, please?", "hint": "x"},
                   {"meaning_feedback": "Clear.", "grammar_feedback": "Good.", "goal_met": True, "corrections": []})
    out = roleplay_response(scene("hotel"), history, spanish_bank, "ES", "Spain", "hotel", retriever, final=True)
    assert "?" not in out["reply"] and "?" not in out["english"]
    assert out["reply"].endswith("¡Adiós!")


@pytest.mark.parametrize("bad, problem", [
    ("Traigamos un vaso de agua, por favor.", "customer"),
    ("Perfect. Y para las bebidas, tenemos agua y café.", "English"),
])
def test_waiter_lines_from_the_final_run_are_retried(fake_llm, spanish_bank, retriever, bad, problem):
    """What: A Spanish waiter line in the customer's voice ("Traigamos un vaso de agua") or with English mixed in ("Perfect.") is asked for again with the problem named.
    Why: The judges found both in the café role-play of the final run.
    If it fails: The waiter asks for water like a customer, or the learner sees English inside the Spanish line."""
    fake_llm.queue({"reply": bad, "english": "x", "hint": "y"}, TURN)
    history = [{"role": "assistant", "text": "¡Hola! ¿Qué desea?"},
               {"role": "user", "text": "Un café con leche, por favor."}]
    roleplay_response(scene(), history, spanish_bank, "ES", "Spain", "restaurant", retriever)
    assert problem in fake_llm.calls[1]["messages"][-1]["content"]


# --- Fixes from the reviewed run 2026-11-24_203138 (final round) ---------------------

CAFE = [{"role": "assistant", "text": "¿Qué va a tomar?"},
        {"role": "user", "text": "Hola, un café con leche, por favor."},
        {"role": "assistant", "text": "¿Qué va a comer?"},
        {"role": "user", "text": "No, gracias. Solo el café."}]


def test_english_that_copies_the_spanish_is_retried(fake_llm, spanish_bank, retriever):
    """What: A Spanish partner turn whose English line just repeats the Spanish is asked for again.
    Why: The reviewer saw "Perfecto, un café y un vaso de agua." as its own English meaning.
    If it fails: "Show English meaning" shows Spanish, and the learner gets no translation."""
    fake_llm.queue({"reply": "Perfecto, un café y un vaso de agua.", "english": "Perfecto, un café y un vaso de agua.", "hint": "x"},
                   {"reply": "Perfecto, un café y un vaso de agua.", "english": "Perfect, a coffee and a glass of water.", "hint": "x"})
    roleplay_response(scene(), CAFE, spanish_bank, "ES", "Spain", "restaurant", retriever)
    assert "repeats the Spanish" in fake_llm.calls[1]["messages"][-1]["content"]


def test_bill_is_not_brought_before_it_is_asked_for(fake_llm, spanish_bank, retriever):
    """What: A Spanish waiter who brings the bill before the traveler asked for it is asked for a new reply.
    Why: The reviewer saw "Aquí tiene la cuenta." right after "No, gracias. Solo el café." The learner leads the scene.
    If it fails: The waiter ends the scene and the learner never gets to ask for the bill."""
    fake_llm.queue({"reply": "Aquí tiene la cuenta.", "english": "Here is your bill.", "hint": "x"},
                   {"reply": "Muy bien, enseguida.", "english": "Very well, right away.", "hint": "x"})
    roleplay_response(scene(), CAFE, spanish_bank, "ES", "Spain", "restaurant", retriever)
    assert "has not asked for it" in fake_llm.calls[1]["messages"][-1]["content"]


def test_bill_is_fine_once_asked_for(fake_llm, spanish_bank, retriever):
    """What: Once the traveler asks for the bill, a Spanish waiter line with the bill passes the checks.
    Why: The bill check must only stop the waiter from rushing ahead.
    If it fails: The waiter can never bring the bill, and the scene cannot finish."""
    history = CAFE + [{"role": "assistant", "text": "Enseguida."},
                      {"role": "user", "text": "La cuenta, por favor."}]
    fake_llm.queue({"reply": "Aquí tiene la cuenta.", "english": "Here is your bill.", "hint": "Say thank you."})
    roleplay_response(scene(), history, spanish_bank, "ES", "Spain", "restaurant", retriever)
    assert len(fake_llm.calls) == 1


def test_name_is_not_asked_again(fake_llm, spanish_bank, retriever):
    """What: A Spanish receptionist who asks for the name after the guest gave it is asked for a new reply.
    Why: The reviewer saw "¿A nombre de quién?" right after "Me llamo Ana García."
    If it fails: The receptionist ignores the guest and the scene goes in circles."""
    history = [{"role": "assistant", "text": "Buenas noches. ¿Tiene reserva?"},
               {"role": "user", "text": "Me llamo Ana García."}]
    fake_llm.queue({"reply": "¿A nombre de quién?", "english": "Under what name?", "hint": "x"},
                   {"reply": "Muy bien, señora García. Aquí tiene la llave.", "english": "Very good, Ms García. Here is the key.", "hint": "x"})
    roleplay_response(scene("hotel"), history, spanish_bank, "ES", "Spain", "hotel", retriever)
    assert "already gave it" in fake_llm.calls[1]["messages"][-1]["content"]


def test_hint_that_quotes_the_partner_is_replaced(fake_llm, spanish_bank, retriever):
    """What: A Spanish hint that tells the learner to say the partner's own line ("You could say '¿Cuánto tiempo piensa quedarse?'") is replaced with the scene goal.
    Why: The reviewer saw hints written from the clerk's side.
    If it fails: The learner is told to say the receptionist's line and plays the wrong role."""
    history = [{"role": "assistant", "text": "Buenas noches. ¿Tiene reserva?"},
               {"role": "user", "text": "Sí, tengo una reserva."}]
    fake_llm.queue({"reply": "Muy bien. Aquí tiene la llave.", "english": "Very good. Here is the key.",
                    "hint": "You could say '¿Cuánto tiempo piensa quedarse?' to ask how long you plan to stay."})
    out = roleplay_response(scene("hotel"), history, spanish_bank, "ES", "Spain", "hotel", retriever)
    assert out["hint"].startswith("Remember your goal")


def test_nonsense_last_line_gets_a_reaction(fake_llm, spanish_bank, retriever):
    """What: When the last Spanish role-play line is not Spanish ("jkjkjk"), the closing starts with "Perdón, no entiendo."
    Why: The reviewer saw the partner thank the learner for nonsense.
    If it fails: The partner ignores the learner's last line and the ending makes no sense."""
    history = HISTORY[:-1] + [{"role": "user", "text": "jkjkjk"}]
    fake_llm.queue({"meaning_feedback": "Partly.", "grammar_feedback": "Good.", "goal_met": False, "corrections": []})
    out = roleplay_response(scene(), history, spanish_bank, "ES", "Spain", "restaurant", retriever, final=True)
    assert out["reply"].startswith("Perdón, no entiendo.") and out["english"].startswith("Sorry")


def test_correct_answer_feedback_is_short_with_no_hint(fake_llm, spanish_bank):
    """What: For a Spanish answer that is correct in meaning and grammar, the hint is empty and each feedback text keeps at most two sentences.
    Why: The reviewer saw long praise with an inaccurate claim ("puedes" as the informal form in Argentina).
    If it fails: Correct answers get long explanations where the SLM can mislead the learner."""
    fake_llm.queue({"meaning_ok": True, "grammar_ok": True, "hint": "Using 'puede' instead of 'puedes' is more polite.",
                    "meaning_feedback": "Clear. Very clear. Extremely clear.", "grammar_feedback": "Fine. Good. Great. Perfect."})
    result = sentence_feedback(TASK, "Quisiera la cuenta, por favor", [], spanish_bank, "Spain")
    assert result["hint"] == "" and result["meaning_feedback"] == "Clear. Very clear."
    assert result["grammar_feedback"] == "Fine. Good."


def test_round_with_copied_english_is_repaired(fake_llm, spanish_bank, valid_round):
    """What: When an English prompt in a Spanish practice round is a copy of the Spanish, only that sentence is sent for an English translation, and the round is kept.
    Why: The model often writes restaurant prompts in Spanish ("¡Camarero! ¿Me puede traer un té...?"); throwing such rounds away made "The generated sentences were not usable" appear in the app.
    If it fails: The learner is asked to translate Spanish into Spanish, or the round fails."""
    bad = json.loads(json.dumps(valid_round))
    bad["tasks"][0]["english"] = bad["tasks"][0]["spanish"]
    fake_llm.queue(bad, {"english": ["I would like water, please."]})
    data = build_sentence_round(spanish_bank, "Spain", "restaurant")
    assert data["tasks"][0]["english"] == "I would like water, please." and len(fake_llm.calls) == 2
    assert json.loads(fake_llm.calls[1]["messages"][1]["content"])["spanish"] == [bad["tasks"][0]["spanish"]]


# --- Turn notes: tell the model what the traveler just did (final round) -------------

DIRECTIONS = [{"role": "assistant", "text": "Hola. ¿Adónde querés ir?", "english": "Hi. Where do you want to go?"},
              {"role": "user", "text": "Perdón, ¿dónde está el museo?"},
              {"role": "assistant", "text": "Está a dos cuadras de acá. ¿Algo más?", "english": "It's two blocks from here. Anything else?"},
              {"role": "user", "text": "¿Te gusta el fútbol?"}]


def test_turn_note_follows_the_learner_line(fake_llm, spanish_bank, retriever):
    """What: Each Spanish partner turn gets a short note right after the traveler's line that quotes it and says to respond to exactly that.
    Why: The 4B model often answered as if the traveler had said nothing (it repeated its last question); the reviewer saw the partner ignore or rush past the learner.
    If it fails: The partner keeps ignoring what the learner just said."""
    fake_llm.queue(TURN)
    roleplay_response(scene(), CAFE[:2], spanish_bank, "ES", "Spain", "restaurant", retriever)
    last = fake_llm.calls[0]["messages"][-1]["content"]
    assert last.startswith("Hola, un café con leche, por favor.") and "respond to exactly that" in last


def test_decline_gets_its_own_note(fake_llm, spanish_bank, retriever):
    """What: When the traveler says no ("No, gracias. Solo el café."), the note says to accept it and not to offer it again or restart the order.
    Why: The reviewer saw "¿Postre?" and "¿Están listos para pedir?" right after the learner declined.
    If it fails: The waiter keeps offering things the learner turned down."""
    fake_llm.queue({"reply": "Muy bien, enseguida.", "english": "Very well, right away.", "hint": "x"})
    roleplay_response(scene(), CAFE, spanish_bank, "ES", "Spain", "restaurant", retriever)
    assert "just said no" in fake_llm.calls[0]["messages"][-1]["content"]


def test_off_topic_line_is_answered_then_the_question_is_asked_again(fake_llm, spanish_bank, retriever):
    """What: An off-topic line ("¿Te gusta el fútbol?") gets a note to react briefly and ask the last question again; asking that question again is not flagged as a repeat; if every try fails, the backup is "Perdón, no entiendo." plus that question.
    Why: The reviewer saw the partner ignore "¿Te gusta el fútbol?", and the old backup was the unrelated timetable line.
    If it fails: Off-topic input is ignored, or the scene gets a line that has nothing to do with it."""
    good = {"reply": "No sé, la verdad. ¿Algo más?", "english": "I don't know, really. Anything else?", "hint": "x"}
    fake_llm.queue(good)
    roleplay_response(scene("directions"), DIRECTIONS, spanish_bank, "AR", "Argentina", "directions", retriever)
    assert "off-topic" in fake_llm.calls[0]["messages"][-1]["content"] and len(fake_llm.calls) == 1
    fake_llm.queue(*[{"reply": "¡Camarero!", "english": "Waiter!", "hint": "x"}] * 3)
    out = roleplay_response(scene("directions"), DIRECTIONS, spanish_bank, "AR", "Argentina", "directions", retriever)
    assert out["reply"] == "Perdón, no entiendo. ¿Algo más?" and out["english"] == "Sorry, I don't understand. Anything else?"


def test_english_line_is_recognised(fake_llm, spanish_bank, retriever):
    """What: A traveler line in English ("How long does it take?") gets the English note (answer in one short, simple Spanish sentence).
    Why: The taxi driver answered English input with the timetable line.
    If it fails: English input is treated as off-topic or ignored."""
    history = [{"role": "assistant", "text": "Súbase, por favor.", "english": "Get in, please."},
               {"role": "user", "text": "How long does it take?"}]
    fake_llm.queue({"reply": "Llegamos pronto.", "english": "We'll be there soon.", "hint": "x"})
    roleplay_response(scene("transport"), history, spanish_bank, "MX", "Mexico", "transport", retriever)
    assert "wrote in English" in fake_llm.calls[0]["messages"][-1]["content"]


def test_question_backup_is_not_the_timetable_line(fake_llm, spanish_bank, retriever):
    """What: When every Spanish try fails after a traveler question, the backup is "Lo siento, no lo sé." (not "Mire el horario, por favor.").
    Why: The timetable line appeared in the hotel and taxi scenes, where it makes no sense.
    If it fails: The partner tells a hotel guest or a taxi passenger to check the timetable."""
    history = [{"role": "assistant", "text": "Buenas noches. ¿Tiene reserva?", "english": "Good evening. Do you have a reservation?"},
               {"role": "user", "text": "¿A qué hora es el desayuno?"}]
    fake_llm.queue(*[{"reply": "¡Camarero!", "english": "Waiter!", "hint": "x"}] * 3)
    out = roleplay_response(scene("hotel"), history, spanish_bank, "ES", "Spain", "hotel", retriever)
    assert out["reply"] == "Lo siento, no lo sé." and "horario" not in out["reply"]


def test_spanish_retries_are_calmer(fake_llm, spanish_bank, retriever):
    """What: Spanish partner retries use temperatures 0.2, 0.4 and 0.6 (Macedonian keeps 0.2, 0.7 and 1.0).
    Why: Hotter retries drifted further from the conversation ("¿Postre?" after a decline).
    If it fails: Each retry gets more random and less connected to what the learner said."""
    fake_llm.queue(*[{"reply": "¡Camarero!", "english": "Waiter!", "hint": "x"}] * 3)
    roleplay_response(scene(), CAFE, spanish_bank, "ES", "Spain", "restaurant", retriever)
    assert [c["config"]["temperature"] for c in fake_llm.calls] == [0.2, 0.4, 0.6]


def test_off_topic_does_not_reask_an_answered_question(fake_llm, spanish_bank, retriever):
    """What: After off-topic input, the partner is only told to ask again the question from its previous turn; when that turn had no question (the destination was already agreed), the note says to bring the traveler back without asking again for what they already said.
    Why: In the turn-note run the driver and the local asked "¿A dónde te llevo?" / "¿Adónde querés ir?" again after the destination was settled (10 turns).
    If it fails: Off-topic input makes the partner restart the scene."""
    history = [{"role": "assistant", "text": "Hola. ¿Adónde querés ir?", "english": "Hi. Where do you want to go?"},
               {"role": "user", "text": "Perdón, ¿dónde está el museo?"},
               {"role": "assistant", "text": "Está a dos cuadras de acá.", "english": "It's two blocks from here."},
               {"role": "user", "text": "¿Te gusta el fútbol?"}]
    fake_llm.queue({"reply": "No sé, la verdad. El museo está cerca.", "english": "I don't really know. The museum is close.", "hint": "x"})
    roleplay_response(scene("directions"), history, spanish_bank, "AR", "Argentina", "directions", retriever)
    note = fake_llm.calls[0]["messages"][-1]["content"]
    assert "Adónde" not in note and "without asking again" in note


def test_english_line_with_the_hint_or_the_learner_line_is_retried(fake_llm, spanish_bank, retriever):
    """What: A Spanish partner turn whose English line contains the hint ("...You could ask...") or copies the traveler's line is asked for again.
    Why: In the turn-note run the English meaning showed the hint text, or the learner's own English question.
    If it fails: "Show English meaning" shows something other than the meaning of the partner's line."""
    history = [{"role": "assistant", "text": "Súbete, por favor.", "english": "Get in, please."},
               {"role": "user", "text": "How long does it take?"}]
    fake_llm.queue({"reply": "Llegamos pronto.", "english": "How long does it take?", "hint": "x"},
                   {"reply": "Llegamos pronto.", "english": "We'll be there soon. You could ask how much it is.", "hint": "x"},
                   {"reply": "Llegamos pronto.", "english": "We'll be there soon.", "hint": "x"})
    roleplay_response(scene("transport"), history, spanish_bank, "MX", "Mexico", "transport", retriever)
    assert "repeats the traveler's line" in fake_llm.calls[1]["messages"][-1]["content"]
    assert "contains the hint" in fake_llm.calls[2]["messages"][-1]["content"]


def test_hint_with_an_unquoted_partner_line_is_replaced(fake_llm, spanish_bank, retriever):
    """What: A Spanish hint that contains the partner's own line without quotation marks ("You could ask: ¿A dónde te llevo?") is replaced with the scene goal.
    Why: The judges saw this hint in the taxi scene; the quote-only check missed it.
    If it fails: The learner is told to say the driver's line."""
    history = [{"role": "assistant", "text": "Buenas tardes. ¿A dónde te llevo?", "english": "Good afternoon. Where can I take you?"},
               {"role": "user", "text": "Buenas tardes, al aeropuerto, por favor."}]
    fake_llm.queue({"reply": "Muy bien, vamos.", "english": "Very good, let's go.", "hint": "You could ask: ¿A dónde te llevo?"})
    out = roleplay_response(scene("transport"), history, spanish_bank, "MX", "Mexico", "transport", retriever)
    assert out["hint"].startswith("Remember your goal")


@pytest.mark.parametrize("english, spanish", [
    ("¡Mesero! ¿Me puede traer el guineo y el sancocho?", "¡Mesero! ¿Me puede traer el guineo y el sancocho?"),
    ("I want to try the el tenedor libre.", "Quiero probar el tenedor libre."),
])
def test_spanish_prompt_that_stays_spanish_is_asked_again(fake_llm, spanish_bank, valid_round, english, spanish):
    """What: If a Spanish-looking English prompt (¿ / ¡, "the el ...", or a copy of the answer) is still not English after the translation step, the round is asked for again.
    Why: The reviewer saw tasks whose prompt was the Spanish answer ("¡Mesero! ¿Me puede traer el guineo y el sancocho?"), which the learner could just copy.
    If it fails: The learner is asked to "translate" a sentence that is already in Spanish."""
    bad = json.loads(json.dumps(valid_round))
    bad["tasks"][0].update(english=english, spanish=spanish)
    fake_llm.queue(bad, {"english": [english]}, valid_round)
    build_sentence_round(spanish_bank, "Spain", "restaurant")
    assert "not English" in fake_llm.calls[2]["messages"][-1]["content"]


def test_round_with_double_article_is_asked_again(fake_llm, spanish_bank, valid_round):
    """What: A Spanish practice round whose answer has two articles together ("Me gusta el la ropa vieja.") is asked for again.
    Why: The reviewer saw "el la ropa vieja" as an example answer.
    If it fails: The learner is taught "el la"."""
    bad = json.loads(json.dumps(valid_round))
    bad["tasks"][0].update(english="I like the ropa vieja.", spanish="Me gusta el la ropa vieja.")
    fake_llm.queue(bad, valid_round)
    build_sentence_round(spanish_bank, "Spain", "restaurant")
    assert "two articles" in fake_llm.calls[1]["messages"][-1]["content"]