"""
SLM-contract set, Macedonian: the code around the Macedonian SLM calls, with the
SLM replaced by FakeLLM. Macedonian has extra checks (Cyrillic only, no Serbian
or Russian forms, no customer-style staff lines, English feedback) and a safe
fallback line from the notes. These tests feed known bad replies and check that
the app catches each one.
"""
import json
import re

import pytest

from llm import LLMError
from retriever import normalize, parse_phrases
from roleplay_practice import SCENES, roleplay_response
from sentence_practice import build_sentence_round, sentence_feedback
from tutor import is_partner_phrase
from conftest import make_round
from practice import topic_sentences

pytestmark = pytest.mark.slm_contract
MK = "North Macedonia"


def scene(topic="restaurant"):
    partner, title, goal = SCENES[topic][0]
    return {"partner": partner, "title": title, "goal": goal, "round_number": 1,
            "previous_openings": []}


def turn(reply, english="Hello, what can I get you?", hint="You could order a drink."):
    return {"reply": reply, "english": english, "hint": hint}


def partner_lines(retriever, topic="restaurant"):
    return {normalize(p["text"]) for c in retriever.chunks
            if c["source"] == f"macedonian/{topic}.md" for p in parse_phrases(c) if is_partner_phrase(p)}


# --- Stage 2: build_sentence_round (Macedonian) -------------------------------------

def test_mk_round_gets_checked_reference_sentences(fake_llm, macedonian_bank):
    """What: The Macedonian Sentences (Stage 2) request contains checked sentences from the notes and the Macedonian rules.
    Why: The small model invents wrong Macedonian forms when it has no correct patterns.
    If it fails: Stage 2 rounds contain more invented or Serbian-like Macedonian."""
    refs = [{"macedonian": "Дали имате мени?", "english": "Do you have a menu?"}]
    fake_llm.queue(make_round(len(macedonian_bank), language="macedonian"))
    build_sentence_round(macedonian_bank, MK, "restaurant", "macedonian", refs)
    sent = json.loads(fake_llm.calls[0]["messages"][1]["content"])
    assert sent["reference_sentences"] == refs
    assert "Cyrillic only" in fake_llm.calls[0]["messages"][0]["content"]


def test_mk_round_in_latin_letters_is_never_shown(fake_llm, macedonian_bank):
    """What: A generated Macedonian sentence in Latin letters is not shown; the round is completed with other sentences.
    Why: The learner must see the answer in Cyrillic.
    If it fails: The "correct" Macedonian answer can show in Latin letters."""
    bad = make_round(len(macedonian_bank), language="macedonian")
    bad["tasks"][0]["macedonian"] = "Bi sakal edna shopska salata."
    fake_llm.queue(bad, make_round(len(macedonian_bank), language="macedonian"))
    data = build_sentence_round(macedonian_bank, MK, "restaurant", "macedonian")
    shown = [p["macedonian"] for p in data["examples"] + data["tasks"]]
    assert "Bi sakal edna shopska salata." not in shown
    assert all(re.search(r"[Ѐ-ӿ]", t) and not re.search(r"[A-Za-z]", t) for t in shown)


def test_mk_round_bad_twice_is_filled_from_the_notes(fake_llm, macedonian_bank, retriever):
    """What: After two bad Macedonian rounds (Spanish: an error), Sentences (Stage 2) fills the round with checked sentences from the notes.
    Why: A Macedonian round should always open; the notes always have correct sentences for the topic.
    If it fails: The learner sees "not usable" and cannot practice Macedonian sentences for that topic."""
    bad = make_round(len(macedonian_bank), language="macedonian")
    for pair in bad["examples"] + bad["tasks"]:
        pair["macedonian"] = "Bi sakal edna shopska salata."
    fake_llm.queue(bad, bad)
    refs = topic_sentences(retriever, "macedonian", "restaurant", "MK")
    data = build_sentence_round(macedonian_bank, MK, "restaurant", "macedonian", refs)
    assert len(fake_llm.calls) == 2
    notes = {r["macedonian"] for r in refs}
    assert len(data["examples"]) == 3 and len(data["tasks"]) == 5
    assert all(p["macedonian"] in notes for p in data["examples"] + data["tasks"])
    assert data["stats"]["from_notes"] == 8


def test_mk_round_timeout_uses_the_notes(fake_llm, macedonian_bank, retriever):
    """What: When the SLM times out, the Macedonian round is built from the notes at once (no second wait).
    Why: On a slow laptop the round can take longer than the time limit.
    If it fails: The learner waits twice and then sees "Ollama didn't answer within 120 seconds"."""
    fake_llm.queue(LLMError("Ollama didn't answer within 120 seconds."))
    refs = topic_sentences(retriever, "macedonian", "hotel", "MK")
    data = build_sentence_round(macedonian_bank, MK, "hotel", "macedonian", refs)
    assert len(fake_llm.calls) == 1 and len(data["tasks"]) == 5


def test_mk_round_drops_words_not_in_the_notes(fake_llm, macedonian_bank, retriever):
    """What: A generated Macedonian sentence with a word that is not in the notes ("батола", "аптечна") is not shown; good generated sentences are kept.
    Why: The small model invents words. Every word in the notes was checked by the native speaker.
    If it fails: The learner sees invented Macedonian as the example answer."""
    good = make_round(len(macedonian_bank), language="macedonian")
    good["tasks"][0]["macedonian"] = "Ми треба шампон и батола паракетамол."
    good["tasks"][0]["english"] = "I need shampoo and a bottle of paracetamol."
    fake_llm.queue(good, good)
    refs = topic_sentences(retriever, "macedonian", "restaurant", "MK")
    data = build_sentence_round(macedonian_bank, MK, "restaurant", "macedonian", refs)
    shown = [p["macedonian"] for p in data["examples"] + data["tasks"]]
    assert "Ми треба шампон и батола паракетамол." not in shown
    assert data["stats"]["generated_rejected"] >= 1


def test_mk_round_uses_at_most_three_generated_sentences(fake_llm, macedonian_bank, retriever):
    """What: A Macedonian round shows at most 3 generated sentences, all as tasks; the 3 examples always come from the notes.
    Why: Generated Macedonian can still have wrong grammar. The examples are what the learner copies, so they must be checked text.
    If it fails: The learner copies a generated example with a mistake, or most of the round is unchecked Macedonian."""
    fake_llm.queue(make_round(len(macedonian_bank), language="macedonian"))
    refs = topic_sentences(retriever, "macedonian", "restaurant", "MK")
    data = build_sentence_round(macedonian_bank, MK, "restaurant", "macedonian", refs)
    notes = {r["macedonian"] for r in refs}
    assert len(fake_llm.calls) == 1
    assert all(p["macedonian"] in notes for p in data["examples"])
    generated = [p for p in data["tasks"] if p["macedonian"] not in notes]
    assert len(generated) <= 3 and data["stats"]["generated_kept"] <= 3
    assert len(data["examples"]) == 3 and len(data["tasks"]) == 5


# --- Stage 2: sentence_feedback (Macedonian) ----------------------------------------

TASK = {"english": "The bill, please.", "macedonian": "Сметката, ве молам.", "vocabulary_ids": [1]}
FEEDBACK = {"meaning_ok": False, "grammar_ok": True,
            "meaning_feedback": "You asked for the menu, not the bill.",
            "grammar_feedback": "The structure is fine.", "hint": "Which word did you study for 'bill'?"}


def test_mk_exact_answer_needs_no_slm_call(fake_llm, macedonian_bank):
    """What: A Macedonian answer that matches the example (any case or punctuation) is correct without an SLM call.
    Why: This is fast and can never be judged wrong.
    If it fails: The SLM can reject a perfect Macedonian answer."""
    result = sentence_feedback(TASK, "сметката ве молам", [], macedonian_bank, MK, "macedonian")
    assert result["meaning_ok"] and result["grammar_ok"] and fake_llm.calls == []


def test_mk_feedback_in_macedonian_is_sent_back(fake_llm, macedonian_bank):
    """What: Feedback written in Macedonian is sent back with "Fix the feedback", and the English version is used.
    Why: A beginner cannot read feedback in Macedonian. The commits list this as a known problem.
    If it fails: The learner gets feedback in (often broken) Macedonian."""
    fake_llm.queue({**FEEDBACK, "meaning_feedback": "Барате мени наместо сметка."}, FEEDBACK)
    result = sentence_feedback(TASK, "Менито, ве молам", [], macedonian_bank, MK, "macedonian")
    assert result["meaning_feedback"] == FEEDBACK["meaning_feedback"]
    assert "Fix the feedback" in fake_llm.calls[1]["messages"][-1]["content"]


def test_mk_invented_words_never_reach_the_learner(fake_llm, macedonian_bank):
    """What: Feedback with an invented Macedonian word is sent back. If it fails twice, the app uses safe English text and keeps the verdict.
    Why: The SLM invents forms like "сладка страва". The learner must not learn them.
    If it fails: The learner can learn a Macedonian word that does not exist."""
    bad = {**FEEDBACK, "hint": "Try saying сладка страва."}
    fake_llm.queue(bad, bad)
    result = sentence_feedback(TASK, "Менито, ве молам", [], macedonian_bank, MK, "macedonian")
    assert "страва" not in result["hint"] and result["meaning_ok"] is False


@pytest.mark.parametrize("label", ["Meaning_ok", "meaning_not_ok", "Meaning_no", "true",
                                   "Meaning_ok Grammar_ok", "Your meaning_ok is false here."])
def test_mk_label_only_feedback_becomes_a_sentence(fake_llm, macedonian_bank, label):
    """What: Macedonian feedback that is a code label ("Meaning_no", "meaning_not_ok") or contains one becomes a real English sentence.
    Why: Learners saw these labels in the app. They look like a bug and explain nothing.
    If it fails: The learner sees a code word instead of feedback."""
    fake_llm.queue({**FEEDBACK, "meaning_feedback": label})
    result = sentence_feedback(TASK, "Менито, ве молам", [], macedonian_bank, MK, "macedonian")
    assert len(result["meaning_feedback"].split()) >= 4 and "_" not in result["meaning_feedback"]


def test_mk_end_feedback_never_looks_like_code(fake_llm, macedonian_bank, retriever):
    """What: Macedonian end feedback never shows code-like words such as "goal_met" or "grammar_ok".
    Why: The end feedback has the same field names as Sentences (Stage 2), so the same problem can happen.
    If it fails: The learner sees code words in the conversation feedback."""
    fake_llm.queue({**FINAL, "meaning_feedback": "goal_met", "grammar_feedback": "Grammar_ok"})
    out = final(fake_llm, macedonian_bank, retriever)
    assert "_" not in out["meaning_feedback"] + out["grammar_feedback"]
    assert out["meaning_feedback"] == "You reached the goal of this conversation."


# --- Stage 3: roleplay_response (Macedonian turns) -------------------------------------

def test_mk_good_reply_is_used_first_time(fake_llm, macedonian_bank, retriever):
    """What: A good Macedonian staff line passes all the checks on the first try.
    Why: The checks must not reject good replies. Each retry adds waiting time.
    If it fails: Good replies are replaced with fallback lines, and the learner waits longer."""
    fake_llm.queue(turn("Добар ден! Повелете."))
    out = roleplay_response(scene(), [], macedonian_bank, "MK", MK, "restaurant", retriever,
                            language="macedonian")
    assert out["reply"] == "Добар ден! Повелете." and len(fake_llm.calls) == 1
    data = json.loads(fake_llm.calls[0]["messages"][1]["content"])
    assert data["partner_lines"], "the SLM must get the staff lines from the notes"


HISTORY_DRINK = [{"role": "assistant", "text": "Повелете."},
                 {"role": "user", "text": "Едно кафе, ве молам."}]


@pytest.mark.parametrize("bad, history, problem", [
    ("Може ли една чаша чај?", [], "asks like a customer"),
    ("Да, ве молам.", [], "says the customer's 'ве молам'"),
    ("Hello, welcome!", [], "not in Cyrillic"),
    ("Хвала, добро.", [], "Serbian form хвала"),
    ("Што сакате за пиење?", HISTORY_DRINK, "asks about drinks again"),
    ("Едно кафе, ве молам.", HISTORY_DRINK, "echoes the traveler"),
])
def test_mk_bad_staff_lines_are_caught(fake_llm, macedonian_bank, retriever, bad, history, problem):
    """What: Known bad Macedonian partner lines are rejected and retried, then replaced by a staff line from the notes.
    Why: The 4B model often speaks as the customer, uses Serbian forms, repeats the traveler, or asks about drinks again.
    If it fails: The learner sees the waiter act as a customer, wrong-language words, or a broken scene."""
    fake_llm.queue(turn(bad), turn(bad), turn(bad))
    out = roleplay_response(scene(), history, macedonian_bank, "MK", MK, "restaurant", retriever,
                            language="macedonian")
    assert out["reply"] != bad, problem
    assert len(fake_llm.calls) == 3
    assert "Write a new reply" in fake_llm.calls[1]["messages"][-1]["content"]


def test_mk_fallback_line_is_a_staff_line_from_the_notes(fake_llm, macedonian_bank, retriever):
    """What: After 3 bad replies, the partner says a checked staff line from the notes.
    Why: A checked line is always correct Macedonian and in role.
    If it fails: The learner sees a bad SLM line or an empty bubble."""
    fake_llm.queue(*[turn("Може ли една чаша чај?")] * 3)
    out = roleplay_response(scene(), [], macedonian_bank, "MK", MK, "restaurant", retriever,
                            language="macedonian")
    assert normalize(out["reply"]) in partner_lines(retriever)


def test_mk_fallback_english_is_a_translation(fake_llm, macedonian_bank, retriever):
    """What: The English under a fallback line is a translation, not a remark from the notes.
    Why: "Show English meaning" must translate the partner's line.
    If it fails: The learner reads notes text (for example "said when inviting you to sit down… You will hear it constantly") as the meaning."""
    fake_llm.queue(*[turn("Може ли една чаша чај?")] * 3)
    out = roleplay_response(scene(), [], macedonian_bank, "MK", MK, "restaurant", retriever,
                            language="macedonian")
    english = out["english"].lower()
    assert "said when" not in english and "you will hear" not in english and len(english) <= 60, out["english"]


def test_mk_staff_price_question_is_asked_again(fake_llm, macedonian_bank, retriever):
    """What: A staff line that asks the price ("Колку е по ноќ?") with English that is not its translation is sent back and replaced.
    Why: The price question is the traveler's line; the staff gives the price. The English must translate the line shown.
    If it fails: The receptionist asks the guest's question, and "Show English meaning" shows a different sentence."""
    fake_llm.queue(turn("Колку е по ноќ?", english="The room costs 60 euros per night."),
                   turn("Добар ден! Повелете."))
    out = roleplay_response(scene(), [], macedonian_bank, "MK", MK, "restaurant", retriever,
                            language="macedonian")
    assert out["reply"] == "Добар ден! Повелете." and len(fake_llm.calls) == 2
    sent_back = fake_llm.calls[1]["messages"][-1]["content"]
    assert "price" in sent_back and "english" in sent_back


def test_mk_empty_fields_get_another_try(fake_llm, macedonian_bank, retriever):
    """What: A Macedonian turn with an empty field is asked again before the app gives up.
    Why: The 4B model often leaves the hint or translation empty.
    If it fails: The learner sees an error that one more try would fix."""
    fake_llm.queue(turn("Добар ден! Повелете.", hint=""), turn("Добар ден! Повелете."))
    out = roleplay_response(scene(), [], macedonian_bank, "MK", MK, "restaurant", retriever,
                            language="macedonian")
    assert out["hint"] and len(fake_llm.calls) == 2


# --- Stage 3: end feedback (Macedonian) -------------------------------------------------

HISTORY = [{"role": "assistant", "text": "Повелете."},
           {"role": "user", "text": "Едно кафе, ве молам."},
           {"role": "assistant", "text": "Веднаш."},
           {"role": "user", "text": "Може ли сметката?"},
           {"role": "assistant", "text": "Секако."},
           {"role": "user", "text": "Благодарам."}]
FINAL = {"meaning_feedback": "You ordered and asked for the bill clearly.",
         "grammar_feedback": "Your sentences are correct.", "goal_met": True, "corrections": []}


def final(fake_llm, macedonian_bank, retriever):
    return roleplay_response(scene(), HISTORY, macedonian_bank, "MK", MK, "restaurant", retriever,
                             final=True, language="macedonian")


def test_mk_end_has_fixed_closing_and_english_feedback(fake_llm, macedonian_bank, retriever):
    """What: The Macedonian scene ends with the fixed line "Благодарам, пријатно!" and English feedback from a separate call.
    Why: Inside the role-play, the 4B model stays in character and writes the feedback in Macedonian.
    If it fails: The learner gets no feedback, or feedback in Macedonian."""
    fake_llm.queue(FINAL)
    out = final(fake_llm, macedonian_bank, retriever)
    assert out["reply"] == "Благодарам, пријатно!" and out["meaning_feedback"] == FINAL["meaning_feedback"]
    assert "English-speaking tutor" in fake_llm.calls[0]["messages"][0]["content"]


def test_mk_end_feedback_in_macedonian_is_retried(fake_llm, macedonian_bank, retriever):
    """What: End feedback in Macedonian is asked again in English. Two failures give a friendly error.
    Why: A beginner cannot read the feedback in Macedonian.
    If it fails: The learner gets end feedback that they cannot read."""
    mk = {**FINAL, "meaning_feedback": "Добро нарачавте и ја побаравте сметката."}
    fake_llm.queue(mk, FINAL)
    assert final(fake_llm, macedonian_bank, retriever)["meaning_feedback"] == FINAL["meaning_feedback"]
    fake_llm.queue(mk, mk)
    with pytest.raises(LLMError):
        final(fake_llm, macedonian_bank, retriever)


def test_mk_empty_grammar_feedback_gets_a_sentence(fake_llm, macedonian_bank, retriever):
    """What: Empty grammar feedback becomes "No structure or grammar problems needed revision."
    Why: The model leaves it empty when there is nothing to fix.
    If it fails: The learner sees an empty feedback section."""
    fake_llm.queue({**FINAL, "grammar_feedback": ""})
    assert "No structure" in final(fake_llm, macedonian_bank, retriever)["grammar_feedback"]


@pytest.mark.parametrize("correction, kept", [
    ({"learner_text": "Може ли сметката?", "example_revision": "Може ли сметката, ве молам?",
      "explanation": "Add 've molam' to be more polite."}, True),
    ({"learner_text": "Може ли сметката?", "example_revision": "Може ли сметката?",
      "explanation": "Add a word."}, False),
    ({"learner_text": "Благодарам.", "example_revision": "Ви благодарам.",
      "explanation": "This is correct."}, False),
    ({"learner_text": "I never wrote this", "example_revision": "Ништо.",
      "explanation": "Wrong word."}, False),
])
def test_mk_only_real_corrections_are_kept(fake_llm, macedonian_bank, retriever, correction, kept):
    """What: The app removes Macedonian "corrections" that change nothing, call the line correct, or quote a line that the learner never wrote.
    Why: The model invents mistakes and corrects correct lines.
    If it fails: The learner is told that a correct answer is wrong."""
    fake_llm.queue({**FINAL, "corrections": [correction]})
    out = final(fake_llm, macedonian_bank, retriever)
    assert (len(out["corrections"]) == 1) == kept


def test_mk_end_answers_a_last_question_before_goodbye(fake_llm, macedonian_bank, retriever):
    """What: When the traveler's last (third) line is a question, the partner answers it before saying goodbye.
    Why: The scene ends after the third response; a fixed goodbye ignores a question like "Колку чини една ноќ?".
    If it fails: The receptionist says "Благодарам, пријатно!" and never gives the price the learner asked for."""
    history = HISTORY[:-1] + [{"role": "user", "text": "Колку чини една ноќ?"}]
    fake_llm.queue(turn("Собата чини 60 евра по ноќ.", english="The room costs 60 euros per night."), FINAL)
    out = roleplay_response(scene(), history, macedonian_bank, "MK", MK, "hotel", retriever,
                            final=True, language="macedonian")
    assert out["reply"] == "Собата чини 60 евра по ноќ. Пријатно!"
    assert out["english"] == "The room costs 60 euros per night. Goodbye!"
    assert out["meaning_feedback"] == FINAL["meaning_feedback"] and len(fake_llm.calls) == 2
