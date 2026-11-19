"""Unit set, Macedonian: pronunciation guide, Latin input, Stage 1 and Stage 2 checks. No SLM."""
import re

import pytest

from practice import spanish_feedback, spanish_typing_activity, topic_sentences
from pronounce import say, to_cyrillic, transliterate
from retriever import parse_phrases
from sentence_practice import _feedback_problems, _sentence_count, _validate_round
from conftest import make_round

pytestmark = pytest.mark.unit

STAGE1_EXCLUDE = re.compile(r"\.\.\.|…|__|[/();=]")


@pytest.fixture(scope="module")
def mk_phrases(retriever):
    return [p for c in retriever.chunks if c["language"] == "macedonian" for p in parse_phrases(c)]


# --- pronounce.py --------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("Сита сум.", "Sita sum."), ("Ќе ми донесете ли сметка?", "Kje mi donesete li smetka?"),
    ("Џеб, ѓавол, ѕвезда", "Dzheb, gjavol, dzvezda"), ("Љубов, њива", "Ljubov, njiva"),
])
def test_transliterate(text, expected):
    """What: Macedonian Cyrillic changes to the Latin spelling that the notes use (sh, ch, zh, kj, gj).
    Why: The learner reads this Latin line on every Macedonian card and partner line.
    If it fails: The Latin line under a Macedonian phrase is wrong, and the learner learns a wrong spelling."""
    assert transliterate(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("Сита сум.", "SEE-ta soom."), ("не знам", "NE znam"), ("Може ли", "MO-zhe lee"),
    ("Да, ве молам.", "DA, VE mo-lam."),
])
def test_say_follows_the_stress_rules(text, expected):
    """What: The pronunciation guide puts the stress (CAPITALS) where the Macedonian rules say.
    Why: The SLM writes new partner lines. The app makes their guide with these rules.
    If it fails: The learner reads a wrong stress and pronounces new lines wrongly."""
    assert say(text) == expected


def test_say_matches_the_native_speaker_guides(mk_phrases):
    """What: The generated guide matches the native speaker's hand-checked guide for at least 99% of the notes phrases.
    Why: The hand-checked guides are correct. The generator must give the same result for new lines.
    If it fails: The rules in pronounce.py changed or broke. Many new partner lines can show a wrong pronunciation."""
    checked = [p for p in mk_phrases if p.get("say")]
    same = [p for p in checked if say(p["text"]).strip('".,!? ') == p["say"].strip('".,!? ')]
    differ = [f"{p['text']}: notes {p['say']} / generated {say(p['text'])}" for p in checked if p not in same]
    assert len(same) / len(checked) >= 0.99, "\n".join(differ[:10])


@pytest.mark.parametrize("latin, cyrillic", [
    ("Sakam edno kafe, ve molam.", "Сакам едно кафе, ве молам."),
    ("Kje platam so karticka", "Ќе платам со картицка"),
    ("Blagodaram", "Благодарам"), ("Сакам кафе", "Сакам кафе"),
])
def test_to_cyrillic(latin, cyrillic):
    """What: A Macedonian answer typed in Latin letters changes to Cyrillic. Cyrillic stays as it is.
    Why: Many learners have no Macedonian keyboard.
    If it fails: A learner without a Cyrillic keyboard cannot give a correct answer."""
    assert to_cyrillic(latin) == cyrillic


def test_card_latin_spelling_is_accepted_in_stage1(mk_phrases):
    """What: Each Stage 1 card item, typed exactly as its Latin line on the card, is marked correct.
    Why: The card tells the learner that they can type the Latin spelling.
    If it fails: The learner types what the card shows and the app says "Not quite". This is unfair and confusing."""
    items = [p for p in mk_phrases if p.get("pron") and not STAGE1_EXCLUDE.search(p["text"])
             and len(p["text"].split()) <= 12]
    rejected = [f"{p['text']} | card: {p['pron']} | read as: {to_cyrillic(p['pron'])}"
                for p in items if spanish_feedback(to_cyrillic(p["pron"]), p["text"]) != "correct"]
    assert not rejected, f"{len(rejected)} of {len(items)} card spellings rejected:\n" + "\n".join(rejected)


# --- Stage 1 and Stage 2 content -------------------------------------------------

@pytest.mark.parametrize("topic", ["restaurant", "directions", "transport", "hotel", "shopping"])
def test_macedonian_typing_activity(retriever, topic):
    """What: Each Macedonian topic gives 1–10 Cyrillic cards, and every question has an answer on a card.
    Why: Macedonian Stage 1 now uses the same typing screen as Spanish.
    If it fails: The learner gets an empty activity or a question that no card prepared them for."""
    activity = spanish_typing_activity(retriever, topic, "MK", language="macedonian")
    assert 1 <= len(activity["phrases"]) <= 10
    texts = {p["text"] for p in activity["phrases"]}
    assert all(re.search(r"[Ѐ-ӿ]", t) for t in texts)
    for q in activity["questions"]:
        assert set(q["accepted"]) <= texts


@pytest.mark.parametrize("topic", ["restaurant", "directions", "transport", "hotel"])
def test_topic_sentences_for_stage2(retriever, topic):
    """What: Stage 2 gets 3–12 word checked Macedonian sentences from the notes, with no gaps (...) and no repeats.
    Why: The SLM builds Macedonian rounds from these sentences so it does not invent wrong forms.
    If it fails: The SLM gets bad patterns or none, and Stage 2 rounds have more Macedonian errors."""
    sentences = topic_sentences(retriever, "macedonian", topic, "MK")
    assert sentences
    texts = [s["macedonian"] for s in sentences]
    assert len(set(texts)) == len(texts)
    for t in texts:
        assert 3 <= len(t.split()) <= 12 and "..." not in t and re.search(r"[Ѐ-ӿ]", t)


def test_valid_macedonian_round_passes():
    """What: A correct Macedonian sentence round passes the round check.
    Why: The check must not reject good SLM output.
    If it fails: Macedonian Stage 2 can show "not usable" errors for good rounds."""
    assert _validate_round(make_round(10, language="macedonian"), list(range(10)), "macedonian")


@pytest.mark.parametrize("bad, reason", [
    ("Sakam edno kafe, ve molam.", "Latin letters"),
    ("Сакам edno кафе.", "mixed alphabets"),
    ("Сакам едно кафе. Сакам и вода.", "two sentences for one English sentence"),
])
def test_bad_macedonian_round_is_rejected(bad, reason):
    """What: The round check rejects Macedonian that uses Latin letters, mixes alphabets, or splits one sentence into two.
    Why: The SLM makes these mistakes. The learner must see one clean Cyrillic sentence for each English sentence.
    If it fails: The learner can see Latin or mixed text as the "correct" Macedonian answer."""
    data = make_round(10, language="macedonian")
    data["tasks"][0]["macedonian"] = bad
    data["tasks"][0]["english"] = "I want a coffee."
    with pytest.raises(ValueError):
        _validate_round(data, list(range(10)), "macedonian")


@pytest.mark.parametrize("text, count", [
    ("Извинете. Може ли сметката?", 1), ("Сакам кафе со млеко. Сакам и вода од чешма.", 2),
    ("Gracias. ¿Me trae la cuenta?", 1),
])
def test_sentence_count_ignores_short_openers(text, count):
    """What: A short word in front ("Извинете.", "Gracias.") does not count as a second sentence.
    Why: Polite openers are normal. The one-sentence rule must not reject them.
    If it fails: Good rounds with a polite opener are rejected, so the learner waits for retries."""
    assert _sentence_count(text) == count


def test_feedback_problems_finds_macedonian_and_invented_words():
    """What: The feedback check finds Macedonian feedback text and Macedonian words that are not in the answer, example or bank.
    Why: Feedback must be in English, and the SLM invents Macedonian forms.
    If it fails: The learner can get feedback in broken Macedonian or learn an invented word."""
    allowed = {"сакам", "кафе"}
    assert _feedback_problems({"hint": "Use the word кафе."}, allowed) == []
    assert any("English" in p for p in _feedback_problems({"hint": "Треба да кажете сакам кафе."}, allowed))
    assert any("сладка" in p for p in _feedback_problems({"hint": "Try сладка instead."}, allowed))
