"""
Collect real SLM conversations for the quality evaluation (Spanish and Macedonian).

Plays every scenario in testing/slm_quality/scenarios.yaml through the SAME
functions the app calls (Stage 2 and Stage 3), against the real Ollama model,
`repeats` times each. Saves every SLM reply with its context, timing and
automatic checks (testing/slm_quality/checks.yaml explains each check), plus a
judge packet for the judge agents.

    .venv/bin/python testing/tools/collect_transcripts.py --out testing/reports/<run>
    .venv/bin/python testing/tools/collect_transcripts.py --out testing/reports/<run> --language macedonian
    .venv/bin/python testing/tools/collect_transcripts.py --out testing/reports/<run> --repeats 1 --only roleplay
    .venv/bin/python testing/tools/collect_transcripts.py --out testing/reports/<run> --model gemma3:12b

Writes  <out>/transcripts.json  and  <out>/judge_packet.json
"""
import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]   # the repository root
TESTING = ROOT / "testing"
sys.path.insert(0, str(ROOT))

import llm  # noqa: E402
import sentence_practice  # noqa: E402
from llm import LLMError  # noqa: E402
from practice import spanish_answer, spanish_typing_activity, topic_sentences  # noqa: E402
from retriever import VARIETIES, Retriever, normalize, parse_phrases  # noqa: E402
from roleplay_practice import SCENES, _reference_context, roleplay_response  # noqa: E402
from sentence_practice import build_sentence_round, looks_like_code, sentence_feedback  # noqa: E402
from tutor import is_partner_phrase  # noqa: E402

SCENARIOS = TESTING / "slm_quality" / "scenarios.yaml"
EN_WORDS = {"the", "and", "you", "is", "are", "what", "would", "like", "please", "your", "to",
            "of", "it", "this", "that", "can", "i", "my", "with", "for", "how", "do", "have"}
CYR = re.compile(r"[Ѐ-ӿ]")
LETTER = re.compile(r"[A-Za-zÀ-ÿЀ-ӿ]")
FOREIGN = re.compile(r"[яыэёщъьйюћђ]", re.I)       # Russian / Serbian letters, not Macedonian
NOTES_REMARK = re.compile(r"said when|you will hear|\(the waiter|answers:|^waiter:", re.I)


# --- Recording every SLM call -------------------------------------------------------

class Recorder:
    """Replaces sentence_practice.generate_response (used by Stage 2 and Stage 3)."""

    def __init__(self, model=None):
        self.model = model
        self.calls = []
        self.current = None

    def __call__(self, messages, model_config=None):
        cfg = dict(model_config or {})
        if self.model:
            cfg["model"] = self.model
        start = time.time()
        try:
            stats = llm.generate_response_with_stats(messages, cfg)
        except LLMError as e:
            self.calls.append({"item": self.current, "error": str(e),
                               "seconds": round(time.time() - start, 2)})
            raise
        self.calls.append({"item": self.current, "seconds": round(time.time() - start, 2),
                           "output_tokens": stats["output_tokens"],
                           "tokens_per_second": stats["tokens_per_second"],
                           "raw": stats["text"][:3000]})
        return stats["text"]

    def for_item(self, item_id):
        return [c for c in self.calls if c["item"] == item_id]


def english_share(text):
    words = re.findall(r"[a-záéíóúñü']+", text.lower())
    return sum(w in EN_WORDS for w in words) / len(words) if words else 0.0


def cyrillic_share(text):
    letters = LETTER.findall(text or "")
    return len(CYR.findall(text or "")) / len(letters) if letters else 0.0


def in_target_language(text, language):
    if language == "macedonian":
        return cyrillic_share(text) >= 0.85 and not FOREIGN.search(text)
    return english_share(text) < 0.2


def written_in_english(text, language):
    """Feedback for the learner must be English."""
    return cyrillic_share(text) <= 0.3 if language == "macedonian" else english_share(text) > 0.05


def sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


# --- Scenario players ------------------------------------------------------------------

def play_round(sc, ctx, rec, item_id):
    lang = ctx["language"]
    rec.current = item_id
    refs = topic_sentences(ctx["retriever"], lang, sc["topic"], sc["variety"]) if lang == "macedonian" else None
    data = build_sentence_round(ctx["bank"], ctx["label"], sc["topic"], lang, refs)
    bank_words = {w for p in ctx["bank"] for w in spanish_answer(p["text"]).split() if len(w) > 3}
    pairs = data["examples"] + data["tasks"]
    uses_bank = [bool(set(spanish_answer(p[lang]).split()) & bank_words) for p in pairs]
    checks = {
        "schema_valid": True,  # build_sentence_round only returns validated rounds
        "target_language": all(in_target_language(p[lang], lang) for p in pairs),
        "uses_studied_words": sum(uses_bank) >= 6,
    }
    return [{"item_id": item_id, "kind": "sentence_round", "input": None, "output": data,
             "checks": checks}]


def play_feedback(sc, ctx, rec, base_id):
    lang, items = ctx["language"], []
    for i, ans in enumerate(sc["answers"]):
        item_id = f"{base_id}.a{i + 1}"
        rec.current = item_id
        typed = ans["text"]
        answer = sentence_practice.to_cyrillic(typed) if lang == "macedonian" else typed  # as the app does
        try:
            out = sentence_feedback(sc["task"], answer, [], ctx["bank"], ctx["label"], lang)
        except LLMError as e:
            items.append({"item_id": item_id, "kind": "sentence_feedback", "input": {"task": sc["task"], **ans},
                          "output": None, "error": str(e), "checks": {"no_error": False}})
            continue
        checks = {"no_error": True}
        for key, want in (ans.get("expect") or {}).items():
            checks[f"verdict_{key}"] = out.get(key) == want
        text = out["meaning_feedback"] + " " + out["grammar_feedback"] + " " + out["hint"]
        checks["feedback_in_english"] = written_in_english(text, lang)
        checks["no_code_words"] = not any(looks_like_code(out[k]) for k in ("meaning_feedback", "grammar_feedback")) \
            and not (out["hint"].strip() and looks_like_code(out["hint"]))
        items.append({"item_id": item_id, "kind": "sentence_feedback",
                      "input": {"task": sc["task"], **ans, "read_as": answer}, "output": out, "checks": checks})
    return items


def play_roleplay(sc, ctx, rec, base_id):
    lang = ctx["language"]
    partner, title, goal = SCENES[sc["topic"]][sc.get("scene", 0)]
    scene = {"partner": partner, "title": title, "goal": goal, "round_number": 1,
             "previous_openings": []}
    history, items = [], []
    turns = [None] + list(sc["learner"])
    for t, learner in enumerate(turns):
        final = t == len(turns) - 1
        item_id = f"{base_id}.t{t}"
        rec.current = item_id
        typed = (sentence_practice.to_cyrillic(learner) if lang == "macedonian" else learner) if learner else None
        pending = history + ([{"role": "user", "text": typed}] if typed else [])
        kind = "roleplay_final" if final else "roleplay_turn"
        try:
            out = roleplay_response(scene, pending, ctx["bank"], sc["variety"], ctx["label"],
                                    sc["topic"], ctx["retriever"], final=final, language=lang)
        except LLMError as e:
            items.append({"item_id": item_id, "kind": kind, "input": {"learner": typed, "history": history},
                          "output": None, "error": str(e), "checks": {"no_error": False}})
            break   # the app also stops here until the learner retries
        reply = out["reply"]
        n_calls = len(rec.for_item(item_id))
        checks = {
            "no_error": True,
            "reply_in_target_language": in_target_language(reply, lang),
            "short_reply": len(sentences(reply)) <= 3 and len(reply) <= 220,
            "no_echo": not typed or spanish_answer(typed) not in spanish_answer(reply)
                       or len(typed.split()) < 3,
            "english_meaning_given": english_share(out["english"]) > 0.1 or len(out["english"].split()) <= 3,
            "english_is_a_translation": not NOTES_REMARK.search(out["english"]) and len(out["english"]) <= 120,
        }
        fallback = (lang == "macedonian" and not final and n_calls >= 3
                    and normalize(reply) in ctx["partner_lines"])
        if lang == "macedonian" and not final:
            checks["no_fallback_needed"] = not fallback
        if final:
            checks["feedback_in_english"] = written_in_english(
                out["meaning_feedback"] + " " + out["grammar_feedback"], lang)
            checks["closing_has_no_question"] = "?" not in reply
            checks["no_code_words"] = not any(looks_like_code(t) for t in
                                              [out["meaning_feedback"], out["grammar_feedback"]]
                                              + [c["explanation"] for c in out.get("corrections") or []])
            if "goal_met" in (sc.get("expect") or {}):
                checks["verdict_goal_met"] = out["goal_met"] == sc["expect"]["goal_met"]
        items.append({"item_id": item_id, "kind": kind, "fallback": fallback,
                      "input": {"learner": typed, "typed": learner, "history": [dict(m) for m in history]},
                      "output": out, "checks": checks})
        history = pending + [{"role": "assistant", "text": reply, "english": out["english"]}]
    return items


PLAYERS = {"sentence_round": play_round, "sentence_feedback": play_feedback,
           "roleplay": play_roleplay}


# --- Main ------------------------------------------------------------------------------

def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def check_ollama(model):
    try:
        llm.generate_response_with_stats([{"role": "user", "content": "Say OK."}],
                                         {"num_predict": 5, **({"model": model} if model else {})})
        return None
    except LLMError as e:
        return str(e)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="run folder, e.g. testing/reports/2026-11-17_1530")
    ap.add_argument("--repeats", type=int, help="runs per scenario (default: from scenarios.yaml)")
    ap.add_argument("--model", help="Ollama model to test (default: llm.DEFAULT_MODEL)")
    ap.add_argument("--language", choices=["spanish", "macedonian"], help="run one language only")
    ap.add_argument("--only", help="comma-separated scenario ids or kinds to run")
    args = ap.parse_args()

    spec = yaml.safe_load(SCENARIOS.read_text(encoding="utf-8"))
    settings = spec["settings"]
    repeats = args.repeats or settings["repeats"]
    scenarios = [{"language": "spanish", **s} for s in spec["scenarios"]]
    if args.language:
        scenarios = [s for s in scenarios if s["language"] == args.language]
    if args.only:
        keep = {x.strip() for x in args.only.split(",")}
        scenarios = [s for s in scenarios if s["id"] in keep or s["kind"] in keep]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = args.model or llm.DEFAULT_MODEL
    meta = {"model": model, "commit": git_commit(), "started": datetime.now().isoformat(timespec="seconds"),
            "repeats": repeats, "settings": settings, "languages": sorted({s["language"] for s in scenarios})}

    problem = check_ollama(args.model)
    if problem:
        meta["skipped"] = f"SLM not available: {problem}"
        (out / "transcripts.json").write_text(json.dumps({"meta": meta, "scenarios": []}, indent=2))
        print(f"SKIPPED slm_quality: {problem}")
        return 2

    rec = Recorder(args.model)
    sentence_practice.generate_response = rec   # Stage 3 calls it through _json_reply too
    retriever = Retriever()
    results = []
    for sc in scenarios:
        lang = sc["language"]
        label = VARIETIES.get(sc["variety"], sc["variety"])   # the same label the app passes
        bank = spanish_typing_activity(retriever, sc["topic"], sc["variety"], language=lang)["phrases"]
        partner = {normalize(p["text"]) for c in retriever.chunks if c["language"] == lang
                   for p in parse_phrases(c) if is_partner_phrase(p)}
        ctx = {"bank": bank, "label": label, "retriever": retriever, "language": lang,
               "partner_lines": partner}
        runs = []
        for r in range(1, repeats + 1):
            base_id = f"{sc['id']}.r{r}"
            print(f"  {base_id} ...", flush=True)
            start = time.time()
            try:
                items = PLAYERS[sc["kind"]](sc, ctx, rec, base_id)
                error = None
            except LLMError as e:
                items, error = [{"item_id": base_id, "kind": sc["kind"], "input": None,
                                 "output": None, "error": str(e), "checks": {"no_error": False}}], str(e)
            for it in items:
                calls = rec.for_item(it["item_id"])
                it["slm_calls"] = len(calls)
                it["retries"] = max(0, len(calls) - 1)
                it["seconds"] = round(sum(c["seconds"] for c in calls), 2)
                it["tokens_per_second"] = max((c.get("tokens_per_second", 0) for c in calls), default=0)
                it["failed_raw"] = [c.get("raw") or c.get("error") for c in calls[:-1]][:2]
            runs.append({"repeat": r, "seconds": round(time.time() - start, 1), "error": error,
                         "items": items})
        results.append({**{k: sc[k] for k in ("id", "kind", "topic", "variety", "about", "language")},
                        "variety_name": label if lang == "spanish" else "Macedonian (MK)",
                        "bank": [p["text"] for p in bank],
                        "notes_excerpt": _reference_context(retriever, sc["topic"], sc["variety"], lang)[:1500]
                        if sc["kind"] == "roleplay" else "",
                        "runs": runs})

    meta["finished"] = datetime.now().isoformat(timespec="seconds")
    (out / "transcripts.json").write_text(
        json.dumps({"meta": meta, "scenarios": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_judge_packet(out, results)
    n = sum(len(r["items"]) for s in results for r in s["runs"])
    print(f"Saved {n} SLM replies from {len(results)} scenarios to {out / 'transcripts.json'}")
    return 0


def write_judge_packet(out, results):
    """Everything the judge needs, and nothing else (no timings, no auto checks)."""
    items = []
    for sc in results:
        for run in sc["runs"]:
            for it in run["items"]:
                if it.get("output") is None:
                    continue
                items.append({
                    "item_id": it["item_id"], "kind": it["kind"], "language": sc["language"],
                    "scenario": sc["about"], "variety": sc["variety_name"], "topic": sc["topic"],
                    "studied_bank": sc["bank"], "notes_excerpt": sc["notes_excerpt"],
                    "learner_input": it["input"], "slm_reply": it["output"],
                })
    (out / "judge_packet.json").write_text(json.dumps(items, ensure_ascii=False, indent=2),
                                           encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
