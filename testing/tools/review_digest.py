"""
Digest the human review of one test run, for the language-tutor-conversation-reviewer skill.

    .venv/bin/python testing/tools/review_digest.py testing/reports/<run>

Reads the run folder like build_report.py (transcripts, judgments, human_review*.json).
Writes  <run>/review_digest.json  and  <run>/review_digest.md:
  - coverage: how much of the review queue people decided, per language
  - problem clusters: replies that PEOPLE confirmed as not acceptable, grouped by
    language + where the reply comes from + problem type, with notes and examples
  - false alarms: replies the machine failed but people accepted (judge or check is wrong)
  - test triage: failed code tests marked as real bug / out of date / not sure
  - open: high-priority items that nobody decided yet
A decision on an example reply also covers its group of similar replies (as in the report).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_report as br  # noqa: E402

# Where each kind of SLM reply is produced, and what usually fixes it.
SOURCE = {
    "sentence_round": {
        "where": "sentence_practice.build_sentence_round (system prompt) + _validate_round (guard); "
                 "Macedonian also prompts.LANGUAGE_RULES['macedonian'] and practice.topic_sentences",
        "levers": ["prompt", "guard", "grounding data"]},
    "sentence_feedback": {
        "where": "sentence_practice.sentence_feedback (system prompt) + _feedback_problems and the "
                 "label/answer-leak post-processing",
        "levers": ["prompt", "guard", "post-processing", "deterministic pre-check"]},
    "roleplay_turn": {
        "where": "roleplay_practice.roleplay_response (system prompt, Macedonian partner_lines) + "
                 "tutor.check_reply / Macedonian checks in roleplay_response + _safe_line fallback; "
                 "Spanish and Macedonian rules in prompts.py",
        "levers": ["prompt", "guard + retry", "fallback", "language rules"]},
    "roleplay_final": {
        "where": "roleplay_practice.roleplay_response(final=True) for Spanish; "
                 "roleplay_practice._macedonian_final for Macedonian; correction filters",
        "levers": ["prompt", "post-processing", "split into a separate call"]},
}


def main(run):
    checks_doc = br.load_yaml(br.QDIR / "checks.yaml", {"code_checks": {}})
    check_text = {k: v.get("fails", "") for k, v in checks_doc.get("code_checks", {}).items()}
    settings = (br.load_yaml(br.QDIR / "scenarios.yaml", {}) or {}).get("settings", {})
    q = br.read_quality(run, settings, check_text) if (run / "transcripts.json").exists() else {"items": []}
    human = br.read_human(run)
    items = {i["key"]: i for i in q.get("items", [])}

    # Effective decision: own decision, else the decision on the example that covers it.
    covered_by = {k: i["key"] for i in items.values() for k in i.get("group", [])}

    def effective(key):
        d = human.get(key, {})
        if d.get("verdict") in ("ok", "fail"):
            return d, False
        rep = covered_by.get(key)
        rd = human.get(rep, {}) if rep else {}
        return (rd, True) if rd.get("verdict") in ("ok", "fail") else ({}, False)

    coverage = defaultdict(lambda: {"replies": 0, "queue": 0, "decided": 0, "confirmed_fail": 0,
                                    "accepted": 0, "agree": 0, "reviewed": 0})
    clusters = defaultdict(list)
    false_alarms = []
    open_high = []
    for i in items.values():
        c = coverage[i["language"]]
        c["replies"] += 1
        d, via = effective(i["key"])
        if i["review"]:
            c["queue"] += 1
            if human.get(i["key"], {}).get("verdict"):
                c["decided"] += 1
            elif i["priority"] == "high":
                open_high.append({"key": i["key"], "why": i["review"][0]["title"]})
        if not d:
            continue
        c["reviewed"] += 1
        c["agree"] += (d["verdict"] == "ok") == (i["machine"] == "ok")
        example = {
            "key": i["key"], "scenario": i["scenario"], "about": i["about"], "learner": i["learner"],
            "typed": i["typed"], "reply": br_reply(i), "judge_summary": i["summary"],
            "judge_low": {k: v for k, v in br.numeric(i["scores"]).items() if v <= 2},
            "judge_reasons": i["reasons_by_item"], "failed_checks": [k for k, v in i["checks"].items() if not v],
            "human_note": d.get("note", ""), "reviewer": d.get("reviewer", ""), "via_group": via}
        if d["verdict"] == "fail":
            c["confirmed_fail"] += 1
            tag = d.get("tag") or (i["tags"][0] if i["tags"] else "untagged")
            clusters[(i["language"], i["kind"], tag)].append(example)
        elif i["machine"] == "fail":
            c["accepted"] += 1
            false_alarms.append({**example, "language": i["language"], "kind": i["kind"]})

    out_clusters = []
    for (lang, kind, tag), ex in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        own = [e for e in ex if not e["via_group"]]
        out_clusters.append({
            "language": lang, "kind": kind, "problem": tag, "replies": len(ex),
            "scenarios": sorted({e["scenario"] for e in ex}),
            "human_notes": sorted({e["human_note"] for e in ex if e["human_note"]}),
            "where": SOURCE[kind]["where"], "levers": SOURCE[kind]["levers"],
            "examples": (own or ex)[:3]})

    tests = []
    for t in (_failed_tests(run)):
        d = human.get("test:" + t["id"], {})
        tests.append({"test": t["id"], "verdict": d.get("verdict", "not decided"), "note": d.get("note", ""),
                      "message": t["message"][:400]})

    digest = {"run": run.name, "coverage": {k: {**v, "judge_agreement": round(v["agree"] / v["reviewed"], 2)
                                                 if v["reviewed"] else None} for k, v in coverage.items()},
              "problem_clusters": out_clusters, "false_alarms": false_alarms, "test_triage": tests,
              "open_high_priority": open_high,
              "reviewers": sorted({d.get("reviewer", "") for d in human.values() if d.get("reviewer")})}
    (run / "review_digest.json").write_text(json.dumps(digest, ensure_ascii=False, indent=2), encoding="utf-8")
    (run / "review_digest.md").write_text(to_md(digest), encoding="utf-8")
    print(to_md(digest))
    if not human:
        print("WARNING: no human_review*.json in this run folder. Ask the reviewers to download their "
              "decisions from report.html and put the files here.")
        return 2
    return 0


def br_reply(i):
    out = i.get("output") or {}
    if i["kind"] == "sentence_round":
        lang = i["language"]
        return " | ".join(f"{p['english']} → {p[lang]}" for p in out.get("tasks", []))
    if i["kind"] == "sentence_feedback":
        return (f"meaning_ok={out.get('meaning_ok')} grammar_ok={out.get('grammar_ok')}: "
                f"{out.get('meaning_feedback', '')} {out.get('grammar_feedback', '')} Hint: {out.get('hint', '')}")
    text = f"{out.get('reply', '')} ({out.get('english', '')})"
    if i["kind"] == "roleplay_final":
        text += f" Feedback: {out.get('meaning_feedback', '')} {out.get('grammar_feedback', '')}"
    return text


def _failed_tests(run):
    tests = []
    for xml in run.glob("pytest-*.xml"):
        tests += [t for t in br.read_junit(xml)["tests"] if t["status"] == "failed"]
    return tests


def to_md(d):
    L = [f"# Review digest — run {d['run']}", "", f"Reviewers: {', '.join(d['reviewers']) or 'none found'}", ""]
    for lang, c in d["coverage"].items():
        L.append(f"- {lang.title()}: {c['decided']} of {c['queue']} queue items decided; "
                 f"{c['reviewed']} replies covered by a decision; {c['confirmed_fail']} confirmed not acceptable; "
                 f"{c['accepted']} machine failures accepted by people"
                 + (f"; judge agreed {c['judge_agreement']:.0%}" if c["judge_agreement"] is not None else ""))
    L += ["", "## Problems that people confirmed"]
    for c in d["problem_clusters"]:
        L.append(f"- **{c['language']} · {c['kind']} · {c['problem']}** — {c['replies']} replies in "
                 f"{', '.join(c['scenarios'])}. Notes: {' / '.join(c['human_notes']) or '–'}")
        for e in c["examples"][:2]:
            L.append(f"  - {e['key']}: learner “{e['learner'] or '(opening)'}” → “{e['reply'][:200]}”")
    if not d["problem_clusters"]:
        L.append("- None.")
    L += ["", "## False alarms (people accepted what the machine failed)"]
    L += [f"- {e['key']} ({e['language']}, {e['kind']}): failed checks {e['failed_checks'] or '–'}, judge low "
          f"{e['judge_low'] or '–'}. Note: {e['human_note'] or '–'}" for e in d["false_alarms"]] or ["- None."]
    L += ["", "## Failed code tests (triage)"]
    L += [f"- {t['test']}: {t['verdict']}. {t['note']}" for t in d["test_triage"]] or ["- None."]
    if d["open_high_priority"]:
        L += ["", f"## Not decided yet (must-check items): {len(d['open_high_priority'])}"]
        L += [f"- {o['key']}: {o['why']}" for o in d["open_high_priority"][:10]]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(Path(sys.argv[1]).resolve()))
