"""
Build the interactive test report for one run folder.

    .venv/bin/python testing/tools/build_report.py testing/reports/<run>

Reads (whatever is there):
  sets.txt                  which sets ran
  catalog.json              What / Why / If it fails for every test (pytest --catalog)
  pytest-<set>.xml          unit / slm_contract / ui results (JUnit XML)
  transcripts.json          slm_quality conversations + code checks
  judgments*.json           slm_quality rubric scores from the judge agents
  human_review*.json        decisions exported from report.html by human reviewers
  ui_exploration.md         notes from an exploratory browser pass (optional)
Writes:
  report.html   one self-contained page: review queue, live results, known problems, test catalog
  results.json  final numbers (machine + human decisions found in the folder)
  summary.md    a short text summary
Exit code 0 when nothing failed (open human reviews do not count as failures).
"""
import json
import random
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]   # the repository root
TESTING = ROOT / "testing"
QDIR = TESTING / "slm_quality"
RUBRIC = ["makes_sense", "language", "grounded", "feedback", "level_tone", "roleplay", "robustness"]
RUBRIC_LABEL = {"makes_sense": "Makes sense", "language": "Language", "grounded": "Grounded",
                "feedback": "Feedback", "level_tone": "Level & tone", "roleplay": "Role-play",
                "robustness": "Odd input"}
SET_INFO = {
    "unit": ("Logic tests", "Checks the app's own rules with no SLM: plans, calendars, Stage 1 answers, "
             "pronunciation, and the checks that guard SLM output. Fast and exact."),
    "slm_contract": ("SLM safety-net tests", "Gives the app fake SLM replies (good, broken, wrong language, "
                     "wrong role) and checks that the app catches each problem. It does not measure the real SLM."),
    "ui": ("Screen tests", "Clicks through the real app screens without a browser, with a fake SLM. "
           "Checks that a learner can reach and finish each activity."),
    "slm_quality": ("SLM quality (real model)", "Plays learner situations with the real local SLM. "
                    "Code checks and judge agents score each reply. Humans confirm the doubtful ones."),
}
CHECK_REASON_SKIP = {"no_error"}


def load_yaml(path, default):
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else default


# --- pytest sets --------------------------------------------------------------------

def read_junit(path):
    tests = []
    for case in ET.parse(path).getroot().iter("testcase"):
        status, message = "passed", ""
        for tag in ("failure", "error", "skipped"):
            el = case.find(tag)
            if el is not None:
                status = {"failure": "failed", "error": "failed", "skipped": "skipped"}[tag]
                message = (el.get("message") or el.text or "").strip()[:1500]
        stem = case.get("classname", "").split(".")[-1]
        name = case.get("name", "")
        tests.append({"id": f"{stem}::{name}", "func": f"{stem}::{name.split('[')[0]}",
                      "case": name[len(name.split('[')[0]):], "status": status,
                      "seconds": round(float(case.get("time") or 0), 2), "message": message})
    counts = Counter(t["status"] for t in tests)
    return {"status": "fail" if counts["failed"] else ("pass" if tests else "skipped"),
            "passed": counts["passed"], "failed": counts["failed"], "skipped": counts["skipped"],
            "seconds": round(sum(t["seconds"] for t in tests), 1), "tests": tests}


# --- SLM quality ----------------------------------------------------------------------

def read_judgments(run):
    out = {}
    for path in sorted(run.glob("judgments*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        for j in (raw if isinstance(raw, list) else raw.get("judgments", [])):
            out[j["item_id"]] = j
    return out


def numeric(scores):
    return {k: v for k, v in (scores or {}).items() if isinstance(v, (int, float))}


def review_reasons(it, j, lang, settings, check_text):
    """Why a human should look at this reply. Each reason: code, priority, title, why."""
    reasons = []
    if it.get("output") is None:
        return reasons
    scores, why_by = numeric(j.get("scores")) if j else {}, (j or {}).get("reasons") or {}
    low = sorted(((v, k) for k, v in scores.items() if v <= 2))
    for v, k in low:
        reasons.append({"code": "judge_low", "priority": "medium", "rubric": k, "score": v,
                        "why_short": why_by.get(k) or (j or {}).get("summary", ""),
                        "title": f"The judge says this can hurt learners ({RUBRIC_LABEL[k]} {v}/5)",
                        "why": f"The judge gave “{RUBRIC_LABEL[k]}” {v} out of 5: {why_by.get(k) or (j or {}).get('summary', '')} "
                               "If the judge is right, learners see this problem. Confirm it before the team spends time on a fix."})
    verdict_fail = [k for k, ok in it["checks"].items() if k.startswith("verdict_") and not ok]
    fb = scores.get("feedback")
    if verdict_fail and fb is not None and fb >= 4:
        k = verdict_fail[0]
        reasons.append({"code": "disagree", "priority": "high",
                        "title": "The code check and the judge disagree",
                        "why": f"The scenario expects {k[8:]} = {expected(it, k)}, but the SLM said "
                               f"{(it['output'] or {}).get(k[8:])}. The judge still gave the feedback {fb}/5. "
                               "One of them is wrong. Only a person can decide."})
    other_fail = [k for k, ok in it["checks"].items() if not ok and not k.startswith("verdict_")
                  and k not in CHECK_REASON_SKIP]
    if other_fail and not low:
        k = other_fail[0]
        reasons.append({"code": "check_only", "priority": "medium", "check": k,
                        "title": "A code check failed, but the judge saw no big problem",
                        "why": f"The check “{k}” failed. "
                               + (f"If this is real: {check_text[k]} " if check_text.get(k) else "")
                               + "The judge gave no score below 3. Look at the reply and decide if a learner would see a problem."})
    if lang == "macedonian" and j and ((scores.get("language") or 5) <= 3 or j.get("confidence") == "low"):
        doubt = (f"gave the Macedonian {scores['language']}/5" if (scores.get("language") or 5) <= 3
                 else "was not sure about the Macedonian")
        lang_score = scores.get("language") or 5
        reasons.append({"code": "native", "priority": "high" if lang_score <= 2 else "medium" if lang_score == 3 else "low",
                        "title": "Only a native speaker can confirm this Macedonian",
                        "why": f"Machine judges are less reliable for Macedonian. The judge {doubt}"
                               + (f": {why_by.get('language')}" if why_by.get("language") else ".")
                               + " A native speaker must say if the Macedonian is correct."})
    return reasons


def group_judge_problems(items):
    """Many replies of one scenario often have the same problems. A person confirms ONE
    example (the worst reply); the decision then covers the similar replies. Judge findings
    are grouped by scenario; failed code checks by scenario and check."""
    def regroup(code, keyfn, make):
        groups = defaultdict(list)
        for i in items:
            rs = [r for r in i["review"] if r["code"] == code]
            if rs:
                for r in rs:
                    i["review"].remove(r)
                groups[keyfn(i, rs)].append((i, rs))
        for key, members in groups.items():
            members.sort(key=lambda m: (min(r.get("score", 9) for r in m[1]), m[0]["key"]))
            rep, rs = members[0]
            rep["review"].append(make(rep, rs, members))
            rep["group"] = sorted(set(rep.get("group", []) + [m[0]["key"] for m in members[1:]]))

    def judge_card(rep, rs, members):
        n = len(members)
        counts = Counter(r["rubric"] for _, rr in members for r in rr)
        worst = "; ".join(f"{RUBRIC_LABEL[r['rubric']]} {r['score']}/5: {r['why_short']}" for r in rs)
        others = ", ".join(f"{RUBRIC_LABEL[k]} in {v}" for k, v in counts.most_common())
        return {"code": "judge_low", "priority": "medium", "score": min(r["score"] for r in rs),
                "title": f"Confirm the judge's finding: {n} repl{'ies' if n > 1 else 'y'} in this scenario can hurt learners",
                "why": f"This is the worst example. The judge says: {worst}. "
                       + (f"Low scores across the {n} replies: {others}. " if n > 1 else "")
                       + "If the judge is right, learners see this problem. Confirm it before the team spends time on a fix."
                       + (" Your decision also applies to the similar replies, unless someone decides one separately." if n > 1 else "")}

    def check_card(rep, rs, members):
        n, r = len(members), rs[0]
        return {**r, "title": r["title"] + (f" ({n} replies)" if n > 1 else ""),
                "why": r["why"] + (" Your decision also applies to the similar replies." if n > 1 else "")}

    regroup("judge_low", lambda i, rs: i["scenario"], judge_card)
    regroup("check_only", lambda i, rs: (i["scenario"], rs[0]["check"]), check_card)


def expected(it, check):
    inp = it.get("input") or {}
    return (inp.get("expect") or {}).get(check[8:], "?") if check != "verdict_goal_met" else "see scenario"


def read_quality(run, settings, check_text):
    data = json.loads((run / "transcripts.json").read_text(encoding="utf-8"))
    meta = data["meta"]
    if meta.get("skipped"):
        return {"status": "skipped", "reason": meta["skipped"], "meta": meta, "scenarios": [], "items": []}
    judgments = read_judgments(run)
    settings = {**settings, **meta.get("settings", {})}
    min_score = settings.get("acceptable_min_score", 3)
    scenarios, items = [], []
    for sc in data["scenarios"]:
        sc.setdefault("language", "spanish")   # runs from before Macedonian support
        expect = {}
        sc_items = []
        for run_ in sc["runs"]:
            for it in run_["items"]:
                j = judgments.get(it["item_id"])
                scores = numeric(j.get("scores")) if j else {}
                checks_ok = all(it["checks"].values())
                if it.get("output") is None:
                    machine = "fail"
                elif j:
                    machine = "ok" if checks_ok and min(scores.values() or [5]) >= min_score else "fail"
                else:
                    machine = "ok" if checks_ok else "fail"
                inp = it.get("input") or {}
                if sc["kind"] == "roleplay" and it["kind"] == "roleplay_final":
                    expect = {}
                entry = {
                    "key": it["item_id"], "scenario": sc["id"], "language": sc["language"],
                    "kind": it["kind"], "topic": sc["topic"], "variety": sc["variety_name"],
                    "about": sc["about"], "learner": inp.get("learner") or inp.get("read_as") or inp.get("text"),
                    "typed": inp.get("typed") or inp.get("text"), "history": inp.get("history") or [],
                    "task": inp.get("task"), "expect": inp.get("expect"), "note": inp.get("note"),
                    "output": it.get("output"), "error": it.get("error"), "fallback": it.get("fallback", False),
                    "checks": it["checks"], "seconds": it.get("seconds", 0), "retries": it.get("retries", 0),
                    "judged": bool(j), "scores": (j or {}).get("scores") or {},
                    "reasons_by_item": (j or {}).get("reasons") or {}, "tags": (j or {}).get("tags") or [],
                    "summary": (j or {}).get("summary", ""), "confidence": (j or {}).get("confidence", ""),
                    "machine": machine,
                }
                entry["review"] = review_reasons(it, j, sc["language"], settings, check_text)
                sc_items.append(entry)
        items += sc_items
        scenarios.append({"id": sc["id"], "kind": sc["kind"], "language": sc["language"],
                          "about": sc["about"], "variety": sc["variety_name"], "topic": sc["topic"],
                          "replies": len(sc_items)})
    group_judge_problems(items)
    # Random spot checks among replies the machine passed, to measure the judge.
    rng = random.Random(run.name)
    for lang in {i["language"] for i in items}:
        passed = [i for i in items if i["language"] == lang and i["machine"] == "ok" and i["judged"]
                  and not i["review"]]
        for i in rng.sample(passed, min(settings.get("spot_checks_per_language", 4), len(passed))):
            i["review"].append({"code": "spot", "priority": "low", "title": "Random spot check",
                                "why": "The machine passed this reply. A person checks a few passed replies to measure "
                                       "how often the judge is right. If the judge is often wrong, the machine scores "
                                       "cannot be trusted."})
    for i in items:
        order = {"high": 0, "medium": 1, "low": 2}
        i["review"].sort(key=lambda r: order[r["priority"]])
        i["priority"] = i["review"][0]["priority"] if i["review"] else None
    return {"status": "ok", "meta": meta, "settings": settings, "judged": bool(judgments),
            "scenarios": scenarios, "items": items}


# --- Human decisions, results ------------------------------------------------------------

def read_human(run):
    decisions = {}
    for path in sorted(run.glob("human_review*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        for key, d in (raw.get("decisions") or {}).items():
            d = {**d, "reviewer": d.get("reviewer") or raw.get("reviewer", "")}
            if key not in decisions or d.get("at", "") > decisions[key].get("at", ""):
                decisions[key] = d
    return decisions


def final_numbers(data):
    """The same rules as the page's JavaScript (keep them in step)."""
    human, q = data["human"], data["slm"]
    share = q.get("settings", {}).get("scenario_pass_share", 0.9)
    human = dict(human)
    for i in q.get("items", []):     # a decision on an example covers its similar replies
        d = human.get(i["key"], {})
        if d.get("verdict") in ("ok", "fail"):
            for k in i.get("group", []):
                if not human.get(k, {}).get("verdict"):
                    human[k] = {**d, "via": i["key"]}
    final_ok = {}
    for i in q.get("items", []):
        d = human.get(i["key"], {}).get("verdict")
        final_ok[i["key"]] = (d == "ok") if d in ("ok", "fail") else i["machine"] == "ok"
    langs = {}
    for lang in sorted({i["language"] for i in q.get("items", [])}):
        its = [i for i in q["items"] if i["language"] == lang]
        reviewed = [i for i in its if human.get(i["key"], {}).get("verdict") in ("ok", "fail")]
        agree = [i for i in reviewed if (human[i["key"]]["verdict"] == "ok") == (i["machine"] == "ok")]
        langs[lang] = {"replies": len(its),
                       "acceptable_machine": round(sum(i["machine"] == "ok" for i in its) / len(its), 3),
                       "acceptable_final": round(sum(final_ok[i["key"]] for i in its) / len(its), 3),
                       "reviewed": len(reviewed), "open_reviews": sum(1 for i in its if i["review"]
                                                                      and i not in reviewed),
                       "judge_agreement": round(len(agree) / len(reviewed), 3) if reviewed else None}
    scen = []
    for s in q.get("scenarios", []):
        its = [i for i in q["items"] if i["scenario"] == s["id"]]
        ok = sum(final_ok[i["key"]] for i in its) / len(its) if its else 0
        scen.append({"id": s["id"], "language": s["language"], "ok_share": round(ok, 3),
                     "status": "pass" if ok >= share else "fail",
                     "open_reviews": sum(1 for i in its if i["review"] and not human.get(i["key"], {}).get("verdict"))})
    tests = {}
    for name, s in data["sets"].items():
        failed = [t for t in s.get("tests", []) if t["status"] == "failed"]
        kinds = Counter(human.get("test:" + t["id"], {}).get("verdict", "open") for t in failed)
        tests[name] = {"failed": len(failed), "real_bug": kinds["bug"], "test_out_of_date": kinds["outdated"],
                       "not_decided": kinds["open"] + kinds["unsure"]}
    return {"languages": langs, "scenarios": scen, "tests": tests}


def known_issue_evidence(issues, data):
    q, sets = data["slm"], data["sets"]
    failed_funcs = {t["func"].split("::")[1] for s in sets.values() for t in s.get("tests", []) if t["status"] == "failed"}
    ran_funcs = {t["func"].split("::")[1] for s in sets.values() for t in s.get("tests", [])}
    out = []
    for iss in issues:
        tests_failed = [t for t in iss.get("tests", []) if t in failed_funcs]
        tests_ran = [t for t in iss.get("tests", []) if t in ran_funcs]
        keys = [i["key"] for i in q.get("items", []) if i["language"] in iss.get("languages", [])
                and (any(not i["checks"].get(c, True) for c in iss.get("checks", []))
                     or set(i["tags"]) & set(iss.get("tags", [])))]
        out.append({**iss, "tests_failed": tests_failed, "tests_ran": tests_ran, "item_keys": keys,
                    "tested": bool(tests_ran) or any(i["language"] in iss.get("languages", [])
                                                     for i in q.get("items", []))})
    return out


def previous_run(run):
    runs = sorted(p for p in run.parent.iterdir()
                  if p.is_dir() and p.name < run.name and (p / "results.json").exists())
    return runs[-1] if runs else None


def compare(final, prev_dir):
    if not prev_dir:
        return None
    try:
        prev = json.loads((prev_dir / "results.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    out = {"previous": prev_dir.name, "lines": []}
    pf = prev.get("final") or {}
    for lang, v in final["languages"].items():
        old = (pf.get("languages") or {}).get(lang)
        if old:
            a, b = old["acceptable_final"], v["acceptable_final"]
            if abs(a - b) >= 0.05:
                out["lines"].append(f"{lang.title()}: acceptable replies {a:.0%} → {b:.0%}")
        else:
            out["lines"].append(f"{lang.title()}: first run with this language")
    before = {s["id"]: s["status"] for s in (pf.get("scenarios") or [])}
    if not before:   # older report format
        before = {s["id"]: s.get("status") for s in ((prev.get("sets") or {}).get("slm_quality") or {}).get("scenarios", [])}
    for s in final["scenarios"]:
        if s["id"] in before and before[s["id"]] not in (s["status"], None):
            out["lines"].append(f"Scenario {s['id']}: {before[s['id']]} → {s['status']}")
    prev_tests = {t["id"]: t["status"] for st in (prev.get("sets") or {}).values() for t in st.get("tests", [])}
    return out | {"prev_tests": prev_tests}


# --- Build ----------------------------------------------------------------------------------

def build(run):
    checks_doc = load_yaml(QDIR / "checks.yaml", {"code_checks": {}, "rubric": {}})
    check_text = {k: v.get("fails", "") for k, v in checks_doc.get("code_checks", {}).items()}
    settings = (load_yaml(QDIR / "scenarios.yaml", {}) or {}).get("settings", {})
    sets_ran = (run / "sets.txt").read_text().split() if (run / "sets.txt").exists() else []
    catalog = json.loads((run / "catalog.json").read_text(encoding="utf-8")) if (run / "catalog.json").exists() else {}

    sets = {}
    for name in sets_ran:
        if name == "slm_quality":
            continue
        if (run / f"pytest-{name}.xml").exists():
            sets[name] = read_junit(run / f"pytest-{name}.xml")
        else:
            sets[name] = {"status": "fail", "reason": f"No results. See {name}.log.", "tests": [],
                          "passed": 0, "failed": 0, "skipped": 0}
    if "slm_quality" in sets_ran and (run / "transcripts.json").exists():
        slm = read_quality(run, settings, check_text)
    elif "slm_quality" in sets_ran:
        slm = {"status": "missing", "reason": "No transcripts.json. See slm_quality.log.", "items": [], "scenarios": []}
    else:
        slm = {"status": "not run", "items": [], "scenarios": []}

    # Failed code tests also go to the review queue: bug, or out-of-date test?
    test_reviews = []
    for name, s in sets.items():
        for t in s["tests"]:
            if t["status"] != "failed":
                continue
            doc = catalog.get(t["func"], {})
            test_reviews.append({
                "key": "test:" + t["id"], "set": name, "func": t["func"], "case": t["case"],
                "language": doc.get("language", "both"), "message": t["message"], "doc": doc,
                "priority": "high",
                "review": [{"code": "test_failed", "priority": "high", "title": "A code test failed",
                            "why": (f"This test checks: {doc.get('what', t['func'])} If it fails: {doc.get('fails', '')} "
                                    "Decide: is this a real bug, or is the test out of date because of an intended "
                                    "change (for example the UI update)?")}]})

    data = {"meta": {"run": run.name, "built": datetime.now().isoformat(timespec="seconds"),
                     "model": (slm.get("meta") or {}).get("model"), "commit": (slm.get("meta") or {}).get("commit"),
                     "sets_ran": sets_ran},
            "set_info": SET_INFO, "rubric_label": RUBRIC_LABEL, "sets": sets, "slm": slm,
            "test_reviews": test_reviews, "checks_doc": checks_doc, "catalog": catalog,
            "human": read_human(run),
            "exploration": (run / "ui_exploration.md").read_text(encoding="utf-8")
            if (run / "ui_exploration.md").exists() else ""}
    data["known"] = known_issue_evidence(load_yaml(QDIR / "known_issues.yaml", []), data)
    final = final_numbers(data)
    data["comparison"] = compare(final, previous_run(run))

    results = {"meta": data["meta"], "final": final,
               "sets": {k: {kk: vv for kk, vv in v.items()} for k, v in sets.items()},
               "review": {"items": sum(1 for i in slm.get("items", []) if i["review"]) + len(test_reviews),
                          "decided": sum(1 for d in data["human"].values() if d.get("verdict"))},
               "known_issues": [{"id": k["id"], "seen": bool(k["tests_failed"] or k["item_keys"])} for k in data["known"]]}
    (run / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (run / "summary.md").write_text(summary_md(data, final), encoding="utf-8")
    template = (TESTING / "tools" / "report_template.html").read_text(encoding="utf-8")
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    (run / "report.html").write_text(template.replace("__REPORT_DATA__", blob), encoding="utf-8")
    print((run / "summary.md").read_text(encoding="utf-8"))
    print(f"Report: {run / 'report.html'}")
    bad = any(s["status"] == "fail" for s in sets.values()) or any(s["status"] == "fail" for s in final["scenarios"])
    return 1 if bad else 0


def summary_md(data, final):
    m = data["meta"]
    lines = [f"# Test report {m['run']}", ""]
    if m.get("model"):
        lines += [f"Model `{m['model']}` · commit `{m.get('commit', '')}`", ""]
    open_n = sum(1 for i in data["slm"].get("items", []) if i["review"] and not data["human"].get(i["key"], {}).get("verdict"))
    open_n += sum(1 for t in data["test_reviews"] if not data["human"].get(t["key"], {}).get("verdict"))
    lines += [f"**Human review: {open_n} items are waiting.** Open report.html and use the Review tab.", ""]
    for name, s in data["sets"].items():
        t = final["tests"][name]
        lines.append(f"- {SET_INFO[name][0]}: {s['passed']} passed, {s['failed']} failed"
                     + (f" ({t['real_bug']} real bugs, {t['test_out_of_date']} out-of-date tests, "
                        f"{t['not_decided']} not decided)" if s["failed"] else ""))
    for lang, v in final["languages"].items():
        lines.append(f"- SLM quality, {lang.title()}: {v['acceptable_final']:.0%} of {v['replies']} replies acceptable"
                     f" (machine estimate {v['acceptable_machine']:.0%}; {v['reviewed']} reviewed by a person"
                     + (f", judge agreed {v['judge_agreement']:.0%}" if v["judge_agreement"] is not None else "") + ")")
    failing = [s for s in final["scenarios"] if s["status"] == "fail"]
    if failing:
        lines += ["", "Scenarios that fail: " + ", ".join(s["id"] for s in failing)]
    seen = [k for k in data["known"] if k["tests_failed"] or k["item_keys"]]
    if seen:
        lines += ["", "Known problems seen again in this run:"]
        lines += [f"- {k['problem']}" for k in seen]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(build(Path(sys.argv[1]).resolve()))
