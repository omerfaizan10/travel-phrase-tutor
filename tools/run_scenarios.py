"""Play every roleplay scenario in both languages: the learner always taps the
suggested sentence (as in the app). Writes a readable transcript per run.

usage: python run_scenarios.py OUT_DIR [lang ...] [--only scen,scen] [--turns N] [--chat] [--model NAME]
                                  [--strategy first|random|detour] [--seed N]

Learner strategies:
  first   always tap the first suggestion (the main path)
  random  tap a random suggestion each turn (explores other branches)
  detour  like random, but at turn 2 ask a checked phrase from a DIFFERENT scene
          (off-topic) and at turn 4 repeat the learner's first line (memory/repeats)
"""
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # this project folder

from planner import build_plan            # noqa: E402
from prompts import SCENARIOS, TutorSettings  # noqa: E402
from retriever import Retriever            # noqa: E402
from tutor import _traveler_phrases, chat_turn, roleplay_turn  # noqa: E402

args = sys.argv[1:]
out = Path(args.pop(0))
out.mkdir(parents=True, exist_ok=True)
only = None
turns = 6
turn_fn = roleplay_turn
strategy, seed = "first", 0
if "--strategy" in args:
    i = args.index("--strategy"); strategy = args[i + 1]; del args[i:i + 2]
if "--seed" in args:
    i = args.index("--seed"); seed = int(args[i + 1]); del args[i:i + 2]
mc = None
if "--model" in args:
    i = args.index("--model"); mc = {"model": args[i + 1]}; del args[i:i + 2]
if "--chat" in args:
    args.remove("--chat"); turn_fn = chat_turn
if "--only" in args:
    i = args.index("--only"); only = args[i + 1].split(","); del args[i:i + 2]
if "--turns" in args:
    i = args.index("--turns"); turns = int(args[i + 1]); del args[i:i + 2]
langs = args or ["macedonian", "spanish"]
TRIP = {"macedonian": "I'm going to Skopje and Ohrid for a week",
        "spanish": "I'm going to Madrid and Seville for a week"}

r = Retriever()
summary = []
for lang in langs:
    for scen in SCENARIOS:
        if only and scen not in only:
            continue
        plan = build_plan(lang, [scen], 7, r)
        session = next((x for x in plan if x["scenario"] == scen and not x.get("review")), plan[0])
        s = TutorSettings(language=lang, scenario=scen, task="roleplay", mode="rag",
                          goal=TRIP[lang], timeline="7 days")
        hist, lines, nfb, nprob = [], [], 0, 0
        rng = random.Random(f"{seed}-{lang}-{scen}")
        # Off-topic line for "detour": a checked traveler phrase from another scene.
        other = rng.choice([x for x in SCENARIOS if x != scen])
        detour_pool = _traveler_phrases(TutorSettings(language=lang, scenario=other, task="roleplay"), r)
        detour = rng.choice(detour_pool) if detour_pool else None
        first_line = None
        t0 = time.time()
        t = turn_fn(s, [], session, TRIP[lang], r, model_config=mc)
        for k in range(turns + 1):
            nfb += t["fallback"]; nprob += bool(t["problems"])
            lines.append(f"PARTNER: {t['reply']}\n   = {t['reply_english']}"
                         + (f"\n   [attempts={t['attempts']} problems={t['problems']} fallback={t['fallback']}]"
                            if t["attempts"] > 1 or t["fallback"] else ""))
            for rj in t.get("rejected", []):
                lines.append(f"   x rejected: {rj['reply']}  <- {'; '.join(x[:60] for x in rj['problems'])}")
            if t.get("tip"):
                lines.append(f"   tip: {t['tip']}")
            hist.append({"role": "assistant", "content": t["reply"], "english": t["reply_english"]})
            for o in t.get("suggestions", [])[1:]:
                lines.append(f"   (also: {o['text']} = {o['meaning']}{' [pb]' if o.get('phrasebook') else ''})")
            options = t.get("suggestions") or ([t["suggestion"]] if t.get("suggestion") else [])
            sg, note = (options[0] if options else None), ""
            if options and strategy in ("random", "detour"):
                sg = rng.choice(options)
            if strategy == "detour" and not t.get("scene_done") and k < turns:
                if k == 2 and detour:
                    sg, note = {"text": detour["text"], "meaning": detour["meaning"]}, f"  <- DETOUR (from {other})"
                elif k == 4 and first_line:
                    sg, note = first_line, "  <- REPEAT of first line"
            if not sg or k == turns:
                lines.append("   (scene done)" if t.get("scene_done") else "   (no suggestion)" if not sg else "")
                break
            first_line = first_line or sg
            lines.append(f"YOU:     {sg['text']}{' [pb]' if sg.get('phrasebook') else ''}{note}\n   = {sg['meaning']}")
            hist.append({"role": "user", "content": sg["text"]})
            t = turn_fn(s, hist, session, TRIP[lang], r, model_config=mc)
        dt = time.time() - t0
        text = f"### {lang} / {scen} / {strategy}  ({dt:.0f}s, fallbacks={nfb})\n" + "\n".join(lines) + "\n"
        (out / f"{lang}_{scen}.txt").write_text(text)
        print(text, flush=True)
        summary.append({"lang": lang, "scen": scen, "seconds": round(dt), "fallbacks": nfb,
                        "turns_with_problems": nprob})
(out / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
