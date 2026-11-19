---
name: test-app
description: Test the Travel Phrase Tutor (Spanish and Macedonian) and write an interactive report. Runs the logic, SLM safety-net and screen test sets, collects real conversations with the local SLM (Ollama), has judge agents score every SLM reply against a quality rubric, and builds testing/reports/<run>/report.html with a human review queue (with the reason for each item), live results, known problems and a test catalog. Use when the user runs /test-app or asks to test the app, check SLM reply quality, compare models or prompts, review results, or get a test report.
---

# /test-app [quick | full | slm | ui] [--language spanish|macedonian] [--repeats N] [--model NAME] [--only IDS]

Pick the mode from the user's words. Default: `quick`.

| Mode | When | Sets | Time |
|---|---|---|---|
| `quick` | after any code change | unit, slm_contract | < 30 s |
| `full` | before a PR or merge | unit, slm_contract, ui, slm_quality + judge | 10–25 min |
| `slm` | after changing prompts, the model, `sentence_practice.py`, `roleplay_practice.py`, `tutor.py` checks or the knowledge base | slm_quality + judge | 5–20 min |
| `ui` | after changing `app.py` or the practice screens | ui + exploratory browser pass | 2–5 min |

Always use `.venv/bin/python`. Never commit `testing/reports/`. Code tests always cover both
languages; `--language` limits only the SLM quality set (one reviewer per language).

## 1. Run the sets

```bash
.venv/bin/python testing/tools/run_tests.py --sets quick
.venv/bin/python testing/tools/run_tests.py --sets all                          # full
.venv/bin/python testing/tools/run_tests.py --sets slm_quality --language macedonian   # slm, one language
.venv/bin/python testing/tools/run_tests.py --sets ui                           # ui
```

Pass `--repeats`, `--model`, `--only` and `--language` through. Run modes that include
slm_quality in the background. The first line of output names the run folder.

If slm_quality prints `SKIPPED` (Ollama is down or the model is missing), tell the user the
fix that it printed (for example `brew services start ollama`). Do not judge anything.

## 2. Judge the SLM replies (full and slm modes)

`<run>/judge_packet.json` is a list of SLM replies to score. Split it into parts of at most
40 items, and start one `general-purpose` agent per part, in parallel, at most 4. Keep
Macedonian and Spanish items in separate parts when possible. Give each agent this brief,
with the paths and the index range filled in:

> You are judging the quality of replies from a small language model that tutors
> English-speaking travelers in Spanish or Macedonian (see each item's `language`). Read
> `testing/slm_quality/rubric.md` (the scoring rules, with a Macedonian section) and
> `testing/slm_quality/calibration.json` (examples that were already judged; score the same way).
> Then read items [A, B) of `<run>/judge_packet.json`. Each item has the scenario, the variety,
> the learner's input (for Macedonian, `typed` is what the learner typed and `learner` is the
> Cyrillic that the app read), the studied vocabulary, reference notes for role-plays, and the
> SLM's reply. Judge each reply as a careful, fair teacher of that language. Use the full 1–5
> scale and do not be generous: a reply that would confuse or mislead a learner scores 2 or
> less. Use null for items that do not apply (see the rubric's table). For role-play turns, look
> at the history too: does the reply make sense after what was said?
> Write `<run>/judgments.part<N>.json`: a JSON list with one object per item:
> `{"item_id": str, "scores": {"makes_sense": 1-5|null, "language": ..., "grounded": ...,
> "feedback": ..., "level_tone": ..., "roleplay": ..., "robustness": ...},
> "reasons": {<item>: "one plain-English sentence a non-expert understands, for every score below 4"},
> "tags": [rubric problem tags], "summary": "one sentence on the reply overall",
> "confidence": "high" | "low"}`. Set confidence "low" when you are not sure that the
> Macedonian (or a regional Spanish form) is correct. Judge every item in your range. Check that
> the file is valid JSON. Reply with only the number of items you judged and the 3 worst item_ids.

When all the agents are done, check that every item_id in the packet has a judgment, and
re-run any missing ones. Then rebuild the report:

```bash
.venv/bin/python testing/tools/build_report.py testing/reports/<timestamp>
```

## 3. Exploratory UI pass (ui and full modes)

Start the app with `preview_start` (name `streamlit`, from `.claude/launch.json`). In the browser pane:
- Build a Spanish plan with "Choose with buttons" and a Macedonian plan with "Describe it in a chat".
- Build one plan without Vocabulary: the word list must show before the first Sentences activity.
- On the calendar, use Next lesson / Continue, then ‹ Previous / Next › inside a practice.
- Shelf this plan, open it again from Your plans, and check that the progress is kept.
- Open one activity of each lesson type (Vocabulary, Sentences, Role play), in both languages.
- Try edge inputs: empty answers, accents (é ñ ¿), Macedonian typed in Latin letters, very long
  text, English instead of the target language, and pressing buttons twice.
- In Stage 3, have a short real conversation with the SLM.

Write what you saw to `<run>/ui_exploration.md`: what worked, and each problem with steps to
reproduce. Then rebuild the report (step 2's command). Stop the preview server when done.

## 4. Report back

- Read `<run>/summary.md`. Tell the user in plain English: which sets passed, how many items
  wait for human review (and that the Review tab says why for each one), the 3 most important
  problems (most harmful to a learner first), and what changed since the last run.
- Give the path `testing/reports/<timestamp>/report.html`. The page is interactive: reviewers decide
  each item, the results update at once, and "Download my decisions" saves a
  `human_review_*.json`. Put those files in the run folder and rebuild to fold them into
  `results.json`. Offer to publish the page as a private Artifact; publish only if they say yes.
- Macedonian items marked "Only a native speaker can confirm" go to the Macedonian reviewer.
- If a code test failed: say whether it looks like a real bug or an out-of-date test (for
  example after an intended UI change). Ask before changing app code. If a test is out of
  date, fix the test AND its What / Why / If it fails docstring.
- Every test function must have a docstring with `What:`, `Why:` and `If it fails:` lines in
  plain language. The report's Test catalog shows "no description" for any that do not.
