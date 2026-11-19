---
name: language-tutor-conversation-reviewer
description: Turn a human-reviewed test report of the Travel Phrase Tutor's SLM conversations (Spanish and Macedonian) into fixes. Reads the reviewers' decisions, summarizes how it interprets them and waits for approval (gate 1), recommends fixes (mostly prompt tuning, but also guards, post-processing, grounding data, tests or the judge) with an execution plan and waits for approval (gate 2), then implements the fixes and reruns the tests to show before/after. Use when the user runs /language-tutor-conversation-reviewer, or asks to act on, interpret or fix the results of a reviewed test report / human-in-the-loop decisions.
---

# Language tutor conversation reviewer

You turn **people's decisions** about SLM replies into **fixes that are proven by a re-test**.
The people's decisions are the ground truth. The judge agents and code checks are only
evidence. You work in four phases with **two hard approval gates**. Before gate 2 is approved
you change NO project files (the digest that the script writes into `reports/` is the only
exception).

Background you need (read when relevant, not all up front):
- `testing/slm_quality/` — `scenarios.yaml` (learner situations), `rubric.md`, `checks.yaml`
  (what each check means for learners), `calibration.json` (judged examples), `known_issues.yaml`.
- `testing/skills/test-app/SKILL.md` — how to run the tests and the judge agents.
- Where SLM replies come from (`testing/tools/review_digest.py` prints this per problem):
  | Reply kind | Produced by | Guards / post-processing |
  |---|---|---|
  | `sentence_round` | `sentence_practice.build_sentence_round` system prompt | `_validate_round`; Macedonian: `practice.topic_sentences` grounding, `prompts.LANGUAGE_RULES` |
  | `sentence_feedback` | `sentence_practice.sentence_feedback` system prompt | exact-match pre-check, `_feedback_problems`, label and answer-leak clean-up |
  | `roleplay_turn` | `roleplay_practice.roleplay_response` system prompt (+ `partner_lines` for Macedonian) | `tutor.check_reply`, Macedonian checks in `roleplay_response`, `_safe_line` fallback |
  | `roleplay_final` | `roleplay_response(final=True)` (Spanish), `_macedonian_final` (Macedonian) | correction filters |
- The model is a small local model (`llm.DEFAULT_MODEL`, Qwen3-4B, through Ollama).

## Phase 1 — Read the human review

1. Find the run folder: the one that the user names, else the newest `testing/reports/<timestamp>/`
   that contains `human_review*.json`. If there are no decision files, stop. Tell the user to
   click "Download my decisions" in `report.html` and to put the file(s) in the run folder.
2. Run the digest and read it:
   ```bash
   .venv/bin/python testing/tools/review_digest.py testing/reports/<run>
   ```
   It writes `review_digest.md` / `.json`: coverage per language, **problem clusters** that
   people confirmed (with notes, examples, source and the possible fixes), **false alarms** (the
   machine failed it, people accepted it), **test triage** (real bug / out of date / not sure),
   and **must-check items that are still open**.
3. For each cluster, open 1–3 example replies in `transcripts.json` (and, if needed, the raw
   failed tries in `failed_raw`) and the code that produced them. Form a **root-cause hypothesis**:
   - the prompt does not say it, says it unclearly, or two rules conflict;
   - the prompt says it, but a 4B model does not follow it reliably (this needs a guard or
     code, not more prompt text);
   - the grounding data is missing or wrong (notes, `partner_lines`, reference sentences);
   - post-processing changes the reply wrongly (for example the fallback English);
   - the test expectation or the judge is wrong (the false alarms).

## Gate 1 — Approve the interpretation

Write a **brief** interpretation, at most about 15 lines, in plain language:
- What was reviewed: who, how many items, which languages. Say clearly what was NOT reviewed
  (open must-check items, a language with few decisions).
- The confirmed problems, largest first. One line each: what learners see, how many replies,
  in which scenarios, and your root-cause hypothesis (labelled as a hypothesis).
- False alarms: where the judge or a code check was wrong, and what that means for trust.
- Code tests: real bugs vs out-of-date tests.

Then ask with `AskUserQuestion` (options: "Approve interpretation", "Correct it"), and **stop**.
If the user corrects you, update the interpretation and ask again. Do not design fixes until
the user approves.

## Phase 2 — Recommend fixes and design the plan

For each approved problem, choose the **lowest lever that will reliably work**:
1. **Deterministic code**, when the rule is exact (a pre-check, a mapping, a filter). Exact
   rules must not depend on the model.
2. **Guard + retry with the problem named**, when the model can fix its own mistake if told
   (the pattern that `roleplay_response` uses for Macedonian). Then a safe fallback.
3. **Prompt tuning**, when the instruction is missing, unclear or conflicting. For a 4B model:
   short, concrete rules; one rule per idea; a correct example taken from the notes (never an
   invented one); the most important rule near the end; no rule that contradicts another; keep
   prompts short (context is 8k tokens with the notes). If you change shared prompt text, both
   languages are affected.
4. **Grounding data** (notes, `partner_lines`, reference sentences). Macedonian and Spanish
   notes belong to the native speakers: propose the change, and mark it "needs native-speaker
   approval". Do not invent target-language wording yourself without flagging it.
5. **Model or settings** (temperature, `num_predict`, another model, for example the `v2`
   branch's Gemma 3 12B): recommend only, with evidence. Do not switch.
6. **The judge side**, for false alarms: add the reviewer's case to `calibration.json` (with
   their note), clarify `rubric.md`, or correct a code check in `testing/tools/collect_transcripts.py`
   and `checks.yaml`. Do not change app code for a false alarm.
7. **Tests**: "out of date" → update the test AND its `What:` / `Why:` / `If it fails:`
   docstring. "Real bug" → app fix plus the test stays as it is.

Rules for every fix:
- Never weaken a test, a check or a scenario expectation to make it pass. Change an
  expectation only when a reviewer said that the expectation was wrong.
- Never hard-code scenario text (for example the exact learner lines) into the app.
- Each guard or post-processing change gets a regression test in
  `testing/tests/test_slm_contract*.py` (fake SLM), with a `What:` / `Why:` / `If it fails:` docstring.
- A problem that no scenario covers well gets a new scenario in `scenarios.yaml`, and new
  problems get an entry in `known_issues.yaml`.

Write the plan, then get approval at **gate 2** by `EnterPlanMode` → write the plan file →
`ExitPlanMode`. If plan mode is not available, show the plan and ask with `AskUserQuestion`
("Approve plan", "Change it"), then stop. The plan contains:
- A table: problem → root cause → lever → files and functions → the change in one line →
  risk (including which language could regress) → how the re-test shows that it worked.
- The order of the work (cheapest and most certain first), and what you will NOT change.
- The re-test: which scenario ids, how many repeats (at least 3; 5 for prompt changes),
  which unchanged scenarios you run as a regression check in each language, and the target
  (for example "customer-voice in mk-roleplay-* below 10% of turns").
- Items that need a native speaker or a product decision.

## Gate 2 — Approve the plan

Do not edit any project file before the user approves. If they change the plan, update it
and ask again.

## Phase 3 — Implement and re-test

1. If the current branch is `main`, create a branch first (for example
   `fix/slm-quality-<run>`). Do not commit or push unless the user asks.
2. Make the changes in the approved order. After each guard or post-processing change, run
   its new regression test.
3. Run the code tests, then the targeted SLM re-test (in the background):
   ```bash
   .venv/bin/python testing/tools/run_tests.py --sets quick
   .venv/bin/python testing/tools/run_tests.py --sets ui          # only if screens or shared code changed
   .venv/bin/python testing/tools/run_tests.py --sets slm_quality --only <ids> --repeats 5
   ```
   Judge the new replies exactly as in `testing/skills/test-app/SKILL.md` step 2, then
   `tools/build_report.py` on the new run folder.
4. Compare before and after for each approved problem, and write `fix_results.md` in the new
   run folder: problem → before (people's confirmed rate in the reviewed run) → after (the
   machine estimate in the new run) → target met or not. Say clearly that "after" is not
   reviewed by a person yet.
5. If a target is not met, say so. Propose at most one more iteration, and get approval again
   before you change more.

## Phase 4 — Report back

Tell the user, briefly: what changed (files, one line each), the before/after table, the code
test result, anything that still needs a native speaker, and that the new run's
`report.html` has a fresh review queue for a person to confirm the improvement. Nothing is
committed.
