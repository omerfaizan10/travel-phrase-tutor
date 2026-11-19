# Testing the Travel Phrase Tutor

This folder holds all the tests for the app (Spanish and Macedonian), the test report, and two
AI-agent skills. Do all steps from the repository root.

| Item | Contents |
|---|---|
| `tests/` | Code tests: logic, SLM safety net (fake SLM), and screens |
| `slm_quality/` | Learner situations for the real SLM (`scenarios.yaml`), scoring rules, known problems |
| `tools/` | The test runner, the report builder, and the review digest |
| `skills/` | `test-app` and `language-tutor-conversation-reviewer` (instructions for an AI agent) |
| `reports/` | Your test runs. Git ignores this folder. |

## 1. Set up (one time)

1. Install Ollama and the model:
   ```bash
   brew install ollama && brew services start ollama && ollama pull qwen3:4b-instruct
   ```
2. Make the Python environment, if you do not have `.venv`:
   ```bash
   python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
   ```
3. Do a quick test:
   ```bash
   .venv/bin/python testing/tools/run_tests.py
   ```
   The last lines give the report path.

## 2. Run the tests

| You want to… | Command | Time |
|---|---|---|
| Check after a code change | `.venv/bin/python testing/tools/run_tests.py` | 30 s |
| Check the screens | `.venv/bin/python testing/tools/run_tests.py --sets ui` | 1 min |
| Test the real SLM, one language | `.venv/bin/python testing/tools/run_tests.py --sets slm_quality --language macedonian` | 5–15 min |
| Test everything | `.venv/bin/python testing/tools/run_tests.py --sets all` | 10–25 min |

1. Start Ollama before an SLM run.
2. Each run makes a new folder `testing/reports/<date_time>/`.
3. Each person runs on their own computer. Runs do not share files, so you can run at the same time.

**SLM runs need judges.** Without judges, the report shows only the code checks for SLM replies.

- **With Claude Code:** type `/test-app full` (or `/test-app slm --language macedonian`). The
  agent runs the tests, starts the judges, and builds the report.
- **With another AI tool:** tell it: *"Follow step 2 of testing/skills/test-app/SKILL.md for
  testing/reports/<run>"*. Then run:
  ```bash
  .venv/bin/python testing/tools/build_report.py testing/reports/<run>
  ```

## 3. Do your review in the report

1. Open `testing/reports/<run>/report.html` in Chrome, Safari or Firefox.
2. Type your name in **Your name**.
3. Set **Language** to your language.
4. Open the **Review** tab. Each card is one item that needs a person.
5. Read the **coloured box** on the card. It says **why** the item needs you:
   - **Must check** (red): the machine is not sure, or a code test failed.
   - **Should check** (yellow): confirm a problem that the judge found. One card can cover
     several similar replies.
   - **Spot check** (blue): a reply that the machine passed. This measures the judge.
6. Read **what happened**: the learner's line, the SLM reply, and the judge's scores.
7. Click one choice:
   - SLM reply: **Acceptable for a learner**, **Not acceptable**, or **Not sure / skip**.
   - Code test: **Real bug**, **Test is out of date**, or **Not sure**.
8. If a reply is not acceptable, select the **Main problem**. Write the correct wording in the note.
9. Do the next card. The banner and the **Results** tab change immediately.
10. To change a decision, select **show decided items**, then click **undo**.

Your decisions stay in this browser. Use the other tabs for more information:
- **Results:** pass and fail for each test set, language, and scenario.
- **Known problems:** the problems that the team reported before, and whether this run saw them again.
- **Test catalog:** each test in plain language, with what it tests, why, and what a failure means.

## 4. Share your decisions

1. Click **Download my decisions**. You get `human_review_<run>_<name>.json`.
2. Send the file to the person who owns the run, or put it in `testing/reports/<run>/`.
3. To see another reviewer's decisions, click **Load decisions** and select their file.
4. To store the final numbers, put all decision files in the run folder, then run:
   ```bash
   .venv/bin/python testing/tools/build_report.py testing/reports/<run>
   ```

## 5. Add the skills to your AI tool

- **Claude Code, this project:** `bash testing/install_skills.sh`
- **Claude Code, all your projects:** `bash testing/install_skills.sh --user`
- **Another tool:** `bash testing/install_skills.sh <your tool's skill or rules folder>`. If your
  tool has no skill folder, tell it: *"Read testing/skills/<skill>/SKILL.md and follow it."*
- Run the script again after you pull changes to `testing/skills/`.

## 6. Fix problems with the reviewer agent

The `language-tutor-conversation-reviewer` agent uses the decisions to find and fix the problems.

1. Finish your review (step 3). Put the decision files in the run folder (step 4).
2. Start the agent: in Claude Code, type `/language-tutor-conversation-reviewer testing/reports/<run>`.
   In another tool, use the sentence from step 5.
3. **Gate 1:** the agent gives a short summary of what the reviewers found. Read it.
   Approve it, or correct it.
4. **Gate 2:** the agent shows a plan. The plan gives each problem, its cause, the fix (usually a
   prompt change), the files, and the re-test. Approve the plan, or change it. The agent changes
   no code before this approval.
5. The agent makes the fixes on a new branch, adds tests, and runs the tests again.
6. Read `fix_results.md` in the new run folder. Review the new `report.html` (step 3) to confirm
   that the fix worked. The agent does not commit. You decide.

## 7. Add tests

- **New learner situation for the SLM:** add it to `slm_quality/scenarios.yaml`. You do not need code.
- **New code test:** add a `test_*` function in `tests/`. Give it a docstring with three lines:
  `What:`, `Why:` and `If it fails:`. The report's catalog shows them.
- **New known problem:** add it to `slm_quality/known_issues.yaml`.
- **Macedonian:** `scenarios.yaml` and `calibration.json` have draft Macedonian items. A native
  speaker must confirm them.
