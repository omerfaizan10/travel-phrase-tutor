# Spanish & Macedonian Travel Tutor — Version 2

Local, AI-powered language tutor. Everything runs on your laptop — no cloud APIs.

> ### This is v2 (the `v2` branch)
>
> v2 is an updated version of the tutor with the changes listed below. The original
> version is **v1**, kept unchanged on the `main` branch and the `v1` tag:
>
> ```bash
> git checkout v2   # this version: chat roleplay, Gemma 3 12B, model switch, coherence fixes
> git checkout v1   # the original version
> ```
>
> **v2 in short:** a natural **chat roleplay** where the conversation follows one thread,
> **Gemma 3 12B** as the default model with a **switch to compare it against Qwen3 4B**, and
> fixes for the incoherent conversations in v1. Details: [What's new in v2](#whats-new-in-v2)
> and [All changes in v2](#all-changes-in-v2).

## What's new in v2

**Run it**

```bash
git clone https://github.com/omerfaizan10/travel-phrase-tutor.git
cd travel-phrase-tutor
git checkout v2

brew install ollama python@3.12
brew services start ollama
ollama pull gemma3:12b              # 8.2 GB, the default model in v2
ollama pull qwen3:4b-instruct       # 2.5 GB, to compare with the model switch

/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

Gemma 3 12B uses about 7 GB of memory while running (it fits on a 16 GB+ Apple Silicon Mac).
A turn takes roughly 6–15 seconds.

**Chat roleplay (default mode).** The model sees the whole conversation as a normal chat,
writes the partner's reply, and proposes up to three things the learner could say next, so
both sides follow the same thread. Each proposal is matched by *meaning* against the scene's
checked traveler phrases: when one means the same, the phrasebook phrase is shown instead,
with its own translation (marked **✓ phrasebook**). When no phrase fits (e.g. ordering for
the rest of a group of five), the model's own sentence is shown without the mark, but only
after strict checks (script, not English, not a staff line, a real translation, not a
repeat). Each scene has a goal (e.g. "buy a
ticket", "report a lost bag"), so the conversation has a direction and an end. Kept checks
on the partner's line: script and Serbian/Russian forms, the partner talking like the
customer, echoing, and timetable times that are not in the notes (made-up practice
timetables are given for scenes whose notes have none). The original roleplay is still
available as _Full system_ in the sidebar.

**Model switch.** The _Model_ switch at the top of the page runs the same pipeline with
Gemma 3 12B or Qwen3 4B. Switching restarts the current lesson and practice, so each
conversation uses one model.

**Fixes to the original roleplay** (also in _Full system_ mode): suggestions only come from
the learner's variety (no Venezuelan lines for a Spain learner) and fit the scene; the
partner no longer orders, asks prices or says the traveler's lines; the scene no longer
repeats a finished step (e.g. suggesting a second meal after food and drinks are ordered)
or ends on a polite "thank you"; times written in words are checked; vocabulary chips no
longer show speaker labels ("Taxi driver: ...").

**Test transcripts.** `tools/run_scenarios.py` plays every scenario in both languages and
writes readable transcripts. Three learner strategies take the conversation in different
directions: `first` (always the first suggestion), `random` (a random suggestion each turn)
and `detour` (random, plus an off-topic phrase from another scene at turn 2 and a repeat of
the learner's first line at turn 4):

```bash
.venv/bin/python tools/run_scenarios.py results/my_run --chat --strategy detour --model gemma3:12b
.venv/bin/python tools/run_scenarios.py results/my_run            # original roleplay
```

The latest chat-mode test: all 8 scenarios × 2 languages × 3 learner strategies × 2 models
(96 conversations, up to 8 learner turns each), in `results/v2_test/<model>_<strategy>/`:

| Run | Fallbacks | Turns with problems | Learner turns | Scenes ended | Suggestions per turn | Seconds per turn |
|---|---|---|---|---|---|---|
| Gemma 3 12B · first | 1 | 2 | 113 | 8/16 | 2.8 | 6.7 |
| Gemma 3 12B · random | 0 | 0 | 120 | 7/16 | 2.8 | 5.8 |
| Gemma 3 12B · detour | 2 | 2 | 127 | 3/16 | 2.9 | 7.2 |
| Qwen3 4B · first | 9 | 11 | 60 | 15/16 | 2.6 | 4.0 |
| Qwen3 4B · random | 8 | 8 | 78 | 12/16 | 2.7 | 3.3 |
| Qwen3 4B · detour | 5 | 6 | 57 | 15/16 | 2.8 | 2.8 |

"Scenes ended" counts scenes that reached a goodbye within 8 learner turns; Gemma's longer
conversations often hit the limit first, while Qwen's scenes end after 3–5 turns. Each
folder has one transcript per scenario and a `summary.json`. `results/gemma12b_final/` is
the original roleplay (*Full system*) on Gemma 3 12B.

**Known limits.** See [Suggestions for the team](#suggestions-for-the-team) at the end of
this section.

### Suggestions for the team

Open questions and ideas for whoever continues the project; the transcripts in `results/`
show each point.

1. **How strict should suggestions be?** Three versions were tried. (a) Model-written
   suggestions: varied and fitting, but with slips (English in the suggestion, waiter lines
   suggested to the learner, imperfect Macedonian). (b) Phrasebook-only, picked by number:
   always correct sentences, but the model favoured the first numbers in the list and could
   not order for a group of five because no phrase fits. (c) The current version: the model
   proposes, proposals are snapped to the phrasebook by meaning, and the rest are kept only
   after strict checks (shown without ✓). It still sometimes proposes a correct line that is
   off the thread (asking for the menu after ordering) or writes a weak own sentence.
2. **Gemma 3 12B vs. Qwen3 4B** (table above). Chat mode is meant for Gemma: it handles
   off-topic questions and repeats naturally (asked about a ring, the passer-by says "I
   don't work here, ask in the shop"; when the learner repeats "I lost my bag", the officer
   says "Yes, you already said. Where did you lose it?"), with 0–2 fallbacks per 16 scenes.
   Qwen3 4B often speaks as the traveler (the checks catch it, so it falls back to a generic
   line: 5–9 fallbacks per 16 scenes), copies the partner's line into suggestions and ends
   scenes early. It is about 2× faster per turn and needs 2.5 GB instead of 8 GB. The app
   shows a note when chat mode runs on Qwen3 4B.
3. **Grow the traveler phrase lists.** Suggestion quality depends on the notes having
   short, complete traveler answers for each step (numbers of people, "one-way, please",
   "still water"). Adding these to `knowledge_base/` helps both modes.
4. **Native-speaker review.** The partner's lines are still free text; a native speaker
   should spot-check the Macedonian transcripts, and the two new fixed lines
   ("Извинете, не сум сигурен.", "Lo siento, no estoy seguro.").
5. **Practice timetables.** `PRACTICE_FACTS` in `prompts.py` are made-up schedules so the
   transport and hotel scenes can answer time questions. Replace them with checked data if
   the notes get real timetables.
6. **Evaluation.** The test harness always taps the first suggestion; a fairer evaluation
   would also include learners typing their own (sometimes wrong) answers.
7. **v1 calendar interface.** v2 is built on the trip → plan → lesson → practice flow. The
   team's newer calendar interface on `main` (v1) does not have chat mode or the model
   switch yet; bringing them over is a separate step.

### All changes in v2

**Why.** In v1 the roleplays were often incoherent: the waiter said the traveler's lines
("A table for two, please"), ordered food himself, or asked "May I have a glass of water?",
and the suggested learner line often did not follow the conversation (e.g. "Without sugar"
when nobody had asked about sugar). The causes were in the pipeline, not only the model:
the learner's suggestion was picked from the phrasebook by keyword rules instead of being
written in context, the partner was held back by many strict rules, and each turn only
looked at the learner's last line.

**Conversation context**

- Every turn the model gets an English recap of the whole conversation, so it remembers what
  was asked, ordered and answered.
- Retrieval uses the last few turns (not one isolated line) and prefers the current scene's
  notes, so a restaurant scene no longer pulls hotel or taxi lines.
- Every phrasebook line shown to the model is tagged with who says it (traveler or staff).

**Chat roleplay (new default mode, `chat_turn` in `tutor.py`)**

- The model writes the partner's reply and chooses up to three suggested learner replies in
  one call, with the whole conversation in view.
- Each scene has a goal (`SCENE_GOALS` in `prompts.py`); after six learner turns the
  partner is nudged to wrap up once the goal is reached.
- Scene endings are decided in code, not left to the model: once the partner says goodbye
  only checked thank-you/goodbye lines are suggested (never "Do you take credit cards?"
  after "Have a nice day!"); once the learner says goodbye the scene ends after the
  partner's reply; the model's own "scene over" counts only after three learner turns.
- The scene's notes are retrieved once and stay the same for the whole conversation.
- The model proposes the learner's next lines (English + target language); each is matched
  by meaning (embedding similarity ≥ 0.85) against the scene's checked traveler phrases
  (learner's variety only) and replaced by the phrasebook phrase when one means the same.
  Otherwise the model's sentence is kept only if it passes strict checks: script, not
  English, not a staff offer ("¿Qué desea?"), a real translation (fetched if missing or
  copied), swapped text/translation fixed. If nothing is usable, the best-fitting phrases
  are ranked by code. (An intermediate version made the model pick phrases by number; it
  favoured the first numbers in the list regardless of meaning, so it was replaced.)
- Repeats are dropped: a line the learner already said, the same kind of question already
  asked ("how much ...", "what time ..."), or nearly the same meaning (embedding similarity).
  Lines the learner typed or shortened get an English meaning for this (phrasebook entry,
  else a one-off translation).
- Missing or copied translations of the partner's line are filled in with a short extra call.
- Made-up practice timetables (`PRACTICE_FACTS`) for scenes whose notes have no times, so
  the partner can answer "when does it leave?" instead of going in circles.
- The partner is a stranger doing their job: no small talk about the traveler's trip, no
  stage directions in brackets.

**Rules removed or relaxed in chat mode, and why**

The original roleplay (*Full system*) kept the small model safe with many strict rules, but
together they made the partner stiff and the conversation jump around. Chat mode drops or
relaxes these, so the conversation can follow one thread:

| Rule in *Full system* | In chat mode | Why |
|---|---|---|
| The learner's suggestion is picked by code: keyword "scene steps" (`SCENE_STAGES`) plus similarity to the partner's last line | Removed. The model proposes the next lines with the whole conversation in view; they are matched to checked phrasebook phrases by meaning (code ranking is only a fallback) | Keyword steps put lines in the wrong step ("not spicy" counted as a comment on the meal) and ignored what had been said |
| A fixed scene script tells the partner its next move every turn (`STAFF_MOVES`, e.g. "confirm the food and ask about drinks") | Removed. The scene has a goal (`SCENE_GOALS`) and the partner responds to what was actually said | The script pushed the next step even when the moment called for something else |
| Unknown-words rule: a reply is rejected if more than 34% of its words are not in the notes (`MAX_UNKNOWN_SHARE`) | Removed | It rejected natural, correct lines (e.g. "Да, проверете го кабинето" = check the fitting room) and forced stiff, copied wording |
| Notes re-retrieved every turn from the learner's last line | Notes retrieved once per scene and kept fixed | The background kept changing under the conversation, pulling unrelated lines |
| Each turn's prompt adds instructions to the learner's last line ("Reply as the waiter, do not repeat it...") | A normal chat history: learner lines as user messages, the partner's past turns as its own replies | The model sees a real conversation instead of a rebuilt prompt |
| Replies limited to one or two short sentences; "too long", "filler" and "answers its own question" count as problems | One to three short sentences; these soft problems are ignored | Natural replies were cut down or regenerated |
| "May I help you?" sentences removed from replies after the opening | Not applied | Not needed once the partner follows the conversation |
| Feedback tip is a fixed "Exactly right: ... is straight from the phrasebook" when the line is a phrasebook sentence | The model writes the tip about the learner's last line | More useful feedback on what was actually said |

**Kept in chat mode** (these protect correctness and role): Macedonian/Spanish script only;
no Serbian/Russian letters or forms; no Latin words inside Macedonian lines; the partner
never speaks as the customer (ordering, asking the price, asking for directions or
timetable times, "I'm lost"); no exact traveler lines from the notes in the partner's mouth;
no echoing the learner or repeating its own previous line; no stage directions in brackets;
timetable times (digits or words) must be in the notes or the practice timetable; yes/no
answers must not contradict themselves.

**Partner checks (both modes)**

- New checks for the partner talking like the customer: ordering ("Може ли една чаша
  вода?", "¿Me trae...?"), asking the price, asking timetable questions, claiming the
  traveler's problem ("I'm lost"), talking about food outside the restaurant, handing the
  traveler's request back with "please", and repeating one word over and over.
- When one sentence of a reply breaks a rule, only that sentence is dropped instead of
  replacing the whole turn with "Of course."
- Fallback lines come from the current scene (no receptionist line in a restaurant) and
  "check the timetable" is only used at the ticket desk.
- Times written in words ("во осум и дваесет", "a las diez y media") are read as clock
  times and checked against the notes, including times the notes give in words.
- Yes/no answers that contradict themselves ("Yes, it has no meat") are caught.
- Fixes for wrong forms: Serbian "кључ" → "клуч", Latin words inside Macedonian lines,
  "¿Hola!" / "¿Buenos días?" → "¡Hola!" / "¡Buenos días!".

**Suggested learner line in the original roleplay (_Full system_ mode)**

- Suggestions only come from the learner's language variety and from sections that fit the
  scene (no driving lines at a bus ticket desk, no small talk in an emergency).
- Ranked against the conversation so far, not just the partner's last line.
- Scene steps no longer go backwards (no second meal after food and drinks are ordered);
  keywords match whole words ("eat" no longer matches "great"); a polite "thank you"
  mid-scene no longer ends the scene; answers to a question nobody asked ("Without sugar")
  are held back.

**Interface and models**

- Default model: Gemma 3 12B (`llm.py`), with a switch at the top of the page to compare it
  with Qwen3 4B; the previous model is unloaded on switch.
- Chat mode shows up to three tappable suggestions.
- Vocabulary chips show the plain meaning, without speaker labels.
- `roleplay_turn` and `chat_turn` also return the replies that failed the checks
  (`rejected`), for debugging and evaluation.

**What's in this repository:** the runnable app (`app.py`, `planner.py`, `tutor.py`,
`prompts.py`, `retriever.py`, `llm.py`), the curated notes in `knowledge_base/`, and the
UI theme. The evaluation (`evaluate.py`, `results/`), tests, LoRA training (`lora/`) and
source logs (`sources/`) are part of the full project and will be added later; the
sections below that mention them describe the full project. The two "+ LoRA" modes in the
sidebar need the fine-tuned `tutor-lora` model, which is not in this repo; use
_Full system_.

## Setup (macOS)

```bash
brew install ollama python@3.12
brew services start ollama          # runs Ollama in the background
ollama pull qwen3:4b-instruct       # 2.5 GB, one time

/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv

.venv/bin/pip install -r requirements.txt
```

## Check it works

```bash
.venv/bin/python -m pytest -v       # unit tests (+1 live test if Ollama is up)
.venv/bin/python smoke_test.py      # Spanish + Macedonian sample, with timings
```

## Run the app

```bash
.venv/bin/streamlit run app.py
```

1. Describe your trip in English ("I'm going to Skopje in two weeks and want to order at restaurants").
2. Check what was detected (language, days, goals) and fix anything wrong.
3. Get a day-by-day lesson plan built from the curated notes.
4. Open a day: a lesson, then a roleplay starts automatically (e.g. the waiter
   asks what you'd like), with vocabulary help and a suggested sentence every turn.
   Tap the suggested sentence to say it.

The interface is styled like a printed phrasebook (theme in `.streamlit/config.toml`, CSS in
`app.py`). It uses system fonts only, so it works fully offline. The evaluation "Mode" setting is
in the collapsed sidebar.

## Run the evaluation

```bash
.venv/bin/python evaluate.py            # 8 test trips x 3 modes (~10 min)
.venv/bin/python evaluate.py --quick    # 1 trip per language
```

Writes to `results/`: all outputs with automatic checks and timings, a
**blind rating sheet** for the native speakers (rows shuffled, mode hidden;
grammar, naturalness, goal alignment 1-5), the answer key, and a summary.

## Architecture

```
 app.py / evaluate.py
   |
   |  "describe your trip"
   v
 planner.py   parse_trip()  -> language, days, goals          (plain code)
              build_plan()  -> day-by-day sessions from notes  (plain code)
   |
   v
 tutor.py     mode = "rag" (Full system)        mode = "prompt" / "baseline"
              ----------------------------      ---------------------------
              make_lesson()                     tutor_reply()
                model picks phrase NUMBERS        free text from the model
                code prints phrases exactly       (comparison conditions)
                from the notes + quiz
              roleplay_turn()
                partner line + translation,
                vocabulary, suggested reply, tip
                checks: script, wrong-language
                forms, echoing, traveler lines,
                times not in the timetable
                -> regenerate (max 2) -> safe fallback
   |                         |
   v                         v
 retriever.py  knowledge_base/<lang>/*.md -> sections -> multilingual embeddings
               phrase index ("- phrase [pron] = meaning" lines)
 prompts.py    system prompts, language rules, JSON schemas, the 3 modes
 llm.py        Ollama -> Qwen3-4B (local), optional JSON-schema output
```

| File           | Job                                                                |
| -------------- | ------------------------------------------------------------------ |
| `app.py`       | UI: trip -> plan -> lesson -> roleplay (phrasebook-style design)   |
| `planner.py`   | Reads the trip description; builds the lesson plan                 |
| `tutor.py`     | Lessons and roleplay turns for all 3 modes, plus the output checks |
| `prompts.py`   | Settings, prompts, language rules, the 3 modes                     |
| `retriever.py` | Loads/searches `knowledge_base/`; phrase index                     |
| `llm.py`       | The only file that talks to Ollama                                 |
| `evaluate.py`  | Baseline vs prompt-only vs full system, blind rating sheet         |

**Modes (the proposal's comparison):**

- _Baseline_: raw model, no system prompt.
- _Prompt only_: tutor prompt and language rules, no notes.
- _Full system_: notes, phrases copied from the notes, checked roleplay lines.

**Knowledge base:** `knowledge_base/<language>/` uses the same file names for
both languages (`basics.md`, `restaurant.md`, `directions.md`, `shopping.md`,
`hotel_and_time.md`, `transport.md`, `trains.md`, `weather.md`,
`sightseeing.md`, `everyday_conversation.md`). Each `## Section` is one
passage and one possible lesson unit. Phrase lines must look like
`- phrase [pronunciation] = English meaning`. Sources and corrections are
logged in `sources/`.

Macedonian lines carry two brackets: the transliteration (sh / ch / zh / kj / gj)
and an English-style respelling with the stressed syllable in capitals, shown on
the phrase card as "say: ...":
`- Сита сум. [Sita sum.] [SEE-ta soom.] = I'm full.`
Respelling: и = ee, у = oo, ј = y, ц = ts, ќ / ѓ = ky / gy, џ = j, љ / њ = ly / ny.

## Using the model from your code (Member A — `llm.py`)

```python
from llm import generate_response

reply = generate_response([
    {"role": "system", "content": "You are a friendly Spanish tutor."},
    {"role": "user", "content": "How do I order a coffee?"},
])
```

- Optional overrides: `generate_response(messages, {"temperature": 0.2})`
- Need timings for evaluation? Use `generate_response_with_stats()` — returns
  `text`, `total_seconds`, `load_seconds`, `output_tokens`, `tokens_per_second`.
- Failures raise `llm.LLMError` with a fix-it message (e.g. Ollama not running).

## Model & hardware (recorded 2026-11-11)

|          |                                                                         |
| -------- | ----------------------------------------------------------------------- |
| Model    | `qwen3:4b-instruct` (= `qwen3:4b-instruct-2507-q4_K_M`, 4-bit, 2.5 GB)  |
| Runtime  | Ollama 0.40.16, ollama-python 0.6.3                                     |
| Settings | temperature 0.7 (0.5 in tutor modes), top_p 0.8, top_k 20, num_ctx 8192 |
| Hardware | Apple M5 Pro, 24 GB RAM                                                 |
| Speed    | ~85 tokens/s, ~1 s per short reply (model already loaded)               |
