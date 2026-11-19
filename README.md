# Travel Phrase Tutor (Spanish & Macedonian)

A local, AI-assisted travel-language tutor built for the CSIT 559 team project. The learner
chooses a language, a study window and travel topics; the app builds a dated study calendar and
walks them through three kinds of practice: vocabulary, sentence building and short role plays.

Everything runs on your laptop. A small language model (Qwen3-4B) runs through
[Ollama](https://ollama.com), and every phrase that is taught comes from hand-curated notes in
`knowledge_base/`. No cloud API is used and no data leaves the machine.

## What the app does

### 1. Preferences: buttons or a chat

The learner gives their preferences in one of two ways:

- **Choose with buttons:** language; Spanish variety (Spain, Latin America in general, or one of
  19 Latin American countries); a study window picked on a date-range calendar (no upper limit);
  practice weekdays; minutes per session (1–240); topics, ranked by priority; and lesson types.
- **Describe it in a chat:** the learner writes in English, Spanish or Macedonian (for example,
  "I'm going to Mexico in March, I can study Tuesdays and Thursdays for 20 minutes, mostly
  restaurants"). Code reads the dates, weekdays, minutes, topics and country with word rules for
  all three languages. The model only fills a gap when code found nothing. The tutor asks for one
  missing item at a time, never asks for the same item twice (it suggests a value instead), and
  replies in the learner's language, with a "Show in English" toggle.

Topics: restaurants, directions, shopping, hotels, transportation, weather, sightseeing and
getting help. The app keeps the choices until the plan is approved, so "Change my preferences"
brings the learner back to the same screen with the same values.

### 2. Summary

Before showing any dates, the app explains the plan: the number of sessions and total minutes,
the share of time for each lesson type (vocabulary 30% / sentences 30% / role play 40%, adjusted
when a type is left out), the share of time for each topic by rank, and for Spanish, which variety
each topic uses. If a country has no notes for a topic, Spain Spanish is used and the learner can
switch that topic to another variety with notes.

### 3. Calendar

The approved plan becomes dated sessions on the learner's practice weekdays. The calendar shows
overall progress, a **Next lesson** card with a Continue button, and each session's state
(complete / in progress / upcoming). Inside a practice, a bar gives ← Calendar, ‹ Previous and
Next ›, plus chips for the activities in the session. An activity counts as complete as soon as
it has a result.

### Practice stages (both languages)

| Stage | What the learner does | Model use |
|---|---|---|
| **1 · Vocabulary** | Study phrase cards from the notes, type each item three times, then answer recall questions. Answers are checked locally (case, spacing and punctuation are ignored; a missing accent is flagged separately). Macedonian accepts Cyrillic or Latin typing. Items answered right on the first try leave the bank. | None. All checking is plain code. |
| **2 · Sentences** | The model writes 3 example pairs and 5 English prompts built only from the words the learner studied in Stage 1. The learner types whole sentences and gets meaning and grammar feedback, then a recap. | Qwen3-4B with JSON output. Macedonian rounds are grounded in checked sentences from the notes. |
| **3 · Role play** | A short scene with a waiter, passer-by, shop assistant, receptionist and so on. The learner gets three responses per scene, with a translation and a hint each turn, then feedback on meaning, grammar and whether the goal was met. There is no limit on the number of scenes. | Qwen3-4B. Every partner line is checked (script, wrong-language forms, role slips, invented times) and regenerated or replaced with a safe line from the notes. |

If a plan has no vocabulary lessons, the topic's word list is shown before its first sentence or
role-play activity.

### Saved plans

Approved plans are saved automatically, with all progress. **Shelf this plan** goes to
**Your plans**, where the learner can open, rename or delete any saved plan. Plans are kept after
the app restarts, in `~/.travel_phrase_tutor/plans.json`. Set the `TUTOR_PLANS_FILE` environment
variable to use a different file. The file is outside the repository on purpose, because it holds
learner data.

### Pronunciation (Macedonian)

Every Macedonian phrase shows a transliteration (sh / ch / zh / kj / gj) and an English-style
"say:" guide with the stressed syllable in capitals, for example `Сита сум. → SEE-ta soom.`
Lines in the notes carry hand-checked guides. `pronounce.py` generates guides for the model's
role-play lines.

## Requirements

| | |
|---|---|
| OS | macOS (tested on Apple Silicon). Linux should work with the equivalent installs. |
| Python | 3.12 |
| Ollama | Recent version, running locally on port 11434 |
| Model | `qwen3:4b-instruct` (2.5 GB download) |
| Memory | 16 GB RAM or more recommended |
| Disk | About 5 GB (model, PyTorch, embedding model) |

Python packages (`requirements.txt`): `ollama`, `sentence-transformers`, `streamlit`, `numpy`,
`pytest` and `pyyaml` (used by the test report builder).

Internet is needed once, during setup: to install packages, pull the Ollama model, and download
the multilingual embedding model (`paraphrase-multilingual-MiniLM-L12-v2`) on the first run. After
that the app works offline.

## Setup (macOS)

1. Install Ollama and Python, start Ollama, and pull the model (one time):

   ```bash
   brew install ollama python@3.12
   ```

   ```bash
   brew services start ollama
   ```

   ```bash
   ollama pull qwen3:4b-instruct
   ```

2. Create the virtual environment and install the packages:

   ```bash
   /opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv
   ```

   ```bash
   .venv/bin/pip install -r requirements.txt
   ```

3. Check that it works:

   ```bash
   .venv/bin/python testing/tools/run_tests.py
   ```

If Ollama runs on another machine or port, set `OLLAMA_HOST` (default `http://localhost:11434`).

## Run the app

```bash
.venv/bin/streamlit run app.py
```

Open http://localhost:8501. The first page load takes a few seconds while the embedding model
loads. The first model reply is slower while Ollama loads Qwen3 into memory. After that, replies
take about a second.

In the Claude Code desktop app, `.claude/launch.json` defines a `streamlit` preview on port 8501
(and `streamlit-check` on 8502 for a second copy).

**Sharing a demo:** to put a passcode screen in front of the app (for example, behind a public
tunnel), create `.streamlit/secrets.toml` with `passcode = "..."`. Git ignores this file. With no
passcode set, the app opens directly.

## Testing

All tests live in `testing/`. See [testing/README.md](testing/README.md) for the full guide,
including the human review workflow.

| You want to… | Command | Time |
|---|---|---|
| Check after a code change (logic + SLM safety net with a fake model) | `.venv/bin/python testing/tools/run_tests.py` | ~10–30 s |
| Check the screens (Streamlit AppTest, fake model) | `.venv/bin/python testing/tools/run_tests.py --sets ui` | ~1 min |
| Test real model conversations for one language | `.venv/bin/python testing/tools/run_tests.py --sets slm_quality --language macedonian` | 5–15 min |
| Run everything | `.venv/bin/python testing/tools/run_tests.py --sets all` | 10–25 min |
| Plain pytest | `.venv/bin/python -m pytest testing/tests -m unit` | seconds |

Each run writes `testing/reports/<date_time>/` (ignored by git) with `results.json`,
`summary.md` and an interactive `report.html`. The report has a review queue for native speakers,
results by language and scenario, known problems, and a catalog of every test. Pytest never calls
the real model: a guard fails the test if code reaches Ollama.

**Claude Code skills** (in `testing/skills/`, install with `bash testing/install_skills.sh`):

- `/test-app` runs the tests, collects real model conversations, has judge agents score every
  reply against `testing/slm_quality/rubric.md`, and builds the report.
- `/language-tutor-conversation-reviewer` reads the reviewers' decisions from a report, proposes
  fixes with two approval gates, then implements them and re-runs the tests.

**Current status:** the quick suite has about 265 tests. Two fail on purpose, because they cover
problems listed in `testing/slm_quality/known_issues.yaml` that are not fixed yet: Stage 1 rejects
some Macedonian answers typed exactly as the card's Latin spelling (`lj`, `è`), and some fallback
partner lines show the wrong English meaning.

## Architecture

```
app.py  (Streamlit UI: preferences -> summary -> calendar -> practice; saved plans)
  |
  |-- chat_preferences.py  chat option: word rules (EN/ES/MK) + model extraction as a backup
  |-- planner.py           build_schedule(): dated sessions, stage/topic minutes, variety per topic
  |-- plan_store.py        saves and restores plans in ~/.travel_phrase_tutor/plans.json
  |
  |-- Stage 1  practice.py + spanish_practice.py   phrase bank, typing, recall (no model)
  |-- Stage 2  sentence_practice.py                model-built sentence rounds + feedback
  |-- Stage 3  roleplay_practice.py                three-turn scenes + end feedback
  |               |
  |               '-- tutor.py   reply checks, wrong-language fixes, safe fallback lines
  |
  |-- retriever.py   knowledge_base/<lang>/*.md -> sections, phrase index, embeddings
  |-- prompts.py     system prompts, language and country rules, JSON schemas
  |-- pronounce.py   Macedonian transliteration, "say:" guide, Latin -> Cyrillic
  '-- llm.py         the only module that talks to Ollama
```

| File | Job |
|---|---|
| `app.py` | All screens, styling (printed-phrasebook look, system fonts only), calendar progress, navigation |
| `chat_preferences.py` | Reads chat messages in three languages; asks for missing items; tutor replies |
| `planner.py` | Builds the schedule; picks the Spanish variety for each topic (`plan_varieties`) |
| `plan_store.py` | JSON plan file with atomic writes; snapshot and restore of session state |
| `practice.py` | Builds Stage 1 phrase banks and recall questions; answer checking |
| `spanish_practice.py` | Stage 1 typing screen (both languages, despite the name) |
| `sentence_practice.py` | Stage 2 rounds; checks model output and feedback |
| `roleplay_practice.py` | Stage 3 scenes, turns and end feedback |
| `tutor.py` | Role-play reply checks and fallbacks; also the older lesson/roleplay pipeline |
| `retriever.py` | Loads and searches `knowledge_base/`; Spanish variety labels; phrase parsing |
| `prompts.py` | Prompts, language rules, Spain / Latin America / country rules |
| `pronounce.py` | Macedonian pronunciation helpers |
| `llm.py` | Ollama client: `generate_response()`, `generate_response_with_stats()`, `LLMError` |

**Design rule:** code decides and the model assists. Code builds the plan, checks Stage 1
answers, decides when the chat has every item, and checks every model line. The model writes
sentence tasks, role-play lines and feedback, and is always given the learner's studied words and
the notes.

## Knowledge base

`knowledge_base/spanish/` and `knowledge_base/macedonian/` use the same ten files: `basics`,
`restaurant`, `directions`, `shopping`, `hotel_and_time`, `transport`, `trains`, `weather`,
`sightseeing` and `everyday_conversation`. Each `## Section` is one passage. Phrase lines must
look like this:

```
- phrase [pronunciation] = English meaning
```

**Spanish** sections start with a `Country:` line (`ES`, `419`, `ALL`, or a country code such as
`MX` or `AR`). Most notes are Spain Spanish. General Latin American notes and country notes cover
Argentina, Chile, Colombia, Costa Rica, Cuba, the Dominican Republic, Ecuador, El Salvador,
Guatemala, Honduras, Mexico, Paraguay, Peru, Puerto Rico, Uruguay and Venezuela. Countries without
notes for a topic fall back to Spain Spanish (shown in the app).

**Macedonian** lines carry two brackets, the transliteration and the stress guide, and were
checked by a native speaker:

```
- Сита сум. [Sita sum.] [SEE-ta soom.] = I'm full.
```

Respelling: и = ee, у = oo, ј = y, ц = ts, ќ / ѓ = ky / gy, џ = j, љ / њ = ly / ny.

## Using the model from your code

```python
from llm import generate_response

reply = generate_response([
    {"role": "system", "content": "You are a friendly Spanish tutor."},
    {"role": "user", "content": "How do I order a coffee?"},
])
```

- Overrides: `generate_response(messages, {"temperature": 0.2})`. Pass `"format": <JSON schema>`
  for structured output.
- Timings: `generate_response_with_stats()` also returns `total_seconds`, `load_seconds`,
  `output_tokens` and `tokens_per_second`.
- Failures raise `llm.LLMError` with a fix-it message (for example, "Ollama is not running").

## Model & hardware

| | |
|---|---|
| Model | `qwen3:4b-instruct` (`qwen3:4b-instruct-2507-q4_K_M`, 4-bit, 2.5 GB) |
| Settings | temperature 0.7 (0.5 for tutor turns), top_p 0.8, top_k 20, num_ctx 8192, keep-alive 10 min |
| Embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (falls back to CPU if MPS fails) |
| Tested on | Apple M5 Pro, 24 GB RAM: about 85 tokens/s, about 1 s per short reply once loaded |

## Known limitations

- Two known Stage 1 / role-play bugs are tracked by failing tests (see **Testing**). The other
  problems that reviewers report are listed in `testing/slm_quality/known_issues.yaml`.
- The **Evaluation → Mode** selector in the sidebar is left over from the earlier
  trip → lesson → roleplay flow. It does not change the current calendar practice. The comparison
  modes (baseline / prompt only / full system) and the "+ LoRA" options are still in `tutor.py` and
  `prompts.py`, but the evaluation script and the fine-tuned `tutor-lora` model are not in this
  repository.
- Stage switching inside a plan (moving ahead or back a stage) is not built yet.
- Macedonian scenarios in `testing/slm_quality/` are drafts and need confirmation by a native
  speaker.
- Saved plans live in one file per computer, so everyone who uses the same copy of the app shares
  it.
