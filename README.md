# Spanish & Macedonian Travel Tutor 

Local, AI-powered language tutor. Everything runs on your laptop — no cloud APIs.

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
