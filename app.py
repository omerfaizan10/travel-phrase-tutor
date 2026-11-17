"""
app.py: Streamlit UI for the travel language tutor.

Run:   .venv/bin/streamlit run app.py

Flow (as in the proposal):
  1. Trip     - the learner describes their trip in English.
  2. Plan     - detected language/timeline/goals are confirmed and a
                day-by-day plan is built from the curated notes.
  3. Lesson   - phrase cards for the chosen day, with a short quiz.
  4. Practice - a roleplay starts automatically, with a translation,
                vocabulary and a suggested sentence (tap it to say it).

The sidebar "Mode" is for the evaluation: "Full system" is the grounded,
checked pipeline; "Prompt only" and "Baseline" are the comparison conditions.
"""

from html import escape

import streamlit as st

from llm import LLMError, unload
from planner import GOAL_HINTS, build_plan, parse_trip
from prompts import LANGUAGES, ROLEPLAY_PARTNER, SCENARIOS, TutorSettings
from retriever import Retriever
from tutor import chat_turn, make_lesson, roleplay_turn, start_session, tutor_reply, wrong_language

MODE_LABELS = {
    "chat": "Chat roleplay (natural conversation, phrasebook-checked)",
    "rag": "Full system (notes + checks)",
    "prompt": "Prompt only",
    "baseline": "Baseline (raw model)",
    "rag+lora": "Full system + LoRA model",
    "prompt+lora": "Prompt only + LoRA model",
}
# Models to compare (sidebar switch). Same pipeline and prompts; only the model differs.
MODELS = {"Gemma 3 12B": "gemma3:12b", "Qwen3 4B": "qwen3:4b-instruct"}
GOAL_LABELS = {g: g.capitalize() for g in GOAL_HINTS}
NATIVE_NAME = {"spanish": "Español", "macedonian": "Македонски"}
EXAMPLES = {
    "Madrid · restaurants": "I'm going to Madrid in two weeks and want to order at local restaurants.",
    "Skopje · train to Bitola": "Visiting Skopje next week. I need directions and the train to Bitola.",
    "Seville · markets, hotel": "Trip to Seville in 5 days, I want to shop at markets and check into my hotel.",
}
STEPS = ["Trip", "Plan", "Lesson", "Practice"]

st.set_page_config(page_title="Travel Phrase Tutor", initial_sidebar_state="collapsed")

# --- Styling -----------------------------------------------------------------
# System fonts only (no web fonts), so the app works fully offline.
st.markdown("""
<style>
:root {
  --ink: #1F1D1A; --muted: #6E675D; --faint: #9A9287; --line: #E3DDD2;
  --paper: #F7F4EE; --card: #FFFDF9; --accent: #A8492A; --accent-soft: #F3E4DC;
  --serif: Charter, "Iowan Old Style", "Palatino Linotype", Georgia, serif;
  --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  --mono: "SF Mono", Menlo, Consolas, monospace;
}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"] { visibility: hidden; height: 0; }
.block-container { max-width: 760px; padding-top: 2.2rem; padding-bottom: 6rem; }
html, body, [class*="css"], .stMarkdown, p, li, label { font-family: var(--sans); color: var(--ink); }
h1, h2, h3 { font-family: var(--serif) !important; font-weight: 600 !important;
             letter-spacing: -0.01em; color: var(--ink); }

/* Masthead */
.masthead { display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap;
            border-bottom: 1px solid var(--ink); padding-bottom: .6rem; margin-bottom: .4rem; }
.wordmark { font-family: var(--serif); font-size: 1.65rem; font-weight: 600; }
.wordmark span { color: var(--accent); }
.tagline { font-size: .78rem; color: var(--muted); letter-spacing: .04em; text-transform: uppercase; }

/* Step bar */
.steps { display: flex; gap: 1.6rem; margin: .2rem 0 2rem; font-size: .8rem; flex-wrap: wrap;
         letter-spacing: .06em; text-transform: uppercase; color: var(--faint); }
.steps .on { color: var(--ink); border-bottom: 2px solid var(--accent); padding-bottom: 2px; }
.steps .done { color: var(--muted); }

.section-label { font-size: .72rem; letter-spacing: .1em; text-transform: uppercase;
                 color: var(--accent); margin: 1.8rem 0 .3rem; font-weight: 600; }
.lede { font-family: var(--serif); font-size: 1.12rem; line-height: 1.55; color: var(--ink); }
.quiet { color: var(--muted); font-size: .88rem; }

/* Plan timeline */
.day-num { font-family: var(--serif); font-size: 1.05rem; color: var(--accent); padding-top: .35rem; }
.day-topic { font-size: .95rem; padding-top: .4rem; }
.tag { font-size: .7rem; letter-spacing: .06em; text-transform: uppercase; color: var(--muted);
       border: 1px solid var(--line); border-radius: 3px; padding: 1px 6px; white-space: nowrap; }

/* Phrase cards */
.phrase { background: var(--card); border: 1px solid var(--line); border-radius: 6px;
          padding: .85rem 1rem .8rem; margin-bottom: .6rem; }
.phrase .n { font-size: .72rem; color: var(--faint); float: right; }
.phrase .t { font-family: var(--serif); font-size: 1.22rem; line-height: 1.3; }
.phrase .p { font-family: var(--mono); font-size: .78rem; color: var(--muted); margin-top: .15rem; }
.phrase .m { font-size: .92rem; margin-top: .35rem; }
.phrase .who { font-size: .75rem; color: var(--accent); margin-top: .2rem; }
.note { border-left: 2px solid var(--accent); padding: .3rem 0 .3rem .9rem; margin: 1rem 0;
        font-family: var(--serif); font-size: 1rem; color: var(--ink); }
.sources { font-size: .75rem; color: var(--faint); margin-top: .6rem; }

/* Roleplay */
.bubble { max-width: 84%; padding: .7rem .95rem; border-radius: 10px; margin: .5rem 0; line-height: 1.45; }
.partner { background: var(--card); border: 1px solid var(--line); border-top-left-radius: 2px; }
.me { background: var(--ink); color: var(--paper); margin-left: auto; border-top-right-radius: 2px; }
.me * { color: var(--paper) !important; }
.who-label { font-size: .68rem; letter-spacing: .08em; text-transform: uppercase; color: var(--faint); margin-bottom: .15rem; }
.partner .line { font-family: var(--serif); font-size: 1.15rem; }
.partner .tr { font-size: .85rem; color: var(--muted); margin-top: .2rem; }
.vocab { margin-top: .5rem; font-size: .82rem; }
.vocab b { font-family: var(--serif); font-weight: 600; }
.chip { display: inline-block; background: var(--accent-soft); border-radius: 3px; padding: 1px 7px;
        margin: 2px 4px 2px 0; }
.tipline { font-size: .82rem; color: var(--muted); margin-top: .45rem; font-style: italic; }

/* Widgets */
.stButton > button { border-radius: 4px; border: 1px solid var(--ink); background: transparent;
                     color: var(--ink); font-family: var(--sans); font-size: .88rem; padding: .35rem .9rem; }
.stButton > button:hover { border-color: var(--accent); color: var(--accent); }
.stButton > button[kind="primary"] { background: var(--ink); color: var(--paper); }
.stButton > button[kind="primary"]:hover { background: var(--accent); border-color: var(--accent); color: var(--paper); }
.stButton > button p { color: inherit !important; }
textarea, input, [data-baseweb="select"] > div { border-radius: 4px !important; }
[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 6px; background: var(--card); }
[data-testid="stHeaderActionElements"], .stMarkdown h2 a, .stMarkdown h3 a { display: none !important; }
</style>
""", unsafe_allow_html=True)


# --- Helpers -------------------------------------------------------------------
@st.cache_resource
def get_retriever():
    return Retriever()  # loaded once, shared across reruns


def reset(*keys):
    for k in keys:
        st.session_state.pop(k, None)


def html(s):
    st.markdown(s, unsafe_allow_html=True)


def label(text):
    html(f'<div class="section-label">{escape(text)}</div>')


def step_bar(current):
    parts = []
    for i, name in enumerate(STEPS):
        cls = "on" if i == current else ("done" if i < current else "")
        parts.append(f'<span class="{cls}">{i + 1} &nbsp;{name}</span>')
    html('<div class="steps">' + "".join(parts) + "</div>")


def fail(e):
    st.error(f"The local model didn't respond: {e}")
    st.stop()


def phrase_card(i, p):
    pron = f'<div class="p">{escape(p["pron"])}</div>' if p.get("pron") else ""
    who = f'<div class="who">the {escape(p["speaker"])} says this</div>' if p.get("speaker") else ""
    html(f'<div class="phrase"><span class="n">{i:02d}</span><div class="t">{escape(p["text"])}</div>'
         f'{pron}<div class="m">{escape(p["meaning"])}</div>{who}</div>')


def partner_bubble(t, partner):
    # The translation is model-written (not from the notes), so it is marked as approximate.
    tr = (f'<div class="tr" title="Approximate translation by the model; the vocabulary '
          f'below is from the checked notes">≈ {escape(t["reply_english"])}</div>'
          if t.get("reply_english") else "")
    vocab = ""
    if t.get("vocab"):
        vocab = '<div class="vocab"><span class="quiet">In this line: </span>' + "".join(
            f'<span class="chip"><b>{escape(v["text"])}</b> &nbsp;{escape(v["meaning"])}</span>'
            for v in t["vocab"]) + "</div>"
    tip = f'<div class="tipline">{escape(t["tip"])}</div>' if t.get("tip") else ""
    html(f'<div class="bubble partner"><div class="who-label">{escape(partner)}</div>'
         f'<div class="line">{escape(t["reply"])}</div>{tr}{vocab}{tip}</div>')


def my_bubble(text):
    html(f'<div class="bubble me"><div class="who-label">You</div>{escape(text)}</div>')


# --- Masthead + sidebar ----------------------------------------------------------
html('<div class="masthead"><div class="wordmark">Travel Phrase <span>Tutor</span></div>'
     '<div class="tagline">Español · Македонски · runs on your laptop</div></div>')

# Team passcode, only when the app is shared over a public link. Set it in
# .streamlit/secrets.toml (passcode = "...", not pushed to GitHub); with no
# passcode set, the app opens directly, as on your own laptop.
try:
    PASSCODE = st.secrets.get("passcode", "")
except Exception:          # no secrets.toml at all
    PASSCODE = ""
if PASSCODE and not st.session_state.get("unlocked"):
    html('<div class="lede">This is the CSIT 559 team demo. Enter the team passcode to try it.</div>')
    code = st.text_input("Passcode", type="password")
    if code:
        if code.strip() == PASSCODE:
            st.session_state.unlocked = True
            st.rerun()
        st.error("That passcode isn't right.")
    st.stop()

# Model switch on the page itself (the sidebar starts collapsed), so the two
# models are easy to compare.
model_name = st.radio("Model", list(MODELS), horizontal=True, key="model_switch",
                      help="Compare the two models on the same pipeline. "
                           "Switching starts the current lesson and practice again.")
if st.session_state.get("model_name", model_name) != model_name:
    # Keep each conversation to one model, so a comparison is fair, and free the
    # other model's memory so Ollama never has to juggle both.
    reset("lesson", "roleplay", "free_chat")
    unload(MODELS[st.session_state.model_name])
st.session_state.model_name = model_name

with st.sidebar:
    st.markdown("### Evaluation")
    mode_label = st.selectbox("Mode", list(MODE_LABELS.values()),
                              help="Full system is the one to demo. The others are the "
                                   "comparison conditions from the proposal.")
    mode_key = next(k for k, v in MODE_LABELS.items() if v == mode_label)
    # "chat" = the full system with the chat roleplay (chat_turn) instead of roleplay_turn.
    use_chat = mode_key == "chat"
    mode, use_lora = ("rag" if use_chat else mode_key.split("+")[0]), mode_key.endswith("+lora")
    turn_fn = chat_turn if use_chat else roleplay_turn
    # The LoRA modes keep their fine-tuned model; every other mode uses the switch.
    mc = None if use_lora else {"model": MODELS[model_name]}
    if st.button("Start over"):
        reset("trip", "parsed", "plan", "session", "lesson", "roleplay", "free_chat", "draft")
        st.rerun()

stage = (3 if "lesson" in st.session_state
         else 2 if "session" in st.session_state
         else 1 if "parsed" in st.session_state else 0)
step_bar(stage)

# --- 1. Trip ---------------------------------------------------------------------------
html('<div class="lede">Tell me about your trip: where you are going, when, and what you '
     'want to be able to do there. I will plan your lessons around it.</div>')
trip = st.text_area("Your trip", key="draft", height=90, label_visibility="collapsed",
                    placeholder="e.g. I'm going to Madrid in two weeks and want to order at local restaurants.")
html('<div class="quiet" style="margin:.2rem 0 .3rem">Or start from an example:</div>')
def use_example(text):
    st.session_state.draft = text   # runs before the text box is drawn


for col, (short, ex) in zip(st.columns(len(EXAMPLES)), EXAMPLES.items()):
    col.button(short, key=f"ex-{short}", help=ex, use_container_width=True,
               on_click=use_example, args=(ex,))

if st.button("Plan my trip", type="primary"):
    if not trip.strip():
        st.warning("Describe your trip first.")
    else:
        reset("plan", "session", "lesson", "roleplay", "free_chat")
        st.session_state.trip = trip.strip()
        st.session_state.parsed = parse_trip(trip)
        st.rerun()

if "parsed" not in st.session_state:
    st.stop()

# --- 2. Confirm + plan ------------------------------------------------------------------
p = st.session_state.parsed
label("What I understood")
if not p["found"]["language"]:
    st.warning("I couldn't tell which language you need. Please choose it.")
langs = list(LANGUAGES)
c1, c2 = st.columns([3, 2])
lang_label = c1.selectbox("Language", [LANGUAGES[l] for l in langs],
                          index=langs.index(p["language"]) if p["language"] else 0)
language = langs[[LANGUAGES[l] for l in langs].index(lang_label)]
days = c2.number_input("Days until you leave", 1, 60, value=int(p["days"]))
goal_labels = st.multiselect("You want to be able to", list(GOAL_LABELS.values()),
                             default=[GOAL_LABELS[g] for g in p["goals"]])
goals = [g for g, lab in GOAL_LABELS.items() if lab in goal_labels]
missing = [k for k in ("days", "goals") if not p["found"][k]]
if missing:
    html(f'<div class="quiet">Not in your description, so I guessed: {", ".join(missing)}. '
         'Change them above if needed.</div>')

if st.button("Build my plan", type="primary"):
    if not goals:
        st.warning("Choose at least one goal.")
    else:
        reset("session", "lesson", "roleplay", "free_chat")
        st.session_state.plan = build_plan(language, goals, int(days), get_retriever())
        st.session_state.plan_lang = language
        st.session_state.plan_days = int(days)
        st.rerun()

if "plan" not in st.session_state:
    st.stop()

plan = st.session_state.plan
language = st.session_state.plan_lang
if not plan:
    st.error(f"There are no {LANGUAGES[language]} notes for these goals yet.")
    st.stop()

label(f"Your plan · {NATIVE_NAME[language]} · {st.session_state.plan_days} days")
is_any_open = "session" in st.session_state
plan_box = (st.expander(f"Day {plan[st.session_state.session]['day']} is open · change day")
            if is_any_open else st.container())
with plan_box:
  for i, s in enumerate(plan):
    c_day, c_topic, c_btn = st.columns([1.1, 6, 1.6])
    c_day.markdown(f'<div class="day-num">Day {s["day"]}</div>', unsafe_allow_html=True)
    c_topic.markdown(f'<div class="day-topic">{escape(s["title"])} &nbsp;'
                     f'<span class="tag">{escape(GOAL_LABELS[s["scenario"]])}</span></div>',
                     unsafe_allow_html=True)
    is_open = st.session_state.get("session") == i
    if c_btn.button("Open", key=f"open-{i}", type="primary" if is_open else "secondary",
                    use_container_width=True):
        reset("lesson", "roleplay", "free_chat")
        st.session_state.session = i
        st.rerun()

if "session" not in st.session_state:
    st.stop()

session = plan[st.session_state.session]
settings = TutorSettings(language=language, scenario=session["scenario"], task="roleplay",
                         mode=mode, lora=use_lora, goal=st.session_state.trip,
                         timeline=f"{st.session_state.plan_days} days")
partner = ROLEPLAY_PARTNER[session["scenario"]].removeprefix("the ").removeprefix("a ").capitalize()
r = get_retriever()

# --- Baseline / prompt-only: plain chat (comparison conditions) -------------------------
if mode != "rag":
    label(f"Day {session['day']} · {MODE_LABELS[mode_key]}")
    if "free_chat" not in st.session_state:
        lesson_settings = TutorSettings(**{**settings.__dict__, "task": "lesson"})
        history = start_session(lesson_settings)
        with st.spinner("Writing your lesson…"):
            try:
                out = tutor_reply(lesson_settings, history, r, model_config=mc)
            except LLMError as e:
                fail(e)
        st.session_state.free_chat = history + [{"role": "assistant", "content": out["text"]}]
    for m in st.session_state.free_chat[1:]:
        if m["role"] == "user":
            my_bubble(m["content"])
        else:
            html(f'<div class="bubble partner">{escape(m["content"]).replace(chr(10), "<br>")}</div>')
    if text := st.chat_input("Reply…"):
        st.session_state.free_chat.append({"role": "user", "content": text})
        try:
            out = tutor_reply(settings, st.session_state.free_chat, r, model_config=mc)
            st.session_state.free_chat.append({"role": "assistant", "content": out["text"]})
        except LLMError as e:
            fail(e)
        st.rerun()
    st.stop()

# --- 3. Lesson ---------------------------------------------------------------------------
if "lesson" not in st.session_state:
    with st.spinner("Preparing today's lesson…"):
        try:
            st.session_state.lesson = make_lesson(settings, session, st.session_state.trip, r, model_config=mc)
        except LLMError as e:
            fail(e)
    st.rerun()

L = st.session_state.lesson
label(f"Day {session['day']}")
st.markdown(f"## {session['title']}")
html(f'<div class="lede">{escape(L["intro"])}</div>')
st.write("")
cols = st.columns(2)
for i, ph in enumerate(L["phrases"], 1):
    with cols[(i - 1) % 2]:
        phrase_card(i, ph)
if L["tip"]:
    html(f'<div class="note">{escape(L["tip"])}</div>')
if L["quiz"]:
    with st.expander("Check yourself"):
        for q in L["quiz"]:
            st.markdown(f"**{q['q']}**")
            st.markdown(f'<span class="quiet">{escape(q["a"])}</span>', unsafe_allow_html=True)
html(f'<div class="sources">From the curated notes: {escape(", ".join(L["sources"]))}</div>')

# --- 4. Practice (starts automatically) -------------------------------------------------
label(f"Practice · {model_name if not use_lora else 'LoRA model'}")
if use_chat and MODELS[model_name] != "gemma3:12b":
    html('<div class="note">Chat roleplay works best with Gemma 3 12B. Qwen3 4B often mixes up '
         'the roles (the partner says the traveler\'s lines), so expect more generic fallback lines.</div>')
st.markdown(f"## {SCENARIOS[session['scenario']].capitalize()}")
if "roleplay" not in st.session_state:
    with st.spinner(f"The {partner.lower()} is on the way…"):
        try:
            first = turn_fn(settings, [], session, st.session_state.trip, r, model_config=mc)
        except LLMError as e:
            fail(e)
    st.session_state.roleplay = [{"role": "assistant", "content": first["reply"], "turn": first}]


def send(text):
    # Keep the partner's English translations: tutor.py uses them to track how far
    # the scene has got (so suggestions only move forward).
    history = [{"role": m["role"], "content": m["content"],
                **({"english": m["turn"].get("reply_english", "")} if m["role"] == "assistant" else {})}
               for m in st.session_state.roleplay]
    history.append({"role": "user", "content": text})
    my_bubble(text)                       # show the learner's line straight away
    with st.spinner(f"The {partner.lower()} is replying…"):
        try:
            t = turn_fn(settings, history, session, st.session_state.trip, r, model_config=mc)
        except LLMError as e:
            fail(e)
    st.session_state.roleplay += [{"role": "user", "content": text},
                                  {"role": "assistant", "content": t["reply"], "turn": t}]
    st.rerun()


for m in st.session_state.roleplay:
    if m["role"] == "assistant":
        partner_bubble(m["turn"], partner)
    else:
        my_bubble(m["content"])

last = st.session_state.roleplay[-1]["turn"]
if last.get("scene_done"):
    # The scene has ended: more lines would only go off-script, so offer a restart.
    html('<div class="note">Scene complete. Open the next day in your plan, or run this '
         'scene again to practise it.</div>')
    if st.button("Practise this scene again", key="restart-scene"):
        del st.session_state["roleplay"]
        st.rerun()
else:
    # Chat mode offers several replies (marked when they are checked phrasebook
    # sentences); the full system offers one.
    options = last.get("suggestions") or ([last["suggestion"]] if last.get("suggestion") else [])
    if options:
        html('<div class="quiet" style="margin-top:.6rem">You could say:</div>')
        for i, sg in enumerate(options):
            badge = "  ✓ phrasebook" if sg.get("phrasebook") else ""
            if st.button(f"{sg['text']}  ·  {sg['meaning']}{badge}",
                         key=f"say-{len(st.session_state.roleplay)}-{i}"):
                send(sg["text"])
    if text := st.chat_input(f"Answer in {LANGUAGES[language]}…"):
        if wrong_language(text, language):
            # Don't pass English to the partner: it would answer nonsense in character.
            script = " (Cyrillic letters)" if language == "macedonian" else ""
            st.warning(f"Please write it in {LANGUAGES[language]}{script}, "
                       "or tap the suggested line above.")
        else:
            send(text)
