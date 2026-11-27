"""
app.py: Streamlit UI for the travel language tutor.

Run:   .venv/bin/streamlit run app.py

Calendar review version: questionnaire -> summary -> confirmation -> dated sessions.

Original app flow (practice screens will be reconnected in a later update):
  1. Preferences - guided language, timeline, availability and topic questionnaire.
  2. Plan     - confirmed language/timeline/goals are used to build a
                day-by-day plan is built from the curated notes.
  3. Lesson   - phrase cards for the chosen day, with a short quiz.
  4. Practice - a roleplay starts automatically, with a translation,
                vocabulary and a suggested sentence (tap it to say it).

The sidebar "Mode" is for the evaluation: "Full system" is the grounded,
checked pipeline; "Prompt only" and "Baseline" are the comparison conditions.
"""

from practice import spanish_typing_activity, topic_sentences
from spanish_practice import render_spanish_practice
from sentence_practice import render_sentence_practice
from roleplay_practice import render_roleplay
from html import escape
import json
from datetime import date, datetime, timedelta

import streamlit as st

from llm import LLMError
from chat_preferences import READING, detect_language, new_chat, plan_inputs, read_message, reply_parts
from planner import GOAL_HINTS, STAGE_WEIGHTS, build_schedule, plan_varieties, STAGE_NAMES
import plan_store as store
from pronounce import say, transliterate
from prompts import LANGUAGES, ROLEPLAY_PARTNER, SCENARIOS, TutorSettings
from retriever import DEFAULT_VARIETY, LATIN_AMERICA, VARIETIES, Retriever
from tutor import make_lesson, roleplay_turn, start_session, tutor_reply, wrong_language

MODE_LABELS = {
    "rag": "Full system (notes + checks)",
    "prompt": "Prompt only",
    "baseline": "Baseline (raw model)",
    "rag+lora": "Full system + LoRA model",
    "prompt+lora": "Prompt only + LoRA model",
}
GOAL_LABELS = {g: g.capitalize() for g in GOAL_HINTS}
NATIVE_NAME = {"spanish": "Español", "macedonian": "Македонски"}
STEPS = ["Preferences", "Summary", "Calendar"]

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
.bubble .pron { font-family: var(--mono); font-size: .78rem; color: var(--muted); margin-top: .1rem; }
.chip .pron { font-family: var(--mono); font-size: .74rem; color: var(--muted); }
.vocab { margin-top: .5rem; font-size: .82rem; }
.vocab b { font-family: var(--serif); font-weight: 600; }
.chip { display: inline-block; background: var(--accent-soft); border-radius: 3px; padding: 1px 7px;
        margin: 2px 4px 2px 0; }
.tipline { font-size: .82rem; color: var(--muted); margin-top: .45rem; font-style: italic; }

/* Calendar progress (update 4) */
.next-label { font-size: .72rem; letter-spacing: .1em; text-transform: uppercase; color: var(--accent);
              font-weight: 600; }
.next-title { font-family: var(--serif); font-size: 1.1rem; }
[class*="st-key-next_lesson"] { border: 2px solid var(--accent) !important; background: var(--card); }
.progress-line { display: flex; justify-content: space-between; font-size: .85rem; margin-top: .8rem; }
/* The practice bar stays one row on narrow screens (Streamlit stacks columns below 640px). */
[class*="st-key-practice_nav"] [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; gap: .5rem; }
[class*="st-key-practice_nav"] [data-testid="stColumn"] { min-width: max-content !important;
                                                          width: auto !important; flex: 0 0 auto !important; }
[class*="st-key-practice_nav"] [data-testid="stColumn"]:nth-child(2) { min-width: 0 !important;
                                                                        flex: 1 1 0 !important; }
[class*="st-key-practice_nav"] .stButton > button { padding: .3rem .6rem; white-space: nowrap;
                                                    width: max-content; }
.nav-chips { display: flex; gap: 6px; flex-wrap: wrap; margin: .4rem 0 1rem; }
.nav-chips span { font-size: .72rem; padding: 2px 9px; border-radius: 10px; border: 1px solid var(--line);
                  color: var(--muted); }
.nav-chips .dn { background: #E4EDDA; border-color: #E4EDDA; color: #27500A; }
.nav-chips .cu { background: var(--ink); border-color: var(--ink); color: var(--paper); }

/* Your plans (update 5) */
.plan-name { font-weight: 600; font-size: 1rem; }
.plan-tag { font-size: .66rem; letter-spacing: .06em; text-transform: uppercase; border-radius: 3px;
            padding: 1px 6px; margin-left: 6px; background: var(--line); color: var(--muted); font-weight: 500; }
.plan-tag.done { background: var(--accent-soft); color: #7A3219; }

/* Preferences chat */
.chk { display: grid; grid-template-columns: 10.5rem 1fr; gap: .1rem .8rem; font-size: .88rem; margin-top: .35rem; }
.chk .k { color: var(--muted); }
.chk .miss { color: var(--accent); font-weight: 600; }
.chk .sug { color: var(--muted); font-style: italic; }
.bubble .problem { color: var(--accent); font-size: .88rem; margin-top: .4rem; }
.bubble .ask { font-family: var(--serif); font-size: 1.05rem; margin-top: .5rem; }

/* Widgets */
.stButton > button { border-radius: 4px; border: 1px solid var(--ink); background: transparent;
                     color: var(--ink); font-family: var(--sans); font-size: .88rem; padding: .35rem .9rem; }
.stButton > button:hover { border-color: var(--accent); color: var(--accent); }
.stButton > button[kind="primary"] { background: var(--ink); color: var(--paper); }
.stButton > button[kind="primary"]:hover { background: var(--accent); border-color: var(--accent); color: var(--paper); }
.stButton > button p { color: inherit !important; }
/* Streamlit's "Press Enter to submit/apply" hint sits on top of the typed text; hide it everywhere. */
[data-testid="InputInstructions"] { display: none !important; }
.stButton > button:disabled, .stButton > button[kind="primary"]:disabled {
  background: transparent; color: var(--faint); border-color: var(--line); }
[data-testid="stFormSubmitButton"] > button { border-radius: 4px; border: 1px solid var(--ink);
  background: transparent; color: var(--ink); }
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
    if "practice_progress" in keys:
        st.session_state.pop("studied_banks", None)
        for widget_key in list(st.session_state):
            if widget_key.startswith(("answer_", "es_", "s2_", "s3_")):
                del st.session_state[widget_key]
    for k in keys:
        st.session_state.pop(k, None)


def reset_topic_varieties():
    """Forget per-topic variety switches (they belong to one built schedule)."""
    for k in [k for k in st.session_state if k.startswith("tv_")]:
        del st.session_state[k]


def calendar_blocks(session, schedule):
    """Combine matching stage/topic/variety blocks in first-appearance order.

    This changes the calendar display only. Separate stages never merge and
    the original scheduling data is left intact.
    """
    grouped = {}
    for block in session["blocks"]:
        topic = block["topic"]
        used = schedule.get("topic_variety", {}).get(topic, {}).get("used")
        key = (block["stage"], topic, used)
        if key not in grouped:
            grouped[key] = {**block, "variety": used}
        else:
            grouped[key]["minutes"] += block["minutes"]
    return list(grouped.values())


CALENDAR_TOPICS = {
    "restaurant": "Restaurants", "directions": "Directions", "shopping": "Shopping",
    "hotel": "Hotels", "transport": "Transportation", "weather": "Weather",
    "sightseeing": "Sightseeing", "emergency": "Getting help",
}


def variety_name(code):
    return VARIETIES.get(code, code)


def nice_date(d):
    return d.strftime("%b %d, %Y")


def study_window(prefs, schedule):
    """'Dec 01, 2026 to Jan 15, 2027': the first and last study day, both included."""
    start = date.fromisoformat(prefs.get("start_date", schedule["start_date"]))
    end = (date.fromisoformat(prefs["end_date"]) if "end_date" in prefs else
           date.fromisoformat(schedule["departure_date"]) - timedelta(days=1))
    return f"{nice_date(start)} to {nice_date(end)}"


def html(s):
    st.markdown(s, unsafe_allow_html=True)


def label(text):
    html(f'<div class="section-label">{escape(text)}</div>')


def step_bar(current, with_plans=False):
    """current: 0-2 for Preferences / Summary / Calendar, -1 for the Your plans page."""
    parts = [f'<span class="{"on" if current == -1 else ""}">Your plans</span>'] if with_plans else []
    for i, name in enumerate(STEPS):
        cls = "on" if i == current else ("done" if i < current else "")
        parts.append(f'<span class="{cls}">{i + 1} &nbsp;{name}</span>')
    html('<div class="steps">' + "".join(parts) + "</div>")


def fail(e):
    st.error(f"The local model didn't respond: {e}")
    st.stop()


def phrase_card(i, p):
    pron = f'<div class="p">{escape(p["pron"])}</div>' if p.get("pron") else ""
    if p.get("say"):  # stress guide: capitals mark the stressed syllable
        pron += (f'<div class="p" title="Capitals mark the stressed syllable">'
                 f'say: {escape(p["say"])}</div>')
    who = f'<div class="who">the {escape(p["speaker"])} says this</div>' if p.get("speaker") else ""
    html(f'<div class="phrase"><span class="n">{i:02d}</span><div class="t">{escape(p["text"])}</div>'
         f'{pron}<div class="m">{escape(p["meaning"])}</div>{who}</div>')


def pron_lines(text):
    """Transliteration + "say:" guide under a Macedonian line ("" for other languages)."""
    guide = say(text)
    if not guide:
        return ""
    return (f'<div class="pron">{escape(transliterate(text))}</div>'
            f'<div class="pron" title="Capitals mark the stressed syllable">say: {escape(guide)}</div>')


def partner_bubble(t, partner):
    # The translation is model-written (not from the notes), so it is marked as approximate.
    tr = (f'<div class="tr" title="Approximate translation by the model; the vocabulary '
          f'below is from the checked notes">≈ {escape(t["reply_english"])}</div>'
          if t.get("reply_english") else "")
    vocab = ""
    if t.get("vocab"):
        vocab = '<div class="vocab"><span class="quiet">In this line: </span>' + "".join(
            f'<span class="chip"><b>{escape(v["text"])}</b>'
            + (f' <span class="pron">{escape(v.get("say") or say(v["text"]))}</span>'
               if v.get("say") or say(v["text"]) else "")
            + f' &nbsp;{escape(v["meaning"])}</span>'
            for v in t["vocab"]) + "</div>"
    tip = f'<div class="tipline">{escape(t["tip"])}</div>' if t.get("tip") else ""
    html(f'<div class="bubble partner"><div class="who-label">{escape(partner)}</div>'
         f'<div class="line">{escape(t["reply"])}</div>{pron_lines(t["reply"])}{tr}{vocab}{tip}</div>')


def my_bubble(text):
    html(f'<div class="bubble me"><div class="who-label">You</div>{escape(text)}{pron_lines(text)}</div>')


# --- Preferences: shared plan builder and the chat option -------------------------
PREF_MODES = ["Choose with buttons", "Describe it in a chat"]


# Lesson types = the planner's stages: (name, short description for the choice cards).
LESSON_TYPES = {1: ("Vocabulary", "Learn and recall words and short expressions."),
                2: ("Sentences", "Build full sentences and get corrections."),
                3: ("Role play", "Use your words in a short travel conversation.")}


def kept(key, default):
    """The value a preference widget had when the learner last left the preferences page.
    Streamlit drops widget values once a widget is off screen (e.g. on the summary page),
    so the values are copied to "pref_kept" and used as the widget defaults on return."""
    return st.session_state.setdefault("pref_kept", {}).get(key, default)


def keep(key, value):
    st.session_state.setdefault("pref_kept", {})[key] = value
    return value


def forget_preferences():
    """Clear every preference choice and the chat: the plan was approved (it is locked
    now), or the learner chose Start over."""
    reset("pref_chat", "pref_kept")
    for key in [k for k in st.session_state if k.startswith(("q_", "chat_en_", "chat_type_"))]:
        del st.session_state[key]


def lesson_type_cards(key_prefix, chosen):
    """The three lesson type cards; returns the stages that are ticked."""
    label("Lesson types")
    st.caption("Choose the types of lessons in your plan. You need at least one.")
    stages = []
    for stage, column in zip(LESSON_TYPES, st.columns(3)):
        name, description = LESSON_TYPES[stage]
        with column.container(border=True):
            if st.checkbox(name, value=stage in chosen, key=f"{key_prefix}{stage}"):
                stages.append(stage)
            st.caption(description)
    if not stages:
        st.warning("Choose at least one lesson type.")
    else:
        lesson_types_note(stages)
    return stages


def lesson_types_note(stages):
    """One line under the lesson type choice (both preferences screens)."""
    if 1 not in stages:
        text = ("Sentences and role play use a topic's words. Without vocabulary lessons, you "
                "see each topic's word list before its first activity.")
    elif len(stages) < len(LESSON_TYPES):
        text = "Your practice time is shared between the lesson types you chose."
    else:
        return
    html(f'<div class="note">{escape(text)}</div>')


def topic_vocabulary(topic, variety):
    """The phrases studied for a topic and variety (Stage 1 banks or a word list), no repeats."""
    vocabulary, seen = [], set()
    for bank in st.session_state.get("studied_banks", {}).values():
        if bank['topic'] == topic and bank['variety'] == variety:
            for phrase in bank['phrases']:
                if phrase['text'] not in seen:
                    vocabulary.append(phrase)
                    seen.add(phrase['text'])
    return vocabulary


def word_list_first(topic, variety, language, share):
    """A plan without vocabulary lessons: show the topic's word list (read only) before
    its first sentence or role-play activity, then use those words. Stops the page."""
    activity = spanish_typing_activity(get_retriever(), topic, variety, share=share,
                                       language=language)
    label("Words for this topic")
    st.write("Your plan has no vocabulary lessons, so read these words first. "
             "This activity uses them.")
    if not activity['phrases']:
        st.warning("There are no suitable phrases for this topic yet.")
        st.stop()
    for i, phrase in enumerate(activity['phrases'], 1):
        phrase_card(i, phrase)
    if st.button("I've read the words — start", type="primary", key=f"wordlist_{topic}_{variety}"):
        banks = dict(st.session_state.get('studied_banks', {}))
        banks[f"wordlist_{topic}_{variety}"] = {'topic': topic, 'variety': variety,
                                                'phrases': [dict(p) for p in activity['phrases']]}
        st.session_state.studied_banks = banks
        st.rerun()
    st.stop()


def build_plan_and_continue(language, variety, goals, days, weekdays, minutes, start_date, end_date,
                            stages=(1, 2, 3)):
    """Build the schedule from either preferences screen and open the summary page.
    A planner refusal (e.g. no practice weekday in the window) is shown as a warning."""
    try:
        topic_variety = plan_varieties(get_retriever(), language, variety, goals)
        schedule = build_schedule(language, goals, int(days), weekdays, int(minutes), start_date,
                                  variety=variety, topic_variety=topic_variety, stages=stages)
    except ValueError as error:
        st.warning(str(error))
        return
    st.session_state.schedule_draft = schedule
    st.session_state.preferences = {
        "language": language, "variety": variety, "days": int(days),
        "start_date": start_date.isoformat(), "end_date": end_date.isoformat(),
        "weekdays": weekdays, "minutes": int(minutes), "goals": goals,
        "stages": list(schedule["stages"]), "starting_stage": schedule["stages"][0],
    }
    st.session_state.schedule_confirmed = False
    reset("active_practice", "practice_progress", "practice_results")
    reset_topic_varieties()
    st.rerun()


def tutor_bubble(msg, lang=None):
    """One tutor message in the preferences chat, in `lang` (default: the reply's language)."""
    parts = reply_parts(msg, lang)
    if "text" in parts:
        body = escape(parts["text"])
    else:
        rows = "".join(
            f'<span class="k">{escape(label)}</span>'
            + (f'<span class="miss">{escape(parts["missing"])}</span>' if value is None else
               f'<span>{escape(value)}' + (f' <span class="sug">({escape(parts["suggested"])})</span>'
                                           if suggested else "") + '</span>')
            for label, value, suggested in parts["rows"])
        body = f'<div>{escape(parts["have"])}</div><div class="chk">{rows}</div>'
        body += "".join(f'<div class="problem">{escape(p)}</div>' for p in parts["problems"])
        body += f'<div class="ask">{escape(parts["question"] or parts["closing"])}</div>'
        if parts["model_note"]:
            body += f'<div class="tipline">{escape(parts["model_note"])}</div>'
    html(f'<div class="bubble partner"><div class="who-label">Tutor</div>{body}</div>')


def chat_preferences_screen():
    chat = st.session_state.setdefault("pref_chat", new_chat())
    for i, msg in enumerate(chat["messages"]):
        if msg["role"] == "you":
            html(f'<div class="bubble me"><div class="who-label">You</div>{escape(msg["text"])}</div>')
        elif msg.get("lang", "en") == "en":
            tutor_bubble(msg)
        else:
            # A reply in Spanish or Macedonian can be shown in English instead.
            key = f"chat_en_{i}"
            english = st.session_state.get(key, kept(key, False))
            tutor_bubble(msg, "en" if english else None)
            keep(key, st.toggle("Show in English", value=kept(key, False), key=key))
    # The learner's message is already on screen; read it now, below it.
    if chat.get("pending"):
        before = chat["found"].get("stages")
        with st.spinner(READING[detect_language(chat["pending"], chat.get("lang", "en"))]):
            read_message(chat, chat.pop("pending"), record=False)
        if chat["found"].get("stages") != before:
            # The message changed the lesson types: draw fresh cards with the new choice.
            chat["cards"] = chat.get("cards", 0) + 1
        st.rerun()
    with st.form("pref_chat_form", clear_on_submit=True, border=False):
        box, button = st.columns([6, 1], vertical_alignment="bottom")
        text = box.text_input("Your message", key="q_chat_text", label_visibility="collapsed")
        sent = button.form_submit_button("Send", use_container_width=True)
    if sent and text.strip():
        chat["messages"].append({"role": "you", "text": text.strip()})
        chat["pending"] = text.strip()
        st.rerun()
    # Lesson types: say them in the chat, or tick the cards (the same choice either way).
    found = chat["found"]
    chosen = found["stages"] if found.get("stages") is not None else [1, 2, 3]
    ticked = lesson_type_cards(f"chat_type_{chat.get('cards', 0)}_", chosen)
    if ticked != chosen:
        found["stages"] = ticked
    inputs = plan_inputs(chat)
    build_col, restart_col = st.columns([1, 1])
    if build_col.button("Build my plan", type="primary", disabled=inputs is None, key="chat_build"):
        build_plan_and_continue(**inputs)
    if restart_col.button("Start over", key="chat_restart",
                          help="Clear the chat and your preferences, and start again."):
        st.session_state.pop("pref_chat", None)
        for key in [k for k in st.session_state if k.startswith(("chat_en_", "chat_type_"))]:
            del st.session_state[key]
        kept_values = st.session_state.get("pref_kept", {})
        for key in [k for k in kept_values if k.startswith("chat_en_")]:
            del kept_values[key]
        st.rerun()
    if inputs is None:
        st.caption("Build my plan is available when every necessary item is complete. "
                   "You can change or restart your preferences at any time.")


# --- Calendar progress and navigation (update 4) -----------------------------------
START_LABELS = {1: "Start vocabulary practice", 2: "Start sentence practice", 3: "Start roleplay"}


def plan_activities(schedule):
    """Every activity of the plan in calendar order (one per merged calendar block)."""
    activities = []
    for s_index, session in enumerate(schedule["sessions"]):
        for b_index, block in enumerate(calendar_blocks(session, schedule)):
            activities.append({
                "id": f"{session['date']}_{block['stage']}_{block['topic']}_{block['variety']}",
                "session": s_index, "block": b_index, "date": session["date"],
                "stage": block["stage"], "topic": block["topic"], "variety": block["variety"],
                "minutes": block["minutes"]})
    return activities


def done_ids():
    return set(st.session_state.get("practice_progress", {}))


def next_activity(activities, done):
    """The first activity in calendar order that is not complete (None when all are)."""
    return next((a for a in activities if a["id"] not in done), None)


def activity_label(a, language):
    where = f" · {variety_name(a['variety'])}" if language == "spanish" and a["variety"] else ""
    return f"{LESSON_TYPES[a['stage']][0]} · {CALENDAR_TOPICS[a['topic']]}{where} · {a['minutes']} min"


def open_activity(a):
    st.session_state.active_practice = {"id": a["id"], "session": a["session"], "block": a["block"]}
    st.rerun()


def record_progress(activity_id, block, variety, date_text, result):
    """Mark an activity complete as soon as it has a result, so leaving it with Next,
    Previous or Calendar (instead of a Finish button) still counts."""
    progress = dict(st.session_state.get("practice_progress", {}))
    progress[activity_id] = {"stage": block["stage"], "topic": block["topic"], "variety": variety,
                             "date": date_text, "result": result}
    st.session_state.practice_progress = progress
    autosave_plan()


# --- Saved plans (update 5) ------------------------------------------------------------
def plan_name(prefs):
    name = LANGUAGES[prefs["language"]]
    if prefs["language"] == "spanish":
        name += f" ({variety_name(prefs.get('variety', 'ES'))})"
    return name


def plan_progress(schedule, progress):
    """{"done": complete sessions, "total": sessions} for the Your plans page."""
    done = set(progress or {})
    sessions = {}
    for a in plan_activities(schedule):
        sessions.setdefault(a["session"], []).append(a["id"] in done)
    return {"done": sum(all(v) for v in sessions.values()), "total": len(schedule["sessions"])}


def autosave_plan():
    """Save the open (approved) plan to the plans file when anything in it changed."""
    if not st.session_state.get("schedule_confirmed") or "plan_id" not in st.session_state:
        return
    state = store.snapshot(st.session_state)
    fingerprint = json.dumps(state, sort_keys=True)
    if st.session_state.get("_saved_fingerprint") == fingerprint:
        return
    store.save(st.session_state.plan_id, plan_name(state["preferences"]), state,
               plan_progress(state["schedule_draft"], state.get("practice_progress")))
    st.session_state._saved_fingerprint = fingerprint


def leave_plan(view):
    """Shelve the open plan (it is saved first) and go to Your plans or a new plan."""
    autosave_plan()
    store.clear_plan_state(st.session_state)
    reset("schedule_confirmed", "active_practice", "plan_id", "_saved_fingerprint")
    reset_topic_varieties()
    st.session_state.view = view
    st.rerun()


def open_saved_plan(record):
    store.restore(st.session_state, record)
    st.session_state.plan_id = record["id"]
    st.session_state.schedule_confirmed = True
    reset("active_practice", "view", "_saved_fingerprint")
    reset_topic_varieties()
    store.touch(record["id"])
    st.rerun()


def when(stamp):
    day = datetime.fromisoformat(stamp).date()
    return ("today" if day == date.today() else "yesterday" if day == date.today() - timedelta(days=1)
            else day.strftime("%b %d"))


def your_plans_page(saved):
    title_col, new_col = st.columns([3, 1.4], vertical_alignment="bottom")
    title_col.markdown("## Your plans")
    if new_col.button("Create a new plan", type="primary", key="plans_new", use_container_width=True):
        st.session_state.view = "new"
        st.rerun()
    if not saved:
        st.info("You have no saved plans. Create a new plan to start.")
        return
    st.caption("Your progress is saved. Open a plan to continue where you stopped.")
    for record in sorted(saved.values(), key=lambda r: r.get("last_opened", ""), reverse=True):
        state = record.get("state", {})
        prefs, schedule = state.get("preferences"), state.get("schedule_draft")
        if not prefs or not schedule:
            continue
        progress = record.get("progress") or plan_progress(schedule, state.get("practice_progress"))
        complete = progress["total"] and progress["done"] == progress["total"]
        days = ", ".join(d[:3] for d in prefs["weekdays"])
        topics = ", ".join(CALENDAR_TOPICS[g] for g in prefs["goals"])
        with st.container(border=True, key=f"plan_card_{record['id']}"):
            info, actions = st.columns([4, 1.5], vertical_alignment="center")
            tag = ('<span class="plan-tag done">Complete</span>' if complete
                   else '<span class="plan-tag">Shelved</span>')
            info.markdown(
                f'<div class="plan-name">{escape(record["name"])} {tag}</div>'
                f'<div class="quiet">{escape(study_window(prefs, schedule))} · {escape(days)} · '
                f'{prefs["minutes"]} min · {escape(topics)}</div>', unsafe_allow_html=True)
            info.progress(progress["done"] / max(progress["total"], 1))
            info.caption(f"{progress['done']} of {progress['total']} sessions · "
                         f"last opened {when(record['last_opened'])}")
            if actions.button("Open", key=f"open_{record['id']}", use_container_width=True):
                open_saved_plan(record)
            with actions.popover("⋯ More", use_container_width=True):
                name = st.text_input("Plan name", value=record["name"], key=f"name_{record['id']}")
                if st.button("Save name", key=f"rename_{record['id']}"):
                    store.rename(record["id"], name)
                    st.rerun()
                st.divider()
                sure = st.checkbox("Delete this plan and its progress. This can't be undone.",
                                   key=f"sure_{record['id']}")
                if st.button("Delete plan", key=f"delete_{record['id']}", disabled=not sure):
                    store.delete(record["id"])
                    st.rerun()


def practice_nav(activities, activity_id, language):
    """The bar at the top of every practice: back to the calendar, where you are,
    and the previous / next activity in the plan (across sessions)."""
    done = done_ids()
    i = next(n for n, a in enumerate(activities) if a["id"] == activity_id)
    here = activities[i]
    in_session = [a for a in activities if a["session"] == here["session"]]
    with st.container(border=True, key="practice_nav"):
        back, where, prev_col, next_col = st.columns([1.3, 2.4, 1.2, 1.1], vertical_alignment="center")
        if back.button("← Calendar", key="nav_calendar", help="Back to your calendar"):
            reset("active_practice")
            st.rerun()
        where.markdown(f"**Session {here['session'] + 1}** · activity {in_session.index(here) + 1} "
                       f"of {len(in_session)}")
        before = activities[i - 1] if i > 0 else None
        after = activities[i + 1] if i + 1 < len(activities) else None
        if prev_col.button("‹ Previous", key="nav_prev", disabled=before is None,
                           help=activity_label(before, language) if before else None):
            open_activity(before)
        if next_col.button("Next ›", key="nav_next", disabled=after is None,
                           help=activity_label(after, language) if after else None):
            open_activity(after)
    chips = "".join(
        f'<span class="{"cu" if a["id"] == activity_id else "dn" if a["id"] in done else ""}">'
        f'{"✓ " if a["id"] in done and a["id"] != activity_id else ""}'
        f'{escape(LESSON_TYPES[a["stage"]][0])} · {escape(CALENDAR_TOPICS[a["topic"]])}</span>'
        for a in in_session)
    if here["session"] + 1 < len({a["session"] for a in activities}):
        chips += f'<span>Session {here["session"] + 2} ›</span>'
    html(f'<div class="nav-chips">{chips}</div>')


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

with st.sidebar:
    st.markdown("### Evaluation")
    mode_label = st.selectbox("Mode", list(MODE_LABELS.values()),
                              help="Full system is the one to demo. The others are the "
                                   "comparison conditions from the proposal.")
    mode_key = next(k for k, v in MODE_LABELS.items() if v == mode_label)
    mode, use_lora = mode_key.split("+")[0], mode_key.endswith("+lora")
    if st.button("Start over"):
        reset("trip", "parsed", "plan", "session", "lesson", "roleplay", "free_chat", "draft", "preferences", "schedule_draft", "schedule_confirmed", "plan_id", "view", "_saved_fingerprint")
        forget_preferences()
        reset_topic_varieties()
        reset("active_practice", "practice_progress", "practice_results")
        st.rerun()

saved_plans = store.load_all()
autosave_plan()     # changes made by the last click (callbacks, reruns) reach the file
on_plans_page = (st.session_state.get("view") == "plans" or (
    saved_plans and st.session_state.get("view") != "new"
    and not st.session_state.get("schedule_confirmed") and "schedule_draft" not in st.session_state))
step_bar(-1 if on_plans_page else 2 if st.session_state.get("schedule_confirmed") else
         1 if "schedule_draft" in st.session_state else 0, with_plans=bool(saved_plans) or on_plans_page)
if on_plans_page:
    your_plans_page(saved_plans)
    st.stop()

if "schedule_draft" not in st.session_state:
    # --- 1. Guided questionnaire ---------------------------------------------------
    if saved_plans and st.button("← Your plans", key="back_to_plans"):
        st.session_state.view = "plans"
        st.rerun()
    st.markdown("## Your learning preferences")
    st.caption("Choose with buttons, or describe your plan in your own words.")
    # Every choice below is kept (kept/keep) until the plan is approved, so "Change my
    # preferences" on the summary page comes back to the same screen and values.
    pref_mode = keep("q_mode", st.segmented_control(
        "How do you want to give your preferences?", PREF_MODES,
        default=kept("q_mode", PREF_MODES[0]), required=True, key="q_mode",
        label_visibility="collapsed"))
    if pref_mode == PREF_MODES[1]:
        chat_preferences_screen()
        st.stop()
    language = keep("q_language", st.selectbox(
        "Which language would you like to learn?", list(LANGUAGES),
        index=list(LANGUAGES).index(kept("q_language", "spanish")),
        format_func=lambda value: LANGUAGES[value], key="q_language"))
    variety = DEFAULT_VARIETY[language]
    if language == "spanish":
        # Spain is preselected (the default when no country is chosen). Countries
        # with no notes of their own are marked; their topics use Spain Spanish.
        have = {code for counts in get_retriever().coverage("spanish").values() for code in counts}
        options = ["ES", "419"] + sorted(LATIN_AMERICA, key=variety_name)
        variety = keep("q_variety", st.selectbox(
            "Which Spanish would you like to learn?", options,
            index=options.index(kept("q_variety", "ES")) if kept("q_variety", "ES") in options else 0,
            key="q_variety",
            format_func=lambda c: "Spain (default)" if c == "ES" else
            variety_name(c) + ("" if c in have else " (no country notes yet, uses Spain)"),
            help="Pick the country you are traveling to. Topics without notes for that country "
                 "use Spain Spanish; you can change each topic on the next screen."))
    # One date-range calendar sets the study window (first and last study day,
    # both included). There is no upper limit on its length.
    today = date.today()
    saved_window = kept("q_window", ())
    if not saved_window or min(saved_window) < today:      # nothing kept, or now in the past
        saved_window = (today, today + timedelta(days=13))
    window = keep("q_window", st.date_input(
        "Choose the first and last day of your study window",
        value=saved_window, min_value=today,
        key="q_window", format="MM/DD/YYYY",
        help="Click the first day, then the last day. Your sessions fall on "
             "your practice weekdays inside this window."))
    if len(window) == 2:
        start_date, end_date = window
        days = (end_date - start_date).days + 1
        st.caption(f"{days} days: {nice_date(start_date)} to {nice_date(end_date)}.")
    else:
        start_date = end_date = days = None
        st.info("Choose the last day of your study window.")
    weekdays = keep("q_weekdays", st.multiselect(
        "What days of the week can you practice?",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        default=kept("q_weekdays", ["Monday", "Wednesday", "Friday"]), key="q_weekdays"))
    minutes = keep("q_minutes", st.number_input(
        "How long can you typically practice for, in minutes?",
        min_value=1, max_value=240, value=kept("q_minutes", 15), step=1, key="q_minutes"))
    selected = keep("q_topics", st.multiselect(
        "Which topics would you like to learn?", list(GOAL_LABELS),
        default=kept("q_topics", ["restaurant", "directions"]),
        format_func=lambda value: GOAL_LABELS[value], key="q_topics"))
    st.markdown("**Rank your selected topics**")
    st.caption("Choose a different topic for each position. Rank 1 is most important.")
    goals = []
    # Separate ranking controls allow priorities to be changed without relying on
    # multiselect click order. A selection change starts a fresh set of controls.
    signature = "-".join(selected)
    ranked = kept("q_ranks", [])
    ranked = ranked if sorted(ranked) == sorted(selected) else list(selected)
    for i in range(len(selected)):
        goals.append(st.selectbox(f"Priority {i + 1}", selected, index=selected.index(ranked[i]),
                                  format_func=lambda value: GOAL_LABELS[value],
                                  key=f"q_rank_{signature}_{i}"))
    keep("q_ranks", goals)
    valid_ranks = len(set(goals)) == len(goals)
    if not valid_ranks:
        st.warning("Each topic must appear exactly once in your rankings.")
    if goals and valid_ranks:
        total = len(goals) * (len(goals) + 1) / 2
        st.caption("Target share of practice within each stage:")
        for i, goal in enumerate(goals):
            st.write(f"{i + 1}. {GOAL_LABELS[goal]}: {100 * (len(goals) - i) / total:.1f}%")
    stages = keep("q_types", lesson_type_cards("q_type_", kept("q_types", [1, 2, 3])))

    if st.button("Build my plan", type="primary"):
        if days is None or not weekdays or not goals or not valid_ranks or not stages:
            st.warning("Choose a study window, practice days, topics with unique ranks, "
                       "and at least one lesson type.")
        else:
            build_plan_and_continue(language, variety, goals, int(days), weekdays, int(minutes),
                                    start_date, end_date, stages)
    st.stop()

# --- 2. Explain the plan before showing any dated sessions --------------------
schedule = st.session_state.schedule_draft
prefs = st.session_state.preferences
if not st.session_state.get("schedule_confirmed"):
    st.markdown("## Your suggested learning plan")
    st.write(f"You will practice {LANGUAGES[prefs['language']]} from "
             f"{study_window(prefs, schedule)}.")
    st.write(f"Practice days: {', '.join(prefs['weekdays'])}. "
             f"Each session: {prefs['minutes']} minutes.")
    st.write(f"Your schedule has {len(schedule['sessions'])} sessions, "
             f"with {schedule['total_minutes']} minutes of practice in total.")
    stages = schedule.get("stages", [1, 2, 3])
    st.markdown("**Your lesson types**")
    descriptions = {
        1: "Learn and recall useful words and short expressions from the reference materials.",
        2: "Expand phrases into sentences, revisit vocabulary, and receive focused corrections.",
        3: "Use what you have learned in short travel conversations, with feedback afterward.",
    }
    weight_total = sum(STAGE_WEIGHTS[s] for s in stages)
    for n, stage in enumerate(stages, 1):
        target = round(100 * STAGE_WEIGHTS[stage] / weight_total)
        st.markdown(f"**{n}. {LESSON_TYPES[stage][0]} — about {target}% "
                    f"({schedule['stage_minutes'][stage - 1]} minutes)**  \n{descriptions[stage]}")
    for stage in LESSON_TYPES:
        if stage not in stages:
            html(f'<div class="quiet" style="text-decoration: line-through; margin: -.3rem 0 1rem">'
                 f'{escape(LESSON_TYPES[stage][0])} · not in this plan</div>')
    st.markdown("**Your topic priorities**")
    for i, goal in enumerate(prefs['goals'], 1):
        st.write(f"{i}. {GOAL_LABELS[goal]} — target {schedule['topic_weights'][goal]:.1%}")
    st.write("All selected topics appear in every lesson type. "
             + ("Everyone starts with vocabulary. " if 1 in stages else
                "Without vocabulary lessons, each topic's word list comes before its first activity. ")
             + "Practice minutes are rounded, so topic shares are approximate.")
    if prefs["language"] == "spanish":
        st.markdown(f"**Your Spanish, by topic** (you chose {variety_name(prefs['variety'])})")
        for goal in prefs["goals"]:
            tv = schedule["topic_variety"][goal]
            choice = st.selectbox(
                GOAL_LABELS[goal], tv["alternatives"], index=tv["alternatives"].index(tv["used"]),
                format_func=variety_name, key=f"tv_{goal}",
                help="Only varieties with notes for this topic are listed.")
            if choice != tv["used"]:
                tv["used"], tv["switched"] = choice, True
            if tv["fallback"] and not tv.get("switched") and tv["used"] == DEFAULT_VARIETY["spanish"]:
                st.caption(f"No {variety_name(tv['requested'])} notes for this topic yet, "
                           f"so Spain Spanish is used. You can pick another variety above.")
    st.caption("A short schedule may offer only a brief introduction to lower-priority topics.")
    st.write("This is your starting plan. You can choose to move forward or return to a stage "
             "when the practice activities are available.")
    if st.button("Yes, this sounds good — show my calendar", type="primary"):
        # The plan is approved and locked: the preference choices and chat are cleared.
        st.session_state.schedule_confirmed = True
        st.session_state.plan_id = store.new_plan_id()
        reset("view")
        forget_preferences()
        autosave_plan()
        st.rerun()
    if st.button("Change my preferences",
                 help="Go back to your preferences; your choices and chat are kept."):
        reset("schedule_draft", "schedule_confirmed", "active_practice", "practice_progress", "practice_results")
        reset_topic_varieties()
        st.rerun()
    st.stop()

# --- 3. Calendar -------------------------------------------------------------
if "active_practice" not in st.session_state:
    title_col, shelf_col = st.columns([3, 1.4], vertical_alignment="bottom")
    title_col.markdown("## Your learning calendar")
    if shelf_col.button("Shelf this plan", key="shelf_plan", use_container_width=True,
                        help="Save this plan with its progress and go to Your plans. "
                             "Open it again any time."):
        leave_plan("plans")
    lang_label = LANGUAGES[prefs['language']]
    if prefs['language'] == "spanish":
        lang_label += f" ({variety_name(prefs.get('variety', 'ES'))})"
    st.write(f"{lang_label} · {prefs['minutes']} minutes per session · "
             f"Study window: {study_window(prefs, schedule)}")
    plan_stages = schedule.get("stages", [1, 2, 3])
    later = [LESSON_TYPES[s][0].lower() for s in plan_stages if s != 1]
    if 1 not in plan_stages:
        st.caption(f"Your plan has {' and '.join(later)} lessons. Each topic's word list comes "
                   "before its first activity.")
    elif later:
        st.caption(f"Open a vocabulary activity to study and check your recall. "
                   f"{' and '.join(later).capitalize()} use the vocabulary you studied for the "
                   "same topic. Stage switching is coming next.")
    else:
        st.caption("Open a vocabulary activity to study and check your recall.")
    # Progress, the next lesson, and each session's state. Only the session that holds
    # the next lesson is open; complete sessions show a check.
    activities = plan_activities(schedule)
    done = done_ids()
    upcoming = next_activity(activities, done)
    n_sessions = len(schedule['sessions'])
    complete = [all(a["id"] in done for a in activities if a["session"] == i) for i in range(n_sessions)]
    html(f'<div class="progress-line"><span>{sum(complete)} of {n_sessions} sessions complete</span>'
         f'<span class="quiet">{round(100 * sum(complete) / n_sessions)}%</span></div>')
    st.progress(sum(complete) / n_sessions)
    if upcoming:
        day = date.fromisoformat(upcoming["date"])
        with st.container(border=True, key="next_lesson"):
            text_col, button_col = st.columns([4, 1.3], vertical_alignment="center")
            text_col.markdown(
                f'<div class="next-label">Next lesson</div>'
                f'<div class="next-title">Session {upcoming["session"] + 1} · '
                f'{escape(LESSON_TYPES[upcoming["stage"]][0])} · {escape(CALENDAR_TOPICS[upcoming["topic"]])}</div>'
                f'<div class="quiet">{day.strftime("%A, %b %d")} · {upcoming["minutes"]} min</div>',
                unsafe_allow_html=True)
            if button_col.button("Continue", type="primary", key="continue_next",
                                 use_container_width=True):
                open_activity(upcoming)
    else:
        st.success("You completed every session in this plan. You can practice any activity again.")
    for index, session in enumerate(schedule['sessions'], 1):
        day = date.fromisoformat(session['date'])
        mine = [a for a in activities if a["session"] == index - 1]
        finished = sum(a["id"] in done for a in mine)
        if complete[index - 1]:
            state, icon = "Complete", ":material/check_circle:"
        elif upcoming and upcoming["session"] == index - 1:
            state, icon = f"In progress · {finished} of {len(mine)}", ":material/play_circle:"
        elif finished:
            state, icon = f"{finished} of {len(mine)} done", ":material/timelapse:"
        else:
            state, icon = "Upcoming", ":material/radio_button_unchecked:"
        heading = f"Session {index} · {day.strftime('%a, %b %d')} · {session['minutes']} min · {state}"
        with st.expander(heading, icon=icon,
                         expanded=bool(upcoming) and upcoming["session"] == index - 1):
            for a in mine:
                is_done, is_next = a["id"] in done, upcoming is not None and a["id"] == upcoming["id"]
                mark = (":green[:material/check_circle:]" if is_done else
                        ":orange[:material/play_circle:]" if is_next else
                        ":gray[:material/radio_button_unchecked:]")
                text_col, button_col = st.columns([5, 2.2], vertical_alignment="center")
                text_col.markdown(f"{mark} {'**' if is_next else ''}{activity_label(a, prefs['language'])}"
                                  f"{'**' if is_next else ''}")
                if button_col.button("Practice again" if is_done else START_LABELS[a["stage"]],
                                     key=f"practice_{a['id']}", type="primary" if is_next else "secondary",
                                     use_container_width=True):
                    open_activity(a)
            st.caption("Minutes are pacing suggestions; they do not measure mastery.")
    if st.button("Start a new plan",
                 help="This plan is saved in Your plans with its progress. Start a new plan "
                      "from new preferences."):
        forget_preferences()
        leave_plan("new")
    
    
# --- Stage 1: study -> recall -> reference-based feedback ----------------------
if "active_practice" in st.session_state:
    active = st.session_state.active_practice
    session = schedule['sessions'][active['session']]
    block = calendar_blocks(session, schedule)[active['block']]
    activity_id = active['id']
    variety = block['variety'] or DEFAULT_VARIETY[prefs['language']]
    language = prefs['language']
    # Stages 2 and 3 share one structure for both languages; only Spanish has varieties.
    variety_caption = (f"Spanish variety: {variety_name(variety)}. Practice minutes are flexible."
                       if language == 'spanish' else "Practice minutes are flexible.")
    bank_label = "this topic and Spanish variety" if language == 'spanish' else "this topic"
    plan_stages = schedule.get("stages", [1, 2, 3])
    topic_weights = schedule.get('topic_weights') or {}
    topic_share = (topic_weights.get(block['topic'], 0) / max(topic_weights.values())
                   if topic_weights else 1.0)
    practice_nav(plan_activities(schedule), activity_id, language)
    if block['stage'] == 3:
        st.markdown(f"## Stage 3: {CALENDAR_TOPICS[block['topic']]}")
        st.caption(variety_caption)
        vocabulary = topic_vocabulary(block['topic'], variety)
        if not vocabulary and 1 not in plan_stages:
            word_list_first(block['topic'], variety, language, topic_share)
        if not vocabulary:
            st.info(f"Complete a Stage 1 activity for {bank_label} first. "
                    "The roleplay will use your studied vocabulary.")
            st.stop()
        examples = []
        for completed in st.session_state.get("practice_progress", {}).values():
            if (completed['stage'] == 2 and completed['topic'] == block['topic']
                and completed['variety'] == variety):
                examples.extend({'english': row['english'], language: row['reference']}
                                for row in completed.get('result', {}).get('records', []))
        result = render_roleplay(activity_id, vocabulary, variety, variety_name(variety),
                                 block['topic'], get_retriever(), examples[-10:], language=language)
        if result is not None:
            # Save every finished scenario, even when the learner continues.
            record_progress(activity_id, block, variety, session['date'], result)
            if st.button("Finish roleplay and return to calendar", key=f"finish_roleplay_{activity_id}"):
                reset("active_practice")
                st.rerun()
        st.caption("Your progress is saved automatically. Shelf this plan and come back to it any time.")
        st.stop()
    if block['stage'] == 2:
        st.markdown(f"## Stage 2: {CALENDAR_TOPICS[block['topic']]}")
        st.caption(variety_caption)
        # Use snapshots of the vocabulary the learner actually typed three times (or, in a
        # plan without vocabulary lessons, the word list they read), not a presumed list.
        vocabulary = topic_vocabulary(block['topic'], variety)
        if not vocabulary and 1 not in plan_stages:
            word_list_first(block['topic'], variety, language, topic_share)
        if not vocabulary:
            st.info(f"Complete a Stage 1 activity for {bank_label} first. "
                    "Stage 2 will use that vocabulary.")
            st.stop()
        # Both languages: give the model checked sentences from the topic notes to build from
        # (Spanish built from isolated words made odd sentences like "un agua con leche").
        reference = topic_sentences(get_retriever(), language, block['topic'], variety)
        result = render_sentence_practice(activity_id, vocabulary, variety_name(variety), block['topic'],
                                          language, reference)
        if result is not None:
            record_progress(activity_id, block, variety, session['date'], result)
            if st.button("Finish this sentence practice", key=f"finish_sentences_{activity_id}"):
                reset("active_practice")
                st.rerun()
        st.caption("Your progress is saved automatically. Shelf this plan and come back to it any time.")
        st.stop()
    previous_minutes = sum(
        b['minutes'] for earlier in schedule['sessions'][:active['session']]
        for b in earlier['blocks'] if b['stage'] == 1 and b['topic'] == block['topic'])
    # Cards and questions per activity follow the topic's share of the plan
    # (the top-ranked topic gets 10 items and 5 questions; see practice.bank_size).
    weights = schedule.get('topic_weights') or {}
    share = (weights.get(block['topic'], 0) / max(weights.values())) if weights else 1.0
    activity = spanish_typing_activity(get_retriever(), block['topic'], variety,
                                       offset=previous_minutes // 2, share=share, language=language)
    st.markdown(f"## Stage 1: {CALENDAR_TOPICS[block['topic']]}")
    st.write("Goal: learn useful words and expressions for this topic.")
    st.caption("The suggested minutes are flexible; there is no timer.")
    # Same typing practice for both languages (Macedonian accepts Cyrillic or Latin input).
    study = st.session_state.get(f'es_study_{activity_id}', {})
    if activity['phrases'] and study.get('studied_once'):
        # Stages 2 and 3 for this topic build on exactly the items typed here.
        banks = dict(st.session_state.get('studied_banks', {}))
        banks[activity_id] = {'topic': block['topic'], 'variety': variety,
                              'phrases': [dict(p) for p in activity['phrases']]}
        st.session_state.studied_banks = banks
    result = render_spanish_practice(activity, activity_id, language)
    if result is not None:
        record_progress(activity_id, block, variety, session['date'], result)
        if st.button("Finish this practice", key=f"finish_spanish_{activity_id}"):
            reset("active_practice")
            st.rerun()
    st.caption("Your progress is saved automatically. Shelf this plan and come back to it any time.")
