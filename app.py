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

from html import escape
from datetime import date

import streamlit as st

from llm import LLMError
from planner import GOAL_HINTS, MAX_DAYS, build_schedule, parse_timeline, plan_varieties, STAGE_NAMES
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


def reset_topic_varieties():
    """Forget per-topic variety switches (they belong to one built schedule)."""
    for k in [k for k in st.session_state if k.startswith("tv_")]:
        del st.session_state[k]


def variety_name(code):
    return VARIETIES.get(code, code)


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
    if p.get("say"):  # stress guide: capitals mark the stressed syllable
        pron += (f'<div class="p" title="Capitals mark the stressed syllable">'
                 f'say: {escape(p["say"])}</div>')
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

with st.sidebar:
    st.markdown("### Evaluation")
    mode_label = st.selectbox("Mode", list(MODE_LABELS.values()),
                              help="Full system is the one to demo. The others are the "
                                   "comparison conditions from the proposal.")
    mode_key = next(k for k, v in MODE_LABELS.items() if v == mode_label)
    mode, use_lora = mode_key.split("+")[0], mode_key.endswith("+lora")
    if st.button("Start over"):
        reset("trip", "parsed", "plan", "session", "lesson", "roleplay", "free_chat", "draft", "preferences", "schedule_draft", "schedule_confirmed")
        for key in list(st.session_state):
            if key.startswith("q_"):
                del st.session_state[key]
        reset_topic_varieties()
        st.rerun()

step_bar(2 if st.session_state.get("schedule_confirmed") else
         1 if "schedule_draft" in st.session_state else 0)

if "schedule_draft" not in st.session_state:
    # --- 1. Guided questionnaire ---------------------------------------------------
    st.markdown("## Your learning preferences")
    st.caption("Choose your topics and practice availability.")
    start_date = date.today()
    st.caption("Your calendar starts today, using your selected practice weekdays, and ends the day before departure.")
    language = st.selectbox("Which language would you like to learn?", list(LANGUAGES),
                            format_func=lambda value: LANGUAGES[value], key="q_language")
    variety = DEFAULT_VARIETY[language]
    if language == "spanish":
        # Spain is preselected (the default when no country is chosen). Countries
        # with no notes of their own are marked; their topics use Spain Spanish.
        have = {code for counts in get_retriever().coverage("spanish").values() for code in counts}
        options = ["ES", "419"] + sorted(LATIN_AMERICA, key=variety_name)
        variety = st.selectbox(
            "Which Spanish would you like to learn?", options, index=0, key="q_variety",
            format_func=lambda c: "Spain (default)" if c == "ES" else
            variety_name(c) + ("" if c in have else " (no country notes yet, uses Spain)"),
            help="Pick the country you are traveling to. Topics without notes for that country "
                 "use Spain Spanish; you can change each topic on the next screen.")
    timeline = st.text_input("How long until your trip?", value="two weeks", key="q_timeline",
                             help="For example: 10 days, two weeks, next month, or simply 14.")
    detected_days = parse_timeline(timeline)
    if detected_days is None:
        st.info("Please give a duration such as 'two weeks' or '10 days'.")
        days = None
    else:
        if detected_days > MAX_DAYS:
            st.warning(f"This initial version supports up to {MAX_DAYS} preparation days. "
                       "Choose a shorter study window below if appropriate.")
        days = st.number_input("Confirm days until departure / study window", min_value=1,
                               max_value=MAX_DAYS, value=min(detected_days, MAX_DAYS),
                               key=f"q_days_{detected_days}")
    weekdays = st.multiselect("What days of the week can you practice?",
                             ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                             default=["Monday", "Wednesday", "Friday"], key="q_weekdays")
    minutes = st.number_input("How long can you typically practice for, in minutes?",
                              min_value=1, max_value=240, value=15, step=1, key="q_minutes")
    selected = st.multiselect("Which topics would you like to learn?", list(GOAL_LABELS),
                              default=["restaurant", "directions"],
                              format_func=lambda value: GOAL_LABELS[value], key="q_topics")
    st.markdown("**Rank your selected topics**")
    st.caption("Choose a different topic for each position. Rank 1 is most important.")
    goals = []
    # Separate ranking controls allow priorities to be changed without relying on
    # multiselect click order. A selection change starts a fresh set of controls.
    signature = "-".join(selected)
    for i in range(len(selected)):
        goals.append(st.selectbox(f"Priority {i + 1}", selected, index=i,
                                  format_func=lambda value: GOAL_LABELS[value],
                                  key=f"q_rank_{signature}_{i}"))
    valid_ranks = len(set(goals)) == len(goals)
    if not valid_ranks:
        st.warning("Each topic must appear exactly once in your rankings.")
    if goals and valid_ranks:
        total = len(goals) * (len(goals) + 1) / 2
        st.caption("Target share of practice within each stage:")
        for i, goal in enumerate(goals):
            st.write(f"{i + 1}. {GOAL_LABELS[goal]}: {100 * (len(goals) - i) / total:.1f}%")
    
    if st.button("Build my plan", type="primary"):
        if days is None or not weekdays or not goals or not valid_ranks:
            st.warning("Enter a valid timeline, select practice days and topics, and use unique ranks.")
        else:
            try:
                topic_variety = plan_varieties(get_retriever(), language, variety, goals)
                schedule = build_schedule(language, goals, int(days), weekdays, int(minutes), start_date,
                                          variety=variety, topic_variety=topic_variety)
            except ValueError as error:
                st.warning(str(error))
            else:
                st.session_state.schedule_draft = schedule
                st.session_state.preferences = {
                    "language": language, "variety": variety, "days": int(days), "timeline": timeline,
                    "weekdays": weekdays, "minutes": int(minutes), "goals": goals,
                    "starting_stage": 1,
                }
                st.session_state.schedule_confirmed = False
                reset_topic_varieties()
                st.rerun()
    st.stop()

# --- 2. Explain the plan before showing any dated sessions --------------------
schedule = st.session_state.schedule_draft
prefs = st.session_state.preferences
if not st.session_state.get("schedule_confirmed"):
    st.markdown("## Your suggested learning plan")
    st.write(f"You will practice {LANGUAGES[prefs['language']]} before your trip on "
             f"{schedule['departure_date']}.")
    st.write(f"Practice days: {', '.join(prefs['weekdays'])}. "
             f"Each session: {prefs['minutes']} minutes.")
    st.write(f"Your schedule has {len(schedule['sessions'])} sessions, "
             f"with {schedule['total_minutes']} minutes of practice in total.")
    st.markdown("**Your three learning stages**")
    descriptions = [
        "Learn and recall useful words and short expressions from the reference materials.",
        "Expand phrases into sentences, revisit vocabulary, and receive focused corrections.",
        "Use what you have learned in short travel conversations, with feedback afterward.",
    ]
    for i, (name, description, budget) in enumerate(zip(STAGE_NAMES.values(), descriptions,
                                                       schedule['stage_minutes']), 1):
        target = [30, 30, 40][i - 1]
        st.markdown(f"**{i}. {name} — about {target}% ({budget} minutes)**  \n{description}")
    st.markdown("**Your topic priorities**")
    for i, goal in enumerate(prefs['goals'], 1):
        st.write(f"{i}. {GOAL_LABELS[goal]} — target {schedule['topic_weights'][goal]:.1%}")
    st.write("All selected topics appear in every stage. Everyone starts with vocabulary. "
             "Practice minutes are rounded, so topic shares are approximate.")
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
        st.session_state.schedule_confirmed = True
        st.rerun()
    if st.button("Change my preferences"):
        reset("schedule_draft", "schedule_confirmed")
        reset_topic_varieties()
        st.rerun()
    st.stop()

# --- 3. Dated calendar for instructional review -------------------------------
st.markdown("## Your learning calendar")
lang_label = LANGUAGES[prefs['language']]
if prefs['language'] == "spanish":
    lang_label += f" ({variety_name(prefs.get('variety', 'ES'))})"
st.write(f"{lang_label} · {prefs['minutes']} minutes per session · "
         f"Departure: {schedule['departure_date']}")
st.info("Calendar review: the dates, topics, and stages are ready to review. "
        "Stage-specific practice activities and stage-switching controls are the next update.")
for index, session in enumerate(schedule['sessions'], 1):
    day = date.fromisoformat(session['date'])
    heading = f"Session {index} · {day.strftime('%A, %b %d, %Y')} · {session['minutes']} minutes"
    with st.expander(heading, expanded=index == 1):
        for block in session['blocks']:
            used = schedule.get("topic_variety", {}).get(block["topic"], {}).get("used")
            where = f" · {variety_name(used)}" if prefs['language'] == "spanish" and used else ""
            st.write(f"Stage {block['stage']}: {block['stage_name']} · "
                     f"{GOAL_LABELS[block['topic']]}{where} · {block['minutes']} min")
        st.caption("Minutes are pacing suggestions; they do not measure mastery.")
if st.button("Change my preferences"):
    reset("schedule_draft", "schedule_confirmed")
    reset_topic_varieties()
    st.rerun()
