"""Stage 1 typing screen (Spanish and Macedonian). Every answer is checked locally against the bank.

A round: type each remaining item correctly three times, then answer the recall
questions. Items answered correctly on the first try are learned and leave the
bank; the learner can keep practicing the rest until the bank is empty.
"""
import streamlit as st
from practice import spanish_feedback, typing_questions
from pronounce import say, to_cyrillic, transliterate
from prompts import LANGUAGES

CHARS = ('á', 'é', 'í', 'ó', 'ú', 'ü', 'ñ', '¿', '¡')
REPEATS = 3


def _append_character(key, character):
    # Callback runs before the input is rendered, preserving the current text.
    st.session_state[key] = st.session_state.get(key, '') + character


def _accent_buttons(key):
    st.caption('Click a character to add it to the end of your answer.')
    for column, char in zip(st.columns(len(CHARS)), CHARS):
        column.button(char, key=f'{key}_char_{char}', on_click=_append_character,
                      args=(key, char))


def _typed(input_key, language):
    """The learner's answer; Macedonian typed in Latin letters is read as Cyrillic."""
    answer = st.session_state.get(input_key, '')
    return to_cyrillic(answer) if language == 'macedonian' else answer


def _answer_box(label, key, language):
    st.text_input(label, key=key)
    if language == 'macedonian':
        st.caption('Type in Cyrillic or in Latin letters as in the notes '
                   '(sh = ш, ch = ч, zh = ж, kj = ќ, gj = ѓ, c = ц, j = ј, dz = ѕ, dzh = џ, lj = љ, nj = њ).')
        typed = st.session_state.get(key, '')
        if typed.strip() and to_cyrillic(typed) != typed:
            st.caption(f'Read as: **{to_cyrillic(typed)}**')
    else:
        _accent_buttons(key)


def _study_submit(state_key, input_key, expected, bank_size, language='spanish'):
    state = dict(st.session_state[state_key])
    answer = _typed(input_key, language)
    if not answer.strip():
        state['feedback'] = f'Please type the {LANGUAGES[language]} word or expression first.'
    elif spanish_feedback(answer, expected) == 'correct':
        state['repetitions'] += 1
        st.session_state[input_key] = ''
        if state['repetitions'] == REPEATS:
            state['index'] += 1
            state['repetitions'] = 0
            state['feedback'] = (f'Correct! Typed {REPEATS} out of {REPEATS}. Next item.'
                                 if state['index'] < bank_size else '')
        else:
            state['feedback'] = f"Correct! Typed {state['repetitions']} out of {REPEATS}"
    else:
        state['feedback'] = 'Not quite. Look at the card and try again.'
    st.session_state[state_key] = state


def _recall_submit(state_key, input_key, question, language='spanish'):
    state = dict(st.session_state[state_key])
    answer = _typed(input_key, language)
    if not answer.strip():
        state['feedback'] = 'Please enter an answer.'
    else:
        statuses = [spanish_feedback(answer, target) for target in question['accepted']]
        if 'correct' in statuses:
            state['records'] = state['records'] + [{'question': state['index'], 'answer': answer,
                                                    'misses': state['misses'], 'skipped': False}]
            state.update(index=state['index'] + 1, misses=0, feedback='Correct!')
            st.session_state[input_key] = ''
        else:
            state['misses'] += 1
            state['feedback'] = ('Not quite. Check the accent marks or ü and try again.'
                                 if 'accent' in statuses and language == 'spanish' else 'Not quite. Try again.')
    st.session_state[state_key] = state


def _skip_question(state_key, input_key):
    """“I don't know”: skip only this question; the round continues."""
    state = dict(st.session_state[state_key])
    state['records'] = state['records'] + [{'question': state['index'], 'answer': '',
                                            'misses': state['misses'], 'skipped': True}]
    state.update(index=state['index'] + 1, misses=0, feedback='Skipped. The answer is shown at the end of the round.')
    st.session_state[input_key] = ''
    st.session_state[state_key] = state


def _learned(questions, records):
    """Bank items answered correctly on the first try this round."""
    texts = set()
    for record in records:
        if not record['skipped'] and record['misses'] == 0:
            texts.update(questions[record['question']]['accepted'])
    return texts


def render_spanish_practice(activity, activity_id, language='spanish'):
    """Render study/recall rounds; return a result once a round is finished, else None."""
    name = LANGUAGES[language]
    phrases = activity['phrases']
    question_count = activity.get('question_count', len(activity['questions']))
    if not phrases or not activity['questions']:
        st.warning(f'This topic{" and Spanish variety" if language == "spanish" else ""} does not yet have '
                   'enough suitable items for a practice round. Choose another topic.')
        return None
    st.caption('Reference materials: ' + ', '.join(sorted({p['source'] for p in phrases})))
    study_key, recall_key = f'es_study_{activity_id}', f'es_recall_{activity_id}'
    if study_key not in st.session_state:
        st.session_state[study_key] = {'index': 0, 'repetitions': 0, 'feedback': '', 'round': 1,
                                       'bank': [p['text'] for p in phrases], 'studied_once': False,
                                       'learned': [], 'rounds': []}
    study = st.session_state[study_key]
    bank = [p for p in phrases if p['text'] in study['bank']]

    if not bank:
        st.balloons() if not study.get('celebrated') else None
        study['celebrated'] = True
        st.success('Good Job! You’ve completed this section!')
        st.write(f"You learned all {len(phrases)} items in this bank.")
        for p in phrases:
            st.write(f"{p['text']} — {p['meaning']}")
        return {'answered': sum(r['answered'] for r in study['rounds']),
                'total': sum(r['total'] for r in study['rounds']),
                'rounds': study['rounds'], 'learned': study['learned'], 'completed': True}

    if study['index'] < len(bank):
        st.markdown('### Study the vocabulary and expressions')
        if study['round'] > 1:
            st.caption(f"Round {study['round']}: {len(bank)} item(s) left to learn.")
        st.write(f"Item {study['index'] + 1} of {len(bank)} · "
                 f"{study['repetitions']} of {REPEATS} correct entries")
        p = bank[study['index']]
        with st.container(border=True):
            st.markdown(f"### {p['text']}")
            st.write(f"Means: {p['meaning']}")
            if language == 'macedonian':
                st.caption(f"{p.get('pron') or transliterate(p['text'])} · say: {p.get('say') or say(p['text'])}")
            elif p.get('pron'):
                st.caption(p['pron'])
        key = f"es_input_study_{activity_id}_{study['round']}"
        _answer_box(f'Type the {name} word or expression shown above', key, language)
        st.button('Check typing', key=f'es_check_study_{activity_id}',
                  on_click=_study_submit, args=(study_key, key, p['text'], len(bank), language))
        if study['feedback']:
            (st.success if study['feedback'].startswith('Correct') else st.info)(study['feedback'])
        st.caption('Punctuation and capital letters are not required. '
                   f'Type the item, then click Check typing. Repeat until you have {REPEATS} correct entries.')
        return None

    if not study['studied_once']:
        study['studied_once'] = True     # app.py saves the studied bank for Stages 2 and 3
        st.session_state[study_key] = study
    if recall_key not in st.session_state:
        questions = typing_questions(bank, question_count, f"{activity.get('seed', activity_id)}:{study['round']}")
        n = len(questions)
        st.success(f'Study practice complete! You typed every item correctly {REPEATS} times.')
        st.write(f'Next: {n} question{"s" if n != 1 else ""} from this bank, with the cards hidden. '
                 'A wrong answer lets you retry without showing the answer. '
                 '“I don’t know” skips just that question; all answers are shown at the end.')
        if st.button(f'Start the {n}-question check', key=f'es_start_recall_{activity_id}_{study["round"]}'):
            st.session_state[recall_key] = {'index': 0, 'misses': 0, 'records': [],
                                            'feedback': '', 'questions': questions}
            st.rerun()
        return None

    recall = st.session_state[recall_key]
    questions = recall['questions']
    if recall['index'] < len(questions):
        st.markdown('### Check your recall')
        st.write(f"Question {recall['index'] + 1} of {len(questions)}")
        question = questions[recall['index']]
        st.write(f"How do you say ‘{question['meaning']}’ in {name}?")
        key = f"es_input_recall_{activity_id}_{study['round']}"
        _answer_box(f'Your {name} answer', key, language)
        st.button('Check answer', key=f'es_check_recall_{activity_id}',
                  on_click=_recall_submit, args=(recall_key, key, question, language))
        if recall['feedback']:
            st.info(recall['feedback'])
        if recall['misses']:
            st.button('I don’t know — skip this question', key=f'es_skip_{activity_id}',
                      on_click=_skip_question, args=(recall_key, key))
        return None

    # Round finished: summary, then items learned this round leave the bank.
    learned = _learned(questions, recall['records'])
    first_try = sum(not r['skipped'] and r['misses'] == 0 for r in recall['records'])
    skipped = sum(r['skipped'] for r in recall['records'])
    st.markdown('### Round complete')
    st.write(f"{first_try} of {len(questions)} correct on the first try"
             + (f", {skipped} skipped" if skipped else "") + '.')
    for i, (q, r) in enumerate(zip(questions, recall['records']), 1):
        mark = '✓' if not r['skipped'] and r['misses'] == 0 else ('skipped' if r['skipped'] else 'after a retry')
        st.write(f"{i}. {q['meaning']} → {' / '.join(q['accepted'])} ({mark})")
    st.caption(f'These are answers from the studied bank; other natural {name} translations may also exist.')
    remaining = [p['text'] for p in bank if p['text'] not in learned]
    round_result = {'round': study['round'], 'answered': len(questions) - skipped, 'total': len(questions),
                    'first_try': first_try, 'skipped': skipped, 'records': recall['records']}
    rounds = [r for r in study['rounds'] if r['round'] != study['round']] + [round_result]
    learned_all = sorted(set(study['learned']) | learned)
    if remaining:
        st.write(f"{len(remaining)} item(s) still to learn: " + ', '.join(remaining))
        if st.button('Keep practicing the remaining items', key=f'es_continue_{activity_id}_{study["round"]}'):
            st.session_state[study_key] = {**study, 'index': 0, 'repetitions': 0, 'feedback': '',
                                           'round': study['round'] + 1, 'bank': remaining,
                                           'learned': learned_all, 'rounds': rounds}
            del st.session_state[recall_key]
            st.rerun()
    else:
        if st.button('Continue', key=f'es_done_{activity_id}_{study["round"]}'):
            st.session_state[study_key] = {**study, 'bank': [], 'learned': learned_all, 'rounds': rounds}
            del st.session_state[recall_key]
            st.rerun()
    return {'answered': sum(r['answered'] for r in rounds), 'total': sum(r['total'] for r in rounds),
            'ended_with_reveal': False, 'records': recall['records'], 'rounds': rounds,
            'learned': learned_all, 'completed': not remaining}
