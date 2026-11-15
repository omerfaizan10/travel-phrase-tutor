"""Spanish Stage 1 screen. Every answer is checked locally against the bank."""
import streamlit as st
from practice import spanish_feedback

CHARS = ('á', 'é', 'í', 'ó', 'ú', 'ü', 'ñ', '¿', '¡')


def _append_character(key, character):
    # Callback runs before the input is rendered, preserving the current text.
    st.session_state[key] = st.session_state.get(key, '') + character


def _accent_buttons(key):
    st.caption('Click a character to add it to the end of your answer.')
    for column, char in zip(st.columns(len(CHARS)), CHARS):
        column.button(char, key=f'{key}_char_{char}', on_click=_append_character,
                      args=(key, char))


def _study_submit(state_key, input_key, expected):
    state = dict(st.session_state[state_key])
    answer = st.session_state.get(input_key, '')
    if not answer.strip():
        state['feedback'] = 'Please type the Spanish word or expression first.'
    elif spanish_feedback(answer, expected) == 'correct':
        state['repetitions'] += 1
        state['feedback'] = 'Correct!'
        st.session_state[input_key] = ''
        if state['repetitions'] == 3:
            state['index'] += 1
            state['repetitions'] = 0
    else:
        state['feedback'] = 'Not quite. Look at the card and try again.'
    st.session_state[state_key] = state


def _recall_submit(state_key, input_key, question):
    state = dict(st.session_state[state_key])
    answer = st.session_state.get(input_key, '')
    if not answer.strip():
        state['feedback'] = 'Please enter an answer.'
    else:
        statuses = [spanish_feedback(answer, target) for target in question['accepted']]
        if 'correct' in statuses:
            records = list(state['records'])
            records.append({'question': state['index'], 'answer': answer,
                            'misses': state['misses']})
            state.update(index=state['index'] + 1, records=records, misses=0,
                         feedback='Correct!')
            st.session_state[input_key] = ''
        else:
            state['misses'] += 1
            state['feedback'] = ('Not quite. Check the accent marks or ü and try again.'
                                 if 'accent' in statuses else 'Not quite. Try again.')
    st.session_state[state_key] = state


def render_spanish_practice(activity, activity_id):
    """Render study/recall and return a completed-round result, else None."""
    phrases, questions = activity['phrases'], activity['questions']
    if len(phrases) < 5 or len(questions) < 5:
        st.warning('This topic and Spanish variety do not yet have enough suitable items '
                   'for a five-question round. Choose another available variety or topic.')
        return None
    if len(phrases) < 10:
        st.info(f'This reference bank has {len(phrases)} suitable items. '
                'You will study those items and answer five questions from them.')
    st.caption('Reference materials: ' + ', '.join(sorted({p['source'] for p in phrases})))
    study_key, recall_key = f'es_study_{activity_id}', f'es_recall_{activity_id}'
    if study_key not in st.session_state:
        st.session_state[study_key] = {'index': 0, 'repetitions': 0, 'feedback': ''}
    study = st.session_state[study_key]
    if study['index'] < len(phrases):
        st.markdown('### Study the vocabulary and expressions')
        st.write(f"Item {study['index'] + 1} of {len(phrases)} · "
                 f"{study['repetitions']} of 3 correct entries")
        p = phrases[study['index']]
        with st.container(border=True):
            st.markdown(f"### {p['text']}")
            st.write(f"Means: {p['meaning']}")
            if p.get('pron'):
                st.caption(p['pron'])
        key = f'es_input_study_{activity_id}'
        st.text_input('Type the Spanish word or expression shown above', key=key)
        _accent_buttons(key)
        st.button('Check typing', key=f'es_check_study_{activity_id}',
                  on_click=_study_submit, args=(study_key, key, p['text']))
        if study['feedback']:
            st.info(study['feedback'])
        st.caption('Type the item, then click Check typing. Repeat until you have three correct entries.')
        return None
    if recall_key not in st.session_state:
        st.success('Study practice complete! You typed every item correctly three times.')
        st.write('Next: five questions from this bank, with the cards hidden. '
                 'A wrong answer lets you retry without showing the answer. '
                 'After a mistake, “I don’t know” ends the whole round and reveals all five answers.')
        if st.button('Start the 5-question check', key=f'es_start_recall_{activity_id}'):
            st.session_state[recall_key] = {'index': 0, 'misses': 0, 'records': [],
                                          'feedback': '', 'revealed': False}
            st.rerun()
        return None
    recall = st.session_state[recall_key]
    if recall['index'] < 5 and not recall['revealed']:
        st.markdown('### Check your recall')
        st.write(f"Question {recall['index'] + 1} of 5")
        question = questions[recall['index']]
        st.write(f"How do you say ‘{question['meaning']}’ in Spanish?")
        key = f'es_input_recall_{activity_id}'
        st.text_input('Your Spanish answer', key=key)
        _accent_buttons(key)
        st.button('Check answer', key=f'es_check_recall_{activity_id}',
                  on_click=_recall_submit, args=(recall_key, key, question))
        if recall['feedback']:
            st.info(recall['feedback'])
        if recall['misses']:
            if st.button('I don’t know — end round and show answers', key=f'es_reveal_{activity_id}'):
                recall['revealed'] = True
                st.session_state[recall_key] = recall
                st.rerun()
        return None
    st.markdown('### Round complete')
    if recall['revealed']:
        st.write('You ended this round. Review all five reference answers below.')
    else:
        st.success('You answered all five questions correctly, including any retries.')
    for i, q in enumerate(questions, 1):
        st.write(f"{i}. {q['meaning']} → {' / '.join(q['accepted'])}")
    st.caption('These are answers from the studied bank; other natural Spanish translations may also exist.')
    if st.button('Practice this bank again', key=f'es_retry_{activity_id}'):
        for key in list(st.session_state):
            if key.startswith('es_') and activity_id in key:
                del st.session_state[key]
        st.rerun()
    return {'answered': len(recall['records']), 'total': 5,
            'ended_with_reveal': recall['revealed'], 'records': recall['records']}
