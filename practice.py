"""Stage 1 practice from the reference bank; no generation or embeddings."""
import hashlib
import random
import re

from planner import topic_files
from retriever import accepted_labels, normalize, parse_phrases


def vocabulary_activity(retriever, language, topic, variety, minutes, offset=0):
    """Select short phrases, preferring exact regional notes over shared ones.

    Offset rotates the bank across sessions. Answers are checked against the
    supplied reference meanings; this is practice, not a proficiency test.
    """
    allowed = accepted_labels(language, variety)
    sources = {f"{language}/{name}" for name in topic_files(topic)}
    chunks = [c for c in retriever.chunks
              if c['language'] == language and c['source'] in sources
              and c.get('country') in allowed]
    chunks.sort(key=lambda c: allowed.index(c['country']))
    pool, seen = [], set()
    for chunk in chunks:
        for phrase in parse_phrases(chunk):
            key = normalize(phrase['text'])
            if (key in seen or phrase.get('speaker') or
                re.search(r'\.\.\.|…|__', phrase['text']) or
                len(phrase['text'].split()) > 12):
                continue
            seen.add(key)
            pool.append({**phrase, 'variety': chunk['country']})
    # Preserve regional ordering, use shorter expressions within each variety.
    pool.sort(key=lambda p: (allowed.index(p['variety']), len(p['text'].split())))
    if not pool:
        return {'phrases': [], 'questions': []}
    count = min(len(pool), max(1, min(8, minutes // 2 or 1)))
    start = offset % len(pool)
    selected = (pool[start:] + pool[:start])[:count]
    questions = []
    for phrase in selected:
        alternatives = list(dict.fromkeys(p['meaning'] for p in pool if p['meaning'] != phrase['meaning']))
        seed = int(hashlib.sha256((phrase['text'] + variety).encode()).hexdigest()[:12], 16)
        rng = random.Random(seed)
        rng.shuffle(alternatives)
        options = [phrase['meaning']] + alternatives[:3]
        if len(options) < 2:
            continue
        rng.shuffle(options)
        questions.append({'phrase': phrase['text'], 'answer': phrase['meaning'], 'options': options})
    return {'phrases': selected, 'questions': questions}


def spanish_answer(text):
    """Ignore case, spacing and sentence punctuation, but preserve spelling."""
    import unicodedata
    text = unicodedata.normalize('NFC', text).casefold().strip()
    text = re.sub(r'[¿?¡!.,:;]', '', text)
    return ' '.join(text.split())


def spanish_feedback(answer, expected):
    """Return correct/accent/wrong without revealing the expected answer."""
    import unicodedata
    actual, target = spanish_answer(answer), spanish_answer(expected)
    if actual == target:
        return 'correct'
    def without_accents(value):
        return ''.join(c for c in unicodedata.normalize('NFD', value)
                       if c not in ('\u0301', '\u0308'))
    if without_accents(actual) == without_accents(target):
        return 'accent'
    return 'wrong'


def spanish_typing_activity(retriever, topic, variety, offset=0):
    """Ten reference items and five stable recall questions from that bank.

    Prefer single words, then short expressions. Do not invent items when
    the chosen regional/topic notes have a small bank.
    """
    allowed = accepted_labels('spanish', variety)
    sources = {f'spanish/{name}' for name in topic_files(topic)}
    pool, seen = [], set()
    chunks = [c for c in retriever.chunks if c['source'] in sources
              and c['language'] == 'spanish' and c.get('country') in allowed]
    for chunk in chunks:
        for p in parse_phrases(chunk):
            key = spanish_answer(p['text'])
            if (key in seen or p.get('speaker') or len(p['text'].split()) > 12
                or re.search(r'\.\.\.|…|__|[/();=]', p['text'])):
                continue
            # Keep the reference gloss, but omit explanatory parentheticals in the question.
            meaning = re.sub(r'\s*\([^)]*\)', '', p['meaning']).strip().rstrip('.')
            if not meaning or len(meaning.split()) > 20:
                continue
            seen.add(key)
            pool.append({**p, 'meaning': meaning, 'variety': chunk['country']})
    pool.sort(key=lambda p: (len(p['text'].split()) > 3,
                             allowed.index(p['variety']), len(p['text'].split())))
    if not pool:
        return {'phrases': [], 'questions': []}
    start = offset % len(pool)
    selected = (pool[start:] + pool[:start])[:10]
    rng = random.Random(f'{topic}:{variety}:{offset}')
    # Avoid asking the same English prompt twice. Accept studied equivalents
    # for the same English gloss, rather than rejecting an identical meaning.
    by_meaning = {}
    for p in selected:
        by_meaning.setdefault(p['meaning'].casefold(), []).append(p)
    groups = list(by_meaning.values())
    rng.shuffle(groups)
    questions = [{'meaning': group[0]['meaning'], 'answer': group[0]['text'],
                  'accepted': list(dict.fromkeys(p['text'] for p in group))}
                 for group in groups[:5]]
    return {'phrases': selected, 'questions': questions}
