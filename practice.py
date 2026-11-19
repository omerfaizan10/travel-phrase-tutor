"""Stage 1 practice from the reference bank; no generation or embeddings."""
import hashlib
import random
import re

from planner import topic_files
from retriever import accepted_labels, normalize, parse_phrases


# The top-ranked topic gets this many cards and questions per Stage 1 activity; lower-ranked
# topics get the same proportion as their share of the plan (see bank_size).
TOP_WORDS, TOP_QUESTIONS = 10, 5
MIN_WORDS, MIN_QUESTIONS = 3, 2


def bank_size(share=1.0):
    """(cards, questions) for a topic whose plan weight is `share` of the top topic's.
    Ranks 66.7% / 33.3% -> (10, 5) and (5, 3); 50/33/17% -> (10, 5), (7, 3), (3, 2)."""
    share = min(max(share, 0.0), 1.0)
    words = max(MIN_WORDS, int(TOP_WORDS * share + 0.5))
    questions = max(MIN_QUESTIONS, int(TOP_QUESTIONS * share + 0.5))
    return words, min(questions, words)


def vocabulary_activity(retriever, language, topic, variety, minutes, offset=0, share=1.0):
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
    count, question_count = bank_size(share)
    count = min(len(pool), count)
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
    return {'phrases': selected, 'questions': questions[:question_count]}


def spanish_answer(text):
    """Ignore case, spacing and all punctuation (¿ ? ¡ ! . , ; : quotes, dashes, …),
    but preserve spelling and accents. Used for Spanish and Macedonian answers."""
    import unicodedata
    text = unicodedata.normalize('NFC', text).casefold().strip()
    if re.search(r'[Ѐ-ӿ]', text):
        # Macedonian: "сè" is written with a grave accent that the Latin card line ("se")
        # does not show, and the Latin "lj" / "nj" stands for both љ / њ and лј / нј
        # ("iljada" = илјада). Treat each pair as the same spelling.
        text = unicodedata.normalize('NFC', ''.join(
            c for c in unicodedata.normalize('NFD', text) if c != '\u0300'))
        text = (text.replace('e', 'е').replace('ѐ', 'е').replace('ѝ', 'и')
                    .replace('лј', 'љ').replace('нј', 'њ'))
    text = ''.join(' ' if unicodedata.category(c).startswith('P') else c for c in text)
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


def typing_questions(phrases, count, seed):
    """Up to `count` recall questions from `phrases`, one per English meaning. Studied
    equivalents with the same gloss are all accepted, and all count as learned when
    the question is answered."""
    rng = random.Random(seed)
    by_meaning = {}
    for p in phrases:
        by_meaning.setdefault(p['meaning'].casefold(), []).append(p)
    groups = list(by_meaning.values())
    rng.shuffle(groups)
    return [{'meaning': group[0]['meaning'], 'answer': group[0]['text'],
             'accepted': list(dict.fromkeys(p['text'] for p in group))}
            for group in groups[:count]]


def spanish_typing_activity(retriever, topic, variety, offset=0, share=1.0, language='spanish'):
    """Reference items and stable recall questions from that bank; how many depends on
    the topic's share of the plan (bank_size: 10 items and 5 questions for the top topic).

    Prefer single words, then short expressions. Do not invent items when
    the chosen regional/topic notes have a small bank.
    """
    allowed = accepted_labels(language, variety)
    sources = {f'{language}/{name}' for name in topic_files(topic)}
    pool, seen = [], set()
    chunks = [c for c in retriever.chunks if c['source'] in sources
              and c['language'] == language and c.get('country') in allowed]
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
    words, question_count = bank_size(share)
    start = offset % len(pool)
    selected = (pool[start:] + pool[:start])[:words]
    # Avoid asking the same English prompt twice. Accept studied equivalents
    # for the same English gloss, rather than rejecting an identical meaning.
    seed = f'{topic}:{variety}:{offset}'
    questions = typing_questions(selected, question_count, seed)
    return {'phrases': selected, 'questions': questions, 'question_count': question_count, 'seed': seed}


def topic_sentences(retriever, language, topic, variety, limit=40):
    """Checked whole sentences from the topic notes (3+ words, traveler or staff lines),
    given to Stage 2 as correct patterns to reuse, so a small model does not invent forms."""
    allowed = accepted_labels(language, variety)
    sources = {f"{language}/{name}" for name in topic_files(topic)}
    out, seen = [], set()
    for chunk in retriever.chunks:
        if chunk['source'] not in sources or chunk.get('country') not in allowed:
            continue
        for p in parse_phrases(chunk):
            key = normalize(p['text'])
            if key in seen or not 3 <= len(p['text'].split()) <= 12 or re.search(r'\.\.\.|…|__|/', p['text']):
                continue
            seen.add(key)
            out.append({language: p['text'], 'english': re.sub(r'\s*\([^)]*\)', '', p['meaning']).strip()})
    return out[:limit]
