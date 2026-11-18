"""
Macedonian pronunciation helpers, for lines that are not in the notes
(the roleplay partner's model-written replies).

    transliterate("Сита сум.")  -> "Sita sum."       (sh / ch / zh / kj / gj style)
    say("Сита сум.")            -> "SEE-ta soom."    (English-style respelling, stress in capitals)

These follow the same rules as the second bracket in knowledge_base/macedonian:
- Stress falls on the 3rd syllable from the end; 2-syllable words on the first.
- Short unstressed words (не, ќе, ме, се, ...) form one stress unit with the next word
  ("не знам" -> NE znam); ли and clitics after a "-те" imperative join the previous
  word ("Може ли" -> MO-zhe lee, "Дајте ми" -> DAY-te mee).
- Prepositions, conjunctions and forms of "сум" stay unstressed.
Loanwords that break the rule come out wrong; the notes have hand-checked versions.
"""

import re

TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "ѓ": "gj", "е": "e", "ж": "zh",
    "з": "z", "ѕ": "dz", "и": "i", "ј": "j", "к": "k", "л": "l", "љ": "lj", "м": "m",
    "н": "n", "њ": "nj", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "ќ": "kj",
    "у": "u", "ф": "f", "х": "h", "ц": "c", "ч": "ch", "џ": "dzh", "ш": "sh", "ѐ": "e", "ѝ": "i", "è": "e",
}
# Letters an English speaker reads correctly.
RESPELL = {**TRANSLIT, "ѓ": "gy", "и": "ee", "ј": "y", "љ": "ly", "њ": "ny", "ќ": "ky",
           "у": "oo", "ц": "ts", "џ": "j", "ѝ": "ee"}

VOWELS = set("аеиоуѐѝ")
CYR = re.compile(r"[Ѐ-ӿ]")
PROCLITICS = {"не", "ќе", "ме", "те", "го", "ја", "ги", "ми", "ти", "му", "ѝ", "ни", "ви", "им",
              "ве", "се", "си", "би"}
ENCLITICS = {"ли"}
UNSTRESSED = {"во", "на", "до", "од", "за", "со", "по", "кај", "низ", "без", "пред", "зад", "над",
              "под", "при", "меѓу", "и", "а", "но", "сум", "е", "сме", "сте", "се", "да", "ко"}
IMPERATIVES = {"дај", "кажи", "донеси", "покажи", "помогни", "јави", "чекај"}
LIGHT = PROCLITICS | ENCLITICS | UNSTRESSED


def transliterate(text: str) -> str:
    out = []
    for ch in text:
        lat = TRANSLIT.get(ch.lower())
        out.append(ch if lat is None else (lat.capitalize() if ch != ch.lower() else lat))
    return "".join(out)


def syllables(word: str) -> list[str]:
    """Split a Cyrillic word into syllables (syllabic р counts as a vowel)."""
    w = word.lower()
    nuclei = [i for i, c in enumerate(w)
              if c in VOWELS or (c == "р" and len(w) > 1
                                 and (i == 0 or w[i - 1] not in VOWELS)
                                 and (i == len(w) - 1 or w[i + 1] not in VOWELS))]
    if not nuclei:
        return [word]
    cuts = []
    for a, b in zip(nuclei, nuclei[1:]):
        n_cons = b - a - 1
        c1 = w[a + 1] if n_cons >= 1 else ""
        c2 = w[a + 2] if n_cons >= 2 else ""
        # V-CV; a stop + р/л stays together (do-bro); otherwise VC-CV
        together = n_cons <= 1 or (n_cons == 2 and c2 in "рл" and c1 not in "рлмнј")
        cuts.append(a + 1 if together else a + 2)
    bounds = [0] + cuts + [len(word)]
    return [word[x:y] for x, y in zip(bounds, bounds[1:])]


def _respell(text: str) -> str:
    return "".join(RESPELL.get(c, c) for c in text.lower())


def say(phrase: str) -> str:
    """English-style pronunciation guide for a Macedonian phrase; "" if it has no Cyrillic."""
    if not CYR.search(phrase or ""):
        return ""
    phrase = re.sub(r"\s*\([^)Ѐ-ӿ]*\)", "", phrase)          # drop (m), (f), English notes
    tokens = re.findall(r"[Ѐ-ӿ]+|[^Ѐ-ӿ]+", phrase)
    words = [i for i, t in enumerate(tokens) if CYR.match(t)]
    low = {i: tokens[i].lower() for i in words}

    def joinable(i, j):          # only spaces between word tokens i and j
        return all(tokens[k].strip() == "" for k in range(i + 1, j))

    # Group words into stress units.
    groups, pending = [], []
    for n, i in enumerate(words):
        nxt = words[n + 1] if n + 1 < len(words) else None
        prev = groups[-1] if groups else None
        leans_back = prev and not pending and joinable(prev[-1], i) and (
            low[i] in ENCLITICS
            or (low[i] in PROCLITICS and (
                low[prev[-1]] in IMPERATIVES
                or (low[prev[-1]].endswith("те") and len(syllables(tokens[prev[-1]])) >= 2)
                or nxt is None or not joinable(i, nxt))))
        if leans_back:
            prev.append(i)
        elif low[i] in PROCLITICS and nxt is not None and joinable(i, nxt):
            pending.append(i)
        else:
            groups.append(pending + [i])
            pending = []
    if pending:
        groups.append(pending)

    out = {i: _respell(tokens[i]) for i in words}
    for g in groups:
        sylls = [(i, s) for i in g for s in syllables(tokens[i])]
        lows = [low[i] for i in g]
        after = tokens[g[-1] + 1] if g[-1] + 1 < len(tokens) else ""
        is_yes = lows[0] == "да" and after[:1] in {",", ".", "!"}
        all_light = all(w in LIGHT for w in lows)
        if all_light and len(words) > 1 and ("не" in lows or "ќе" in lows):
            stressed = 0                                   # "не е" -> NE e
        elif all_light and len(words) > 1 and not is_yes:
            stressed = None                                # "Мило ми е" -> MEE-lo mee e
        else:
            stressed = max(len(sylls) - 3, 0)
        per_word: dict[int, list[str]] = {}
        for k, (i, s) in enumerate(sylls):
            r = _respell(s)
            per_word.setdefault(i, []).append(r.upper() if k == stressed else r)
        for i, parts in per_word.items():
            out[i] = "-".join(parts)
    return re.sub(r"\s+", " ", "".join(out.get(i, t) for i, t in enumerate(tokens))).strip()


# Latin -> Cyrillic, so learners without a Macedonian keyboard can type "Sakam kafe"
# (same sh / ch / zh / kj / gj style as the notes). Longest spellings are tried first.
_FROM_LATIN = sorted(
    [("dzh", "џ"), ("dž", "џ"), ("sh", "ш"), ("ch", "ч"), ("zh", "ж"), ("kj", "ќ"), ("gj", "ѓ"),
     ("lj", "љ"), ("nj", "њ"), ("dz", "ѕ"), ("š", "ш"), ("č", "ч"), ("ž", "ж"), ("ќ", "ќ"),
     ("ć", "ќ"), ("ǵ", "ѓ"), ("ḱ", "ќ"), ("y", "ј"), ("w", "в"), ("x", "кс"), ("q", "к")]
    + [(lat, cyr) for cyr, lat in TRANSLIT.items() if len(lat) == 1 and cyr not in "ѐѝè"],
    key=lambda pair: -len(pair[0]))


def to_cyrillic(text: str) -> str:
    """'Sakam edno kafe, ve molam.' -> 'Сакам едно кафе, ве молам.' Cyrillic is left as is."""
    out, i = [], 0
    while i < len(text):
        for lat, cyr in _FROM_LATIN:
            chunk = text[i:i + len(lat)]
            if chunk.lower() == lat:
                out.append(cyr.capitalize() if chunk[0].isupper() else cyr)
                i += len(lat)
                break
        else:
            out.append(text[i])
            i += 1
    return "".join(out)
