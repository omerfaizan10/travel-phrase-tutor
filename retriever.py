"""
retriever.py — RAG: finds the most relevant passages in our curated notes.

Docs live in knowledge_base/<language>/<topic>.md, for example:
    knowledge_base/spanish/restaurant.md
    knowledge_base/macedonian/directions.md

Each "## heading" section of a doc becomes one searchable chunk. Chunks are
turned into vectors (embeddings) with a multilingual model, so an English
question can match Spanish or Macedonian text.

Usage:
    from retriever import Retriever
    r = Retriever()
    r.search("how do I ask for the bill", language="spanish", k=3)
"""

import re
from pathlib import Path

import numpy as np

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DOCS_DIR = Path(__file__).parent / "knowledge_base"
MAX_CHUNK_CHARS = 1500  # keeps each passage small enough for the prompt


class Retriever:
    def __init__(self, docs_dir=DOCS_DIR, model_name=EMBED_MODEL):
        self.chunks = load_chunks(docs_dir)
        self.model_name = model_name
        self._model = None        # loaded on first search (takes a few seconds)
        self._embeddings = None   # one row per chunk

    def search(self, query, language=None, k=3) -> list[dict]:
        """
        Return the k chunks most similar to `query`, best first.
        language: "spanish" / "macedonian" to only search that folder.
        Each result: {"text", "source", "language", "score"} (score 0..1).
        """
        candidates = [
            i for i, c in enumerate(self.chunks)
            if language is None or c["language"] == language.lower()
        ]
        if not candidates:
            return []
        if self._embeddings is None:
            self._embeddings = self._encode([c["text"] for c in self.chunks])

        query_vec = self._encode([query])[0]
        # Vectors are normalized, so a dot product = cosine similarity.
        scores = self._embeddings[candidates] @ query_vec
        best = np.argsort(-scores)[:k]
        return [
            {**self.chunks[candidates[i]], "score": round(float(scores[i]), 3)}
            for i in best
        ]

    def _encode(self, texts):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            try:
                self._model = SentenceTransformer(self.model_name)
            except RuntimeError:
                # Intel Macs with an AMD GPU report MPS but can't use it ("only supported
                # on devices with unified memory"); the small embedding model runs fine on CPU.
                self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )


def load_chunks(docs_dir=DOCS_DIR) -> list[dict]:
    """Read every knowledge_base/<language>/*.md file and split it into chunks."""
    chunks = []
    for path in sorted(Path(docs_dir).glob("*/*.md")):
        language = path.parent.name.lower()
        text = path.read_text(encoding="utf-8")
        title_match = re.search(r"(?m)^# (.+)$", text)
        title = title_match.group(1).strip() if title_match else path.stem
        for section in _split_sections(text):
            # Text before the first "## " is the title/source header: skip it,
            # it only matches queries vaguely and wastes a retrieval slot.
            if not section.startswith("## "):
                continue
            # Prefix the doc title so a chunk still makes sense on its own.
            body = f"# {title}\n{section}"
            chunks.append({
                "text": body,
                "source": f"{language}/{path.name}",
                "language": language,
            })
    return chunks


def _split_sections(text):
    """
    Split on '## ' headings; split oversized sections by lines. Every piece of
    an oversized section repeats the section's heading, so no piece is lost
    and each still says what it is about.
    """
    pieces = []
    for section in re.split(r"(?m)^(?=## )", text):
        section = section.strip()
        if not section:
            continue
        if len(section) <= MAX_CHUNK_CHARS or not section.startswith("## "):
            pieces.append(section)
            continue
        heading, _, body = section.partition("\n")
        current = ""
        for line in body.splitlines():
            if current.strip() and len(current) + len(line) > MAX_CHUNK_CHARS - len(heading):
                pieces.append(f"{heading}\n{current.strip()}")
                current = ""
            current += line + "\n"
        if current.strip():
            pieces.append(f"{heading}\n{current.strip()}")
    return pieces


# ---------------------------------------------------------------------------
# Phrase index: every "- phrase [pronunciation] = meaning" line in the notes.
# The full ("rag") pipeline only ever shows phrases from here, copied exactly,
# so the model can choose phrases but cannot misspell or invent them.
# ---------------------------------------------------------------------------

# Dialogue labels that mean "the traveler says this" (so not a partner line).
TRAVELER_NAMES = {"viajero", "viajera", "cliente", "turista", "traveler", "traveller",
                  "you", "tú", "usted", "патник", "туристот", "гостин"}

_PHRASE_LINE = re.compile(r"^-\s+(?P<body>.+)$")
_PRON = re.compile(r"\s*\[(?P<pron>[^\]]+)\]\s*")
_CYRILLIC = re.compile(r"[Ѐ-ӿ]")


def parse_phrases(chunk: dict) -> list[dict]:
    """
    Extract phrases from one chunk. Handles:
        - Phrase [pron] = meaning
        - Phrase = meaning
        - A = meaning;  B = meaning        (several pairs on one line)
    Returns dicts: {"text", "pron", "meaning", "source", "section", "language"}.
    """
    lines = chunk["text"].splitlines()
    section = next((l[3:].strip() for l in lines if l.startswith("## ")), "")
    out = []
    for line in lines:
        m = _PHRASE_LINE.match(line.strip())
        if not m or " = " not in m.group("body"):
            continue
        # Several pairs on one line are separated by ";" or by two+ spaces.
        for pair in re.split(r"(?:;\s+|\s{2,})(?=[^;=]+=)", m.group("body")):
            if " = " not in pair:
                continue
            left, meaning = pair.split(" = ", 1)
            pron_m = _PRON.search(left)
            pron = pron_m.group("pron").strip() if pron_m else ""
            text = _PRON.sub(" ", left).strip()
            speaker = ""
            label = re.match(r"^\((waiter|clerk|local|receptionist|seller|driver|staff)\)\s*", text, re.I)
            if label:                      # "(waiter) ¿Qué va a tomar?" -> speaker="waiter"
                speaker, text = label.group(1).lower(), text[label.end():]
            named = re.match(r"^(\w{3,15}):\s+(?=\S)", text)
            if named:                      # dialogue lines: "Viajero: ...", "Келнерот: ..."
                who = named.group(1).lower()
                text = text[named.end():]
                if who not in TRAVELER_NAMES:
                    speaker = speaker or who
            if not text or not meaning.strip() or text.lower().startswith("example"):
                continue
            # The phrase side must be in the target language.
            if chunk["language"] == "macedonian" and not _CYRILLIC.search(text):
                continue
            out.append({
                "text": text, "pron": pron, "meaning": meaning.strip(), "speaker": speaker,
                "source": chunk["source"], "section": section,
                "language": chunk["language"],
            })
    return out


def normalize(s: str) -> str:
    """Lower-case, strip punctuation/brackets, collapse spaces (for matching)."""
    s = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", s.lower())
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()
