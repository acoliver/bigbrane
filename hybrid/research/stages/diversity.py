"""A small TF-IDF cosine similarity over card texts (numpy only)."""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

STOPWORDS = frozenset("""
a an and are as at be by for from has have in is it its of on or that the this to was were with
which each every per into than then there these those their them they we our can may not no one
two all any also such via using used use where when while what who whose how over under
""".split())
TOKEN = re.compile(r"[a-z][a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [word for word in TOKEN.findall(text.lower()) if word not in STOPWORDS and len(word) > 2]


def tfidf_matrix(texts: list[str]) -> np.ndarray:
    docs = [Counter(tokenize(text)) for text in texts]
    vocab = sorted(set().union(*docs)) if docs else []
    index = {word: i for i, word in enumerate(vocab)}
    n = len(docs)
    df = Counter(word for doc in docs for word in doc)
    idf = np.array([math.log((1 + n) / (1 + df[word])) + 1.0 for word in vocab])
    matrix = np.zeros((n, len(vocab)))
    for row, doc in enumerate(docs):
        total = sum(doc.values()) or 1
        for word, count in doc.items():
            matrix[row, index[word]] = count / total
    matrix *= idf
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def most_similar(new_text: str, existing: dict[str, str]) -> tuple[str | None, float]:
    """(id of the most similar existing text, cosine) or (None, 0.0) when there is nothing to compare."""
    if not existing:
        return None, 0.0
    ids = list(existing)
    matrix = tfidf_matrix([existing[i] for i in ids] + [new_text])
    sims = matrix[:-1] @ matrix[-1]
    best = int(np.argmax(sims))
    return ids[best], float(sims[best])
