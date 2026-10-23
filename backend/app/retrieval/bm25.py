"""BM25 (Okapi) keyword ranking.

Implemented here (about 40 lines) rather than imported, to keep the scoring
transparent: score(q, d) = sum over query terms of
    idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))
"""

from __future__ import annotations

import math
from collections import Counter

from app.retrieval.text import tokenize


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.ids: list[str] = []
        self._tfs: list[Counter] = []
        self._lengths: list[int] = []
        self._df: Counter = Counter()

    def __len__(self) -> int:
        return len(self.ids)

    def add(self, chunk_id: str, text: str) -> None:
        tokens = tokenize(text)
        tf = Counter(tokens)
        self.ids.append(chunk_id)
        self._tfs.append(tf)
        self._lengths.append(len(tokens))
        self._df.update(tf.keys())

    def remove(self, chunk_ids: set[str]) -> None:
        keep = [i for i, cid in enumerate(self.ids) if cid not in chunk_ids]
        for i, cid in enumerate(self.ids):
            if cid in chunk_ids:
                self._df.subtract(self._tfs[i].keys())
        self._df += Counter()  # drop zero counts
        self.ids = [self.ids[i] for i in keep]
        self._tfs = [self._tfs[i] for i in keep]
        self._lengths = [self._lengths[i] for i in keep]

    def _idf(self, term: str) -> float:
        n = len(self.ids)
        df = self._df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        terms = set(tokenize(query))
        if not terms or not self.ids:
            return []
        avgdl = sum(self._lengths) / len(self._lengths) or 1.0
        scores: list[tuple[str, float]] = []
        for i, chunk_id in enumerate(self.ids):
            if allowed is not None and chunk_id not in allowed:
                continue
            tf = self._tfs[i]
            length_norm = 1 - self.b + self.b * self._lengths[i] / avgdl
            score = 0.0
            for term in terms:
                freq = tf.get(term, 0)
                if freq:
                    score += self._idf(term) * freq * (self.k1 + 1) / (freq + self.k1 * length_norm)
            if score > 0:
                scores.append((chunk_id, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:k]
