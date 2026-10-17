"""Translation memory: approved past translations, searchable by meaning.

Each target locale gets its own vector index over the *source* sentences, so a
lookup returns the most similar approved English sentences together with how they
were translated into that locale. FAISS is used when installed, otherwise an exact
NumPy search (same results, fine for tens of thousands of entries).
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embeddings import Embedder

try:  # optional accelerator
    import faiss
except ImportError:  # pragma: no cover - exercised when faiss is absent
    faiss = None


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class Entry:
    source: str
    locale: str
    target: str


@dataclass(frozen=True)
class Match:
    entry: Entry
    score: float  # cosine similarity, 1.0 = identical meaning


class VectorIndex:
    def __init__(self, dimensions: int, use_faiss: bool = True):
        self.dimensions = dimensions
        self._faiss = faiss.IndexFlatIP(dimensions) if (use_faiss and faiss is not None) else None
        self._matrix = np.zeros((0, dimensions), dtype=np.float32)

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if self._faiss is not None:
            self._faiss.add(vectors)
        else:
            self._matrix = np.vstack([self._matrix, vectors])

    def __len__(self) -> int:
        return self._faiss.ntotal if self._faiss is not None else len(self._matrix)

    def search(self, query: np.ndarray, k: int):
        k = min(k, len(self))
        if k == 0:
            return np.zeros((len(query), 0)), np.zeros((len(query), 0), dtype=int)
        query = np.ascontiguousarray(query, dtype=np.float32)
        if self._faiss is not None:
            return self._faiss.search(query, k)
        scores = query @ self._matrix.T
        idx = np.argsort(-scores, axis=1)[:, :k]
        return np.take_along_axis(scores, idx, axis=1), idx


class TranslationMemory:
    def __init__(self, embedder: Embedder, use_faiss: bool = True):
        self.embedder = embedder
        self.use_faiss = use_faiss
        self._entries: dict[str, list[Entry]] = defaultdict(list)
        self._index: dict[str, VectorIndex] = {}
        self._exact: dict[tuple, Entry] = {}

    @classmethod
    def load(cls, path: str | Path, embedder: Embedder, use_faiss: bool = True) -> TranslationMemory:
        tm = cls(embedder, use_faiss)
        with open(path, newline="", encoding="utf-8") as fh:
            tm.add([Entry(r["source"], r["locale"], r["target"]) for r in csv.DictReader(fh)])
        return tm

    def save(self, path: str | Path) -> None:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["source", "locale", "target"])
            for entries in self._entries.values():
                writer.writerows((e.source, e.locale, e.target) for e in entries)

    def add(self, entries: Sequence[Entry]) -> int:
        """Add approved translations. Re-approving a sentence replaces the old answer
        for exact lookups; similar-sentence search keeps every approved variant."""
        by_locale: dict[str, list[Entry]] = defaultdict(list)
        for e in entries:
            if normalize(e.source) and normalize(e.target):
                by_locale[e.locale].append(e)
        for locale, items in by_locale.items():
            vectors = self.embedder.embed([e.source for e in items])
            index = self._index.setdefault(locale, VectorIndex(self.embedder.dimensions, self.use_faiss))
            index.add(vectors)
            self._entries[locale].extend(items)
            for e in items:
                self._exact[(locale, normalize(e.source))] = e
        return sum(len(v) for v in by_locale.values())

    def __len__(self) -> int:
        return sum(len(v) for v in self._entries.values())

    def locales(self) -> list[str]:
        return sorted(self._entries)

    def exact(self, source: str, locale: str) -> Entry | None:
        return self._exact.get((locale, normalize(source)))

    def search(self, sources: Sequence[str], locale: str, k: int = 3, min_score: float = 0.45) -> list[list[Match]]:
        """Top-k similar approved translations for each source sentence."""
        index = self._index.get(locale)
        if index is None or not sources:
            return [[] for _ in sources]
        scores, ids = index.search(self.embedder.embed(list(sources)), k)
        entries = self._entries[locale]
        return [
            [Match(entries[i], float(s)) for s, i in zip(row_scores, row_ids) if i >= 0 and s >= min_score]
            for row_scores, row_ids in zip(scores, ids)
        ]
