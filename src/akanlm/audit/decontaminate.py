"""Filter training text against protected word n-grams (default 13).

The ledger records removed documents and affected evaluation-item IDs."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from akanlm.utils.logging import get_logger

logger = get_logger(__name__)

_WORD_RE = re.compile(r"\w+", re.UNICODE)

DEFAULT_NGRAM = 13


def _hash_gram(gram: str) -> int:
    """Stable 64-bit hash of an n-gram.

    Uses blake2b rather than the built-in ``hash`` so the ledger is
    reproducible across processes and Python's hash randomization.
    """
    return int.from_bytes(hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest(), "big")


def ngrams(text: str, n: int = DEFAULT_NGRAM) -> list[str]:
    """Word n-grams of *text*, casefolded.

    Documents shorter than *n* words yield a single gram of the whole token
    sequence, so short evaluation sentences still get protection.
    """
    words = _WORD_RE.findall(text.casefold())
    if not words:
        return []
    if len(words) < n:
        return [" ".join(words)]
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


@dataclass
class EvalItem:
    """One protected evaluation or held-out item."""

    source: str
    index: int
    text: str


@dataclass
class ContaminationLedger:
    """Result of a decontamination pass, serialised into the manifest."""

    ngram_size: int
    normalization: dict = field(default_factory=dict)
    eval_sources: dict[str, int] = field(default_factory=dict)
    eval_grams_indexed: int = 0

    train_docs_seen: Counter = field(default_factory=Counter)
    train_docs_removed: Counter = field(default_factory=Counter)
    #: ``train_source -> {eval_source: hits}``
    hits_by_pair: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    #: Evaluation items that were matched at least once, by ``source:index``.
    eval_items_hit: Counter = field(default_factory=Counter)

    def removal_rate(self, train_source: str) -> float:
        seen = self.train_docs_seen.get(train_source, 0)
        return self.train_docs_removed.get(train_source, 0) / seen if seen else 0.0

    def compromised_eval_items(self, source: str | None = None) -> list[str]:
        """Evaluation items matched by training text, worth manual review."""
        return sorted(
            key for key in self.eval_items_hit
            if source is None or key.startswith(f"{source}:")
        )

    def to_dict(self) -> dict:
        return {
            "ngram_size": self.ngram_size,
            "normalization": self.normalization,
            "eval_sources": dict(self.eval_sources),
            "eval_grams_indexed": self.eval_grams_indexed,
            "train_docs_seen": dict(self.train_docs_seen),
            "train_docs_removed": dict(self.train_docs_removed),
            "removal_rate": {
                s: round(self.removal_rate(s), 6) for s in self.train_docs_seen
            },
            "hits_by_pair": {k: dict(v) for k, v in self.hits_by_pair.items()},
            "eval_items_hit_count": len(self.eval_items_hit),
            "eval_items_hit": dict(self.eval_items_hit.most_common(500)),
        }

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("wrote contamination ledger to %s", path)
        return path


class NgramIndex:
    """Hashed n-gram index over the protected evaluation items."""

    def __init__(self, n: int = DEFAULT_NGRAM) -> None:
        self.n = n
        self._grams: dict[int, str] = {}  # gram hash -> "source:index"

    def add(self, item: EvalItem) -> int:
        """Index one item; returns the number of grams added."""
        key = f"{item.source}:{item.index}"
        added = 0
        for gram in ngrams(item.text, self.n):
            digest = _hash_gram(gram)
            if digest not in self._grams:
                self._grams[digest] = key
                added += 1
        return added

    def add_all(self, items: Iterable[EvalItem]) -> dict[str, int]:
        """Index many items; returns per-source item counts."""
        counts: Counter = Counter()
        for item in items:
            self.add(item)
            counts[item.source] += 1
        return dict(counts)

    def __len__(self) -> int:
        return len(self._grams)

    def find(self, text: str) -> str | None:
        """Return the ``source:index`` of the first matching eval item, if any."""
        for gram in ngrams(text, self.n):
            hit = self._grams.get(_hash_gram(gram))
            if hit is not None:
                return hit
        return None


def build_index(items: Iterable[EvalItem], n: int = DEFAULT_NGRAM) -> tuple[NgramIndex, dict[str, int]]:
    """Build an :class:`NgramIndex` over every protected item."""
    index = NgramIndex(n)
    counts = index.add_all(items)
    logger.info("indexed %d n-grams from %d eval sources", len(index), len(counts))
    return index, counts


def decontaminate(
    train_docs: Iterable[tuple[str, str]],
    index: NgramIndex,
    eval_counts: dict[str, int],
) -> tuple[list[tuple[str, str]], ContaminationLedger]:
    """Return ``(clean_docs, ledger)`` after filtering protected n-gram matches."""
    ledger = ContaminationLedger(
        ngram_size=index.n,
        eval_sources=eval_counts,
        eval_grams_indexed=len(index),
    )
    clean: list[tuple[str, str]] = []

    for source, text in train_docs:
        ledger.train_docs_seen[source] += 1
        hit = index.find(text)
        if hit is None:
            clean.append((source, text))
            continue
        eval_source = hit.rsplit(":", 1)[0]
        ledger.train_docs_removed[source] += 1
        ledger.hits_by_pair[source][eval_source] += 1
        ledger.eval_items_hit[hit] += 1

    for source in ledger.train_docs_seen:
        rate = ledger.removal_rate(source)
        level = logger.warning if rate > 0.01 else logger.info
        level("decontamination: %s removed %.4f%% of documents", source, rate * 100)

    return clean, ledger
