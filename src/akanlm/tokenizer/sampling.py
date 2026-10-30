"""Exact normalized-character budgets for tokenizer training."""

from collections import Counter
from fractions import Fraction
import hashlib
import random


def allocate(total, names, weights):
    """Largest-remainder allocation; integer source quotas sum to the budget."""
    if not names:
        if total:
            raise ValueError("A positive language budget requires a source")
        return {}
    values = [Fraction(str(weights[name])) for name in names]
    if any(value <= 0 for value in values):
        raise ValueError("Source weights must be positive")
    exact = [total * value / sum(values) for value in values]
    counts = [int(value) for value in exact]
    order = sorted(range(len(names)), key=lambda i: exact[i] - counts[i], reverse=True)
    for i in order[:total - sum(counts)]:
        counts[i] += 1
    return dict(zip(names, counts))


def character_quotas(config):
    total = config.total_characters
    ratio = Fraction(str(config.english_ratio))
    if not isinstance(total, int) or isinstance(total, bool) or total <= 0 or not 0 <= ratio <= 1:
        raise ValueError("Require a positive integer character budget and English ratio in [0, 1]")
    english = total * ratio
    if english.denominator != 1:
        raise ValueError("Character budget cannot represent the English ratio exactly")
    names = [*config.twi_sources, *config.english_sources]
    if len(set(names)) != len(names):
        raise ValueError("Source names must be unique across both languages")
    missing = set(names) - config.source_weights.keys()
    if missing:
        raise ValueError(f"Missing within-language source weights: {sorted(missing)}")
    return {**allocate(total - int(english), config.twi_sources, config.source_weights),
            **allocate(int(english), config.english_sources, config.source_weights)}


def text_pieces(text, max_bytes=8192):
    """Preserve line tails; bound every piece by SentencePiece's UTF-8 limit."""
    width = min(2048, max_bytes // 4)
    if width < 1:
        raise ValueError("max_sentence_length must allow at least four UTF-8 bytes")
    for line in text.splitlines():
        line = line.strip()
        while line:
            end = min(width, len(line))
            if end < len(line):
                boundary = line.rfind(" ", 0, end)
                if boundary > 0:
                    end = boundary
            piece = line[:end].strip()
            if piece:
                yield piece
            line = line[end:].lstrip()


class CharacterBudgetIterator:
    """Interleave sources, exhausting exact quotas without rebalancing."""

    def __init__(self, streams, budgets, languages, seed, max_bytes=8192):
        self.streams = streams
        self.budgets = dict(budgets)
        self.languages = languages
        self.seed = seed
        self.max_bytes = max_bytes
        self.counts = {name: Counter() for name in budgets}
        self.digest = hashlib.sha256()
        self.started = False
        self.completed = False

    def __iter__(self):
        if self.started:
            raise RuntimeError("Character-budget stream can only be consumed once")
        self.started = True
        remaining = dict(self.budgets)
        streams = {name: iter(stream) for name, stream in self.streams.items()}
        rng = random.Random(self.seed)
        while any(remaining.values()):
            names = [name for name in remaining if remaining[name]]
            name = rng.choices(names, weights=[remaining[n] for n in names])[0]
            try:
                piece = next(streams[name])[:remaining[name]]
            except StopIteration as error:
                raise ValueError(f"{name} exhausted with {remaining[name]:,} characters still needed; "
                                 "refusing to change the language/source balance") from error
            if not piece.strip() or len(piece.encode("utf-8")) > self.max_bytes:
                raise ValueError(f"Empty or oversized SentencePiece input from {name}")
            remaining[name] -= len(piece)
            self.counts[name].update(characters=len(piece), lines=1, utf8_bytes=len(piece.encode("utf-8")))
            self.digest.update(piece.encode("utf-8"))
            self.digest.update(b"\n")
            yield piece
        self.completed = True

    def report(self):
        languages = {name: Counter() for name in ("twi", "english")}
        for name, counts in self.counts.items():
            languages[self.languages[name]].update(counts)
        total = sum(c["characters"] for c in languages.values())
        lines = sum(c["lines"] for c in languages.values())
        return {"protocol": "exact_character_budget_v1", "completed": self.completed,
                "unit": "normalized Unicode codepoints, excluding record separators",
                "source_budgets": self.budgets, "sources": self.counts,
                "languages": languages, "characters": total, "lines": lines,
                "english_character_share": languages["english"]["characters"] / total if total else None,
                "english_line_share": languages["english"]["lines"] / lines if lines else None,
                "input_sha256": self.digest.hexdigest(), "seed": self.seed}
