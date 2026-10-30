"""Compare Twi/English tokenizers on CPU.

Report tokens per word/character, byte-token share, isolated ɛ/ɔ share,
and prefix-boundary rate among words starting with a configured Akan prefix."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from akanlm.utils.logging import get_logger

logger = get_logger(__name__)

_BYTE_PIECE_RE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")
_WORD_RE = re.compile(r"\S+")

OPEN_VOWELS = ("ɛ", "ɔ", "Ɛ", "Ɔ")

#: Below this many ɛ/ɔ occurrences the isolation ratio is noise, not signal.
MIN_OPEN_VOWELS = 25


@dataclass
class TokenizerStats:
    """Measurements for one tokenizer on one text sample."""

    tokenizer: str
    language: str
    vocab_size: int
    n_docs: int = 0
    n_chars: int = 0
    n_words: int = 0
    n_tokens: int = 0
    n_unk: int = 0
    n_byte_pieces: int = 0
    open_vowel_total: int = 0
    open_vowel_isolated: int = 0
    morph_prefix_total: int = 0
    morph_prefix_aligned: int = 0
    examples: list[dict] = field(default_factory=list)

    @property
    def fertility(self) -> float:
        return self.n_tokens / self.n_words if self.n_words else 0.0

    @property
    def tokens_per_char(self) -> float:
        return self.n_tokens / self.n_chars if self.n_chars else 0.0

    @property
    def unk_rate(self) -> float:
        return self.n_unk / self.n_tokens if self.n_tokens else 0.0

    @property
    def byte_fallback_rate(self) -> float:
        return self.n_byte_pieces / self.n_tokens if self.n_tokens else 0.0

    @property
    def open_vowel_isolation(self) -> float | None:
        """Return None when too few ɛ/ɔ occurrences support a meaningful isolation rate."""
        if self.open_vowel_total < MIN_OPEN_VOWELS:
            return None
        return self.open_vowel_isolated / self.open_vowel_total

    @property
    def morph_boundary_rate(self) -> float:
        return (
            self.morph_prefix_aligned / self.morph_prefix_total
            if self.morph_prefix_total
            else 0.0
        )

    def to_dict(self) -> dict:
        return {
            "tokenizer": self.tokenizer,
            "language": self.language,
            "vocab_size": self.vocab_size,
            "n_docs": self.n_docs,
            "n_words": self.n_words,
            "n_tokens": self.n_tokens,
            "fertility": round(self.fertility, 4),
            "tokens_per_char": round(self.tokens_per_char, 5),
            "unk_rate": round(self.unk_rate, 6),
            "byte_fallback_rate": round(self.byte_fallback_rate, 6),
            "open_vowel_isolation": (
                round(self.open_vowel_isolation, 4)
                if self.open_vowel_isolation is not None
                else None
            ),
            "open_vowel_occurrences": self.open_vowel_total,
            "morph_boundary_rate": round(self.morph_boundary_rate, 4),
            "morph_prefix_occurrences": self.morph_prefix_total,
            "examples": self.examples,
        }


def load_morph_prefixes(language_config: Path) -> list[str]:
    """Akan prefixes worth checking for boundary alignment, longest first."""
    spec = json.loads(language_config.read_text(encoding="utf-8"))
    markers = spec.get("morphological_markers", {})
    prefixes: set[str] = set()
    for key in ("subject_pronoun_prefixes", "possessive_prefixes", "aspect_markers"):
        prefixes.update(markers.get(key, []))
    return sorted(prefixes, key=len, reverse=True)


def _strip_boundary(token: str) -> str:
    """Token text without SentencePiece's word-boundary marker."""
    return token.replace("▁", "")


def analyze(
    tokenizer,
    texts: Sequence[str],
    language: str,
    name: str | None = None,
    morph_prefixes: Sequence[str] = (),
    n_examples: int = 5,
) -> TokenizerStats:
    """Return tokenizer statistics and tokenized examples for the supplied texts."""
    stats = TokenizerStats(
        tokenizer=name or getattr(tokenizer, "name_or_path", "unknown"),
        language=language,
        vocab_size=len(tokenizer),
    )
    unk_id = tokenizer.unk_token_id

    for doc_index, text in enumerate(texts):
        if not text:
            continue
        text = unicodedata.normalize("NFC", text)
        # Whole documents are measured untruncated and never fed to a model, so
        # the model_max_length warning does not apply. tokenize() emits it too,
        # hence tokens come from the ids (out-of-vocabulary text shows as <unk>).
        ids = tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(ids)

        stats.n_docs += 1
        stats.n_chars += len(text)
        stats.n_words += len(_WORD_RE.findall(text))
        stats.n_tokens += len(tokens)
        if unk_id is not None:
            stats.n_unk += sum(1 for i in ids if i == unk_id)
        stats.n_byte_pieces += sum(1 for t in tokens if _BYTE_PIECE_RE.match(t))

        _count_open_vowels(tokens, stats)
        if morph_prefixes:
            _count_morph_boundaries(text, tokenizer, morph_prefixes, stats)

        if len(stats.examples) < n_examples and doc_index < n_examples * 4:
            stats.examples.append(
                {"text": text[:120], "tokens": tokens[:40], "n_tokens": len(tokens)}
            )

    return stats


def _count_open_vowels(tokens: Sequence[str], stats: TokenizerStats) -> None:
    """Count ɛ/ɔ occurrences and how many are stranded as lone tokens."""
    for token in tokens:
        stripped = _strip_boundary(token)
        occurrences = sum(stripped.count(v) for v in OPEN_VOWELS)
        if not occurrences:
            continue
        stats.open_vowel_total += occurrences
        # A token that is nothing but the open vowel means the tokenizer had
        # no subword containing it — mT5's characteristic failure.
        if len(stripped) == 1:
            stats.open_vowel_isolated += occurrences


def _count_morph_boundaries(
    text: str,
    tokenizer,
    prefixes: Sequence[str],
    stats: TokenizerStats,
) -> None:
    """Check whether known Akan prefixes land on token boundaries."""
    for word in _WORD_RE.findall(text):
        bare = word.strip(".,!?;:\"'()[]").lower()
        prefix = next(
            (p for p in prefixes if bare.startswith(p) and len(bare) > len(p) + 1),
            None,
        )
        if prefix is None:
            continue
        stats.morph_prefix_total += 1

        pieces = [_strip_boundary(t) for t in tokenizer.tokenize(word)]
        cumulative = ""
        for piece in pieces:
            cumulative += piece
            if cumulative.lower() == prefix:
                stats.morph_prefix_aligned += 1
                break
            if len(cumulative) > len(prefix):
                break


def compare(
    tokenizers: dict[str, object],
    samples: dict[str, Sequence[str]],
    language_config: Path,
) -> dict:
    """Return diagnostics for each tokenizer/language pair.

    Apply configured morphology prefixes only to Akan/Twi samples."""
    prefixes = load_morph_prefixes(language_config)
    rows: list[dict] = []

    for label, tokenizer in tokenizers.items():
        for language, texts in samples.items():
            is_akan = language.lower().startswith(("ak", "twi"))
            stats = analyze(
                tokenizer,
                texts,
                language=language,
                name=label,
                morph_prefixes=prefixes if is_akan else (),
            )
            rows.append(stats.to_dict())
            isolation = stats.open_vowel_isolation
            logger.info(
                "%-28s %-5s fertility=%.3f unk=%.4f byte=%.4f vowel_isolation=%s morph=%.3f",
                label,
                language,
                stats.fertility,
                stats.unk_rate,
                stats.byte_fallback_rate,
                f"{isolation:.3f}" if isolation is not None else "n/a",
                stats.morph_boundary_rate,
            )

    return {"morph_prefixes": prefixes, "rows": rows}


def render_table(report: dict) -> str:
    """Markdown table of the comparison, for the run report."""
    header = (
        "| tokenizer | lang | vocab | fertility | tok/char | unk | byte-fb | "
        "ɛ/ɔ isolated | morph boundary |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    )
    lines = []
    for row in report["rows"]:
        isolation = row["open_vowel_isolation"]
        lines.append(
            f"| {row['tokenizer']} | {row['language']} | {row['vocab_size']} | "
            f"{row['fertility']:.3f} | {row['tokens_per_char']:.4f} | "
            f"{row['unk_rate']:.4f} | {row['byte_fallback_rate']:.4f} | "
            f"{f'{isolation:.3f}' if isolation is not None else 'n/a':>4} | "
            f"{row['morph_boundary_rate']:.3f} |"
        )
    return header + "\n".join(lines)
