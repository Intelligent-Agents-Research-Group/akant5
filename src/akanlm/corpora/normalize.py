"""Normalize Akan/Twi and English with ordered pure string rules.

Report every rule that changes text, including ɛ/ɔ confusable mapping."""

from __future__ import annotations

import re
import hashlib
from collections import Counter
from pathlib import Path
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

# --------------------------------------------------------------------------
# Character tables
# --------------------------------------------------------------------------

#: Lookalikes that must collapse onto the two Akan-specific Latin letters.
CONFUSABLES: dict[str, str] = {
    # -> ɛ  LATIN SMALL LETTER OPEN E (U+025B)
    "ε": "ɛ",  # ε GREEK SMALL LETTER EPSILON
    "ϵ": "ɛ",  # ϵ GREEK LUNATE EPSILON SYMBOL
    "ᵋ": "ɛ",  # ᵋ MODIFIER LETTER SMALL OPEN E
    "\u0511": "ɛ",  # CYRILLIC SMALL LETTER REVERSED ZE (observed in FLORES+)
    # -> Ɛ  LATIN CAPITAL LETTER OPEN E (U+0190)
    "ℇ": "Ɛ",  # ℇ EULER CONSTANT
    "\u0510": "Ɛ",  # CYRILLIC CAPITAL LETTER REVERSED ZE
    # -> ɔ  LATIN SMALL LETTER OPEN O (U+0254)
    "ᴐ": "ɔ",  # ᴐ LATIN LETTER SMALL CAPITAL OPEN O
    "ͻ": "ɔ",  # ͻ GREEK SMALL REVERSED LUNATE SIGMA SYMBOL
    "ↄ": "ɔ",  # ↄ LATIN SMALL LETTER REVERSED C
    # -> Ɔ  LATIN CAPITAL LETTER OPEN O (U+0186)
    "Ↄ": "Ɔ",  # Ↄ ROMAN NUMERAL REVERSED ONE HUNDRED
    "\u03fd": "Ɔ",  # GREEK CAPITAL REVERSED LUNATE SIGMA SYMBOL
    "Ʃ": "Ɔ",  # Ʃ LATIN CAPITAL LETTER ESH (mis-set for Ɔ in some scans)
}

#: Observed inside otherwise Latin words in Pristine (e.g. wɔрерe).
#: Do not partially transliterate complete foreign-script names/quotations.
MIXED_SCRIPT_CONFUSABLES = {"\u0440": "p", "\u0435": "e"}
_LETTER_RUN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

#: ASCII stand-ins used when no Akan keyboard is available. Ambiguous with
#: real digits and punctuation, so this rule is OFF by default.
ASCII_STANDINS: dict[str, str] = {
    "3": "ɛ",  # 3  -> ɛ
    ")": "ɔ",  # )  -> ɔ
}

#: Invisible characters that carry no linguistic content. Spelled as escapes
#: because literal zero-width characters in source cannot be reviewed.
ZERO_WIDTH = (
    "\u200b"  # ZERO WIDTH SPACE
    "\u200c"  # ZERO WIDTH NON-JOINER
    "\u200d"  # ZERO WIDTH JOINER
    "\u2060"  # WORD JOINER
    "\ufeff"  # ZERO WIDTH NO-BREAK SPACE / BOM
    "\u00ad"  # SOFT HYPHEN
    "\u061c"  # ARABIC LETTER MARK
    "\u200e\u200f"  # LEFT-TO-RIGHT / RIGHT-TO-LEFT MARK
    "\u202a\u202b\u202c\u202d\u202e"  # bidi embeddings, overrides, PDF
    "\u2066\u2067\u2068\u2069"  # bidi isolates and PDI
)

#: Space-like characters folded onto U+0020.
SPACE_LIKE = (
    "\u00a0"  # NO-BREAK SPACE
    "\u1680"  # OGHAM SPACE MARK
    "\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a"  # EN QUAD .. HAIR SPACE
    "\u202f"  # NARROW NO-BREAK SPACE
    "\u205f"  # MEDIUM MATHEMATICAL SPACE
    "\u3000"  # IDEOGRAPHIC SPACE
)

#: Dash-like characters folded onto ASCII hyphen-minus.
DASH_LIKE = (
    "\u2010\u2011\u2012\u2013\u2014\u2015"  # HYPHEN .. HORIZONTAL BAR
    "\u2212"  # MINUS SIGN
)

_SPACE_RE = re.compile(f"[{re.escape(SPACE_LIKE)}]")
_ZERO_WIDTH_RE = re.compile(f"[{re.escape(ZERO_WIDTH)}]")
_DASH_RE = re.compile(f"[{re.escape(DASH_LIKE)}]")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_HSPACE_RUN_RE = re.compile(r"[ \t]{2,}")
_BLANK_LINE_RUN_RE = re.compile(r"\n{3,}")
_TRAILING_WS_RE = re.compile(r"[ \t]+\n")

# Residual markdown emphasis / headings / list bullets left by upstream cleaning.
# Leading indent is [ \t], never \s: \s would match the newlines *before* the
# marker and silently swallow paragraph breaks along with the marker.
_MD_HEADING_RE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+", re.MULTILINE)
_MD_BULLET_RE = re.compile(r"^[ \t]{0,3}[-*+][ \t]+", re.MULTILINE)
_MD_EMPHASIS_RE = re.compile(r"(\*{1,3}|_{2,3})(?=\S)(.+?)(?<=\S)\1", re.DOTALL)
_MD_CODEFENCE_RE = re.compile(r"^\s*```.*$", re.MULTILINE)

# "Speaker:" / "NARRATOR:" style labels at line start, an artefact of the
# dialogue and monologue styles in pristine-twi.
_SPEAKER_LABEL_RE = re.compile(
    r"^[ \t]{0,3}(?:\*{0,2})([A-ZƐƆ][\wÀ-ɏ' .-]{0,30})(?:\*{0,2})[ \t]*:[ \t]+",
    re.MULTILINE,
)

_QUOTE_MAP = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "′": "'", "″": '"',
})


# --------------------------------------------------------------------------
# Rules — each is a pure function
# --------------------------------------------------------------------------


def rule_strip_control(text: str) -> str:
    """Drop C0/C1 control characters, keeping tab, newline and carriage return."""
    return _CONTROL_RE.sub("", text)


def rule_strip_zero_width(text: str) -> str:
    """Drop zero-width joiners, spaces and the BOM."""
    return _ZERO_WIDTH_RE.sub("", text)


def rule_nfc(text: str) -> str:
    """Apply NFC; ɛ́/ɔ́ retain combining accents because no precomposed forms exist."""
    return unicodedata.normalize("NFC", text)


def rule_confusables(text: str) -> str:
    """Fold audited Akan letter lookalikes; preserve canonical accents and tone."""
    text = text.translate(str.maketrans(CONFUSABLES))
    if any(ch in text for ch in MIXED_SCRIPT_CONFUSABLES):
        def repair(match):
            word = match.group()
            if any("LATIN" in unicodedata.name(ch, "") for ch in word):
                return word.translate(str.maketrans(MIXED_SCRIPT_CONFUSABLES))
            return word
        text = _LETTER_RUN_RE.sub(repair, text)
    return rule_nfc(text)


def rule_ascii_standins(text: str) -> str:
    """Fold keyboard stand-ins ``3`` -> ``ɛ`` and ``)`` -> ``ɔ``.

    Destructive and ambiguous with real digits and parentheses. Enable only
    for a source whose audit shows the convention is used consistently.
    """
    return text.translate(str.maketrans(ASCII_STANDINS))


def rule_spaces(text: str) -> str:
    """Fold exotic space characters onto U+0020."""
    return _SPACE_RE.sub(" ", text)


def rule_dashes(text: str) -> str:
    """Fold en/em/figure dashes and the minus sign onto ASCII hyphen-minus."""
    return _DASH_RE.sub("-", text)


def rule_quotes(text: str) -> str:
    """Fold curly quotes and primes onto ASCII apostrophe and quotation mark."""
    return text.translate(_QUOTE_MAP)


def rule_strip_markdown(text: str) -> str:
    """Remove residual markdown headings, bullets, fences and emphasis."""
    text = _MD_CODEFENCE_RE.sub("", text)
    text = _MD_HEADING_RE.sub("", text)
    text = _MD_BULLET_RE.sub("", text)
    text = _MD_EMPHASIS_RE.sub(r"\2", text)
    return text


def rule_strip_speaker_labels(text: str) -> str:
    """Remove ``Speaker:`` prefixes at line start."""
    return _SPEAKER_LABEL_RE.sub("", text)


def rule_collapse_whitespace(text: str) -> str:
    """Collapse horizontal runs, trailing spaces and 3+ blank lines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HSPACE_RUN_RE.sub(" ", text)
    text = _TRAILING_WS_RE.sub("\n", text)
    text = _BLANK_LINE_RUN_RE.sub("\n\n", text)
    return text.strip()


RULES: dict[str, Callable[[str], str]] = {
    "strip_control": rule_strip_control,
    "strip_zero_width": rule_strip_zero_width,
    "nfc": rule_nfc,
    "confusables": rule_confusables,
    "ascii_standins": rule_ascii_standins,
    "spaces": rule_spaces,
    "dashes": rule_dashes,
    "quotes": rule_quotes,
    "strip_markdown": rule_strip_markdown,
    "strip_speaker_labels": rule_strip_speaker_labels,
    "collapse_whitespace": rule_collapse_whitespace,
}

#: Default for Akan text. CorpusSource selects GENERAL_RULES for English.
#: Confusables run after NFC, and recompose any new Latin base/mark pair.
DEFAULT_RULES: tuple[str, ...] = (
    "strip_control",
    "strip_zero_width",
    "nfc",
    "confusables",
    "spaces",
    "dashes",
    "quotes",
    "collapse_whitespace",
)

#: Additional cleanup for LLM-generated prose (pristine-twi and friends).
SYNTHETIC_RULES: tuple[str, ...] = (
    "strip_control",
    "strip_zero_width",
    "nfc",
    "confusables",
    "spaces",
    "dashes",
    "quotes",
    "strip_markdown",
    "strip_speaker_labels",
    "collapse_whitespace",
)

#: Shared text cleanup without interpreting foreign letters as Akan letters.
GENERAL_RULES = tuple(rule for rule in DEFAULT_RULES if rule != "confusables")
NORMALIZATION_VERSION = "akan_latin_v2"


def is_akan(language: str) -> bool:
    return language.lower().replace("_", "-").split("-")[0] in {"ak", "akan", "tw", "twi"}


def rules_for_language(language: str) -> tuple[str, ...]:
    return DEFAULT_RULES if is_akan(language) else GENERAL_RULES


def normalize_for_language(text: str, language: str) -> str:
    """Use this at downstream input/output boundaries with a known language."""
    return normalize_text(text, rules_for_language(language))


def normalization_spec(rules: Sequence[str]) -> dict:
    """Bind artifacts to rule order, implementation and Unicode database."""
    return {"version": NORMALIZATION_VERSION, "rules": list(rules),
            "unicode_version": unicodedata.unidata_version,
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------


@dataclass
class NormalizationReport:
    """Which rules fired, and how the text length moved."""

    applied: list[str] = field(default_factory=list)
    chars_before: int = 0
    chars_after: int = 0
    non_latin_letters: dict[str, int] = field(default_factory=dict)

    @property
    def char_delta(self) -> int:
        return self.chars_after - self.chars_before

    def to_dict(self) -> dict:
        return {
            "version": NORMALIZATION_VERSION,
            "applied": list(self.applied),
            "chars_before": self.chars_before,
            "chars_after": self.chars_after,
            "char_delta": self.char_delta,
            "non_latin_letters": self.non_latin_letters,
        }


def normalize(
    text: str,
    rules: Sequence[str] = DEFAULT_RULES,
) -> tuple[str, NormalizationReport]:
    """Apply named rules in order; return ``(text, report)`` and reject unknown names."""
    report = NormalizationReport(chars_before=len(text))
    current = text
    for name in rules:
        try:
            fn = RULES[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"unknown normalization rule: {name!r}") from exc
        updated = fn(current)
        if updated != current:
            report.applied.append(name)
            current = updated
    report.chars_after = len(current)
    # Audit unknown scripts instead of deleting or guessing transliterations.
    # Tone/combining marks are not letters and remain valid Akan text.
    report.non_latin_letters = {
        ch: count for ch, count in Counter(current).items()
        if unicodedata.category(ch).startswith("L") and "LATIN" not in unicodedata.name(ch, "")
    }
    return current, report


def normalize_text(text: str, rules: Sequence[str] = DEFAULT_RULES) -> str:
    """:func:`normalize` without the report, for call sites that don't audit."""
    return normalize(text, rules)[0]


def normalize_many(
    texts: Iterable[str],
    rules: Sequence[str] = DEFAULT_RULES,
) -> tuple[list[str], dict[str, int]]:
    """Normalize an iterable, returning texts and a per-rule fire count."""
    out: list[str] = []
    counts: dict[str, int] = {name: 0 for name in rules}
    for text in texts:
        normalized, report = normalize(text, rules)
        out.append(normalized)
        for name in report.applied:
            counts[name] += 1
    return out, counts
