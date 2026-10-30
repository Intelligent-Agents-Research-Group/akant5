"""Record corpus counts, hashes, pinned revisions, licenses, and provenance.

Bind run cards to the manifest hash; guard commercial release of NC-trained weights."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from akanlm.audit.sources import CorpusSource, Provenance, Role
from akanlm.utils.logging import get_logger
from akanlm.utils.paths import ensure_dir, sha256_text

logger = get_logger(__name__)


@dataclass
class SourceStats:
    """Record normalized source statistics: ``n_chars`` counts Akan alone;
    ``n_chars_pair`` counts both sides of parallel records."""

    name: str
    n_docs: int = 0
    n_chars: int = 0
    """Target-language characters. For a parallel source, the Akan side only."""

    n_chars_pair: int = 0
    """Both sides plus separator. Zero for monolingual sources."""

    n_words: int = 0
    n_unique_docs: int = 0
    """Distinct target texts under the CPT split's content key."""

    n_chars_unique: int = 0
    content_sha256: str = ""
    normalization_fires: dict[str, int] = field(default_factory=dict)
    non_latin_letters: dict[str, int] = field(default_factory=dict)

    def chars_for(self, as_pairs: bool) -> int:
        """Characters attributable to this source under a consumption mode."""
        return self.n_chars_pair if (as_pairs and self.n_chars_pair) else self.n_chars

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n_docs": self.n_docs,
            "n_chars_target": self.n_chars,
            "n_chars_pair": self.n_chars_pair or None,
            "n_words": self.n_words,
            "n_unique_docs": self.n_unique_docs,
            "n_chars_target_unique": self.n_chars_unique,
            "content_sha256": self.content_sha256,
            "normalization_fires": self.normalization_fires,
            "non_latin_letters": self.non_latin_letters,
        }


def measure(
    source: CorpusSource,
    limit: int | None = None,
    streaming: bool = False,
    max_chars: int | None = None,
) -> SourceStats:
    """Measure normalized target-side counts, hashes, and rule changes.

    Hash targets alone so monolingual and parallel declarations are comparable."""
    import hashlib

    from akanlm.audit.sources import Modality
    from akanlm.corpora.loaders import load_pairs, load_texts

    from akanlm.data.cpt_splits import content_key

    stats = SourceStats(name=source.name)
    digest = hashlib.sha256()
    seen: set[int] = set()

    if source.modality is Modality.MONOLINGUAL:
        stream = (
            (r.text, 0, r.normalization, {})
            for r in load_texts(
                source, limit=limit, streaming=streaming, max_chars=max_chars
            )
        )
    else:
        stream = (
            (r.tgt_text, len(r.src_text) + len(r.tgt_text) + 1,
             r.normalization.get("target", {}), r.normalization.get("source", {}))
            for r in load_pairs(
                source, limit=limit, streaming=streaming, max_chars=max_chars
            )
        )

    for text, pair_chars, target_report, source_report in stream:
        for prefix, report in (("", target_report), ("source/", source_report)):
            for rule in report.get("applied", []):
                key = prefix + rule
                stats.normalization_fires[key] = stats.normalization_fires.get(key, 0) + 1
        for ch, count in target_report.get("non_latin_letters", {}).items():
            stats.non_latin_letters[ch] = stats.non_latin_letters.get(ch, 0) + count
        stats.n_docs += 1
        stats.n_chars += len(text)
        stats.n_chars_pair += pair_chars
        stats.n_words += len(text.split())
        key = int(content_key(text)[:16], 16)
        if key not in seen:
            seen.add(key)
            stats.n_unique_docs += 1
            stats.n_chars_unique += len(text)
        digest.update(text.encode("utf-8"))
        digest.update(b"\x00")

    stats.content_sha256 = digest.hexdigest()
    logger.info(
        "%s: %d docs, %d target chars%s, %d words (sha %s)",
        source.name,
        stats.n_docs,
        stats.n_chars,
        f" ({stats.n_chars_pair} incl. source side)" if stats.n_chars_pair else "",
        stats.n_words,
        stats.content_sha256[:12],
    )
    return stats


def assert_normalization_current(document: dict, sources: dict[str, CorpusSource]) -> None:
    """Reject binding newly normalized training data to an old manifest."""
    from akanlm.corpora.normalize import GENERAL_RULES, normalization_spec

    entries = {entry["name"]: entry for entry in document.get("sources", [])}
    for name, source in sources.items():
        if not source.is_trainable:
            continue
        entry = entries.get(name, {})
        valid = entry.get("normalization") == normalization_spec(source.effective_rules())
        if source.modality.value == "parallel":
            valid = valid and entry.get("source_side_normalization") == normalization_spec(GENERAL_RULES)
        if not valid:
            raise ValueError(
                f"Manifest normalization is missing or outdated for {name}. "
                "Generate a new audit manifest under the current normalization before binding a run."
            )


def build(
    sources: dict[str, CorpusSource],
    stats: dict[str, SourceStats],
    dedup_reports: Iterable[dict] = (),
    contamination: dict | None = None,
    tokenizer_info: dict | None = None,
) -> dict:
    """Assemble the manifest document."""
    from akanlm.corpora.normalize import GENERAL_RULES, normalization_spec

    entries = []
    for name, source in sorted(sources.items()):
        measured = stats.get(name)
        entries.append(
            {
                "name": name,
                "hf_id": source.hf_id,
                "hf_config": source.hf_config,
                "hf_revision": source.hf_revision,
                # MAFAND's bytes come from the parquet branch, not hf_revision,
                # so the manifest must record the revision actually read.
                "hf_parquet_revision": source.hf_parquet_revision,
                "hf_split": source.hf_split,
                "local_path": source.local_path,
                "modality": source.modality.value,
                "language": source.language,
                "claimed_variety": source.claimed_variety,
                "variety_audited": source.variety_audited,
                "provenance": source.provenance.value,
                "licence": source.licence,
                "commercial_use": source.commercial_use,
                "role": source.role.value,
                "normalization_rules": source.effective_rules(),
                "normalization": normalization_spec(source.effective_rules()),
                "source_side_normalization": (
                    normalization_spec(GENERAL_RULES) if source.modality.value == "parallel" else None
                ),
                "expected_rows": source.expected_rows,
                "measured": measured.to_dict() if measured else None,
            }
        )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sources": entries,
        "dedup": list(dedup_reports),
        "contamination": contamination,
        "tokenizer": tokenizer_info,
        "licence_summary": licence_summary(sources.values()),
        "variety_summary": variety_summary(sources.values()),
    }
    manifest["manifest_sha256"] = sha256_text(
        json.dumps({k: v for k, v in manifest.items() if k != "created_utc"}, sort_keys=True)
    )
    return manifest


def licence_summary(sources: Iterable[CorpusSource]) -> dict:
    """Which licences apply, and which block commercial release."""
    sources = list(sources)
    training = [s for s in sources if s.role is Role.TRAIN]
    blocking = sorted(s.name for s in training if not s.commercial_use)
    return {
        "training_sources": sorted(s.name for s in training),
        "licences": sorted({s.licence for s in training}),
        "blocks_commercial_release": blocking,
        "derived_checkpoint_is_noncommercial": bool(blocking),
    }


def variety_summary(sources: Iterable[CorpusSource]) -> dict:
    """Variety labelling status, as the publication plan requires it stated."""
    sources = list(sources)
    return {
        "pooled": False,
        "primary_variety": "akuapem",
        "audited": sorted(s.name for s in sources if s.variety_audited),
        "unaudited": sorted(s.name for s in sources if not s.variety_audited),
        "note": (
            "Sources with variety_audited=false are NOT assigned to the Akuapem "
            "primary condition. FLORES+ separates Akuapem (twi_Latn_akua1239) from "
            "Asante (twi_Latn_asan1239); those are evaluated separately, never pooled."
        ),
    }


def provenance_summary(
    sources: Iterable[CorpusSource],
    stats: dict[str, SourceStats],
    mixture=None,
) -> dict:
    """Return training character counts/shares by provenance.

    With a mixture, weight included sources by repeats; otherwise count all training sources once."""
    by_name = {s.name: s for s in sources}

    # A mixture may list the same source twice under different consumption
    # modes — M3 takes pristine_twi_english both as monolingual Twi and as
    # translation pairs — so components are iterated, not deduplicated by name.
    if mixture is not None:
        components = [(c.source, c.repeat, c.as_pairs) for c in mixture.components]
    else:
        components = [
            (n, 1.0, False) for n, s in by_name.items() if s.role is Role.TRAIN
        ]

    by_provenance: dict[str, float] = {}
    for name, repeat, as_pairs in components:
        source = by_name.get(name)
        measured = stats.get(name)
        if source is None or measured is None:
            continue
        weighted = measured.chars_for(as_pairs) * repeat
        key = source.provenance.value
        by_provenance[key] = by_provenance.get(key, 0.0) + weighted

    total = sum(by_provenance.values())
    return {
        "mixture": getattr(mixture, "name", None),
        "weighted_by_repeat": mixture is not None,
        "chars_by_provenance": {k: int(v) for k, v in by_provenance.items()},
        "share": {k: round(v / total, 4) for k, v in by_provenance.items()} if total else {},
        "human_share": round(by_provenance.get(Provenance.HUMAN.value, 0.0) / total, 4)
        if total
        else 0.0,
        "note": (
            "Shares are character-weighted and include each source's repeat "
            "factor and consumption mode (target-side characters for a "
            "monolingual read of a parallel source, both sides for a pair "
            "read). Without a mixture, every training source counts once as "
            "monolingual, which is not what any run actually sees."
        ),
    }


def write(manifest: dict, path: Path) -> Path:
    """Serialise the manifest."""
    ensure_dir(path.parent)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("wrote manifest to %s (sha %s)", path, manifest["manifest_sha256"][:12])
    return path


def assert_release_safe(manifest: dict) -> None:
    """Reject noncommercial training sources; restrict the release rather than suppressing the error."""
    blocking = manifest["licence_summary"]["blocks_commercial_release"]
    if blocking:
        raise ValueError(
            "training sources carry non-commercial licences: "
            f"{', '.join(blocking)}. Any derived checkpoint is research-only."
        )
