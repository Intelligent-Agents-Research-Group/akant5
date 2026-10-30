"""Load pinned corpus declarations into normalized text.

MAFAND uses converted parquet; ``hf_files`` sources use local extractions."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator

from akanlm.audit.sources import CorpusSource, Modality
from akanlm.corpora.normalize import GENERAL_RULES, normalize
from akanlm.utils.logging import get_logger

if TYPE_CHECKING:
    from akanlm.utils.parallel import OrderedPool

logger = get_logger(__name__)

PARQUET_BRANCH = "refs/convert/parquet"


@dataclass(frozen=True)
class TextRecord:
    """One normalized monolingual document."""

    source: str
    index: int
    text: str
    meta: dict
    normalization: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PairRecord:
    """One normalized translation pair."""

    source: str
    index: int
    src_text: str
    tgt_text: str
    meta: dict
    normalization: dict = field(default_factory=dict)


def _hf_load(source: CorpusSource, streaming: bool = False):
    from datasets import load_dataset

    kwargs: dict = {"split": source.hf_split, "streaming": streaming}
    if source.hf_config:
        kwargs["name"] = source.hf_config
    if source.hf_revision:
        kwargs["revision"] = source.hf_revision
    logger.info(
        "loading %s (config=%s, split=%s, revision=%s, streaming=%s)",
        source.hf_id,
        source.hf_config,
        source.hf_split,
        (source.hf_revision or "HEAD")[:8],
        streaming,
    )
    dataset = load_dataset(source.hf_id, **kwargs)
    _assert_pinned_revision(source, dataset, streaming)
    return dataset


def _assert_pinned_revision(source: CorpusSource, dataset, streaming: bool) -> None:
    """Reject cached builds whose path does not match the pinned revision.

    Offline datasets may silently select a different cached revision."""
    if not source.hf_revision or streaming:
        return
    cache_files = getattr(dataset, "cache_files", None)
    if not cache_files:  # in-memory or streaming build — nothing to verify
        return
    path = cache_files[0].get("filename", "")
    if source.hf_revision not in path:
        raise RuntimeError(
            f"{source.hf_id} resolved to a cached build that is not the pinned "
            f"revision {source.hf_revision[:8]}: {path}\n"
            "Under offline mode datasets ignores revision= and picks the newest "
            "cached build. Re-fetch the pinned revision with "
            "AKANLM_ALLOW_NETWORK=1, or correct hf_revision in the source YAML."
        )


def _load_mafand(source: CorpusSource, streaming: bool = False):
    """Read pinned converted MAFAND parquet because datasets no longer executes its script."""
    from datasets import load_dataset

    config = source.hf_config or "en-twi"
    split = source.hf_split

    # Pin the parquet commit: branch listings are mutable and require network access.
    revision = source.hf_parquet_revision
    if not revision:
        raise ValueError(
            f"{source.name}: hf_parquet_revision is required to read {source.hf_id} "
            f"from {PARQUET_BRANCH}. Resolve the branch to a commit SHA once "
            "(HfApi().list_repo_refs(...).converts) and pin it in the source YAML."
        )

    files = _mafand_data_files(source, revision, config, split)
    if not files:
        raise FileNotFoundError(
            f"no parquet files for {source.hf_id} config={config} split={split} "
            f"at {PARQUET_BRANCH} revision {revision[:8]}"
        )
    logger.info(
        "loading MAFAND %s/%s from %d parquet file(s) at revision %s",
        config, split, len(files), revision[:8],
    )
    return load_dataset("parquet", data_files=files, split="train", streaming=streaming)


def _mafand_data_files(
    source: CorpusSource, revision: str, config: str, split: str
) -> list[str]:
    """Prefer pinned local shard paths to avoid Hub listing and URL resolution.

    Partial snapshots may omit shards; the manifest records measured/expected row counts."""
    from huggingface_hub.constants import HF_HUB_CACHE
    from huggingface_hub.file_download import repo_folder_name

    snapshot_dir = (
        Path(HF_HUB_CACHE)
        / repo_folder_name(repo_id=source.hf_id, repo_type="dataset")
        / "snapshots"
        / revision
        / config
        / split
    )
    if snapshot_dir.is_dir():
        # .exists() follows the symlink into blobs/, so a dangling entry left by
        # an interrupted download is skipped rather than handed to load_dataset.
        local = sorted(f for f in snapshot_dir.glob("*.parquet") if f.exists())
        if local:
            logger.info(
                "MAFAND %s/%s: %d shard(s) from local snapshot at %s",
                config, split, len(local), revision[:8],
            )
            return [str(f) for f in local]

    from huggingface_hub import list_repo_files

    names = list_repo_files(source.hf_id, repo_type="dataset", revision=revision)
    prefix = f"{config}/{split}/"
    wanted = sorted(n for n in names if n.startswith(prefix) and n.endswith(".parquet"))
    return [f"hf://datasets/{source.hf_id}@{revision}/{n}" for n in wanted]


SPECIAL_LOADERS: dict[str, Callable] = {
    "masakhane/mafand": _load_mafand,
}


#: Datasets behind a Hub access request. Loading these without credentials
#: fails with a message that does not say what to do about it.
GATED = {
    "openlanguagedata/flores_plus": (
        "FLORES+ is gated. Accept the terms at "
        "https://huggingface.co/datasets/openlanguagedata/flores_plus and "
        "authenticate with `huggingface-cli login`, or set HF_TOKEN in the "
        "environment. On a cluster, export HF_TOKEN in the job environment, since "
        "compute nodes do not inherit an interactive login."
    ),
}


def _open(source: CorpusSource, streaming: bool = False):
    if source.hf_files is not None:
        # Local, pinned and offline by construction; see akanlm.corpora.extract.
        if streaming:
            raise ValueError(
                f"{source.name}: sources with hf_files are extracted locally and cannot "
                "be streamed; set streaming: false in the mixture"
            )
        from akanlm.corpora.extract import read_rows

        return read_rows(source)
    loader = SPECIAL_LOADERS.get(source.hf_id or "")
    try:
        return loader(source, streaming) if loader else _hf_load(source, streaming)
    except Exception as error:
        hint = GATED.get(source.hf_id or "")
        if hint and _looks_gated(error):
            raise RuntimeError(f"{source.name}: {hint}") from error
        if _looks_offline(error):
            raise RuntimeError(
                f"{source.name}: {source.hf_id}"
                + (f" (config {source.hf_config})" if source.hf_config else "")
                + " is not in the local cache, and the Hub is switched off.\n"
                "Re-run the command with AKANLM_ALLOW_NETWORK=1 to fetch it, e.g.\n"
                "  AKANLM_ALLOW_NETWORK=1 python -m akanlm.cli <command>"
            ) from error
        raise


def _looks_gated(error: Exception) -> bool:
    text = str(error).lower()
    return any(k in text for k in ("gated", "authenticated", "401", "403", "access to model"))


def _looks_offline(error: Exception) -> bool:
    """Recognize Hub-offline and missing cached dataset errors."""
    text = str(error).lower()
    return (
        "offline mode is enabled" in text
        or type(error).__name__ == "OfflineModeIsEnabled"
        or "couldn't find cache" in text
    )


def _extract(row: dict, column: str) -> str | None:
    """Pull a column, handling MAFAND's nested ``{"translation": {...}}`` rows."""
    if column in row:
        value = row[column]
        return value if isinstance(value, str) else None
    translation = row.get("translation")
    if isinstance(translation, dict) and column in translation:
        value = translation[column]
        return value if isinstance(value, str) else None
    return None


def load_texts(
    source: CorpusSource,
    limit: int | None = None,
    streaming: bool = False,
    min_chars: int = 1,
    max_chars: int | None = None,
    pool: "OrderedPool | None" = None,
) -> Iterator[TextRecord]:
    """Yield normalized target-side documents in dataset order.

    Budgets apply after filtering/normalization, including with workers. Keep the
    last document whole; reject sources that cannot meet ``max_chars``.
    Parallel sources yield ``text_column``; use ``load_pairs`` for both sides."""
    from akanlm.utils.parallel import OrderedPool

    normalize_row = partial(_text_record, source=source, rules=source.effective_rules(), min_chars=min_chars)
    dataset = _open(source, streaming=streaming)
    emitted = 0
    chars = 0

    for record in (pool or OrderedPool()).map(normalize_row, enumerate(dataset)):
        if record is None:
            continue
        yield record
        emitted += 1
        chars += len(record.text)
        if limit is not None and emitted >= limit:
            return
        if max_chars is not None and chars >= max_chars:
            return
    _check_budget(source, chars, max_chars)


def _text_record(_shared, item: tuple[int, dict], *, source: CorpusSource, rules, min_chars: int) -> TextRecord | None:
    """Normalize one ``(index, row)``; ``None`` when the text is missing or too short."""
    index, row = item
    raw = _extract(row, source.text_column or "text")
    if not raw:
        return None
    text, report = normalize(raw, rules)
    if len(text) < min_chars:
        return None
    return TextRecord(
        source=source.name,
        index=index,
        text=text,
        meta={k: v for k, v in row.items() if k != source.text_column and isinstance(v, (str, int, float))},
        normalization=report.to_dict(),
    )


def _check_budget(source: CorpusSource, chars: int, max_chars: int | None) -> None:
    if max_chars is not None:
        raise ValueError(f"{source.name} ended after {chars:,} normalized characters, "
                         f"short of its {max_chars:,}-character budget")


def load_pairs(
    source: CorpusSource,
    limit: int | None = None,
    streaming: bool = False,
    min_chars: int = 1,
    max_chars: int | None = None,
    pool: "OrderedPool | None" = None,
) -> Iterator[PairRecord]:
    """Yield normalized translation pairs; reject nonparallel sources.

    Apply Akan rules to targets and shared cleanup to English; workers preserve order."""
    if source.modality is not Modality.PARALLEL:
        raise ValueError(f"{source.name} is {source.modality.value}, not parallel")

    from akanlm.utils.parallel import OrderedPool

    normalize_row = partial(_pair_record, source=source, tgt_rules=source.effective_rules(), min_chars=min_chars)
    dataset = _open(source, streaming=streaming)
    emitted = 0
    chars = 0

    for record in (pool or OrderedPool()).map(normalize_row, enumerate(dataset)):
        if record is None:
            continue
        yield record
        emitted += 1
        # Budget on the target side, matching load_texts and SourceStats, so a
        # char cap means the same volume of Akan either way it is consumed.
        chars += len(record.tgt_text)
        if limit is not None and emitted >= limit:
            return
        if max_chars is not None and chars >= max_chars:
            return
    _check_budget(source, chars, max_chars)


def _pair_record(_shared, item: tuple[int, dict], *, source: CorpusSource, tgt_rules, min_chars: int) -> PairRecord | None:
    """Normalize one ``(index, row)`` pair; ``None`` when a side is missing or too short."""
    index, row = item
    raw_src = _extract(row, source.source_column or "en")
    raw_tgt = _extract(row, source.text_column or "twi")
    if not raw_src or not raw_tgt:
        return None
    src_text, src_report = normalize(raw_src, GENERAL_RULES)
    tgt_text, tgt_report = normalize(raw_tgt, tgt_rules)
    if len(src_text) < min_chars or len(tgt_text) < min_chars:
        return None
    return PairRecord(
        source=source.name,
        index=index,
        src_text=src_text,
        tgt_text=tgt_text,
        meta={},
        normalization={"source": src_report.to_dict(), "target": tgt_report.to_dict()},
    )
