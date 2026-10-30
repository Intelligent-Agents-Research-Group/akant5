"""Load declared corpora into normalized text.

A single generic loader is driven by the :class:`~akanlm.audit.sources.CorpusSource`
declaration — the YAML already names the dataset, revision, config, split and
columns, so per-source loader modules would only restate it. Sources needing
genuinely different mechanics are registered in :data:`SPECIAL_LOADERS`.

Only one such special case exists today: ``masakhane/mafand`` is a script-based
dataset whose repository contains nothing but ``mafand.py``. Modern ``datasets``
releases no longer execute loading scripts, so MAFAND is read from the
auto-converted parquet branch instead.

A source declaring ``hf_files`` is read from its local extraction
(:mod:`akanlm.corpora.extract`) and never from the Hub at load time.
"""

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
    """Fail loudly if the build we got back is not the pinned revision.

    Under ``HF_HUB_OFFLINE=1`` ``datasets`` does not honour ``revision=``: it
    falls back to the ``Cache`` module, which selects the *most recently
    modified* cached build for the config and logs it only at WARNING. Passing a
    revision that does not exist at all still returns rows. Verified directly —
    a nonexistent ``DEADBEEF...`` revision returned the 1012-row
    ``5fec6c13...`` build without error.

    That silence is the danger: the manifest hash bound into each run card is
    only meaningful if the bytes actually came from the pinned commit. The
    cached build's path carries the revision it was built from, so check it
    rather than trust it.
    """
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
    """Read MAFAND from the auto-converted parquet branch.

    ``masakhane/mafand`` ships only a loading script, which ``datasets`` will
    not run. The parquet conversion at ``refs/convert/parquet`` carries the
    same rows under ``<config>/<split>/*.parquet``.
    """
    from datasets import load_dataset

    config = source.hf_config or "en-twi"
    split = source.hf_split

    # Resolve against the pinned parquet-branch SHA, never the branch name.
    # refs/convert/parquet is mutable — the Hub regenerates it whenever the
    # source dataset changes — so listing it by name leaves MAFAND the one
    # source whose file set is decided at load time rather than pinned, which
    # would quietly move the manifest hash bound into every run card. Listing a
    # branch name is also an uncacheable live API call, so it is the one thing
    # that cannot run offline even with a complete cache.
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
    """``data_files`` for the pinned parquet shards, from local cache when possible.

    Two separate things reach for the network here, and both must be avoided for
    a cached MAFAND to join an offline audit: ``list_repo_files`` is a live API
    call ``huggingface_hub`` does not cache, and an ``hf://`` URL is resolved
    through the Hub even when the bytes are already downloaded. So when the hub
    snapshot for *revision* is on disk, hand ``load_dataset`` ordinary local
    paths and neither happens.

    The snapshot directory is named by the commit SHA, so reading it is exactly
    as pinned as asking the Hub. Caveat worth stating plainly: a *partially*
    downloaded snapshot lists fewer shards with no local signal that it is
    short. ``expected_rows`` in the source YAML is the backstop, recorded beside
    the measured count in the manifest.
    """
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
    """Offline mode refused, or refused to find a cache entry.

    Both spellings matter: huggingface_hub raises OfflineModeIsEnabled for a
    blocked request, while datasets raises a bare ValueError from its Cache
    module when the config was never downloaded. Neither message mentions
    AKANLM_ALLOW_NETWORK, which is the only knob a user of this repo has.
    """
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
    """Yield normalized documents from *source*, target language only.

    A **parallel** source is accepted here and yields its ``text_column`` — the
    Akan side — discarding the other language. That is how the synthetic Twi
    monolingual stream is drawn: ``pristine-twi-english`` is the project's only
    Ghana NLP synthetic source, and M1/M2 take its Twi column while M3
    additionally takes it as pairs via :func:`load_pairs`.

    Args:
        source: Validated corpus declaration.
        limit: Stop after this many yielded records.
        streaming: Stream instead of materialising. Required for C4 and
            English Wikipedia, which are far too large to download whole.
        min_chars: Drop documents shorter than this after normalization.
        max_chars: Stop once this many normalized characters have been yielded.
            Prefer this to *limit* whenever the intent is a volume budget: a row
            cap is only a proxy for one, and an unstable proxy when document
            length varies. English Wikipedia in streaming order averages 30,664
            chars over its first 200 articles, 4,910 over its first 69,000 and
            3,036 across all 6.4M — so a row cap chosen from any sample of the
            front misses its intended volume by multiples. The last document is
            emitted whole rather than truncated, so the budget is a floor.
        pool: Normalize rows in this pool's worker processes. Records still
            arrive in dataset order, and the budgets are applied here, so the
            output is the same as without a pool.
    Yields:
        :class:`TextRecord` in dataset order.
    Raises:
        ValueError: If the source ends before *max_chars*; a shortfall would
            silently change the mixture's declared composition.
    """
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
    """Yield normalized translation pairs from *source*.

    The target-language side is normalized with the source's Akan rules; the
    English side gets shared cleanup without Akan letter substitutions or
    synthetic speaker/Markdown stripping. *pool* normalizes in worker
    processes, as in :func:`load_texts`.

    Raises:
        ValueError: If *source* is not parallel.
    """
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
