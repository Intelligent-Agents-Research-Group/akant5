"""Path helpers, SHA-256 identities, and atomic JSON writes."""

import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

from slugify import slugify

DATA_ROOT = Path(os.environ.get("AKANLM_DATA_ROOT", "data"))
RUNS_ROOT = Path(os.environ.get("AKANLM_RUNS_ROOT", "runs"))

RAW_DIR = DATA_ROOT / "raw"
INTERIM_DIR = DATA_ROOT / "interim"
PROCESSED_DIR = DATA_ROOT / "processed"
MANIFEST_DIR = DATA_ROOT / "manifests"

# Leaf directory names a training run reuses verbatim, which therefore carry no
# information about *which* run a checkpoint belongs to.
GENERIC_CHECKPOINT_LEAVES = frozenset({"final", "best", "last"})


def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Full SHA-256 of a file, streamed."""
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def sha256_text(text: str) -> str:
    """Full SHA-256 of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest(value) -> str:
    """SHA-256 of a value's canonical JSON (sorted keys, UTF-8)."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(destination: str | Path, report):
    """Write atomically and return the report as later reads will see it."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    temporary.replace(destination)
    return json.loads(destination.read_text())


def run_label(checkpoints: Iterable[str | Path]) -> str:
    """A filename-safe label naming the run(s) an evaluation covers.

    Eval reports are written per CPT mixture, so a timestamp alone makes a
    directory of reports unreadable: every mixture's numbers land in files
    whose names are indistinguishable. This derives the run name from the
    checkpoints themselves.

    ``runs/cpt/akant5-small-m1x_en/final`` -> ``akant5-small-m1x_en``, since the leaf
    of a training run is the same word for every run and names nothing.
    A non-final leaf is kept, so an intermediate checkpoint stays distinct:
    ``runs/cpt/akant5-small-m1x_en/checkpoint-20000`` ->
    ``akant5-small-m1x_en-checkpoint-20000``. Hub ids keep both halves
    (``google/mt5-base`` -> ``google-mt5-base``).
    """
    parts: list[str] = []
    for checkpoint in checkpoints:
        raw = str(checkpoint).rstrip("/")
        path = PurePosixPath(raw)
        leaf = path.name
        parent = path.parent.name
        if leaf in GENERIC_CHECKPOINT_LEAVES or leaf.startswith(("checkpoint-", "step-")):
            # The leaf of a training run is the same word for every run, so on
            # its own it names nothing; the run directory above it is the name.
            stem = parent or leaf
            if leaf != "final":
                stem = f"{stem}-{leaf}"
        elif len(path.parts) == 2 and not raw.startswith((".", "/")):
            stem = f"{parent}-{leaf}"  # a hub id: google/mt5-base
        else:
            stem = leaf
        parts.append(slugify(stem, lowercase=True))

    if not parts:
        return "unknown"
    if len(parts) > 3:
        parts = parts[:3] + [f"and-{len(parts) - 3}-more"]
    return "+".join(parts)
