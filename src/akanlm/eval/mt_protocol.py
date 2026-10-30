"""Frozen MT probe protocol: learning-rate selection on development data, then seeded test runs.

The probe measures, per model size and direction, how translation quality
after identical fine-tuning changes with continued pretraining. The procedure
below is fixed before any test decoding and does not depend on the language,
so a Kpelle or Vai study reuses it unchanged; only the task (language code and
data) differs.

Each stage is resumable and writes its artifacts under one study directory:

1. ``freeze`` runs the preflight on training and development text
   (deduplication against development and test text, the length rule,
   tokenizer coverage, the normalization guard) and writes ``protocol.json``.
   It decodes nothing.
2. ``select`` fine-tunes one system in one direction at every grid learning
   rate with the selection seed, restores the lowest-development-loss weights,
   decodes the development split, and chooses the rate by development chrF++.
   It never loads test data.
3. ``test`` writes a receipt binding the protocol and the selection, then
   fine-tunes the remaining seeds at the selected rate and decodes every test
   set for every seed. The selection run is the first seed's run.
4. ``report`` computes seed means and standard deviations and paired
   bootstrap differences between systems from the saved predictions alone.

Failure modes, listed before the code was written:

- F1: the scored weights are not the lowest-development-loss checkpoint
  (stop-step weights, the patience reference step, or a deleted checkpoint);
- F2: an interrupted run resumes with lost best-checkpoint or patience state;
- F3: learning-rate selection reads or decodes test data;
- F4: test decoding runs under a procedure, task data or selection that
  differs from the frozen one;
- F5: the learning-rate choice is ambiguous (ties, missing grid points);
- F6: seeds or rates share a directory, or the seed is not applied;
- F7: predictions are misaligned with references, or a partial file from an
  interrupted run counts as complete;
- F8: truncation or tokenizer coverage gaps bias one system without notice;
- F9: weights are deleted before a stage that still needs them;
- F10: language-specific constants change the procedure for a new language.
- F11: a dropout override changes source artifacts, misses instantiated
  dropout modules, or differs between selection, test, and resumed runs.
"""

from __future__ import annotations

import hashlib
import copy
import json
import shutil
import statistics
import subprocess
import unicodedata
from dataclasses import dataclass, field
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from akanlm.eval.mt_probe import (
    LANGUAGE_NAMES,
    ProbeConfig,
    direction_languages,
    directions_for,
    task_prefix,
)
from akanlm.utils.logging import get_logger
from akanlm.utils.paths import ensure_dir, run_label

logger = get_logger(__name__)

PROTOCOL_VERSION = "mt_probe_best_dev_v5"

#: The frozen procedure. Its canonical-JSON hash is recorded in every study,
#: selection and receipt; changing any value requires a new version.
PROCEDURE: dict = {
    "version": PROTOCOL_VERSION,
    "fine_tuning": {
        "optim": "adafactor",
        "lr_scheduler_type": "linear",
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 4,
        "max_steps": 6000,
        "eval_steps": 50,
        "early_stopping_patience": 10,
        "early_stopping_threshold": 0.001,
        "bf16": True,
        "pad_to_multiple_of": 32,
        "scored_weights": "checkpoint with the lowest development loss, first occurrence",
    },
    "learning_rate_selection": {
        "grid": [3e-5, 1e-4, 3e-4, 1e-3, 3e-3],
        "seed": 13,
        "split": "dev",
        "metric": "chrf2pp",
        "ties": (
            "equal development chrF++ at two decimals, then the lower restored "
            "development loss, then the lower learning rate"
        ),
        "scope": "each system and direction separately",
    },
    "seeds": [13, 14, 15],
    "lengths": {
        "default": 256,
        "fallback": 512,
        "max_dev_fraction_over_default": 0.01,
        "rule": (
            "use the fallback for every system when, for any tokenizer audited "
            "at freeze time, any direction and either side, more than the "
            "stated fraction of development examples exceed the default"
        ),
    },
    "decoding": {"num_beams": 4, "generation_batch_size": 16},
    "prompt": "translate {source} to {target}: ",
    "normalization": (
        "akanlm.corpora.normalize.normalize_for_language on model inputs, "
        "references and hypotheses"
    ),
    "deduplication": (
        "drop a training pair when its normalized English side or its "
        "normalized other-language side equals that side of any development "
        "or test pair"
    ),
    "normalization_guard": (
        "stop when normalization removes a combining mark (NFD category Mn) "
        "from any training or development sentence"
    ),
    "coverage_flag_unk_rate": 0.001,
    "metrics": {
        "primary": "chrF++ (sacreBLEU CHRF, word_order=2)",
        "secondary": "BLEU (sacreBLEU defaults)",
    },
    "uncertainty": {
        "seeds": "mean and sample standard deviation of corpus scores over seeds",
        "paired_bootstrap": {
            "resamples": 2000,
            "seed": 13,
            "unit": "test sentence, aligned by id",
            "statistic": "difference of seed-averaged corpus chrF++",
            "interval": "95% percentile",
        },
    },
    "revision_history": [
        ("v4 (2026-10-26): 3,000-update cap and grid {1e-4, 3e-4, 1e-3}. Its English-to-Twi "
         "development pilot fired the budget trigger (an initialization restored at update "
         "2,950 of 3,000 at its selected rate) and 6 of 10 systems selected a grid edge. v5 "
         "doubles the cap and extends the grid by one point beyond each edge for every system."),
    ],
    "revision_triggers": [
        ("at the selected rate, the restored step is the first development "
         "check for any system (the check interval or the grid is too coarse)"),
        ("at the selected rate, the restored step falls in the last patience "
         "window before max_steps for any system (the budget is too small)"),
    ],
}

#: Hard-capped token lengths at which the length audit reports fractions.
AUDITED_LENGTHS = (128, 256, 512)
#: Source files whose hashes are recorded at freeze time and in each receipt.
CODE_FILES = (
    "src/akanlm/eval/mt_protocol.py",
    "src/akanlm/eval/mt_probe.py",
    "src/akanlm/corpora/normalize.py",
)


# ---------------------------------------------------------------------------
# Hashing and small file helpers
# ---------------------------------------------------------------------------


def canonical_sha256(value) -> str:
    """sha256 of *value* as canonical JSON (sorted keys, no whitespace)."""
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, document: dict) -> Path:
    """Write atomically, so an interrupted write never leaves a valid-looking file."""
    ensure_dir(path.parent)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
    return path


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def code_provenance(repo_root: Path | None = None) -> dict:
    """Full git commit, dirty flag and hashes of the files that define the probe."""
    root = repo_root or Path(__file__).resolve().parents[3]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
    ).stdout.strip() or "unknown"
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", *CODE_FILES],
        cwd=root, capture_output=True, text=True, check=False,
    ).stdout.strip()
    return {
        "git_commit": commit,
        "probe_files_modified": bool(dirty),
        "files": {
            name: file_sha256(root / name) for name in CODE_FILES if (root / name).exists()
        },
    }


# ---------------------------------------------------------------------------
# Tasks: one language, its training/development split and its test sets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Item:
    """One parallel sentence; ``other`` is the non-English side."""

    id: str
    english: str
    other: str

    def side(self, language: str) -> str:
        return self.english if language == "en" else self.other


@dataclass
class Task:
    language: str
    spec: dict
    train: list[Item]
    dev: list[Item]
    tests: dict[str, list[Item]] = field(default_factory=dict)
    test_varieties: dict[str, str | None] = field(default_factory=dict)
    #: ``(language, text before probe normalization)`` for the guard.
    raw_train_dev: list[tuple[str, str]] = field(default_factory=list)


def items_sha256(items: Sequence[Item]) -> str:
    return canonical_sha256([[item.id, item.english, item.other] for item in items])


def _check_ids(name: str, items: Sequence[Item]) -> None:
    ids = [item.id for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{name}: sentence ids are not unique")
    if not items:
        raise ValueError(f"{name}: no sentence pairs")


def load_task(spec: dict, *, include_tests: bool) -> Task:
    """Rebuild a task from its recorded spec.

    ``include_tests=False`` is how selection guarantees it never reads test
    text (F3).
    """
    if spec["kind"] == "twi_mafand_flores":
        task = _twi_task(spec, include_tests)
    elif spec["kind"] == "local_jsonl":
        task = _local_task(spec, include_tests)
    else:
        raise ValueError(f"Unknown MT task kind: {spec['kind']}")
    for name, items in [("train", task.train), ("dev", task.dev), *task.tests.items()]:
        _check_ids(name, items)
    return task


def twi_task_spec(corpora_dir: Path, limit: int | None = None) -> dict:
    return {"kind": "twi_mafand_flores", "language": "twi",
            "corpora_dir": str(corpora_dir), "limit": limit}


def local_task_spec(language: str, data_dir: Path, limit: int | None = None) -> dict:
    """A local JSONL task: ``train.jsonl``, ``dev.jsonl`` and ``test_<name>.jsonl``.

    Each line is ``{"id": ..., "translation": {"en": ..., "<code>": ...}}``;
    ``id`` is optional and defaults to ``<file stem>-<line number>``.
    """
    if language == "en" or language not in LANGUAGE_NAMES:
        raise ValueError(f"Unsupported probe language: {language}")
    return {"kind": "local_jsonl", "language": language,
            "data_dir": str(Path(data_dir).resolve()), "limit": limit}


def _twi_task(spec: dict, include_tests: bool) -> Task:
    from akanlm.audit.sources import load_sources
    from akanlm.corpora.loaders import load_pairs
    from akanlm.eval.mt_probe import load_flores_items

    limit = spec.get("limit")
    sources = load_sources(Path(spec["corpora_dir"]))
    mafand = sources["mafand_en_twi"]

    def mafand_items(split: str) -> list[Item]:
        records = load_pairs(mafand.model_copy(update={"hf_split": split}), limit=limit)
        return [Item(f"mafand-{split}-{r.index}", r.src_text, r.tgt_text) for r in records]

    task = Task(language="twi", spec=spec, train=mafand_items("train"),
                dev=mafand_items("validation"))
    # The loader has already normalized MAFAND; the guard checks that the
    # probe's own normalization removes nothing further.
    task.raw_train_dev = [
        (language, item.side(language))
        for item in task.train + task.dev for language in ("en", "twi")
    ]
    if include_tests:
        for variety in ("akuapem", "asante"):
            english, twi, ids = load_flores_items(sources, variety, limit)
            name = f"flores_{variety}"
            task.tests[name] = [Item(f"flores-{i}", e, t) for i, e, t in zip(ids, english, twi)]
            task.test_varieties[name] = variety
        task.tests["mafand_test"] = mafand_items("test")
        task.test_varieties["mafand_test"] = None
    return task


def _local_task(spec: dict, include_tests: bool) -> Task:
    from akanlm.corpora.normalize import normalize_for_language

    language = spec["language"]
    data_dir = Path(spec["data_dir"])
    limit = spec.get("limit")
    raw: list[tuple[str, str]] = []

    def read(path: Path, keep_raw: bool) -> list[Item]:
        items = []
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    english, other = row["translation"]["en"], row["translation"][language]
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise ValueError(f"{path}:{number}: expected translation.en and "
                                     f"translation.{language}") from exc
                if not isinstance(english, str) or not isinstance(other, str):
                    raise ValueError(f"{path}:{number}: translation text is not a string")
                if keep_raw:
                    raw.extend([("en", english), (language, other)])
                english = normalize_for_language(english, "en")
                other = normalize_for_language(other, language)
                if not english or not other:
                    raise ValueError(f"{path}:{number}: empty translation text")
                items.append(Item(str(row.get("id", f"{path.stem}-{number}")), english, other))
                if limit is not None and len(items) >= limit:
                    break
        return items

    task = Task(language=language, spec=spec,
                train=read(data_dir / "train.jsonl", True),
                dev=read(data_dir / "dev.jsonl", True))
    task.raw_train_dev = raw
    if include_tests:
        paths = sorted(data_dir.glob("test_*.jsonl"))
        if not paths:
            raise ValueError(f"{data_dir}: no test_<name>.jsonl files")
        for path in paths:
            name = path.stem.removeprefix("test_")
            if name == "dev":
                raise ValueError(f"{path}: 'dev' is reserved for the development split")
            task.tests[name] = read(path, False)
            task.test_varieties[name] = None
    return task


def deduplicate_training(task: Task) -> tuple[list[Item], list[str]]:
    """Drop training pairs that share either side with development or test text."""
    held = [*task.dev, *(item for items in task.tests.values() for item in items)]
    english = {item.english for item in held}
    other = {item.other for item in held}
    kept, removed = [], []
    for item in task.train:
        (removed.append(item.id) if item.english in english or item.other in other
         else kept.append(item))
    return kept, removed


def training_items(task: Task, protocol: dict) -> list[Item]:
    """Training pairs after the deduplication recorded at freeze time."""
    removed = set(protocol["task"]["train"]["removed_ids"])
    return [item for item in task.train if item.id not in removed]


# ---------------------------------------------------------------------------
# Preflight audits
# ---------------------------------------------------------------------------


def _combining_marks(text: str) -> int:
    return sum(
        1 for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) == "Mn"
    )


def normalization_guard(raw: Sequence[tuple[str, str]]) -> dict:
    """Fail when probe normalization removes a combining mark (tone, nasal)."""
    from akanlm.corpora.normalize import normalize_for_language

    changed = 0
    losses = []
    for language, text in raw:
        normalized = normalize_for_language(text, language)
        changed += normalized != text
        if _combining_marks(normalized) < _combining_marks(text):
            losses.append({"language": language, "text": text, "normalized": normalized})
    if losses:
        raise ValueError(
            f"Normalization removed combining marks from {len(losses)} sentences; "
            f"first: {losses[0]}"
        )
    return {"sentences": len(raw), "changed_by_normalization": changed, "marks_removed": 0}


def audit_tokenizer(tokenizer, train: Sequence[Item], dev: Sequence[Item], language: str,
                    caps: Sequence[int] = AUDITED_LENGTHS) -> dict:
    """Development lengths per direction and side, and coverage per language.

    Lengths count the tokens the probe feeds the model (prefix and EOS
    included). Coverage counts ``<unk>`` ids and exact decode round trips over
    training and development text, without special tokens.
    """
    unk = tokenizer.unk_token_id
    audit: dict = {"directions": {}, "coverage": {}}
    for direction in directions_for(language):
        source_language, target_language = direction_languages(direction)
        prefix = task_prefix(direction)
        sources = [prefix + item.side(source_language) for item in dev]
        targets = [item.side(target_language) for item in dev]
        lengths = {
            "source": [len(ids) for ids in tokenizer(sources)["input_ids"]],
            "target": [len(ids) for ids in tokenizer(text_target=targets)["input_ids"]],
        }
        audit["directions"][direction] = {
            side: {
                "max": max(values),
                "fraction_over": {
                    str(cap): sum(v > cap for v in values) / len(values)
                    for cap in caps
                },
            }
            for side, values in lengths.items()
        }
    for side in ("en", language):
        texts = [item.side(side) for item in [*train, *dev]]
        encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
        tokens = sum(len(ids) for ids in encoded)
        unknown = sum(ids.count(unk) for ids in encoded) if unk is not None else 0
        exact = sum(
            tokenizer.decode(ids, skip_special_tokens=False,
                             clean_up_tokenization_spaces=False) == text
            for ids, text in zip(encoded, texts)
        )
        audit["coverage"][side] = {
            "sentences": len(texts),
            "tokens": tokens,
            "unk_tokens": unknown,
            "unk_rate": unknown / tokens if tokens else 0.0,
            "exact_round_trip_fraction": exact / len(texts),
        }
    return audit


def worst_dev_fraction(audit: dict, cap: int) -> float:
    return max(
        side["fraction_over"][str(cap)]
        for direction in audit["directions"].values() for side in direction.values()
    )


# ---------------------------------------------------------------------------
# Freeze
# ---------------------------------------------------------------------------


def protocol_path(study: Path) -> Path:
    return study / "protocol.json"


def system_path(study: Path, label: str) -> Path:
    return study / "systems" / f"{label}.json"


def procedure_with_dropout(procedure: dict, dropout_rate: float | None) -> dict:
    """Return a separately versioned procedure without changing the legacy one."""
    result = copy.deepcopy(procedure)
    if dropout_rate is not None:
        # ProbeConfig supplies the same validation used before model loading.
        ProbeConfig(dropout_rate=dropout_rate)
        result["fine_tuning"]["dropout_rate"] = float(dropout_rate)
        suffix = "_dropout_override"
        if not result["version"].endswith(suffix):
            result["version"] += suffix
    return result


def freeze(
    study: Path,
    task_spec: dict,
    checkpoints: Sequence[str],
    procedure: dict | None = None,
    dropout_rate: float | None = None,
) -> dict:
    """Run the preflight and write ``protocol.json`` and one file per system.

    A study is frozen once. Calling this again with the same task and
    procedure only registers new systems, and only when their tokenizer passes
    the length rule at the length already resolved (F4, F8).

    Args:
        procedure: Defaults to :data:`PROCEDURE`. Custom budgets mark a study
            as smoke; an explicit dropout override is a supported variant.
        dropout_rate: Applied to every system and frozen into the procedure.
            Omission preserves checkpoint rates. Repeat the same override
            when registering additional systems under an existing study.
    """
    from transformers import AutoTokenizer

    procedure = procedure_with_dropout(PROCEDURE if procedure is None else procedure,
                                       dropout_rate)
    canonical = procedure_with_dropout(PROCEDURE, procedure["fine_tuning"].get("dropout_rate"))
    smoke = (canonical_sha256(procedure) != canonical_sha256(canonical)
             or task_spec.get("limit") is not None)
    language = task_spec["language"]
    for direction in directions_for(language):
        if procedure["prompt"].format(
            source=LANGUAGE_NAMES[direction_languages(direction)[0]],
            target=LANGUAGE_NAMES[direction_languages(direction)[1]],
        ) != task_prefix(direction):
            raise ValueError("The procedure's prompt does not match the probe prefix")

    labels = {run_label([checkpoint]): checkpoint for checkpoint in checkpoints}
    if len(labels) != len(checkpoints):
        raise ValueError("Two systems share a run label")

    existing = protocol_path(study)
    task = load_task(task_spec, include_tests=True)
    train, removed = deduplicate_training(task)
    caps = sorted({*AUDITED_LENGTHS, procedure["lengths"]["default"],
                   procedure["lengths"]["fallback"]})
    # One tokenizer in memory at a time; mT5's 250k vocabulary is large.
    audits = {label: audit_tokenizer(AutoTokenizer.from_pretrained(path), train, task.dev,
                                     language, caps)
              for label, path in labels.items()}

    if existing.exists():
        protocol = load_protocol(study)
        if protocol["task"]["spec"] != task_spec or protocol["procedure"] != procedure:
            raise ValueError(f"{study} is frozen under a different task or procedure")
        resolved = protocol["lengths"]["resolved"]
    else:
        lengths = procedure["lengths"]
        worst = {label: worst_dev_fraction(audit, lengths["default"])
                 for label, audit in audits.items()}
        resolved = (lengths["fallback"]
                    if max(worst.values()) > lengths["max_dev_fraction_over_default"]
                    else lengths["default"])
        protocol = {
            "protocol_version": procedure["version"],
            "smoke": smoke,
            "procedure": procedure,
            "procedure_sha256": canonical_sha256(procedure),
            "task": {
                "spec": task_spec,
                "language": language,
                "directions": list(directions_for(language)),
                "train": {"raw": len(task.train), "used": len(train),
                          "removed_ids": removed, "items_sha256": items_sha256(task.train)},
                "dev": {"n": len(task.dev), "items_sha256": items_sha256(task.dev)},
                "tests": {
                    name: {"n": len(items), "items_sha256": items_sha256(items),
                           "variety": task.test_varieties[name]}
                    for name, items in task.tests.items()
                },
            },
            "normalization_guard": normalization_guard(task.raw_train_dev),
            "lengths": {
                "resolved": resolved,
                "worst_dev_fraction_over_default": worst,
                "audited_at_freeze": sorted(labels),
            },
            "code": code_provenance(),
            "created": _now(),
        }
        _write_json(existing, protocol)
        (study / "protocol.sha256").write_text(file_sha256(existing) + "\n", encoding="utf-8")
        protocol = load_protocol(study)
        logger.info("froze %s: %s, max length %d", study, procedure["version"], resolved)

    threshold = protocol["procedure"]["lengths"]["max_dev_fraction_over_default"]
    for label, checkpoint in labels.items():
        path = system_path(study, label)
        record = {
            "system": label,
            "checkpoint": checkpoint,
            "audit": audits[label],
            "dev_fraction_over_resolved": worst_dev_fraction(audits[label], resolved),
            "coverage_flag": any(
                side["unk_rate"] > protocol["procedure"]["coverage_flag_unk_rate"]
                for side in audits[label]["coverage"].values()
            ),
        }
        if record["dev_fraction_over_resolved"] > threshold:
            raise ValueError(
                f"{label}: {record['dev_fraction_over_resolved']:.2%} of development "
                f"examples exceed the frozen length {resolved}"
            )
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != record:
                raise ValueError(f"{label} is registered with a different audit")
            continue
        _write_json(path, record)
    return protocol


def load_protocol(study: Path) -> dict:
    """Read ``protocol.json`` and check it against its hash and the code (F4)."""
    path = protocol_path(study)
    if not path.exists():
        raise FileNotFoundError(f"{study}: no protocol.json; run eval mt-freeze first")
    recorded = (study / "protocol.sha256").read_text(encoding="utf-8").strip()
    if file_sha256(path) != recorded:
        raise ValueError(f"{path} was edited after it was frozen")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if canonical_sha256(protocol["procedure"]) != protocol["procedure_sha256"]:
        raise ValueError(f"{path}: procedure does not match its recorded hash")
    canonical = procedure_with_dropout(PROCEDURE, protocol["procedure"]["fine_tuning"].get("dropout_rate"))
    if not protocol["smoke"] and protocol["procedure_sha256"] != canonical_sha256(canonical):
        raise ValueError(
            f"{path} was frozen under procedure {protocol['procedure']['version']}, "
            f"which differs from the code's {PROTOCOL_VERSION}"
        )
    protocol["_sha256"] = recorded
    return protocol


def registered_system(study: Path, checkpoint: str) -> dict:
    path = system_path(study, run_label([checkpoint]))
    if not path.exists():
        raise ValueError(f"{checkpoint} is not registered in {study}; run eval mt-freeze")
    record = json.loads(path.read_text(encoding="utf-8"))
    record["_sha256"] = file_sha256(path)
    return record


def verified_task(protocol: dict, *, include_tests: bool) -> Task:
    """Reload the task and require the exact text frozen in the protocol (F4)."""
    recorded = protocol["task"]
    task = load_task(recorded["spec"], include_tests=include_tests)
    checks = [("train", task.train, recorded["train"]["items_sha256"]),
              ("dev", task.dev, recorded["dev"]["items_sha256"])]
    if include_tests:
        if sorted(task.tests) != sorted(recorded["tests"]):
            raise ValueError("Test sets differ from the frozen protocol")
        checks += [(name, items, recorded["tests"][name]["items_sha256"])
                   for name, items in task.tests.items()]
    for name, items, expected in checks:
        if items_sha256(items) != expected:
            raise ValueError(f"{name} text differs from the frozen protocol")
    return task


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def probe_config(procedure: dict, direction: str, learning_rate: float, seed: int,
                 max_length: int) -> ProbeConfig:
    tuning, decoding = procedure["fine_tuning"], procedure["decoding"]
    return ProbeConfig(
        direction=direction,
        max_steps=tuning["max_steps"],
        eval_steps=tuning["eval_steps"],
        early_stopping_patience=tuning["early_stopping_patience"],
        early_stopping_threshold=tuning["early_stopping_threshold"],
        learning_rate=learning_rate,
        optim=tuning["optim"],
        lr_scheduler_type=tuning["lr_scheduler_type"],
        per_device_train_batch_size=tuning["per_device_train_batch_size"],
        gradient_accumulation_steps=tuning["gradient_accumulation_steps"],
        max_source_length=max_length,
        max_target_length=max_length,
        generation_max_length=max_length,
        num_beams=decoding["num_beams"],
        generation_batch_size=decoding["generation_batch_size"],
        seed=seed,
        bf16=tuning["bf16"],
        pad_to_multiple_of=tuning["pad_to_multiple_of"],
        restore_best_dev_loss=True,
        dropout_rate=tuning.get("dropout_rate"),
    )


def run_dir(study: Path, direction: str, label: str, learning_rate: float, seed: int) -> Path:
    """One directory per system, direction, rate and seed (F6)."""
    return study / "runs" / direction / label / f"lr{learning_rate:.0e}_seed{seed}"


def model_rows(items: Sequence[Item], direction: str) -> list[dict]:
    """Prefixed, normalized fine-tuning rows, as :func:`finetune` expects."""
    from akanlm.corpora.normalize import normalize_for_language

    source_language, target_language = direction_languages(direction)
    prefix = task_prefix(direction)
    return [
        {"source": prefix + normalize_for_language(item.side(source_language), source_language),
         "target": normalize_for_language(item.side(target_language), target_language)}
        for item in items
    ]


def _decode_paths(run: Path, name: str) -> tuple[Path, Path]:
    return run / "decodes" / f"{name}.jsonl", run / "decodes" / f"{name}.json"


def decode_complete(run: Path, name: str, items: Sequence[Item]) -> dict | None:
    """The score record when both files exist and match the items (F7)."""
    predictions, scores = _decode_paths(run, name)
    if not predictions.exists() or not scores.exists():
        return None
    record = json.loads(scores.read_text(encoding="utf-8"))
    if (record.get("items_sha256") != items_sha256(items)
            or record.get("predictions_sha256") != file_sha256(predictions)):
        raise ValueError(f"{scores} does not match its predictions or the frozen items")
    return record


def decode_and_score(model, tokenizer, items: Sequence[Item], config: ProbeConfig,
                     run: Path, name: str, system: str, variety: str | None) -> dict:
    """Decode *items*, score them, and write every prediction with its id."""
    from akanlm.corpora.normalize import normalize_for_language
    from akanlm.eval.mt_probe import generate, score

    direction = config.direction
    source_language, target_language = direction_languages(direction)
    sources = [item.side(source_language) for item in items]
    references = [item.side(target_language) for item in items]
    hypotheses = generate(model, tokenizer, sources, config, normalize_outputs=False)
    result = score(hypotheses, references, system, name, direction, variety)
    at_cap = sum(
        len(ids) >= config.generation_max_length - 1
        for ids in tokenizer(result.raw_predictions, add_special_tokens=True)["input_ids"]
    )
    predictions, scores = _decode_paths(run, name)
    ensure_dir(predictions.parent)
    temporary = predictions.with_name(predictions.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for item, source, reference, prediction, raw in zip(
            items, sources, references, result.predictions, result.raw_predictions
        ):
            handle.write(json.dumps({
                "id": item.id,
                "source": normalize_for_language(source, source_language),
                "reference": normalize_for_language(reference, target_language),
                "prediction": prediction,
                "raw_prediction": raw,
            }, ensure_ascii=False) + "\n")
    temporary.replace(predictions)
    record = result.to_dict(keep_predictions=0)
    for key in ("sample_predictions", "sample_raw_predictions"):
        record.pop(key)
    record.update({
        "items_sha256": items_sha256(items),
        "predictions_sha256": file_sha256(predictions),
        "retokenized_predictions_at_length_cap": at_cap,
    })
    # The score file is written last, so its presence marks a complete decode.
    _write_json(scores, record)
    return record


def _trained_model(run: Path, checkpoint: str, train_rows, dev_rows, config: ProbeConfig):
    from akanlm.eval.mt_probe import finetune

    if (run / "training_summary.json").exists() and not (run / "final").exists():
        raise RuntimeError(f"{run}: weights were pruned, but a decode is still missing (F9)")
    resume = run.exists() and any(run.iterdir())
    return finetune(checkpoint, train_rows, dev_rows, config, run, resume=resume)


def _prune(run: Path, weights: bool) -> None:
    """Delete optimizer checkpoints, and the final weights when *weights*."""
    for checkpoint in run.glob("checkpoint-*"):
        shutil.rmtree(checkpoint)
    final = run / "final"
    if weights and final.exists():
        record = {path.name: file_sha256(path) for path in sorted(final.glob("*.safetensors"))}
        _write_json(run / "pruned_weights.json", {"files": record, "pruned": _now()})
        shutil.rmtree(final)


def run_decodes(run: Path, checkpoint: str, system: str, items_by_name: dict,
                varieties: dict, train_items, dev_items, config: ProbeConfig) -> dict:
    """Train if needed and decode every missing split; return all score records."""
    import gc

    records = {name: decode_complete(run, name, items) for name, items in items_by_name.items()}
    missing = [name for name, record in records.items() if record is None]
    if missing:
        model, tokenizer = _trained_model(
            run, checkpoint, model_rows(train_items, config.direction),
            model_rows(dev_items, config.direction), config,
        )
        for name in missing:
            records[name] = decode_and_score(
                model, tokenizer, items_by_name[name], config, run, name, system,
                varieties.get(name),
            )
        del model
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass
    _prune(run, weights=False)
    summary = json.loads((run / "training_summary.json").read_text(encoding="utf-8"))
    for record in records.values():
        record["training"] = {key: summary[key] for key in (
            "completed_steps", "stopping_reason", "restored_step", "restored_dev_loss",
            "reevaluated_dev_loss", "patience_reference_step",
        )}
    return records


def choose_learning_rate(candidates: Sequence[dict], grid: Sequence[float]) -> dict:
    """Highest development chrF++, then lower restored loss, then lower rate (F5)."""
    if sorted(c["learning_rate"] for c in candidates) != sorted(grid):
        raise ValueError("Selection requires exactly one run per grid learning rate")
    best = max(round(c["dev_chrf2pp"], 2) for c in candidates)
    tied = [c for c in candidates if round(c["dev_chrf2pp"], 2) == best]
    return min(tied, key=lambda c: (c["restored_dev_loss"], c["learning_rate"]))


def selection_path(study: Path, direction: str, label: str) -> Path:
    return study / "selection" / direction / f"{label}.json"


def select(study: Path, checkpoint: str, direction: str, keep_weights: bool = False) -> dict:
    """Learning-rate selection for one system and direction, on development data only."""
    protocol = load_protocol(study)
    system = registered_system(study, checkpoint)
    label = system["system"]
    if direction not in protocol["task"]["directions"]:
        raise ValueError(f"{direction} is not a direction of this study")
    task = verified_task(protocol, include_tests=False)
    procedure = protocol["procedure"]
    rule = procedure["learning_rate_selection"]
    train = training_items(task, protocol)
    candidates = []
    for learning_rate in rule["grid"]:
        config = probe_config(procedure, direction, learning_rate, rule["seed"],
                              protocol["lengths"]["resolved"])
        run = run_dir(study, direction, label, learning_rate, rule["seed"])
        dev = run_decodes(run, checkpoint, label, {"dev": task.dev}, {}, train, task.dev,
                          config)["dev"]
        candidates.append({
            "learning_rate": learning_rate,
            "run": str(run.relative_to(study)),
            "dev_chrf2pp": dev["chrf2pp"],
            "dev_bleu": dev["bleu"],
            "restored_dev_loss": dev["training"]["restored_dev_loss"],
            "restored_step": dev["training"]["restored_step"],
            "completed_steps": dev["training"]["completed_steps"],
            "stopping_reason": dev["training"]["stopping_reason"],
        })
    chosen = choose_learning_rate(candidates, rule["grid"])
    tuning = procedure["fine_tuning"]
    last_window = tuning["max_steps"] - tuning["early_stopping_patience"] * tuning["eval_steps"]
    flags = []
    if chosen["restored_step"] == tuning["eval_steps"]:
        flags.append("restored_at_first_check")
    if chosen["restored_step"] > last_window:
        flags.append("restored_in_last_patience_window")
    if chosen["learning_rate"] in (min(rule["grid"]), max(rule["grid"])):
        flags.append("selected_at_grid_edge")
    document = {
        "protocol_sha256": protocol["_sha256"],
        "procedure_sha256": protocol["procedure_sha256"],
        "system": label,
        "checkpoint": checkpoint,
        "system_record_sha256": system["_sha256"],
        "direction": direction,
        "candidates": candidates,
        "selected": chosen,
        "flags": flags,
        "created": _now(),
    }
    _write_json(selection_path(study, direction, label), document)
    for candidate in candidates:
        if candidate is not chosen:
            _prune(study / candidate["run"], weights=not keep_weights)
    logger.info("%s %s: selected learning rate %g (dev chrF++ %.2f); flags: %s",
                label, direction, chosen["learning_rate"], chosen["dev_chrf2pp"], flags or "none")
    return document


def receipt_path(study: Path, direction: str, label: str) -> Path:
    return study / "receipts" / direction / f"{label}.json"


def results_path(study: Path, direction: str, label: str) -> Path:
    return study / "results" / direction / f"{label}.json"


def test(study: Path, checkpoint: str, direction: str, keep_weights: bool = False) -> dict:
    """Seeded test runs at the selected rate, after writing the receipt (F4)."""
    protocol = load_protocol(study)
    system = registered_system(study, checkpoint)
    label = system["system"]
    selection_file = selection_path(study, direction, label)
    if not selection_file.exists():
        raise ValueError(f"{label} {direction}: run eval mt-select first")
    selection = json.loads(selection_file.read_text(encoding="utf-8"))
    if selection["protocol_sha256"] != protocol["_sha256"]:
        raise ValueError(f"{selection_file} was made under a different protocol")
    if selection["system_record_sha256"] != system["_sha256"]:
        raise ValueError(f"{selection_file} was made for a different system record")
    task = verified_task(protocol, include_tests=True)
    procedure = protocol["procedure"]
    if procedure["learning_rate_selection"]["seed"] not in procedure["seeds"]:
        raise ValueError("The selection seed must be one of the test seeds")
    learning_rate = selection["selected"]["learning_rate"]
    receipt = {
        "protocol_sha256": protocol["_sha256"],
        "procedure_sha256": protocol["procedure_sha256"],
        "selection_sha256": file_sha256(selection_file),
        "system": label,
        "checkpoint": checkpoint,
        "direction": direction,
        "learning_rate": learning_rate,
        "seeds": procedure["seeds"],
        "tests": {name: items_sha256(items) for name, items in task.tests.items()},
    }
    path = receipt_path(study, direction, label)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if {key: existing[key] for key in receipt} != receipt:
            raise ValueError(f"{path} records a different protocol, selection or test set")
    else:
        _write_json(path, {**receipt, "code": code_provenance(), "created": _now()})
    train = training_items(task, protocol)
    splits = {"dev": task.dev, **task.tests}
    per_seed = {}
    for seed in procedure["seeds"]:
        config = probe_config(procedure, direction, learning_rate, seed,
                              protocol["lengths"]["resolved"])
        run = run_dir(study, direction, label, learning_rate, seed)
        per_seed[seed] = run_decodes(run, checkpoint, label, splits, task.test_varieties,
                                     train, task.dev, config)
        _prune(run, weights=not keep_weights)
    summary = {}
    for name in splits:
        chrf = [per_seed[seed][name]["chrf2pp"] for seed in procedure["seeds"]]
        bleu = [per_seed[seed][name]["bleu"] for seed in procedure["seeds"]]
        summary[name] = {
            "n": per_seed[procedure["seeds"][0]][name]["n_examples"],
            "variety": task.test_varieties.get(name),
            "chrf2pp_mean": statistics.fmean(chrf),
            "chrf2pp_sd": statistics.stdev(chrf) if len(chrf) > 1 else 0.0,
            "bleu_mean": statistics.fmean(bleu),
            "bleu_sd": statistics.stdev(bleu) if len(bleu) > 1 else 0.0,
            "chrf2pp_by_seed": dict(zip(map(str, procedure["seeds"]), chrf)),
            "bleu_by_seed": dict(zip(map(str, procedure["seeds"]), bleu)),
        }
    document = {
        "receipt_sha256": file_sha256(path),
        "system": label,
        "checkpoint": checkpoint,
        "direction": direction,
        "learning_rate": learning_rate,
        "signatures": per_seed[procedure["seeds"][0]]["dev"]["signatures"],
        "scores": summary,
        "runs": {
            str(seed): {
                "run": str(run_dir(study, direction, label, learning_rate, seed).relative_to(study)),
                "training": records["dev"]["training"],
            }
            for seed, records in per_seed.items()
        },
        "created": _now(),
    }
    _write_json(results_path(study, direction, label), document)
    return document


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _predictions(run: Path, name: str) -> dict[str, dict]:
    path, _ = _decode_paths(run, name)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return {row["id"]: row for row in rows}


def paired_bootstrap(a_by_seed: Sequence[Sequence[str]], b_by_seed: Sequence[Sequence[str]],
                     references: Sequence[str], resamples: int, seed: int) -> dict:
    """Bootstrap the difference of seed-averaged corpus chrF++ over sentences.

    Each resample draws sentence indices with replacement and recomputes
    every seed's corpus chrF++ from per-sentence statistics, so the interval
    covers sentence sampling; seed variation is reported separately.
    """
    import numpy as np
    import sacrebleu

    metric = sacrebleu.CHRF(word_order=2)

    def statistics_of(hypotheses):
        return np.asarray(
            metric._extract_corpus_statistics(list(hypotheses), [list(references)]),
            dtype=np.int64,
        )

    def corpus(stats, index=None):
        rows = stats if index is None else stats[index]
        return metric._compute_score_from_stats([int(v) for v in rows.sum(axis=0)]).score

    a_stats = [statistics_of(h) for h in a_by_seed]
    b_stats = [statistics_of(h) for h in b_by_seed]
    observed = (statistics.fmean(corpus(s) for s in a_stats)
                - statistics.fmean(corpus(s) for s in b_stats))
    rng = np.random.default_rng(seed)
    n = len(references)
    differences = np.empty(resamples)
    for r in range(resamples):
        index = rng.integers(0, n, n)
        differences[r] = (statistics.fmean(corpus(s, index) for s in a_stats)
                          - statistics.fmean(corpus(s, index) for s in b_stats))
    low, high = np.percentile(differences, [2.5, 97.5])
    return {"difference": observed, "ci95": [float(low), float(high)],
            "fraction_at_or_below_zero": float((differences <= 0).mean()),
            "resamples": resamples, "seed": seed}


def _controls(checkpoint: str) -> tuple[str | None, str | None]:
    """``(initialization, matching mT5)`` for a CPT run, from its run card."""
    from akanlm.eval.baselines import MT5_BACKBONES

    path = Path(checkpoint)
    if not path.is_dir():
        return None, None
    for parent in (path, *path.parents):
        card = parent / "run_card.json"
        if card.is_file():
            config = json.loads((path / "config.json").read_text(encoding="utf-8"))
            shape = tuple(config.get(k) for k in ("d_model", "d_ff", "num_layers", "num_heads"))
            initial = json.loads(card.read_text(encoding="utf-8"))["config"]["base_checkpoint"]
            return initial, MT5_BACKBONES.get(shape)
    return None, None


def comparison_pairs(checkpoints: Sequence[str]) -> list[tuple[str, str, str]]:
    """``(kind, system, control)``: CPT vs its initialization, and CPT and the
    initialization vs the original mT5 with the same backbone."""
    present = {run_label([c]) for c in checkpoints}
    pairs = []
    for checkpoint in checkpoints:
        initial, backbone = _controls(checkpoint)
        if initial is None:
            continue
        cpt, init = run_label([checkpoint]), run_label([initial])
        mt5 = run_label([backbone]) if backbone else None
        for pair in (("cpt_vs_init", cpt, init), ("cpt_vs_mt5", cpt, mt5),
                     ("init_vs_mt5", init, mt5)):
            if pair[1] in present and pair[2] in present and pair not in pairs:
                pairs.append(pair)
    return pairs


def report(study: Path) -> dict:
    """Seed summaries and paired bootstrap comparisons from saved predictions."""
    protocol = load_protocol(study)
    bootstrap = protocol["procedure"]["uncertainty"]["paired_bootstrap"]
    rows, comparisons = [], []
    for direction in protocol["task"]["directions"]:
        results = {
            path.stem: json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((study / "results" / direction).glob("*.json"))
        }
        for label, result in results.items():
            for name, values in result["scores"].items():
                rows.append({"system": label, "direction": direction, "eval_set": name,
                             "learning_rate": result["learning_rate"], **{
                                 k: values[k] for k in ("n", "variety", "chrf2pp_mean",
                                                        "chrf2pp_sd", "bleu_mean", "bleu_sd")}})
        checkpoints = [result["checkpoint"] for result in results.values()]
        for kind, system, control in comparison_pairs(checkpoints):
            for name in protocol["task"]["tests"]:
                by_system = {}
                for label in (system, control):
                    result = results[label]
                    by_system[label] = [
                        _predictions(study / result["runs"][str(seed)]["run"], name)
                        for seed in protocol["procedure"]["seeds"]
                    ]
                ids = sorted(by_system[system][0])
                for runs in by_system.values():
                    for predictions in runs:
                        if sorted(predictions) != ids:
                            raise ValueError(f"{name}: prediction ids differ between runs")
                references = [by_system[system][0][i]["reference"] for i in ids]
                if any(by_system[control][0][i]["reference"] != r
                       for i, r in zip(ids, references)):
                    raise ValueError(f"{name}: references differ between systems")
                hypotheses = {
                    label: [[run[i]["prediction"] for i in ids] for run in runs]
                    for label, runs in by_system.items()
                }
                comparisons.append({
                    "kind": kind, "system": system, "control": control,
                    "direction": direction, "eval_set": name,
                    **paired_bootstrap(hypotheses[system], hypotheses[control], references,
                                       bootstrap["resamples"], bootstrap["seed"]),
                })
    document = {"protocol_sha256": protocol["_sha256"], "smoke": protocol["smoke"],
                "procedure_version": protocol["procedure"]["version"],
                "rows": rows, "comparisons": comparisons, "created": _now()}
    _write_json(study / "report.json", document)
    return document


def render_report(document: dict) -> str:
    lines = ["| system | direction | eval set | lr | chrF++ | BLEU |",
             "| --- | --- | --- | ---: | ---: | ---: |"]
    for row in document["rows"]:
        lines.append(
            f"| {row['system']} | {row['direction']} | {row['eval_set']} | "
            f"{row['learning_rate']:.0e} | {row['chrf2pp_mean']:.2f} ± {row['chrf2pp_sd']:.2f} | "
            f"{row['bleu_mean']:.2f} ± {row['bleu_sd']:.2f} |"
        )
    if document["comparisons"]:
        lines += ["", "| comparison | system | control | direction | eval set | Δ chrF++ | 95% CI |",
                  "| --- | --- | --- | --- | --- | ---: | --- |"]
        for c in document["comparisons"]:
            lines.append(
                f"| {c['kind']} | {c['system']} | {c['control']} | {c['direction']} | "
                f"{c['eval_set']} | {c['difference']:+.2f} | "
                f"[{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}] |"
            )
    return "\n".join(lines)
