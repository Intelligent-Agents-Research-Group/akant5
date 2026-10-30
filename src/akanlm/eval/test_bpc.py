"""Score frozen CPT test BPC with validation weights, corpora, masks, and scorer.

Verify initialization/final provenance and share nontruncating length guards
across runs with the same tokenizer. Preserve existing results; recheck requires
equal values. Use CPT's launch directory and original run-path form.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import resource
import time
from dataclasses import asdict, replace
from pathlib import Path

VALIDATION = {"initialization": "initialization_char_bpc.json", "final": "final_char_bpc.json"}
# Fields that must be equal when a recheck rescores an existing test result.
RECHECKED = ("bits_per_char", "total_nats", "n_masked_chars", "n_windows", "n_docs", "task_sha256")
SCOPE = ("Frozen CPT test documents, capped whole-document selection in manifest order. "
         "Held out from CPT updates; prior tokenizer/backbone exposure is not excluded.")


def _load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def destination(run: Path, checkpoint: str, recheck: bool = False) -> Path:
    path = run / "evaluation" / VALIDATION[checkpoint].replace("_char_bpc", "_test_char_bpc")
    return path.with_suffix(".recheck.json") if recheck else path


def reference_identity(run: Path, checkpoint: str, *, ignore_scorer_version: bool = False):
    """Verify validation weight identity.

    ``ignore_scorer_version`` permits scorer changes and records both hashes.
    """
    from akanlm.eval.corpus_bpc import CharSpanConfig, implementation_sha256
    from akanlm.eval.denoising import weights_identity
    from akanlm.training.experiment import as_json

    validation = _load_json(run / "evaluation" / VALIDATION[checkpoint])
    model = validation["identity"]["model"]
    weights = weights_identity(model)
    if checkpoint == "final":
        if model != (run / "final").as_posix():
            raise ValueError(f"Validation scored {model}, not this run's final model")
        if weights != {"model.safetensors": _load_json(run / "completed.json")["final_weights_sha256"]}:
            raise ValueError("Final weights differ from completed.json")
    elif model != _load_json(run / "training_plan.json")["config"]["base_checkpoint"]:
        raise ValueError(f"Validation scored {model}, not this run's base checkpoint")
    plan = _load_json(run / "full_run_plan.json")
    spec = CharSpanConfig(**plan["character_evaluation"])
    identity = as_json({"model": model, "weights": weights,
                        "split_manifest_sha256": validation["identity"]["split_manifest_sha256"],
                        "config": asdict(spec), "implementation_sha256": implementation_sha256()})
    recorded = validation["identity"]
    if ignore_scorer_version:
        identity["validation_implementation_sha256"] = recorded["implementation_sha256"]
        recorded = {**recorded, "implementation_sha256": identity["implementation_sha256"]}
        compared = {k: v for k, v in identity.items() if k != "validation_implementation_sha256"}
    else:
        compared = identity
    if compared != recorded:
        raise ValueError("Weights, split, settings or scorer differ from the validation record")
    if len(spec.seeds) != 1:
        raise ValueError("The test protocol uses the single frozen mask seed")
    return validation, plan, spec, identity


def length_overrides(corpora, tokenizer, spec):
    """Longest task per corpus, matching lm.py encoding with EOS.
    Encode unprefixed sentinels directly; insert stock/extended mT5's
    word-prefixed sentinel IDs between tokenized spans.
    """
    import re

    from akanlm.eval.corpus_bpc import select_documents
    from akanlm.eval.lm import build_text_tasks

    sentinels, prefixed = [], False
    for i in range(100):
        token_id = tokenizer.convert_tokens_to_ids(f"<extra_id_{i}>")
        if token_id is None or token_id == tokenizer.unk_token_id:
            token_id = tokenizer.convert_tokens_to_ids(f"▁<extra_id_{i}>")
            prefixed = True
        sentinels.append(token_id)
    if None in sentinels or tokenizer.unk_token_id in sentinels or len(set(sentinels)) != 100:
        raise ValueError("Length preflight requires 100 distinct sentinel tokens")

    def encoded_lengths(texts):
        if not prefixed:
            return [len(ids) + 1 for ids in tokenizer(texts, add_special_tokens=False,
                                                       truncation=False)["input_ids"]]
        lengths = []
        for text in texts:
            parts = [part for part in re.split(r"(<extra_id_\d+>)", text) if part]
            plain = [part for part in parts if not re.fullmatch(r"<extra_id_\d+>", part)]
            pieces = tokenizer(plain, add_special_tokens=False)["input_ids"] if plain else []
            lengths.append(len(parts) - len(plain) + sum(len(ids) for ids in pieces) + 1)
        return lengths

    lengths, overrides = {}, {}
    for (label, _), (texts, _) in corpora.items():
        tasks = build_text_tasks(select_documents(texts, spec), max_chars=spec.max_chars,
                                 noise_density=spec.noise_density,
                                 mean_span_chars=spec.mean_span_chars, seed=spec.seeds[0])
        longest = {side: max(encoded_lengths([getattr(t, side) for t in tasks]))
                   for side in ("source", "target")}
        lengths[label] = {"windows": len(tasks), "max_encoder_tokens": longest["source"],
                          "max_target_tokens": longest["target"]}
        needed = max(longest.values())
        if needed > spec.token_length_limit:
            overrides[label] = {"token_length_limit": needed}
    return lengths, overrides


def reference_overrides(fingerprint: str, run: Path, exclude: Path):
    """Find length overrides for the same tokenizer in sibling run results."""
    for path in sorted(run.parent.glob("*/evaluation/*_test_char_bpc.json")):
        if path == exclude:
            continue
        record = _load_json(path)
        if any(row.get("tokenizer_sha256") == fingerprint for row in record.get("rows", [])):
            return path.as_posix(), record["identity"]["corpus_execution_overrides"]
    return None, None


def recheck_rows(rows, existing):
    reference = {row["dataset"]: row for row in existing["rows"]}
    checks = []
    for row in rows:
        old = reference[row["dataset"]]
        fields = {k: row["result"][k] == old["result"][k] for k in RECHECKED}
        fields["config"] = row["config"] == old["config"]
        checks.append({"dataset": row["dataset"], "equal": fields})
    return checks, all(all(c["equal"].values()) for c in checks) and len(rows) == len(reference)


def score_test_split(run: Path, checkpoint: str = "final", *, recheck: bool = False,
                     preflight: bool = False, ignore_scorer_version: bool = False,
                     output: Path | None = None, batch_size: int = 4) -> Path:
    """Score the test split and return its result path.

    ``output`` sets the result path; ``recheck`` requires equal values.
    """
    if checkpoint not in VALIDATION:
        raise ValueError(f"checkpoint must be one of {sorted(VALIDATION)}")
    run = Path(run)
    started = time.monotonic()
    validation, plan, spec, identity = reference_identity(
        run, checkpoint, ignore_scorer_version=ignore_scorer_version)
    existing = destination(run, checkpoint)
    if recheck and not existing.exists():
        raise FileNotFoundError(f"recheck needs an existing test result: {existing}")
    target = Path(output) if output is not None else destination(run, checkpoint, recheck)
    if target.exists():
        raise FileExistsError(target)
    if preflight:
        return target

    import torch
    import transformers

    from akanlm.data.cpt_splits import evaluation_corpora, read_manifest
    from akanlm.eval import corpus_bpc
    from akanlm.eval.denoising import load_checkpoint, tokenizer_fingerprint
    from akanlm.training.experiment import as_json
    from akanlm.utils.paths import write_json

    if not torch.cuda.is_available():
        raise RuntimeError("Test-split BPC requires a GPU")
    manifest = read_manifest(plan["split_manifest"])
    if manifest["sha256"] != identity["split_manifest_sha256"]:
        raise ValueError("Split manifest differs from the validation record")
    corpora = evaluation_corpora(manifest, "test")
    del manifest
    gc.collect()
    if {label for label, _ in corpora} != {row["dataset"] for row in validation["rows"]}:
        raise ValueError("Test corpus coverage differs from validation")

    tokenizer, model = load_checkpoint(identity["model"])
    fingerprint = tokenizer_fingerprint(tokenizer)
    if any(row["tokenizer_sha256"] != fingerprint for row in validation["rows"]):
        raise ValueError("Tokenizer differs from the validation record")
    lengths, overrides = length_overrides(corpora, tokenizer, spec)
    reference, expected_overrides = reference_overrides(fingerprint, run, exclude=existing)
    if reference is not None and overrides != expected_overrides:
        raise ValueError(f"Guard overrides {overrides} differ from {reference}: {expected_overrides}")
    identity = {**identity, "split": "test", "driver": "akanlm.eval.test_bpc",
                "driver_sha256": _file_sha256(Path(__file__)),
                "corpus_execution_overrides": overrides, "overrides_reference": reference}
    partial = target.with_suffix(".partial.json")
    rows = _load_json(partial)["rows"] if partial.exists() else []
    if rows and _load_json(partial)["identity"] != identity:
        raise ValueError(f"Stale partial result: {partial}")

    model.to("cuda").eval()
    completed = {row["dataset"] for row in rows}
    for key, value in corpora.items():
        label = key[0]
        if label in completed:
            continue
        corpus_spec = replace(spec, **overrides.get(label, {}))
        corpus_started = time.monotonic()
        scored = corpus_bpc.score_corpora(model, tokenizer, {key: value}, identity["model"],
                                          config=corpus_spec, batch_size=batch_size)[0]
        result = scored["result"]
        if (not math.isfinite(result["bits_per_char"]) or result["n_masked_chars"] <= 0
                or result["n_docs"] != scored["n_texts_scored"]):
            raise ValueError(f"Invalid likelihood or counts for {label}")
        rows.append(as_json({"model": identity["model"], "weights": identity["weights"],
                             "tokenizer_sha256": fingerprint,
                             "split_manifest_sha256": identity["split_manifest_sha256"],
                             **scored, "scoring_seconds": time.monotonic() - corpus_started}))
        write_json(partial, {"identity": identity, "rows": rows})
    report = {"identity": identity, "rows": rows, "length_preflight": lengths, "scope": SCOPE,
              "runtime": {"elapsed_seconds": time.monotonic() - started,
                          "scoring_seconds": sum(row["scoring_seconds"] for row in rows),
                          "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                          "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                          "gpu": torch.cuda.get_device_name(), "model_dtype": "float32",
                          "batch_size": batch_size, "torch": torch.__version__,
                          "transformers": transformers.__version__}}
    passed = True
    if recheck:
        report["recheck"], passed = recheck_rows(rows, _load_json(existing))
        report["recheck_passed"] = passed
    write_json(target, report)
    partial.unlink()
    if not passed:
        raise ValueError(f"Recheck found differences; see {target}")
    return target
