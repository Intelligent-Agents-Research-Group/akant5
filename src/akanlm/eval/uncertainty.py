"""Repeated shared masks and bootstrap intervals clustered by source document."""

import hashlib
import json
from dataclasses import replace

import numpy as np

from akanlm.eval.lm import LN2, evaluate


def bootstrap_bpc(document_scores, samples=2000, seed=13):
    """Resample whole documents, retaining all their windows and mask repeats."""
    if samples < 1 or not document_scores:
        raise ValueError("Bootstrap requires positive samples and document scores")
    costs = np.array([d["total_nats"] for d in document_scores], dtype=float)
    chars = np.array([d["n_masked_chars"] for d in document_scores], dtype=float)
    if not np.isfinite(costs).all() or not np.isfinite(chars).all() or np.any(chars <= 0):
        raise ValueError("Document scores must be finite with positive masked counts")
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples)
    for i in range(samples):
        indices = rng.integers(0, len(costs), size=len(costs))
        estimates[i] = costs[indices].sum() / (LN2 * chars[indices].sum())
    lower, upper = np.quantile(estimates, [0.025, 0.975]).tolist()
    return {"method": "document_bootstrap_percentile", "confidence": 0.95,
            "samples": samples, "seed": seed, "lower": lower, "upper": upper,
            "n_documents": len(costs),
            "note": "Conditional on the selected mask seeds; masks of one document remain clustered."}


def evaluate_repeated(model, tokenizer, texts, dataset_name, model_name, *,
                      seeds, bootstrap_samples=2000, bootstrap_seed=13, **kwargs):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide nonempty, distinct mask seeds")
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be nonnegative")
    runs = [evaluate(model, tokenizer, texts, dataset_name, model_name,
                     seed=seed, document_scores=True, **kwargs) for seed in seeds]
    result = runs[0]
    if len(runs) > 1:
        config = {k: v for k, v in result.protocol_config.items() if k != "seed"}
        config["seeds"] = list(seeds)
        fingerprint = hashlib.sha256(json.dumps(
            [result.protocol, [r.task_sha256 for r in runs]], separators=(",", ":")
        ).encode()).hexdigest()
        documents = {}
        for run in runs:
            for row in run.document_scores:
                doc = documents.setdefault(row["document_index"], {
                    "document_index": row["document_index"], "total_nats": 0., "n_masked_chars": 0})
                doc["total_nats"] += row["total_nats"]
                doc["n_masked_chars"] += row["n_masked_chars"]
        totals = {key: sum(getattr(r, key) for r in runs) for key in (
            "n_chars", "n_masked_chars", "n_windows", "n_tokens", "n_target_tokens", "total_nats")}
        result = replace(result, **totals, task_sha256=fingerprint, protocol_config=config,
                         document_scores=list(documents.values()), mask_statistics={
                             key: sum(r.mask_statistics[key] for r in runs)
                             for key in result.mask_statistics})
    result.mask_runs = [{"seed": seed, "task_sha256": run.task_sha256,
                         "bits_per_char": run.bits_per_char,
                         "n_masked_chars": run.n_masked_chars} for seed, run in zip(seeds, runs)]
    if bootstrap_samples:
        result.uncertainty = bootstrap_bpc(result.document_scores, bootstrap_samples, bootstrap_seed)
    return result
