"""Frozen English character-span evaluation of raw CPT checkpoints.

Failure conditions are deliberate: changed texts, masks, scorer code, or weights
invalidate cached rows; mismatched tasks and token overflow abort comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from akanlm.audit.sources import load_sources
from akanlm.corpora.loaders import load_pairs, load_texts
from akanlm.data.cpt_splits import content_key
from akanlm.eval import corpus_bpc
from akanlm.eval.denoising import (
    load_checkpoint, release_memory, tokenizer_fingerprint, weights_identity,
)


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def english_corpora(corpora_dir: Path, max_docs: int | None = None):
    """Two pinned, CPT-held-out English sets; preserve document order."""
    sources = load_sources(corpora_dir)
    flores = sources["flores_plus_english"]
    mafand = sources["mafand_en_twi"]
    assert flores.role.value == "eval_only" and mafand.role.value == "eval_only"
    flores_texts = [row.text for row in load_texts(flores, limit=max_docs)]
    mafand_texts = [row.src_text for row in load_pairs(mafand, limit=max_docs)]
    items = [
        ("flores_plus_english", flores, flores_texts),
        ("mafand_test_english", mafand, mafand_texts),
    ]
    corpora = {}
    for label, source, texts in items:
        if not texts:
            raise ValueError(f"{label}: no English documents")
        raw_count = len(texts)
        if max_docs is None and raw_count != source.expected_rows:
            raise ValueError(f"{label}: expected {source.expected_rows} rows, found {raw_count}")
        seen = set()
        unique_texts = []
        for text in texts:
            key = content_key(text)
            if key not in seen:
                unique_texts.append(text)
                seen.add(key)
        corpora[(label, "test")] = (unique_texts, {
            "source": source.model_dump(mode="json"),
            "documents_sha256": _digest(unique_texts),
            "n_documents_raw": raw_count,
            "excluded_duplicate_rows": raw_count - len(unique_texts),
            "n_documents": len(unique_texts),
        })
    return corpora


def run(models: list[str], output_dir: Path, *, corpora_dir: Path, max_docs: int | None,
        batch_size: int) -> Path:
    if not models or len(models) != len(set(models)):
        raise ValueError("Provide distinct model checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = corpus_bpc.CharSpanConfig(max_chars=512)
    corpora = english_corpora(corpora_dir, max_docs)
    corpus_identity = [
        {"dataset": label, "split": split, **metadata}
        for (label, split), (_, metadata) in corpora.items()
    ]
    shared_identity = {
        "protocol": "english_retention_char_span_v1",
        "char_span_config": {**asdict(config), "seeds": list(config.seeds)},
        "corpora": corpus_identity,
        "implementation_sha256": {
            **corpus_bpc.implementation_sha256(),
            "english_retention.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    shared_hash = _digest(shared_identity)
    manifest_path = output_dir / "input_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != shared_identity:
            raise ValueError("English retention output has a different input manifest")
    else:
        _write_json(manifest_path, shared_identity)
    per_model = []
    for model_path in models:
        model_file = output_dir / f"{Path(model_path).name}-{_digest(model_path)[:10]}.json"
        identity = {
            "model": model_path,
            "weights": weights_identity(model_path),
            "shared_identity_sha256": shared_hash,
        }
        if model_file.exists():
            saved = json.loads(model_file.read_text())
            if saved["identity"] != identity:
                raise ValueError(f"Stale English retention cache: {model_file}")
            per_model.append(saved)
            print(f"[CACHED] {model_path}", flush=True)
            continue
        tokenizer, model = load_checkpoint(model_path)
        tokenizer_hash = tokenizer_fingerprint(tokenizer)
        rows = corpus_bpc.score_corpora(
            model, tokenizer, corpora, model_path, config=config, batch_size=batch_size
        )
        rows = [{
            "model": model_path,
            "weights": identity["weights"],
            "tokenizer_sha256": tokenizer_hash,
            "split_manifest_sha256": shared_hash,
            **row,
        } for row in rows]
        saved = json.loads(json.dumps({"identity": identity, "rows": rows}, ensure_ascii=False))
        _write_json(model_file, saved)
        per_model.append(saved)
        for row in rows:
            print(f"[SCORED] {model_path} {row['dataset']}: "
                  f"{row['result']['bits_per_char']:.4f} bits/masked-char", flush=True)
        del model
        release_memory()
    baseline = {row["dataset"]: row for row in per_model[0]["rows"]}
    comparisons = []
    for saved in per_model[1:]:
        for row in saved["rows"]:
            comparisons.append(corpus_bpc.compare(row, baseline[row["dataset"]]))
    summary = {
        "shared_identity_sha256": shared_hash,
        "baseline": models[0],
        "models": models,
        "scores": [
            {"model": row["model"], "dataset": row["dataset"],
             "bits_per_masked_char": row["result"]["bits_per_char"],
             "n_documents": row["result"]["n_docs"],
             "n_masked_chars": row["result"]["n_masked_chars"],
             "task_sha256": row["result"]["task_sha256"]}
            for saved in per_model for row in saved["rows"]
        ],
        "comparisons_against_initialization": comparisons,
        "interpretation": "Higher conditional bits per masked character means worse English span reconstruction; this is not full-text language-model perplexity or MT quality.",
    }
    summary_path = output_dir / "english_retention_summary.json"
    _write_json(summary_path, summary)
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", help="Initialization first, then raw CPT checkpoints")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--corpora-dir", type=Path, default=Path("configs/corpora"))
    parser.add_argument("--max-docs", type=int, default=None, help="Smoke only: first N documents per corpus")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.max_docs is not None and args.max_docs < 1:
        parser.error("--max-docs must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    print(run(args.models, args.output_dir, corpora_dir=args.corpora_dir,
              max_docs=args.max_docs, batch_size=args.batch_size), flush=True)


if __name__ == "__main__":
    main()
