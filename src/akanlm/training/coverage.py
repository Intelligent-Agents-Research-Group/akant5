"""Strict source coverage checks for a full-mixture CPT run.

Pilot comparisons may intentionally evaluate a common subset of their training
sources. These checks are opt-in for a full run, whose frozen validation and
test splits must instead cover every source and consumption mode.
"""

from collections import Counter
from dataclasses import asdict

from akanlm.data.mixture import MixtureConfig


def validate_full_source_coverage(
    mixture: MixtureConfig, manifest: dict, native_validation: dict | None = None,
) -> dict:
    """Verify a full-mixture split and optionally its native validation report.

    ``manifest`` must already have passed ``read_manifest`` hash validation.
    Count records directly without copying their text or rebuilding tokenized
    tasks. Training counts are candidates before the final exclusion filter;
    the packing report remains the authority on retained training data.
    """
    if not mixture.components:
        raise ValueError("Full-source coverage requires a nonempty mixture")
    expected_components = [asdict(component) for component in mixture.components]
    frozen = manifest["mixture"]
    expected_replay = asdict(mixture.replay) if mixture.replay is not None else None
    if frozen.get("replay") != expected_replay:
        raise ValueError("Full-source coverage requires the split's frozen replay policy")
    if (frozen.get("components") != expected_components
            or frozen.get("seed") != mixture.seed
            or frozen.get("name") != mixture.name):
        raise ValueError("Full-source coverage requires the split's frozen mixture recipe")
    expected_sources = {component.source for component in mixture.components}
    if set(manifest["sources"]) != expected_sources:
        raise ValueError("Full-source split declarations do not match the mixture sources")

    # Mirror evaluation_corpora: one dataset per source/mode, using its first
    # component's cap when a recipe repeats the same source/mode.
    modes = {}
    by_source = {source: [] for source in expected_sources}
    for component in mixture.components:
        label = component.source + ("_parallel" if component.as_pairs else "_monolingual")
        if label not in modes:
            modes[label] = component
            by_source[component.source].append(label)
    source_counts = {source: Counter() for source in expected_sources}
    mode_counts = {label: Counter() for label in modes}
    for record in manifest["records"]:
        source, split = record["source"], record["split"]
        if source not in source_counts or split not in ("validation", "test"):
            raise ValueError("Full-source split contains an undeclared source or non-held-out record")
        if not isinstance(record["target"], str) or not record["target"].strip():
            raise ValueError(f"{source}: empty held-out target text")
        source_counts[source][split] += 1
        for label in by_source[source]:
            component = modes[label]
            if component.limit is not None and record["position"] >= component.limit:
                continue
            if component.limit_chars is not None and record["chars_before"] >= component.limit_chars:
                continue
            if component.as_pairs and (not isinstance(record["other"], str) or not record["other"].strip()):
                raise ValueError(f"{label}: missing held-out parallel text")
            mode_counts[label][split] += 1

    sources = {}
    for source in sorted(expected_sources):
        description = manifest["sources"][source]
        counts = source_counts[source]
        if any(counts[split] < 1 for split in ("validation", "test")):
            raise ValueError(f"{source}: full-source coverage needs validation and test documents")
        if description.get("heldout_documents") != dict(counts):
            raise ValueError(f"{source}: held-out source counts disagree with manifest records")
        train_candidates = description["counts"].get("train", 0)
        if train_candidates < 1:
            raise ValueError(f"{source}: no training document candidates")
        sources[source] = {"train_candidates": train_candidates,
                           "validation_documents": counts["validation"],
                           "test_documents": counts["test"]}

    corpora = {}
    for label, component in modes.items():
        counts = mode_counts[label]
        if any(counts[split] < 1 for split in ("validation", "test")):
            raise ValueError(f"{label}: component caps leave empty validation or test coverage")
        multiplier = 2 if component.as_pairs and component.both_directions else 1
        corpora[label] = {"source": component.source, "as_pairs": component.as_pairs,
                          "validation_documents": counts["validation"],
                          "validation_texts": multiplier * counts["validation"],
                          "test_documents": counts["test"],
                          "test_texts": multiplier * counts["test"]}

    if native_validation is not None:
        if native_validation.get("split_manifest_sha256") != manifest["sha256"]:
            raise ValueError("Native validation is bound to a different split manifest")
        if (native_validation.get("selection_split") != "validation"
                or native_validation.get("selection_metric") != "eval_cpt_loss"):
            raise ValueError("Full-source checkpoint selection must use validation eval_cpt_loss")
        if set(native_validation["sources"]) != set(corpora):
            raise ValueError("Native validation does not cover every mixture source/mode")
        for label, expected in corpora.items():
            actual = native_validation["sources"][label]
            if (actual["source"]["name"] != expected["source"]
                    or actual["as_pairs"] != expected["as_pairs"]
                    or actual["n_documents"] != expected["validation_documents"]
                    or actual["n_texts"] != expected["validation_texts"]):
                raise ValueError(f"{label}: native validation corpus disagrees with the split")
            if actual["n_examples"] < 1 or not actual.get("tasks_sha256"):
                raise ValueError(f"{label}: native validation contains no frozen tasks")
            expected["validation_examples"] = actual["n_examples"]
            expected["validation_tasks_sha256"] = actual["tasks_sha256"]

    return {"mixture": mixture.name, "split_manifest_sha256": manifest["sha256"],
            "n_sources": len(sources), "n_modes": len(corpora),
            "native_validation_checked": native_validation is not None,
            "sources": sources, "corpora": corpora}


def validate_character_tokenization(manifest: dict, tokenizer, char_config=None) -> dict:
    """Exercise the real character scorer's encoding guards on CPU.

    Select the same documents and character masks as ``score_corpora``. A
    stand-in receives the validated batches, avoiding any model-weight load.
    Its zero loss is only an implementation aid: no model metrics are returned
    or written. The caller must verify full-mixture coverage separately.
    """
    from types import SimpleNamespace

    import torch

    from akanlm.data.cpt_splits import evaluation_corpora
    from akanlm.eval.corpus_bpc import DEFAULT_CHAR_SPAN_CONFIG, select_documents
    from akanlm.eval.denoising import tokenizer_fingerprint
    from akanlm.eval.lm import evaluate

    class EncodingOnlyModel:
        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, **kwargs):
            return SimpleNamespace(loss=torch.tensor(0.0))

    spec = DEFAULT_CHAR_SPAN_CONFIG if char_config is None else char_config
    print("[PREFLIGHT] Encoding-only validation on CPU; dummy zero losses below are not model scores.",
          flush=True)
    tasks = []
    corpora = evaluation_corpora(manifest, "validation")
    if not corpora:
        raise ValueError("Character tokenization preflight requires validation corpora")
    model = EncodingOnlyModel()
    for (label, _), (texts, _) in corpora.items():
        selected = select_documents(texts, spec)
        for seed in spec.seeds:
            result = evaluate(
                model, tokenizer, selected, label, "encoding-only preflight (dummy loss)",
                device="cpu", document_scores=False, input_length=spec.token_length_limit,
                max_chars=spec.max_chars, noise_density=spec.noise_density,
                mean_span_chars=spec.mean_span_chars, seed=seed,
            )
            tasks.append({"dataset": label, "seed": seed, "task_sha256": result.task_sha256,
                          "n_docs": result.n_docs, "n_windows": result.n_windows,
                          "n_chars": result.n_chars, "n_masked_chars": result.n_masked_chars})
    return {"protocol": "character_tokenization_preflight_v1", "split": "validation",
            "split_manifest_sha256": manifest["sha256"],
            "tokenizer_sha256": tokenizer_fingerprint(tokenizer),
            "character_config": asdict(spec), "n_corpora": len(corpora), "tasks": tasks}
