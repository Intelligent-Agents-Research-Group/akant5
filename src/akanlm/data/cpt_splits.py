"""Document-level CPT holdouts, fixed before tokenization and repetition."""

import hashlib
import json
import logging
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from akanlm.utils.paths import digest

# v3: inspect every parallel side for exclusion before exact deduplication.
SPLIT_PROTOCOL = "cpt_document_split_v3"


@dataclass(frozen=True)
class SplitConfig:
    seed: int = 20260912
    validation_fraction: float = 0.1
    test_fraction: float = 0.1

    def __post_init__(self):
        if not (0 < self.validation_fraction < 1 and 0 < self.test_fraction < 1
                and self.validation_fraction + self.test_fraction < 1):
            raise ValueError("CPT split fractions must leave nonempty train, validation and test shares")


def content_key(text):
    canonical = " ".join(unicodedata.normalize("NFC", text).casefold().split())
    return hashlib.sha256(canonical.encode()).hexdigest()


DEFAULT_SPLIT_CONFIG = SplitConfig()


def partition(text, config=DEFAULT_SPLIT_CONFIG):
    # No source, mixture, model, tokenizer, record order or repetition in this key.
    # Identical Twi in monolingual and parallel modes always stays in one split.
    key = hashlib.sha256(f"{config.seed}:{content_key(text)}".encode()).digest()
    value = int.from_bytes(key[:8], "big") / 2**64
    if value < config.test_fraction:
        return "test"
    if value < config.test_fraction + config.validation_fraction:
        return "validation"
    return "train"


def prepare(mixture, corpora_dir, output, config=DEFAULT_SPLIT_CONFIG):
    """Prepare a frozen CPT split manifest for held-out source records.

    The manifest is built from component order using the same
    :class:`~akanlm.data.mixture.ExactDuplicates` policy as training, so each
    text is attributed to the same source consistently. Parallel translations
    remain paired with their Twi record. Validation examples that share either
    language verbatim with test records are dropped, and training excludes any
    occurrence of either language from held-out records. Evaluation records are
    not repeated or oversampled, and downstream test sets are exclusion-only.

    Notes:
        - Every parallel side is added to the exclusion set before duplicate
          targets are removed, preventing an alternate translation from
          escaping the manifest.
        - The opposing language is retained for evaluation only when a
          component consumes that source as a parallel pair.
    """
    from akanlm.audit.sources import Modality, assert_trainable, load_sources
    from akanlm.corpora.loaders import load_pairs, load_texts
    from akanlm.data.mixture import ExactDuplicates, derive_source_caps, validate
    from akanlm.training.validation import ProtectedDocuments, heldout_documents

    output = Path(output)
    if mixture.blocked_on:
        raise ValueError(f"Mixture {mixture.name} is blocked: {mixture.blocked_on}")
    if not mixture.components:
        raise ValueError("No CPT sources: M0 is an unadapted evaluation baseline, not a training mixture")
    if output.exists():
        raise FileExistsError(f"CPT split manifest already exists: {output}")
    validate(mixture, corpora_dir)
    sources = load_sources(corpora_dir)
    caps = derive_source_caps([mixture])
    assert_trainable(sources[name] for name in caps)
    downstream, downstream_provenance = heldout_documents(sources)
    guard = ProtectedDocuments(downstream)
    records, summary, exclusion_keys, test_keys = [], {}, set(), set()
    duplicates = ExactDuplicates()
    for name, cap in caps.items():  # component order, as in training
        logging.getLogger(__name__).info("Preparing CPT document splits: %s", name)
        source = sources[name]
        stream = any(c.streaming for c in mixture.components if c.source == name)
        loader = load_pairs if source.modality is Modality.PARALLEL else load_texts
        counts, checksum = Counter(), hashlib.sha256()
        chars = 0
        for record in loader(source, limit=cap.rows, max_chars=cap.chars, streaming=stream):
            target = record.tgt_text if source.modality is Modality.PARALLEL else record.text
            other = record.src_text if source.modality is Modality.PARALLEL else None
            texts = [target] + ([other] if other else [])
            checksum.update(json.dumps([record.index, *texts], ensure_ascii=False).encode() + b"\n")
            counts["read"] += 1
            chars_before = chars
            chars += len(target)
            if not guard.accept(name, *texts):
                counts["downstream_overlap_removed"] += 1
                continue
            group = content_key(target)
            split = partition(target, config)
            # Protect both sides of every held-out occurrence even when this
            # target is an exact duplicate that will not become an eval row.
            if split != "train":
                exclusion_keys.update(content_key(t) for t in texts)
            if split == "test":
                test_keys.update(content_key(t) for t in texts)
            if not duplicates.accept(name, record.index, target):
                counts["exact_duplicate_removed"] += 1
                continue
            counts[split] += 1
            if split == "train":
                continue
            records.append({"source": name, "index": record.index, "group": group,
                            "position": counts["read"] - 1, "chars_before": chars_before,
                            "split": split, "target": target, "other": other})
        summary[name] = {"declaration": source.model_dump(mode="json"),
                         "cap": cap._asdict(), "counts": dict(counts),
                         "consumed_documents_sha256": checksum.hexdigest()}
        logging.getLogger(__name__).info("CPT source %s: %s", name, dict(counts))
    filtered = [r for r in records if r["split"] == "test" or not any(
        content_key(t) in test_keys for t in (r["target"], r["other"]) if t)]
    paired = {c.source for c in mixture.components if c.as_pairs}
    for record in filtered:
        if record["source"] not in paired:
            record["other"] = None
    for name, description in summary.items():
        counts = Counter(r["split"] for r in filtered if r["source"] == name)
        if not all(counts[s] for s in ("validation", "test")):
            raise ValueError(f"{name}: insufficient unique documents for validation and test; "
                             "choose and freeze larger split fractions before training")
        description["heldout_documents"] = dict(counts)
    manifest = {"protocol": SPLIT_PROTOCOL, "config": asdict(config),
                "mixture": asdict(mixture), "sources": summary,
                "downstream_exclusion": downstream_provenance,
                "exact_duplicates": duplicates.report(),
                "validation_test_duplicates_removed": len(records) - len(filtered),
                "heldout_exclusion_keys": sorted(exclusion_keys),
                "records": filtered,
                "scope": "Held out from CPT weight updates; existing tokenizer exposure must be reported."}
    manifest["sha256"] = digest(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False))
    temporary.replace(output)
    return manifest


def read_manifest(path):
    manifest = json.loads(Path(path).read_text())
    if manifest.get("protocol") != SPLIT_PROTOCOL:
        raise ValueError(f"CPT split manifest protocol {manifest.get('protocol')!r} is not "
                         f"{SPLIT_PROTOCOL}; prepare a new split")
    expected = manifest.pop("sha256")
    if digest(manifest) != expected:
        raise ValueError("CPT split manifest content hash mismatch")
    manifest["sha256"] = expected
    return manifest


class TrainingFilter:
    def __init__(self, manifest, downstream):
        self.config = SplitConfig(**manifest["config"])
        # Keep alternate translations of duplicate targets protected too, even
        # when only one target record is retained for evaluation.
        self.keys = set(manifest["heldout_exclusion_keys"])
        self.downstream = downstream
        self.seen, self.removed = Counter(), Counter()

    def accept(self, source, target, *others):
        return self.record(source, self.check(target, *others))

    def check(self, target, *others):
        """Decide without counting, so worker processes can call it.

        Returns ``None`` when the record is outside the train partition or matches
        a held-out key, since the downstream guard is then never consulted.
        Otherwise it returns the downstream guard's own outcome.
        """
        if (partition(target, self.config) != "train"
                or any(content_key(t) in self.keys for t in (target, *others))):
            return None
        return self.downstream.check(target, *others)

    def record(self, source, outcome):
        """Count one :meth:`check` outcome, as ``accept`` would have, and return whether it passed."""
        self.seen[source] += 1
        if outcome is not None:
            self.downstream.record(source, outcome)
        if not outcome:
            self.removed[source] += 1
        return bool(outcome)

    def report(self):
        return {"records_seen": dict(self.seen), "records_removed": dict(self.removed),
                "split_protocol": SPLIT_PROTOCOL, "config": asdict(self.config),
                "downstream": self.downstream.report()}


def evaluation_corpora(manifest, split):
    """One corpus per declared source/consumption mode, without repeat factors."""
    if split not in ("validation", "test"):
        raise ValueError("Evaluation split must be validation or test")
    from types import SimpleNamespace

    from akanlm.data.mixture import _pair_examples

    corpora = {}
    for component in manifest["mixture"]["components"]:
        name = component["source"]
        label = name + ("_parallel" if component["as_pairs"] else "_monolingual")
        if (label, split) in corpora:
            continue
        texts, source_records = [], 0
        for record in manifest["records"]:
            if record["source"] != name or record["split"] != split:
                continue
            if component["limit"] is not None and record["position"] >= component["limit"]:
                continue
            if component["limit_chars"] is not None and record["chars_before"] >= component["limit_chars"]:
                continue
            source_records += 1
            if component["as_pairs"]:
                texts.extend(_pair_examples(SimpleNamespace(tgt_text=record["target"],
                    src_text=record["other"]), component["both_directions"]))
            else:
                texts.append(record["target"])
        corpora[(label, split)] = (texts, {
            "source": manifest["sources"][name]["declaration"],
            "documents_sha256": digest(texts), "n_documents": source_records, "n_texts": len(texts),
            "as_pairs": component["as_pairs"]})
    return corpora


def build_validation(config, tokenizer, corpora_dir, manifest_path):
    from datasets import Dataset

    from akanlm.audit.sources import load_sources
    from akanlm.eval.denoising import (
        DenoisingConfig,
        build_tasks,
        tokenizer_fingerprint,
    )
    from akanlm.training.validation import ProtectedDocuments, heldout_documents

    manifest = read_manifest(manifest_path)
    sources = load_sources(corpora_dir)
    for name, description in manifest["sources"].items():
        if sources[name].model_dump(mode="json") != description["declaration"]:
            raise ValueError(f"CPT source changed since the split was frozen: {name}")
    downstream, provenance = heldout_documents(sources)
    if provenance != manifest["downstream_exclusion"]:
        raise ValueError("Downstream exclusion sources changed since CPT split preparation")
    guard = TrainingFilter(manifest, ProtectedDocuments(downstream))
    spec = DenoisingConfig.from_training(config)
    datasets, reports = {}, {}
    for (label, _), (texts, corpus) in evaluation_corpora(manifest, "validation").items():
        rows = build_tasks(texts, tokenizer, spec)
        datasets[label] = Dataset.from_list(rows)
        reports[label] = {**corpus, "tasks_sha256": digest(rows), "n_examples": len(rows)}
    report = {"protocol": SPLIT_PROTOCOL, "split_manifest_sha256": manifest["sha256"],
              "split_manifest": str(manifest_path), "evaluation_config": asdict(spec),
              "tokenizer_sha256": tokenizer_fingerprint(tokenizer), "sources": reports,
              "selection_metric": "eval_cpt_loss", "selection_split": "validation",
              "aggregation": "total target nats / total nonpadding target tokens across source/mode corpora"}
    report["sha256"] = digest(report)
    return datasets, guard, report
