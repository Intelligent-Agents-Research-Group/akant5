"""Select tokenizer-fitting text under a frozen CPT document split."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path

from akanlm.tokenizer.sampling import CharacterBudgetIterator, character_quotas, text_pieces
from akanlm.utils.paths import digest, file_sha256


class TrainingPartitionIterator(CharacterBudgetIterator):
    """Filter and deduplicate in source order, then interleave exact quotas.

    Only the source prefix needed to fill each quota is read. Selected pieces
    are staged on disk so duplicate ownership follows declared source order,
    independently of the randomized interleaving passed to SentencePiece.
    """

    def __init__(self, config, sources, manifest, manifest_path, guard):
        budgets = character_quotas(config)
        languages = {name: "twi" for name in config.twi_sources}
        languages.update({name: "english" for name in config.english_sources})
        super().__init__({}, budgets, languages, config.seed, config.max_sentence_length)
        self.config, self.sources, self.manifest, self.guard = config, sources, manifest, guard
        self.provenance = {
            "protocol": "tokenizer_frozen_cpt_train_v1", "fit_split": "train",
            "split_manifest": str(manifest_path), "split_manifest_sha256": manifest["sha256"],
            "split_manifest_file_sha256": file_sha256(manifest_path),
            "tokenizer_config_sha256": digest(config.to_dict()),
            "source_order": list(budgets),
            "source_scan": "prefix through the record supplying the last selected character",
            "deduplication": "first accepted (source, index) in declared source order",
            "parallel_policy": "partition by target; protect both sides; fit target only",
            "english_ratio_policy": "fixed tokenizer-fitting share, independent of CPT replay share",
            "sources": {},
        }

    def __iter__(self):
        if self.started:
            raise RuntimeError("Character-budget stream can only be consumed once")
        from akanlm.data.mixture import ExactDuplicates

        duplicates = ExactDuplicates()
        # At the main 200M-character budget these files require at most the
        # UTF-8 bytes of selected text plus line separators, never the full pool.
        with tempfile.TemporaryDirectory(prefix="akanlm-tokenizer-", dir="/tmp") as directory:
            paths = {}
            for position, (name, budget) in enumerate(self.budgets.items()):
                path = Path(directory) / f"{position}.txt"
                self._stage_source(name, budget, path, duplicates)
                paths[name] = path
            self.provenance["exact_duplicates"] = duplicates.report()
            self.provenance["filtering"] = self.guard.report()
            self.streams = {name: self._read_pieces(path) for name, path in paths.items()}
            try:
                yield from super().__iter__()
            finally:
                for stream in self.streams.values():
                    stream.close()

    @staticmethod
    def _read_pieces(path):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                yield line.removesuffix("\n")

    def _stage_source(self, name, budget, path, duplicates):
        from akanlm.audit.sources import Modality
        from akanlm.corpora.loaders import load_pairs, load_texts
        from akanlm.corpora.normalize import normalization_spec

        source = self.sources[name]
        frozen = self.manifest["sources"][name]
        counts = Counter()
        input_hash, selected_hash = hashlib.sha256(), hashlib.sha256()
        selected_characters = 0
        cap = frozen["cap"]
        streaming = any(c["streaming"] for c in self.manifest["mixture"]["components"]
                        if c["source"] == name)
        loader = load_pairs if source.modality is Modality.PARALLEL else load_texts
        records = iter(loader(source, limit=cap["rows"], max_chars=cap["chars"], streaming=streaming))
        try:
            with path.open("w", encoding="utf-8") as handle:
                if budget:
                    for record in records:
                        parallel = source.modality is Modality.PARALLEL
                        target = record.tgt_text if parallel else record.text
                        texts = [target, record.src_text] if parallel else [target]
                        counts["records_read"] += 1
                        input_hash.update(json.dumps([record.index, *texts], ensure_ascii=False).encode() + b"\n")
                        outcome = self.guard.check(*texts)
                        if not self.guard.record(name, outcome):
                            counts["partition_or_heldout_removed" if outcome is None
                                   else "downstream_overlap_removed"] += 1
                            continue
                        if not duplicates.accept(name, record.index, target):
                            counts["exact_duplicate_removed"] += 1
                            continue
                        counts["records_selected"] += 1
                        for piece in text_pieces(target, self.max_bytes):
                            piece = piece[:budget - selected_characters]
                            if not piece.strip():
                                raise ValueError(f"Empty SentencePiece input while filling {name}'s quota")
                            handle.write(piece + "\n")
                            selected_hash.update(piece.encode() + b"\n")
                            selected_characters += len(piece)
                            counts["pieces_selected"] += 1
                            if selected_characters == budget:
                                break
                        if selected_characters == budget:
                            break
        finally:
            if hasattr(records, "close"):
                records.close()
        if selected_characters != budget:
            raise ValueError(f"{name} exhausted with {budget - selected_characters:,} characters still needed; "
                             "refusing to change the language/source balance")
        self.provenance["sources"][name] = {
            "declaration": source.model_dump(mode="json"), "frozen_cap": cap,
            "normalization": normalization_spec(source.effective_rules()),
            "counts": dict(counts), "selected_characters": selected_characters,
            "consumed_prefix_documents_sha256": input_hash.hexdigest(),
            "selected_pieces_sha256": selected_hash.hexdigest(),
            "frozen_consumed_documents_sha256": frozen["consumed_documents_sha256"],
        }

    def report(self):
        return {**super().report(), "training_partition": self.provenance}


def build_training_partition_iterator(config, corpora_dir, manifest_path):
    """Validate provenance before reading any training source text."""
    from akanlm.audit.sources import assert_trainable, load_sources
    from akanlm.data.cpt_splits import TrainingFilter, read_manifest
    from akanlm.training.validation import ProtectedDocuments, heldout_documents

    # Validates quotas and duplicate source names before any corpus is opened.
    budgets = character_quotas(config)
    manifest = read_manifest(manifest_path)
    sources = load_sources(corpora_dir)
    missing = set(budgets) - manifest["sources"].keys()
    if missing:
        raise ValueError(f"Tokenizer fitting sources absent from frozen CPT split: {sorted(missing)}")
    for name in budgets:
        if name not in sources or sources[name].model_dump(mode="json") != manifest["sources"][name]["declaration"]:
            raise ValueError(f"CPT source changed since the split was frozen: {name}")
    assert_trainable(sources[name] for name in budgets)
    downstream, provenance = heldout_documents(sources)
    if provenance != manifest["downstream_exclusion"]:
        raise ValueError("Downstream exclusion sources changed since CPT split preparation")
    guard = TrainingFilter(manifest, ProtectedDocuments(downstream))
    return TrainingPartitionIterator(config, sources, manifest, manifest_path, guard)
