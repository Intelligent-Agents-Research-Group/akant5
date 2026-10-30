"""Fit, serialize, and audit a tokenizer using only frozen training partitions.

Failure modes fixed before implementation: a held-out target or its parallel
English side can enter fitting; downstream evaluation text can enter fitting;
duplicate documents can consume quotas; filtering can change language quotas;
source declarations or downstream data can drift after split preparation;
and a caller can bypass the frozen split through a supplied iterator.
"""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from akanlm.data.cpt_splits import SplitConfig, partition, prepare
from akanlm.data.mixture import MixtureComponent, MixtureConfig
from akanlm.tokenizer.train_spm import TokenizerConfig, build_sentence_iterator, load, train


@pytest.fixture
def fitting_fixture(tmp_path, monkeypatch):
    corpora = tmp_path / "corpora"
    corpora.mkdir()
    split_config = SplitConfig(validation_fraction=0.2, test_fraction=0.2)
    rows = [{"twi": f"Anadwo biara ɔfrɛ no. Nnipa no kɔɔ asɔre mu nnɛ. Document {i}.",
             "en": f"English translation for numbered document {i}."} for i in range(1000)]
    train_rows = [r for r in rows if partition(r["twi"], split_config) == "train"]
    val = next(r for r in rows if partition(r["twi"], split_config) == "validation")
    test = next(r for r in rows if partition(r["twi"], split_config) == "test")
    # The first train-partition target is protected solely by its other side.
    collision = {**train_rows[0], "en": test["en"]}
    downstream_collision = {**train_rows[1], "en": "Reserved evaluation sentence."}
    # Do not add clean counterparts of those two deliberately rejected pairs.
    # A clean pair sharing only its train-partition target is valid under CPT.
    remaining_rows = [r for r in rows if r not in train_rows[:2]]
    pair_rows = [val, test, collision, downstream_collision,
                 train_rows[2], train_rows[2], *remaining_rows]
    english_rows = [{"text": f"The quick brown fox jumps over the lazy dog in document {i}."}
                    for i in range(1000)]
    declarations = {}
    for name, records, modality, language, role in (
        ("twi_pairs", pair_rows, "parallel", "ak", "train"),
        ("english", english_rows, "monolingual", "en", "train"),
        ("downstream", [{"text": "Reserved evaluation sentence."}], "monolingual", "en", "eval_only"),
    ):
        data = tmp_path / f"{name}.json"
        data.write_text(json.dumps(records))
        declaration = {"name": name, "local_path": str(data), "modality": modality,
                       "text_column": "twi" if modality == "parallel" else "text",
                       "language": language, "provenance": "human", "licence": "cc0",
                       "role": role}
        if modality == "parallel":
            declaration["source_column"] = "en"
        (corpora / f"{name}.yaml").write_text(yaml.safe_dump(declaration))
        declarations[name] = declaration
    monkeypatch.setattr("akanlm.corpora.loaders._open", lambda source, **_: (
        json.loads(Path(source.local_path).read_text())))
    mixture = MixtureConfig("fitting-fixture", components=[
        MixtureComponent("twi_pairs"), MixtureComponent("english")])
    split_path = tmp_path / "split.json"
    manifest = prepare(mixture, corpora, split_path, split_config)
    config = TokenizerConfig(
        name="fitting-fixture", vocab_size=384,
        language_config=str(Path("configs/languages/akan.json").resolve()),
        twi_sources=["twi_pairs"], english_sources=["english"],
        english_ratio=0.25, total_characters=20_000,
        source_weights={"twi_pairs": 1, "english": 1}, num_threads=1,
        require_split_manifest=True,
    )
    return config, corpora, split_path, manifest, declarations, (val, test, collision, downstream_collision)


def test_fit_and_reload_preserves_train_only_character_budget(fitting_fixture, tmp_path):
    config, corpora, split_path, manifest, _, excluded = fitting_fixture
    stream = build_sentence_iterator(config, corpora, split_manifest=split_path)
    pieces = list(stream)
    report = stream.report()
    assert report["completed"] and report["characters"] == 20_000
    assert report["english_character_share"] == 0.25
    assert report["source_budgets"] == {"twi_pairs": 15_000, "english": 5_000}
    assert not any(row["twi"] in piece for row in excluded for piece in pieces)
    provenance = report["training_partition"]
    assert provenance["split_manifest_sha256"] == manifest["sha256"]
    assert provenance["fit_split"] == "train"
    counts = provenance["sources"]["twi_pairs"]["counts"]
    assert counts["partition_or_heldout_removed"] >= 3
    assert counts["downstream_overlap_removed"] >= 1
    assert counts["exact_duplicate_removed"] >= 1
    expected_hash = hashlib.sha256("".join(p + "\n" for p in pieces).encode()).hexdigest()
    assert report["input_sha256"] == expected_hash

    output = tmp_path / "trained"
    train(config, output, corpora_dir=corpora, split_manifest=split_path)
    saved = json.loads((output / "sampling.json").read_text())
    assert saved["input_sha256"] == expected_hash
    assert saved["training_partition"] == provenance
    tokenizer = load(output)
    assert tokenizer.pad_token_id == 0 and tokenizer.eos_token_id == 1
    assert tokenizer.convert_tokens_to_ids("<extra_id_0>") == len(tokenizer) - 1
    assert tokenizer.encode("Anadwo biara ɔfrɛ no.", add_special_tokens=False)
    artifact = {"sampling": saved, "tokenizer_len": len(tokenizer),
                "split_manifest_sha256": manifest["sha256"], "verified": True}
    (output / "training_partition_check.json").write_text(json.dumps(artifact, indent=2))


def test_fitting_rejects_drift_and_manifest_bypass(fitting_fixture, tmp_path):
    config, corpora, split_path, _, declarations, _ = fitting_fixture
    with pytest.raises(ValueError, match="split manifest"):
        build_sentence_iterator(config, corpora)
    with pytest.raises(ValueError, match="supplied iterator"):
        train(config, tmp_path / "bypass", sentence_iterator=["unprotected text"],
              corpora_dir=corpora, split_manifest=split_path)
    with pytest.raises(ValueError, match="absent.*split|split.*absent"):
        build_sentence_iterator(replace(config, twi_sources=["not_in_manifest"],
            source_weights={"not_in_manifest": 1, "english": 1}), corpora, split_manifest=split_path)
    changed = {**declarations["twi_pairs"], "notes": "changed after splitting"}
    (corpora / "twi_pairs.yaml").write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError, match="source changed"):
        build_sentence_iterator(config, corpora, split_manifest=split_path)
    (corpora / "twi_pairs.yaml").write_text(yaml.safe_dump(declarations["twi_pairs"]))
    Path(declarations["downstream"]["local_path"]).write_text(json.dumps([
        {"text": "Changed reserved evaluation sentence."}]))
    with pytest.raises(ValueError, match="Downstream exclusion sources changed"):
        build_sentence_iterator(config, corpora, split_manifest=split_path)
