"""Offline CPT training, export, evaluation and resume workflows with tiny models."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from akanlm.data.cpt_splits import SplitConfig, content_key, partition, prepare
from akanlm.data.mixture import MixtureComponent, MixtureConfig
from akanlm.eval.denoising import score
from akanlm.training.cpt import TrainingConfig


@pytest.fixture
def split_fixture(tmp_path, monkeypatch):
    from akanlm.audit.sources import load_sources
    from akanlm.corpora.loaders import PairRecord
    source = load_sources(Path("configs/corpora"))["pristine_twi_english"]
    name = source.name
    rows = [PairRecord(name, i, f"English document {i}.", f"Twi document {i}.", {})
            for i in range(200)]
    config = SplitConfig(validation_fraction=0.2, test_fraction=0.2)
    # Distinct targets sharing an English translation must not cross val/test.
    val = next(r for r in rows if partition(r.tgt_text, config) == "validation")
    test = next(r for r in rows if partition(r.tgt_text, config) == "test")
    rows[rows.index(val)] = replace(val, src_text=test.src_text)
    monkeypatch.setattr("akanlm.audit.sources.load_sources", lambda _: {name: source})
    monkeypatch.setattr("akanlm.data.mixture.load_sources", lambda _: {name: source})
    monkeypatch.setattr("akanlm.training.validation.heldout_documents", lambda _: ([], {}))
    monkeypatch.setattr("akanlm.corpora.loaders.load_pairs", lambda s, **kw: iter(rows[:kw.get("limit")]))
    mixture = MixtureConfig("fixture", components=[
        MixtureComponent(name, repeat=3), MixtureComponent(name, as_pairs=True, limit=100)])
    path = tmp_path / "split.json"
    manifest = prepare(mixture, Path("unused"), path, config)
    return path, manifest, config, rows, mixture


def test_reference_trainer_trains_resumes_and_scores_exported_model(split_fixture, tiny_tokenizer, tmp_path, monkeypatch):
    from datasets import Dataset
    from transformers import MT5Config, MT5ForConditionalGeneration

    from akanlm.data.cpt_splits import build_validation
    from akanlm.training.cpt import TrainingConfig, train

    split_path, manifest, _, rows, mixture = split_fixture
    initial = tmp_path / "initial"
    model_config = MT5Config(vocab_size=len(tiny_tokenizer), d_model=16,
        d_ff=32, d_kv=4, num_heads=2, num_layers=1, num_decoder_layers=1, dropout_rate=0.,
        decoder_start_token_id=0)
    model_config.tie_word_embeddings = False
    model = MT5ForConditionalGeneration(model_config)
    model.encoder.embed_tokens = model.shared
    model.decoder.embed_tokens = model.shared
    model.save_pretrained(initial)
    tiny_tokenizer.save_pretrained(initial)
    mixture_dir = tmp_path / "mixtures"
    mixture_dir.mkdir()
    (mixture_dir / "fixture.yaml").write_text("fixture")
    def dataset(config, tokenizer, *args, protected, **kwargs):
        training = [r for r in rows if protected.accept(r.source, r.tgt_text, r.src_text)]
        heldout = set(manifest["heldout_exclusion_keys"])
        assert training and all(content_key(t) not in heldout for r in training for t in (r.tgt_text, r.src_text))
        tokens = tokenizer("Anadwo biara ɔfrɛ no.", add_special_tokens=False)["input_ids"]
        raw_length = config.resolved_lengths()[0]
        data = Dataset.from_list([{"input_ids": (tokens * raw_length)[:raw_length]} for _ in range(4)])
        summary = {"components": [{"source": rows[0].source}], "preparation": protected.report()}
        return data, summary, None
    monkeypatch.setattr("akanlm.training.cpt.build_dataset", dataset)
    spec = TrainingConfig("tiny", str(initial), mixture=mixture.name, max_steps=2,
        per_device_train_batch_size=1, gradient_accumulation_steps=1,
        per_device_eval_batch_size=2, gradient_checkpointing=False, bf16=False,
        dataloader_num_workers=0, logging_steps=1, eval_steps=1, save_steps=1,
        milestone_steps=[], early_stopping_patience=0)
    output = tmp_path / "trained"
    card = train(spec, output, corpora_dir=Path("unused"), mixtures_dir=mixture_dir, split_manifest=split_path)
    assert card.metrics["completed_steps"] == 2
    assert card.framework["implementation"] == "pytorch_huggingface_mesh_semantic_port"
    assert card.framework["world_size"] == 1
    assert card.framework["parameter_dtype"] == "float32"
    assert card.framework["numerical_parity_claimed"] is False
    assert card.tokens_per_step == spec.effective_batch_size * spec.input_length
    assert card.raw_tokens_per_step == spec.effective_batch_size * spec.resolved_lengths()[0]
    assert "eval_cpt_loss" in card.metrics and "best_twi_loss" not in card.metrics
    assert card.metrics["exported_model"] == "last_step"
    restored = MT5ForConditionalGeneration.from_pretrained(output / "final")
    datasets, _, _ = build_validation(spec, tiny_tokenizer, Path("unused"), split_path)
    values = [score(restored, d, device="cpu") for d in datasets.values()]
    expected = sum(v["total_nats"] for v in values) / sum(v["num_eval_tokens"] for v in values)
    assert card.metrics["eval_cpt_loss"] == pytest.approx(expected, abs=1e-6)
    resumed = train(replace(spec, max_steps=3), output, corpora_dir=Path("unused"),
                    mixtures_dir=mixture_dir, split_manifest=split_path, resume_from_checkpoint=True)
    assert resumed.metrics["completed_steps"] == 3


@pytest.mark.parametrize("optimization", [{}, {
    "lr_scheduler_type": "rsqrt_no_ramp_down", "warmup_steps": 1,
    "adafactor_scale_parameter": True, "z_loss": 1e-4, "dropout_rate": 0.,
}])
def test_real_training_stops_restores_best_and_saves_provenance(tmp_path, monkeypatch, tiny_tokenizer, optimization):
    import torch
    from datasets import Dataset
    from safetensors.torch import load_file
    from transformers import MT5Config, MT5ForConditionalGeneration

    from akanlm.data.collator import DataCollatorForT5MLM
    from akanlm.training.cpt import train

    initial = tmp_path / "init"
    mc = MT5Config(vocab_size=len(tiny_tokenizer), d_model=16, d_ff=32, d_kv=4,
                   num_heads=4, num_layers=1, num_decoder_layers=1,
                   dropout_rate=0, decoder_start_token_id=0)
    mc.tie_word_embeddings = False
    model = MT5ForConditionalGeneration(mc)
    model.encoder.embed_tokens = model.shared
    model.decoder.embed_tokens = model.shared
    model.save_pretrained(initial)
    tiny_tokenizer.save_pretrained(initial)
    config = TrainingConfig("tiny", str(initial), input_length=64, max_steps=5,
                            per_device_train_batch_size=2, gradient_accumulation_steps=1,
                            per_device_eval_batch_size=2, bf16=False,
                            gradient_checkpointing=False, dataloader_num_workers=0,
                            eval_steps=1, save_steps=1, logging_steps=1, warmup_steps=0,
                            lr_scheduler_type="constant_with_warmup",
                            save_total_limit=1, milestone_steps="doubling",
                            early_stopping_patience=1, early_stopping_threshold=100)
    config = replace(config, **optimization)
    raw, target = config.resolved_lengths()
    rows = [{"input_ids": [5 + i % 20 for i in range(raw)]} for _ in range(8)]
    collator = DataCollatorForT5MLM(tiny_tokenizer, input_length=64, target_length=target)
    fixed = Dataset.from_list([
        {k: v[0].tolist() for k, v in collator([row]).items()} for row in rows[:2]
    ])
    names = ("pristine_twi_english_monolingual", "twi_wikipedia_monolingual")
    monkeypatch.setattr("akanlm.data.cpt_splits.build_validation", lambda *a: (
        {name: fixed for name in names}, None, {"sha256": "fixed"}))
    monkeypatch.setattr("akanlm.training.cpt.build_dataset", lambda *a, **k: (
        Dataset.from_list(rows), {"components": [], "preparation": {}}, None))
    out = tmp_path / "run"
    split = tmp_path / "split.json"
    card = train(config, out, split_manifest=split)
    assert card.metrics["completed_steps"] == 2
    assert {f"eval_{name}_loss" for name in names} <= card.metrics.keys()
    assert card.metrics["exported_model"] == "best_cpt_loss"
    assert card.metrics["best_cpt_loss"] == pytest.approx(min(
        entry["eval_cpt_loss"] for entry in json.loads((out / "trainer_state.json").read_text())["log_history"]
        if "eval_cpt_loss" in entry))
    best = Path(card.metrics["best_loss_checkpoint"])
    for name, tensor in load_file(str(best / "model.safetensors")).items():
        torch.testing.assert_close(tensor, load_file(str(out / "final/model.safetensors"))[name])
    assert (out / "milestones/checkpoint-1/tokenizer.json").exists()
    assert (out / "milestones/checkpoint-2/model.safetensors").exists()
    assert card.milestone_steps == [1, 2, 4, 5]
    assert (out / "training_plan.json").exists()
    state = json.loads((out / "trainer_state.json").read_text())
    assert "CPTEarlyStopping" in state["stateful_callbacks"]
    resumed = train(replace(config, max_steps=6), out, resume_from_checkpoint=True, split_manifest=split)
    assert resumed.metrics["completed_steps"] == 3  # patience is restored, not reset
    assert resumed.milestone_steps == [1, 2, 4, 6]
