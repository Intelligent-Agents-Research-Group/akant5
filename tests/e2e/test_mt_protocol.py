"""The frozen best-dev MT protocol, end to end, with tiny offline models.

The failure modes these tests target are listed in the
:mod:`akanlm.eval.mt_protocol` docstring (F1–F10). Each test ends with a study
or run directory whose files are re-verified against their recorded hashes.
"""

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from akanlm.eval import mt_protocol
from akanlm.eval.mt_probe import ProbeConfig, finetune
from akanlm.utils.paths import run_label


def _tiny_model(path, tokenizer, seed=0, dropout_rate=0.0):
    import torch
    from transformers import MT5Config, MT5ForConditionalGeneration

    torch.manual_seed(seed)
    config = MT5Config(
        vocab_size=len(tokenizer), d_model=16, d_ff=32, d_kv=4,
        num_heads=4, num_layers=1, num_decoder_layers=1,
        dropout_rate=dropout_rate, decoder_start_token_id=0,
    )
    config.tie_word_embeddings = False
    model = MT5ForConditionalGeneration(config)
    # Mirror the released mT5 architecture: shared input embeddings and an
    # independent output head, even when the library's tying flag is false.
    model.encoder.embed_tokens = model.shared
    model.decoder.embed_tokens = model.shared
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


def _fingerprint(model) -> str:
    return hashlib.sha256(model.lm_head.weight.detach().cpu().numpy().tobytes()).hexdigest()


@pytest.mark.parametrize("interrupted", [False, True])
def test_restores_first_lowest_dev_loss_checkpoint(
    tmp_path, monkeypatch, tiny_tokenizer, interrupted
):
    """F1, F2: the scored weights are the strict development-loss minimum.

    Step 3 improves on step 2 by less than the patience threshold, so the
    patience reference stays at step 2, training stops at step 5, and the
    restored weights must be step 3's. The mocked loss is keyed by a weight
    fingerprint, so re-evaluating the restored model returns step 3's loss
    only if the step-3 weights were really restored.
    """
    import torch
    from transformers import Seq2SeqTrainer

    initial = _tiny_model(tmp_path / "initial", tiny_tokenizer)
    schedule = {1: 3.0, 2: 2.0, 3: 1.95, 4: 1.99}
    loss_by_weights: dict[str, float] = {}
    saved: dict[int, dict] = {}
    state = {"interrupt": interrupted}

    original_init = Seq2SeqTrainer.__init__
    original_evaluate = Seq2SeqTrainer.evaluation_loop
    original_save = Seq2SeqTrainer._save_checkpoint

    def init(self, **kwargs):
        kwargs["args"].use_cpu = True
        original_init(self, **kwargs)

    def evaluation_loop(self, *args, **kwargs):
        if state["interrupt"] and self.state.global_step == 4:
            state["interrupt"] = False
            raise RuntimeError("simulated interruption")
        output = original_evaluate(self, *args, **kwargs)
        key = _fingerprint(self.model)
        loss_by_weights.setdefault(key, schedule.get(self.state.global_step, 2.5))
        output.metrics[f"{kwargs.get('metric_key_prefix', 'eval')}_loss"] = loss_by_weights[key]
        return output

    def save_checkpoint(self, model, trial):
        saved[self.state.global_step] = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }
        return original_save(self, model, trial)

    monkeypatch.setattr(Seq2SeqTrainer, "__init__", init)
    monkeypatch.setattr(Seq2SeqTrainer, "evaluation_loop", evaluation_loop)
    monkeypatch.setattr(Seq2SeqTrainer, "_save_checkpoint", save_checkpoint)
    config = ProbeConfig(
        max_steps=10, eval_steps=1, early_stopping_patience=3,
        early_stopping_threshold=0.1, bf16=False, restore_best_dev_loss=True,
        per_device_train_batch_size=2, gradient_accumulation_steps=1,
    )
    rows = [{"source": "She calls him.", "target": "Anadwo biara ɔfrɛ no."}] * 4
    output_dir = tmp_path / "probe"
    if interrupted:
        with pytest.raises(RuntimeError, match="simulated interruption"):
            finetune(str(initial), rows, rows, config, output_dir)
        # Only new-best steps are saved, so the resume point is step 3.
        assert sorted(p.name for p in output_dir.glob("checkpoint-*")) == ["checkpoint-3"]
    model, _ = finetune(str(initial), rows, rows, config, output_dir, resume=interrupted)

    summary = json.loads((output_dir / "training_summary.json").read_text())
    assert summary["scored_weights"] == "best_dev_loss"
    assert summary["completed_steps"] == 5
    assert summary["stopping_reason"] == "dev_loss_patience"
    assert summary["restored_step"] == 3
    assert summary["patience_reference_step"] == 2
    assert summary["reevaluated_dev_loss"] == pytest.approx(1.95)
    assert [step for step, _ in summary["dev_loss_history"]] == [1, 2, 3, 4, 5]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, saved[3][name], rtol=0, atol=0)
    assert not torch.equal(model.lm_head.weight, saved[2]["lm_head.weight"])
    assert (output_dir / "final" / "model.safetensors").exists()


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def _pairs(english, other, prefix):
    return [{"id": f"{prefix}-{i}", "translation": {"en": e, "kpe": o}}
            for i, (e, o) in enumerate(zip(english, other))]


def _smoke_procedure():
    procedure = copy.deepcopy(mt_protocol.PROCEDURE)
    procedure["version"] = "smoke"
    procedure["fine_tuning"].update({
        "max_steps": 4, "eval_steps": 1, "early_stopping_patience": 2,
        "per_device_train_batch_size": 2, "gradient_accumulation_steps": 1, "bf16": False,
    })
    procedure["learning_rate_selection"]["grid"] = [1e-3, 3e-3]
    procedure["seeds"] = [13, 14]
    procedure["lengths"].update({"default": 128, "fallback": 256})
    procedure["decoding"] = {"num_beams": 1, "generation_batch_size": 4}
    procedure["uncertainty"]["paired_bootstrap"]["resamples"] = 50
    return procedure


@pytest.mark.parametrize("dropout_rate", [None, 0.1])
def test_freeze_select_test_report(
    tmp_path, monkeypatch, tiny_tokenizer, twi_sentences, english_sentences, dropout_rate
):
    """F3–F10 on a local Kpelle-coded task: no code path is language-specific.

    Ends with a study directory (protocol, receipts, per-seed predictions with
    ids, report) whose hashes the test recomputes.
    """
    import torch
    from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainer

    original_init = Seq2SeqTrainer.__init__
    original_train = Seq2SeqTrainer.train
    observed_dropout = []
    parameter_updates = []

    def init(self, **kwargs):
        kwargs["args"].use_cpu = True
        model = kwargs["model"]
        actual = model.config.dropout_rate
        assert actual == (dropout_rate if dropout_rate is not None else actual)
        assert all(module.p == actual for module in model.modules()
                   if isinstance(module, torch.nn.Dropout))
        observed_dropout.append(actual)
        original_init(self, **kwargs)

    def training(self, *args, **kwargs):
        before = _fingerprint(self.model)
        result = original_train(self, *args, **kwargs)
        parameter_updates.append(before != _fingerprint(self.model))
        assert parameter_updates[-1]
        return result

    monkeypatch.setattr(Seq2SeqTrainer, "__init__", init)
    monkeypatch.setattr(Seq2SeqTrainer, "train", training)

    # Distinct sides, so only the planted duplicate is deduplicated.
    english = [f"{english_sentences[i % len(english_sentences)]} {i}" for i in range(20)]
    other = [f"{twi_sentences[i % len(twi_sentences)]} {i}" for i in range(20)]
    data = tmp_path / "data"
    data.mkdir()
    train = _pairs(english[:12], other[:12], "train")
    dev = _pairs(english[12:14], other[12:14], "dev")
    # One training pair repeats a development English side, and one Kpelle
    # side carries a decomposed tone mark that NFC composes (F8 guard).
    train.append({"id": "train-dup", "translation": {"en": english[12], "kpe": "Ɓ\u025b\u0301l\u025b"}})
    train[0]["translation"]["kpe"] = "e\u0301 k\u025b"
    _write_jsonl(data / "train.jsonl", train)
    _write_jsonl(data / "dev.jsonl", dev)
    _write_jsonl(data / "test_news.jsonl", _pairs(english[14:17], other[14:17], "news"))
    _write_jsonl(data / "test_bible.jsonl", _pairs(english[17:20], other[17:20], "bible"))

    systems = [str(_tiny_model(tmp_path / name, tiny_tokenizer, seed, seed * 0.2))
               for seed, name in enumerate(["sysA", "sysB"])]
    source_hashes = {path: mt_protocol.file_sha256(path)
                     for system in systems
                     for path in (Path(system) / "config.json", Path(system) / "model.safetensors")}
    study = tmp_path / "study"
    spec = mt_protocol.local_task_spec("kpe", data)
    original_procedure = copy.deepcopy(mt_protocol.PROCEDURE)
    canonical_study = tmp_path / "canonical-study"
    canonical = mt_protocol.freeze(canonical_study, spec, systems, dropout_rate=dropout_rate)
    assert canonical["smoke"] is False
    assert mt_protocol.load_protocol(canonical_study)["procedure"] == canonical["procedure"]
    assert mt_protocol.PROCEDURE == original_procedure
    for invalid in (-0.1, 1.0, float("nan")):
        with pytest.raises(ValueError, match="dropout_rate"):
            mt_protocol.freeze(tmp_path / "invalid", spec, systems, dropout_rate=invalid)
        assert not (tmp_path / "invalid").exists()
    protocol = mt_protocol.freeze(study, spec, systems, procedure=_smoke_procedure(),
                                  dropout_rate=dropout_rate)
    assert protocol["procedure"]["fine_tuning"].get("dropout_rate") == dropout_rate
    if dropout_rate is None:
        assert "dropout_rate" not in protocol["procedure"]["fine_tuning"]
    else:
        assert protocol["procedure"]["version"].endswith("_dropout_override")
    frozen_before = (study / "protocol.json").read_bytes()
    with pytest.raises(ValueError, match="different task or procedure"):
        mt_protocol.freeze(study, spec, systems, procedure=_smoke_procedure(), dropout_rate=0.3)
    assert (study / "protocol.json").read_bytes() == frozen_before
    assert protocol["smoke"] is True
    assert protocol["task"]["directions"] == ["en2kpe", "kpe2en"]
    assert protocol["task"]["train"]["removed_ids"] == ["train-dup"]
    lengths = protocol["procedure"]["lengths"]
    worst = max(protocol["lengths"]["worst_dev_fraction_over_default"].values())
    assert protocol["lengths"]["resolved"] == (
        lengths["fallback"] if worst > lengths["max_dev_fraction_over_default"]
        else lengths["default"]
    )
    label_a = run_label([systems[0]])
    record = json.loads((study / "systems" / f"{label_a}.json").read_text())
    assert set(record["audit"]["coverage"]) == {"en", "kpe"}
    # Refreezing with the same inputs is a no-op; a different task is refused.
    mt_protocol.freeze(study, spec, systems, procedure=_smoke_procedure(), dropout_rate=dropout_rate)
    with pytest.raises(ValueError, match="different task or procedure"):
        mt_protocol.freeze(study, {**spec, "limit": 3}, systems, procedure=_smoke_procedure(),
                           dropout_rate=dropout_rate)

    with pytest.raises(ValueError, match="run eval mt-select first"):
        mt_protocol.test(study, systems[0], "en2kpe")

    # F3: selection must work with the test files absent.
    hidden = tmp_path / "hidden"
    hidden.mkdir()
    for path in data.glob("test_*.jsonl"):
        shutil.move(path, hidden / path.name)
    selections = {s: mt_protocol.select(study, s, "en2kpe") for s in systems}
    for path in hidden.iterdir():
        shutil.move(path, data / path.name)

    grid = protocol["procedure"]["learning_rate_selection"]["grid"]
    for selection in selections.values():
        assert len(selection["candidates"]) == len(grid)
        assert selection["selected"] == mt_protocol.choose_learning_rate(
            selection["candidates"], grid
        )
        for candidate in selection["candidates"]:
            run = study / candidate["run"]
            assert not list(run.glob("checkpoint-*"))
            chosen = candidate is selection["selected"] or candidate == selection["selected"]
            assert (run / "final").exists() == chosen
            assert (run / "pruned_weights.json").exists() != chosen
            assert not (run / "decodes" / "news.json").exists()

    results = {s: mt_protocol.test(study, s, "en2kpe", keep_weights=dropout_rate is not None)
               for s in systems}
    seeds = protocol["procedure"]["seeds"]
    for system, result in results.items():
        label = result["system"]
        receipt = json.loads((study / "receipts" / "en2kpe" / f"{label}.json").read_text())
        assert receipt["protocol_sha256"] == (study / "protocol.sha256").read_text().strip()
        assert receipt["selection_sha256"] == mt_protocol.file_sha256(
            study / "selection" / "en2kpe" / f"{label}.json"
        )
        assert set(result["scores"]) == {"dev", "news", "bible"}
        histories = []
        for seed in map(str, seeds):
            run = study / result["runs"][seed]["run"]
            assert (run / "final").exists() == (dropout_rate is not None)
            assert not list(run.glob("checkpoint-*"))
            if dropout_rate is not None:
                exported = AutoModelForSeq2SeqLM.from_pretrained(run / "final")
                assert exported.config.dropout_rate == dropout_rate
                assert exported.encoder.embed_tokens.weight is exported.shared.weight
                assert exported.decoder.embed_tokens.weight is exported.shared.weight
                assert exported.lm_head.weight is not exported.shared.weight
                del exported
            histories.append(result["runs"][seed]["training"]["restored_dev_loss"])
            for name, prefix in (("news", "news"), ("bible", "bible")):
                lines = (run / "decodes" / f"{name}.jsonl").read_text().splitlines()
                rows = [json.loads(line) for line in lines]
                assert [row["id"] for row in rows] == [f"{prefix}-{i}" for i in range(3)]
                score = json.loads((run / "decodes" / f"{name}.json").read_text())
                assert score["predictions_sha256"] == mt_protocol.file_sha256(
                    run / "decodes" / f"{name}.jsonl"
                )
                assert score["signatures"]["chrf2pp"].startswith("nrefs:1|case:mixed")
        # F6: the seed reaches training.
        assert len(set(histories)) == len(seeds)

    # Rerunning a finished system needs no weights and changes nothing (F9).
    again = mt_protocol.test(study, systems[0], "en2kpe")
    assert again["scores"] == results[systems[0]]["scores"]

    # F4: an edited selection or protocol is refused.
    selection_file = study / "selection" / "en2kpe" / f"{label_a}.json"
    original = selection_file.read_text()
    selection_file.write_text(original.replace('"flags"', '"flags" '))
    with pytest.raises(ValueError, match="different protocol, selection or test set"):
        mt_protocol.test(study, systems[0], "en2kpe")
    selection_file.write_text(original)
    protocol_file = study / "protocol.json"
    frozen = protocol_file.read_text()
    protocol_file.write_text(frozen + " ")
    with pytest.raises(ValueError, match="edited after it was frozen"):
        mt_protocol.load_protocol(study)
    protocol_file.write_text(frozen)

    # The bootstrap's observed difference equals the difference of the
    # seed-averaged corpus chrF++ recorded in the results (F7).
    def predictions(system, name):
        result = results[system]
        return [
            [json.loads(line)["prediction"] for line in
             (study / result["runs"][str(seed)]["run"] / "decodes" / f"{name}.jsonl")
             .read_text().splitlines()]
            for seed in seeds
        ]

    references = [json.loads(line)["reference"] for line in (
        study / results[systems[0]]["runs"]["13"]["run"] / "decodes" / "news.jsonl"
    ).read_text().splitlines()]
    comparison = mt_protocol.paired_bootstrap(
        predictions(systems[0], "news"), predictions(systems[1], "news"), references, 50, 13
    )
    expected = (results[systems[0]]["scores"]["news"]["chrf2pp_mean"]
                - results[systems[1]]["scores"]["news"]["chrf2pp_mean"])
    assert comparison["difference"] == pytest.approx(expected, abs=1e-3)
    assert comparison["ci95"][0] <= comparison["difference"] <= comparison["ci95"][1]

    document = mt_protocol.report(study)
    assert document["smoke"] is True
    assert {row["eval_set"] for row in document["rows"]} == {"dev", "news", "bible"}
    assert (study / "report.json").exists()
    assert observed_dropout and parameter_updates and all(parameter_updates)
    assert set(observed_dropout) == ({dropout_rate} if dropout_rate is not None else {0.0, 0.2})
    assert all(mt_protocol.file_sha256(path) == value for path, value in source_hashes.items())
    (study / "dropout_checks.json").write_text(json.dumps({
        "override": dropout_rate, "observed_training_dropout": observed_dropout,
        "parameter_updates": parameter_updates, "source_artifacts_unchanged": True,
    }, indent=2))
