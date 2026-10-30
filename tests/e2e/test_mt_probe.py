"""Offline MT fine-tuning, export and interrupted-resume workflows with tiny models."""

import json

import pytest

from akanlm.eval.mt_probe import ProbeConfig, finetune


@pytest.mark.parametrize(
    "overrides,completed,best_step,reason",
    [
        ({}, 7, 2, "dev_loss_patience"),
        ({"early_stopping_patience": 0}, 8, None, "step_cap"),
        ({"max_steps": 2, "eval_steps": 100}, 2, 2, "step_cap"),
        ({"max_steps": -1, "num_train_epochs": 3.0}, 6, 2, "epoch_cap"),
        ({"early_stopping_patience": 1, "early_stopping_threshold": 1.5},
         2, 1, "dev_loss_patience"),
    ],
)
def test_training_scores_saved_stop_step_weights(
    tmp_path, monkeypatch, tiny_tokenizer, overrides, completed, best_step, reason
):
    import torch
    from transformers import MT5Config, MT5ForConditionalGeneration, Seq2SeqTrainer

    initial = tmp_path / "initial"
    model_config = MT5Config(
        vocab_size=len(tiny_tokenizer), d_model=16, d_ff=32, d_kv=4,
        num_heads=4, num_layers=1, num_decoder_layers=1,
        dropout_rate=0, decoder_start_token_id=0,
    )
    model_config.tie_word_embeddings = False
    model = MT5ForConditionalGeneration(model_config)
    model.save_pretrained(initial)
    tiny_tokenizer.save_pretrained(initial)
    instances = []

    original_init = Seq2SeqTrainer.__init__
    original_evaluate = Seq2SeqTrainer.evaluation_loop
    original_save = Seq2SeqTrainer._save_checkpoint

    def init(self, **kwargs):
        kwargs["args"].use_cpu = True
        original_init(self, **kwargs)
        self.saved_weights = {}
        self.initial_head = self.model.lm_head.weight.detach().clone()
        instances.append(self)

    def evaluation_loop(self, *args, **kwargs):
        output = original_evaluate(self, *args, **kwargs)
        output.metrics["eval_loss"] = {1: 3.0, 2: 2.0}.get(
            self.state.global_step, 4.0
        )
        return output

    def save_checkpoint(self, model, trial):
        self.saved_weights[self.state.global_step] = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }
        return original_save(self, model, trial)

    def reject_best_model_load(self):
        raise AssertionError("The best-dev checkpoint must never be restored")

    monkeypatch.setattr(Seq2SeqTrainer, "__init__", init)
    monkeypatch.setattr(Seq2SeqTrainer, "evaluation_loop", evaluation_loop)
    monkeypatch.setattr(Seq2SeqTrainer, "_save_checkpoint", save_checkpoint)
    monkeypatch.setattr(Seq2SeqTrainer, "_load_best_model", reject_best_model_load)
    config = ProbeConfig(**{
        "max_steps": 8, "eval_steps": 1, "bf16": False,
        "per_device_train_batch_size": 2, "gradient_accumulation_steps": 1,
        **overrides,
    })
    rows = [{"source": "She calls him.", "target": "Anadwo biara ɔfrɛ no."}] * 4
    output_dir = tmp_path / "probe"
    returned, returned_tokenizer = finetune(
        str(initial), rows, rows, config, output_dir
    )
    trainer = instances[0]
    assert not trainer.model_accepts_loss_kwargs
    assert trainer.data_collator.pad_to_multiple_of == config.pad_to_multiple_of
    state = json.loads((output_dir / "trainer_state.json").read_text())
    summary = json.loads((output_dir / "training_summary.json").read_text())
    assert state["global_step"] == completed
    assert state["best_model_checkpoint"] is None
    assert summary["completed_steps"] == completed
    assert summary["stopping_reason"] == reason
    assert summary["best_dev_step"] == best_step
    assert returned is not trainer.model
    assert returned_tokenizer.get_vocab() == tiny_tokenizer.get_vocab()
    for name, value in returned.state_dict().items():
        torch.testing.assert_close(value, trainer.saved_weights[completed][name])
    assert not torch.equal(returned.lm_head.weight, trainer.initial_head)
    if best_step is not None and best_step != completed:
        assert not torch.equal(
            returned.lm_head.weight, trainer.saved_weights[best_step]["lm_head.weight"]
        )


def test_resume_restores_early_stopping_counter(
    tmp_path, monkeypatch, tiny_tokenizer
):
    from transformers import MT5Config, MT5ForConditionalGeneration, Seq2SeqTrainer

    initial = tmp_path / "initial"
    model_config = MT5Config(
        vocab_size=len(tiny_tokenizer), d_model=16, d_ff=32, d_kv=4,
        num_heads=4, num_layers=1, num_decoder_layers=1,
        dropout_rate=0, decoder_start_token_id=0,
    )
    model_config.tie_word_embeddings = False
    MT5ForConditionalGeneration(model_config).save_pretrained(initial)
    tiny_tokenizer.save_pretrained(initial)

    original_init = Seq2SeqTrainer.__init__
    original_eval = Seq2SeqTrainer.evaluation_loop
    original_save = Seq2SeqTrainer._save_checkpoint
    interrupted = False

    def init(self, **kwargs):
        kwargs["args"].use_cpu = True
        original_init(self, **kwargs)

    def evaluation_loop(self, *args, **kwargs):
        output = original_eval(self, *args, **kwargs)
        output.metrics["eval_loss"] = {1: 3.0, 2: 2.0}.get(
            self.state.global_step, 4.0
        )
        return output

    def interrupt_after_checkpoint(self, model, trial):
        nonlocal interrupted
        result = original_save(self, model, trial)
        if self.state.global_step == 3 and not interrupted:
            interrupted = True
            raise RuntimeError("simulated interruption")
        return result

    monkeypatch.setattr(Seq2SeqTrainer, "__init__", init)
    monkeypatch.setattr(Seq2SeqTrainer, "evaluation_loop", evaluation_loop)
    monkeypatch.setattr(
        Seq2SeqTrainer, "_save_checkpoint", interrupt_after_checkpoint
    )
    config = ProbeConfig(
        max_steps=8, eval_steps=1, bf16=False,
        per_device_train_batch_size=2, gradient_accumulation_steps=1,
    )
    rows = [{"source": "She calls him.", "target": "Anadwo biara ɔfrɛ no."}] * 4
    output_dir = tmp_path / "probe"
    with pytest.raises(RuntimeError, match="simulated interruption"):
        finetune(str(initial), rows, rows, config, output_dir)

    assert (output_dir / "checkpoint-3" / "optimizer.pt").exists()
    monkeypatch.setattr(Seq2SeqTrainer, "_save_checkpoint", original_save)
    finetune(str(initial), rows, rows, config, output_dir, resume=True)
    summary = json.loads((output_dir / "training_summary.json").read_text())
    assert summary["completed_steps"] == 7
    assert summary["stopping_reason"] == "dev_loss_patience"
    assert summary["best_dev_step"] == 2
    # A completed run reloads its final artifact without training again.
    finetune(str(initial), rows, rows, config, output_dir, resume=True)
