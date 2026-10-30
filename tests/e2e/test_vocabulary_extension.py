"""Offline vocabulary extension, corruption, optimizer update and serialization.

Failure modes fixed before implementation: shifted original IDs; special/byte
pieces added as ordinary pieces; lexical pieces never selected; stale sentinel
positions; copied weights altered or output head tied; added rows receive no
update; serialization changes segmentation or predictions. The receipt records
the exercised checks; this tiny synthetic model does not establish CPT quality.
"""

import json
import math
import pytest


@pytest.mark.parametrize("width", [16, 24, 32])
def test_extension_trains_and_roundtrips(
    tmp_path, tiny_tokenizer, twi_sentences, english_sentences, width, monkeypatch
):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, MT5Config, MT5ForConditionalGeneration

    from akanlm.data.collator import DataCollatorForT5MLM
    from akanlm.eval.denoising import DenoisingConfig, build_tasks, tokenizer_fingerprint
    from akanlm.tokenizer.extension import apply, extend_tokenizer
    from akanlm.tokenizer.surgery import assert_embeddings_shared, normalize_tying
    from akanlm.tokenizer.train_spm import TokenizerConfig, pad_vocab_size, train

    torch.set_num_threads(1)
    torch.manual_seed(13)
    base = tmp_path / "base"
    config = MT5Config(
        vocab_size=pad_vocab_size(len(tiny_tokenizer)), d_model=width,
        d_ff=width * 2, d_kv=4, num_heads=2, num_layers=1,
        num_decoder_layers=1, dropout_rate=0., decoder_start_token_id=0,
    )
    config.tie_word_embeddings = False
    original = MT5ForConditionalGeneration(config)
    original.encoder.embed_tokens = original.shared
    original.decoder.embed_tokens = original.shared
    with torch.no_grad():
        original.lm_head.weight.mul_(0.1)
    original.save_pretrained(base)
    tiny_tokenizer.save_pretrained(base)
    candidates = tmp_path / "candidates"
    train(TokenizerConfig(name="candidate", vocab_size=420, num_threads=1), candidates,
          sentence_iterator=twi_sentences * 60 + english_sentences * 20)
    extended = tmp_path / "extended"
    extend_tokenizer(str(base), candidates, extended)
    tokenizer = AutoTokenizer.from_pretrained(extended)
    original_vocab = tiny_tokenizer.get_vocab()
    assert len(tokenizer) > len(original_vocab)
    assert all(tokenizer.get_vocab()[piece] == index for piece, index in original_vocab.items())
    assert tokenizer_fingerprint(tokenizer) != tokenizer_fingerprint(tiny_tokenizer)
    for index in range(100):
        piece = f"<extra_id_{index}>"
        assert tokenizer.get_vocab()[piece] == original_vocab[piece]
    added = set(tokenizer.get_vocab()) - original_vocab.keys()
    texts = [*twi_sentences, *english_sentences]
    assert any(added.intersection(tokenizer.tokenize(text)) for text in texts)
    # Boundary handling and fallback have to remain coherent after reloading.
    for text in [*texts, "rare: ẍ 𝔮 ⚡ 中文", "two  spaces", "tone: ɛ́ ɔ̀ á"]:
        assert tokenizer.decode(tokenizer.encode(text, add_special_tokens=False)) == text

    initial = tmp_path / "initial"
    report = apply(str(base), extended, initial)
    model = AutoModelForSeq2SeqLM.from_pretrained(initial)
    normalize_tying(model)
    assert_embeddings_shared(model)
    assert not model.config.tie_word_embeddings
    assert model.shared.weight.data_ptr() != model.lm_head.weight.data_ptr()
    for name, old in original.state_dict().items():
        current = model.state_dict()[name]
        if name in {"shared.weight", "encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"}:
            torch.testing.assert_close(current[:len(tiny_tokenizer)], old[:len(tiny_tokenizer)], rtol=0, atol=0)
        else:
            torch.testing.assert_close(current, old, rtol=0, atol=0)

    collator = DataCollatorForT5MLM(tokenizer, input_length=32, target_length=16)
    new_ids = [index for piece, index in tokenizer.get_vocab().items() if piece in added]
    # Even the highest added ID is lexical input, despite being above sentinels.
    ids = (new_ids * collator.tokens_length)[:collator.tokens_length]
    batch = collator([{"input_ids": ids}])
    assert any(index in new_ids for index in batch["labels"][0].tolist())
    bad = list(ids)
    bad[0] = tokenizer.convert_tokens_to_ids("<extra_id_0>")
    with pytest.raises(ValueError, match="sentinel|reserved"):
        collator([{"input_ids": bad}])
    tasks = build_tasks(texts, tokenizer, DenoisingConfig(
        input_length=32, target_length=16, max_examples=2,
    ))
    assert tasks

    model.train()
    before_input = model.shared.weight.detach().clone()
    before_head = model.lm_head.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    initial_loss = float(model(**batch).loss.detach())
    with torch.no_grad():
        logits = model(**batch).logits
        mass = float(logits.softmax(-1)[..., new_ids].sum(-1).mean())
    for _ in range(15):
        optimizer.zero_grad()
        loss = model(**batch).loss
        loss.backward()
        assert torch.isfinite(model.shared.weight.grad).all()
        assert torch.isfinite(model.lm_head.weight.grad).all()
        optimizer.step()
    final_loss = float(model(**batch).loss.detach())
    assert final_loss < initial_loss and final_loss < math.log(model.config.vocab_size)
    assert not torch.equal(before_input[new_ids], model.shared.weight[new_ids])
    assert not torch.equal(before_head[new_ids], model.lm_head.weight[new_ids])
    final = tmp_path / "final"
    model.eval()
    expected_logits = model(**batch).logits.detach()
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    restored = AutoModelForSeq2SeqLM.from_pretrained(final).eval()
    normalize_tying(restored)
    assert_embeddings_shared(restored)
    assert not restored.config.tie_word_embeddings
    restored_tokenizer = AutoTokenizer.from_pretrained(final)
    assert tokenizer_fingerprint(restored_tokenizer) == tokenizer_fingerprint(tokenizer)
    torch.testing.assert_close(restored(**batch).logits, expected_logits, rtol=0, atol=0)
    receipt = {
        "width": width, "old_tokens": len(tiny_tokenizer),
        "new_tokens": len(tokenizer), "added_pieces": len(added),
        "loss_before": initial_loss, "loss_after": final_loss,
        "initial_added_probability_mass": mass,
        "original_ids_and_rows_preserved": True,
        "added_input_and_output_rows_updated": True,
        "roundtrip_logits_identical": True,
        "decoded_input": tokenizer.decode(batch["input_ids"][0], skip_special_tokens=False),
        "decoded_target": tokenizer.decode(batch["labels"][0][batch["labels"][0] >= 0]),
        "initialization": report.to_dict(),
    }
    (tmp_path / "extension_check.json").write_text(json.dumps(receipt, indent=2))
    if width == 16:
        _train_and_resume(initial, tokenizer, tmp_path, monkeypatch)


def _train_and_resume(initial, tokenizer, tmp_path, monkeypatch):
    """Exercise actual CPT checkpoint state with an extended vocabulary."""
    from dataclasses import replace
    from pathlib import Path

    from datasets import Dataset

    from akanlm.audit.sources import load_sources
    from akanlm.corpora.loaders import PairRecord
    from akanlm.data.cpt_splits import prepare
    from akanlm.data.mixture import MixtureComponent, MixtureConfig
    from akanlm.training.cpt import TrainingConfig, train

    source = load_sources(Path("configs/corpora"))["pristine_twi_english"]
    rows = [PairRecord(source.name, i, f"Source document number {i}.",
                       f"Target document number {i}.", {}) for i in range(80)]
    monkeypatch.setattr("akanlm.audit.sources.load_sources", lambda _: {source.name: source})
    monkeypatch.setattr("akanlm.data.mixture.load_sources", lambda _: {source.name: source})
    monkeypatch.setattr("akanlm.training.validation.heldout_documents", lambda _: ([], {}))
    monkeypatch.setattr("akanlm.corpora.loaders.load_pairs", lambda s, **kw: iter(rows[:kw.get("limit")]))
    mixture = MixtureConfig("fixture", components=[MixtureComponent(source.name)])
    split = tmp_path / "split.json"
    prepare(mixture, Path("unused"), split)
    mixtures = tmp_path / "mixtures"
    mixtures.mkdir()
    (mixtures / "fixture.yaml").write_text("fixture")

    def dataset(config, tokenizer, *args, **kwargs):
        ids = tokenizer("Anadwo biara ɔfrɛ no.", add_special_tokens=False)["input_ids"]
        length = config.resolved_lengths()[0]
        packed = (ids * length)[:length]
        summary = {"components": [{"source": source.name}],
                   "preparation": kwargs["protected"].report()}
        return Dataset.from_list([{"input_ids": packed}] * 4), summary, None

    monkeypatch.setattr("akanlm.training.cpt.build_dataset", dataset)
    config = TrainingConfig(
        "tiny", str(initial), mixture="fixture", max_steps=2,
        input_length=32, target_length=16, per_device_train_batch_size=1,
        gradient_accumulation_steps=1, per_device_eval_batch_size=2,
        gradient_checkpointing=False, bf16=False, dataloader_num_workers=0,
        logging_steps=1, eval_steps=1, save_steps=1, milestone_steps=[],
        ignore_data_skip=False, early_stopping_patience=0,
    )
    output = tmp_path / "cpt"
    first = train(config, output, corpora_dir=Path("unused"),
                  mixtures_dir=mixtures, split_manifest=split)
    assert first.metrics["completed_steps"] == 2
    assert (output / "checkpoint-2" / "optimizer.pt").exists()
    resumed = train(replace(config, max_steps=3), output, corpora_dir=Path("unused"),
                    mixtures_dir=mixtures, split_manifest=split, resume_from_checkpoint=True)
    assert resumed.metrics["completed_steps"] == 3
    assert (output / "checkpoint-3" / "scheduler.pt").exists()
    (tmp_path / "resume_check.json").write_text(json.dumps({
        "initial_steps": 2, "resumed_steps": 3,
        "optimizer_and_scheduler_state_saved": True,
        "tokenizer_tokens": len(tokenizer),
    }, indent=2))
