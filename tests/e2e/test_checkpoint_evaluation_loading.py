"""Saved untied mT5 must preserve its forward values through evaluation loading.

Failure mode: the installed MT5Config can reload untied saved weights with a
tied flag, which changes output scaling without changing any weight tensor.
"""
import json


def test_evaluation_reload_preserves_untied_mt5_logits(tmp_path):
    import torch
    from tokenizers import Tokenizer, models
    from transformers import MT5Config, MT5ForConditionalGeneration, PreTrainedTokenizerFast
    from akanlm.eval.denoising import load_checkpoint

    torch.manual_seed(13)
    config = MT5Config(vocab_size=128, d_model=16, d_kv=4, d_ff=32,
                       num_layers=1, num_decoder_layers=1, num_heads=4,
                       dropout_rate=0.0, decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)
    config.tie_word_embeddings = False
    reference = MT5ForConditionalGeneration(config).eval()
    reference.set_input_embeddings(reference.shared)
    assert reference.encoder.embed_tokens.weight is reference.shared.weight
    assert reference.decoder.embed_tokens.weight is reference.shared.weight
    backend = Tokenizer(models.WordLevel({'<pad>':0, '</s>':1, '<unk>':2,
                         **{f'token_{index}':index for index in range(3,128)}}, unk_token='<unk>'))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='<pad>',
                                      eos_token='</s>', unk_token='<unk>')
    directory = tmp_path / 'checkpoint'
    reference.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    batch = {'input_ids':torch.tensor([[3,4,5,1]]), 'labels':torch.tensor([[6,7,1]])}
    with torch.no_grad():
        expected = reference(**batch)
    _, reloaded = load_checkpoint(directory)
    reloaded.eval()
    with torch.no_grad():
        observed = reloaded(**batch)
    assert not reloaded.config.tie_word_embeddings
    assert reloaded.get_input_embeddings().weight.data_ptr() != reloaded.lm_head.weight.data_ptr()
    torch.testing.assert_close(observed.logits, expected.logits, rtol=0, atol=0)
    torch.testing.assert_close(observed.loss, expected.loss, rtol=0, atol=0)
    (tmp_path / 'evaluation_load_check.json').write_text(json.dumps({
        'untied':True, 'reference_loss':float(expected.loss), 'loaded_loss':float(observed.loss),
        'maximum_logit_difference':float((observed.logits-expected.logits).abs().max()),
        'checkpoint':str(directory)}, indent=2)+'\n')
