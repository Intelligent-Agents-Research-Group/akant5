"""Preserve SentencePiece behavior when importing pretrained tokenizers."""

import json
from pathlib import Path


def restore_sentencepiece_byte_fallback(tokenizer, spiece_path: str | Path) -> bool:
    """Restore native byte fallback without changing IDs or other tokenizer rules.

    Save with PreTrainedTokenizerFast and invalidate old fingerprints; return whether changed."""
    from sentencepiece import sentencepiece_model_pb2
    from tokenizers import Tokenizer

    native = sentencepiece_model_pb2.ModelProto()
    native.ParseFromString(Path(spiece_path).read_bytes())
    if not native.trainer_spec.byte_fallback:
        return False

    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    model = backend["model"]
    if model["type"] != "Unigram":
        raise ValueError("SentencePiece byte-fallback restoration requires a Unigram backend")

    vocab = tokenizer.get_vocab()
    native_pieces = {piece.piece: (index, piece.type)
                     for index, piece in enumerate(native.pieces)}
    byte_type = sentencepiece_model_pb2.ModelProto.SentencePiece.BYTE
    for value in range(256):
        piece = f"<0x{value:02X}>"
        entry = native_pieces.get(piece)
        if (entry is None or entry[1] != byte_type or vocab.get(piece) != entry[0]
                or entry[0] >= len(model["vocab"])
                or model["vocab"][entry[0]][0] != piece):
            raise ValueError(f"Original SentencePiece byte-piece ID does not match: {piece}")

    def has_byte_decoder(decoder):
        return decoder.get("type") == "ByteFallback" or any(
            has_byte_decoder(child) for child in decoder.get("decoders", []))

    decoder = backend.get("decoder")
    if decoder is None:
        raise ValueError("SentencePiece byte-fallback restoration requires an existing decoder")
    decoder_present = has_byte_decoder(decoder)
    if model.get("byte_fallback", False) and decoder_present:
        return False
    model["byte_fallback"] = True
    if not decoder_present:
        backend["decoder"] = {"type": "Sequence",
                              "decoders": [{"type": "ByteFallback"}, decoder]}

    restored = Tokenizer.from_str(json.dumps(backend))
    if restored.get_vocab() != vocab:
        raise ValueError("Byte-fallback restoration would change vocabulary IDs")
    tokenizer._tokenizer = restored
    return True
