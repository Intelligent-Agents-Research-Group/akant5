"""Extend mT5 unigram pieces and initialize only added lexical rows.

Preserve original IDs/normalization and separate output weights.
Copied piece scores are a heuristic, not joint unigram re-estimation."""

from __future__ import annotations

import json
import math
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from akanlm.eval.denoising import tokenizer_fingerprint
from akanlm.tokenizer.surgery import (
    assert_architecture_preserved,
    assert_embeddings_shared,
    load_tokenizer,
    normalize_tying,
)
from akanlm.tokenizer.train_spm import pad_vocab_size
from akanlm.utils.paths import ensure_dir, file_sha256, write_json

TOKENIZER_REPORT = "extension_report.json"
SCORE_RULE = "retain_original_copy_candidate_unigram_scores_v1"


def verify_training_split(tokenizer_dir, split_manifest):
    """Refuse production CPT with unprotected candidates or a different split."""
    from akanlm.data.cpt_splits import read_manifest

    metadata = json.loads((Path(tokenizer_dir) / TOKENIZER_REPORT).read_text())
    if not metadata.get("training_partition_verified"):
        raise ValueError("CPT extension candidates lack verified training-partition provenance")
    if split_manifest is None:
        raise ValueError("Extended-tokenizer CPT requires the candidate-fitting --split-manifest")
    manifest = read_manifest(Path(split_manifest))
    if (metadata["candidate_training_partition"].get("split_manifest_sha256")
            != manifest["sha256"]):
        raise ValueError("CPT split differs from the frozen split used for candidate fitting")
    return manifest["sha256"]


def _empty_destination(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Extension requires an empty destination: {path}")
    return ensure_dir(path)


def _original_tokenizer(reference):
    """Restore native byte fallback when the installed converter dropped it."""
    tokenizer = load_tokenizer(reference)
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    if backend["model"]["type"] != "Unigram":
        raise ValueError("Vocabulary extension requires an original Unigram tokenizer")
    if not backend["model"].get("byte_fallback", False):
        from akanlm.tokenizer.compat import restore_sentencepiece_byte_fallback

        if Path(reference).is_dir():
            native = next((Path(reference) / name for name in ("spiece.model", "spm.model")
                           if (Path(reference) / name).is_file()), None)
        else:
            from transformers.utils.hub import cached_file

            native = cached_file(reference, "spiece.model")
        if native is None:
            raise ValueError("Cannot verify original byte fallback without the native SentencePiece model")
        restore_sentencepiece_byte_fallback(tokenizer, native)
    return tokenizer


def extend_tokenizer(base_model, candidate_dir, output_dir, *, require_training_partitions=False):
    """Append missing NORMAL pieces without changing existing IDs.

    The CLI requires protected training provenance; unprotected API inputs are flagged."""
    from sentencepiece import sentencepiece_model_pb2 as pb
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    candidate_dir, output_dir = Path(candidate_dir), Path(output_dir)
    native_path = candidate_dir / "spm.model"
    proto = pb.ModelProto()
    proto.ParseFromString(native_path.read_bytes())
    if proto.trainer_spec.model_type != pb.TrainerSpec.UNIGRAM:
        raise ValueError("Candidate inventory must be a SentencePiece unigram model")
    candidate_metadata_path = candidate_dir / "akanlm_tokenizer.json"
    candidate_metadata = (json.loads(candidate_metadata_path.read_text())
                          if candidate_metadata_path.exists() else {})
    sampling = candidate_metadata.get("sampling", {})
    partition = sampling.get("training_partition", {})
    protected = (sampling.get("completed") is True and sampling.get("language_budget_verified") is True
                 and partition.get("fit_split") == "train" and bool(partition.get("split_manifest_sha256")))
    if require_training_partitions and not protected:
        raise ValueError("Extension candidates require verified frozen training-partition provenance")

    original = _original_tokenizer(str(base_model))
    original_vocab = original.get_vocab()
    original_size = len(original)
    if sorted(original_vocab.values()) != list(range(original_size)):
        raise ValueError("Original tokenizer IDs must be unique and contiguous")
    backend = json.loads(original.backend_tokenizer.to_str())
    unigram = backend["model"]["vocab"]
    # Some T5 backends store existing sentinels as added tokens. Materialize
    # those already allocated IDs before appending, retaining their matcher.
    id_to_piece = {index: piece for piece, index in original_vocab.items()}
    for index in range(len(unigram), original_size):
        unigram.append([id_to_piece[index], 0.0])
    if any(piece != id_to_piece[index] for index, (piece, _) in enumerate(unigram)):
        raise ValueError("Original backend pieces do not match original token IDs")

    added, overlaps, nonordinary = [], 0, 0
    for candidate_id, entry in enumerate(proto.pieces):
        piece = entry.piece
        if (entry.type != pb.ModelProto.SentencePiece.NORMAL
                or re.fullmatch(r"▁?<extra_id_\d+>|<0x[0-9A-Fa-f]{2}>", piece)):
            nonordinary += 1
            continue
        if piece in original_vocab:
            overlaps += 1
            continue
        if not math.isfinite(entry.score):
            raise ValueError(f"Nonfinite candidate unigram score for {piece!r}")
        index = len(unigram)
        unigram.append([piece, float(entry.score)])
        added.append({"piece": piece, "id": index, "candidate_id": candidate_id,
                      "score": float(entry.score)})
    if not added:
        raise ValueError("Candidate inventory adds no missing ordinary pieces")
    extended = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_str(json.dumps(backend)),
        pad_token=original.pad_token, eos_token=original.eos_token, unk_token=original.unk_token,
        model_max_length=original.model_max_length,
        extra_special_tokens=list(original.extra_special_tokens),
        padding_side=original.padding_side, truncation_side=original.truncation_side,
    )
    extended_vocab = extended.get_vocab()
    if any(extended_vocab.get(piece) != index for piece, index in original_vocab.items()):
        raise ValueError("Extension changed original token IDs")
    output_dir = _empty_destination(output_dir)
    extended.save_pretrained(output_dir)
    reloaded = AutoTokenizer.from_pretrained(output_dir)
    if tokenizer_fingerprint(reloaded) != tokenizer_fingerprint(extended):
        raise ValueError("Extension tokenizer changed during serialization")
    report = {
        "protocol": "mt5_unigram_extension_v1", "base_model": str(base_model),
        "base_tokenizer_sha256": tokenizer_fingerprint(original),
        "extended_tokenizer_sha256": tokenizer_fingerprint(extended),
        "candidate_dir": str(candidate_dir), "candidates_sha256": file_sha256(native_path),
        "candidate_metadata_sha256": (file_sha256(candidate_metadata_path)
                                       if candidate_metadata_path.exists() else None),
        "candidate_training_partition": partition, "training_partition_verified": protected,
        "candidate_input_sha256": sampling.get("input_sha256"), "score_rule": SCORE_RULE,
        "normalization": "original mT5 backend retained; candidate normalization is not substituted",
        "old_tokenizer_size": original_size, "new_tokenizer_size": len(extended),
        "ordinary_overlaps": overlaps, "special_or_byte_candidates_skipped": nonordinary,
        "added_count": len(added), "added": added,
        "sentinels": {piece: index for piece, index in original_vocab.items()
                      if re.fullmatch(r"▁?<extra_id_\d+>", piece)},
    }
    return write_json(output_dir / TOKENIZER_REPORT, report)


@dataclass
class ExtensionReport:
    base_model: str
    tokenizer_dir: str
    old_tokenizer_size: int
    tokenizer_len: int
    old_vocab_size: int
    new_vocab_size: int
    params_before: int
    params_after: int
    original_padding_rows_repurposed: int
    input_rows: dict
    output_rows: dict
    source_checkpoint: dict
    extension_report_sha256: str

    def to_dict(self):
        return {"protocol": "mt5_extension_initialization_v1", **asdict(self),
                "params_added": self.params_after - self.params_before,
                "preserved_rows": "all original tokenizer entries; unused model padding may be repurposed",
                "padding_rule": "pretrained token-row mean",
                "input_rule": "original-subword mean",
                "output_rule": "original-subword mean rescaled to mean norm of original token rows"}


def _extended_matrix(source, old_size, new_size, target_rows, decompositions, rescale):
    import torch

    reference = source.detach().to(device="cpu", dtype=torch.float32)
    mean = reference[:old_size].mean(0)
    result = mean.expand(target_rows, -1).clone()
    result[:old_size] = reference[:old_size]
    reference_norm = float(reference[:old_size].norm(dim=1).mean())
    fallback = 0
    for index, ids in zip(range(old_size, new_size), decompositions, strict=True):
        if not ids:
            fallback += 1
            continue
        row = reference[ids].mean(0)
        if rescale:
            row = row * (reference_norm / row.norm().clamp_min(1e-8))
        result[index] = row
    result = result.to(dtype=source.dtype, device=source.device)
    if not torch.equal(result[:old_size], source[:old_size]):
        raise AssertionError("Original token rows changed during extension")
    return result, {"copied": old_size, "derived": new_size - old_size - fallback,
                    "fallback": fallback, "padding": target_rows - new_size,
                    "mean_original_row_norm": reference_norm, "rescale_derived": rescale}


def apply(base_model, tokenizer_dir, output_dir, dtype="float32"):
    """Build a size-specific mT5 initialization while preserving old token rows."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer_dir, output_dir = Path(tokenizer_dir), Path(output_dir)
    metadata_path = tokenizer_dir / TOKENIZER_REPORT
    metadata = json.loads(metadata_path.read_text())
    old = _original_tokenizer(str(base_model))
    new = AutoTokenizer.from_pretrained(tokenizer_dir)
    if tokenizer_fingerprint(old) != metadata["base_tokenizer_sha256"]:
        raise ValueError("Checkpoint tokenizer differs from the original tokenizer used for extension")
    if tokenizer_fingerprint(new) != metadata["extended_tokenizer_sha256"]:
        raise ValueError("Extended tokenizer content differs from its extension report")
    old_size, new_size = len(old), len(new)
    old_vocab, new_vocab = old.get_vocab(), new.get_vocab()
    if (old_size != metadata["old_tokenizer_size"] or new_size != metadata["new_tokenizer_size"]
            or any(new_vocab.get(piece) != index for piece, index in old_vocab.items())):
        raise ValueError("Extended vocabulary no longer preserves original token IDs")
    output_dir = _empty_destination(output_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(str(base_model), dtype=getattr(torch, dtype))
    if normalize_tying(model):
        raise ValueError("mT5 extension requires an untied pretrained output head")
    assert_embeddings_shared(model)
    before_config = model.config.to_dict()
    old_rows = model.get_input_embeddings().weight.shape[0]
    if old_size > old_rows or model.lm_head.weight.shape[0] != old_rows:
        raise ValueError("Original tokenizer and model vocabulary rows are incompatible")
    target_rows = max(old_rows, pad_vocab_size(new_size))
    params_before = sum(parameter.numel() for parameter in model.parameters())
    pieces = [new.convert_ids_to_tokens(index) for index in range(old_size, new_size)]
    decompositions = old([piece.replace("▁", " ") for piece in pieces],
                         add_special_tokens=False)["input_ids"]
    input_weight, input_report = _extended_matrix(
        model.get_input_embeddings().weight, old_size, new_size, target_rows, decompositions, False)
    head_weight, head_report = _extended_matrix(
        model.lm_head.weight, old_size, new_size, target_rows, decompositions, True)
    model.set_input_embeddings(torch.nn.Embedding.from_pretrained(input_weight, freeze=False))
    head = torch.nn.Linear(model.config.d_model, target_rows, bias=False, device="meta")
    head.weight = torch.nn.Parameter(head_weight)
    model.lm_head = head
    model.config.vocab_size = target_rows
    model.vocab_size = target_rows
    assert_architecture_preserved(before_config, model.config)
    assert_embeddings_shared(model)
    source = {"reference": str(base_model), "revision": getattr(model.config, "_commit_hash", None)}
    if Path(base_model).is_dir():
        directory = Path(base_model)
        weights = sorted(directory.glob("*.safetensors")) or sorted(directory.glob("pytorch_model*.bin"))
        source["files_sha256"] = {path.name: file_sha256(path) for path in weights}
        source["config_sha256"] = file_sha256(directory / "config.json")
    report = ExtensionReport(
        str(base_model), str(tokenizer_dir), old_size, new_size, old_rows, target_rows,
        params_before, sum(parameter.numel() for parameter in model.parameters()),
        min(new_size - old_size, old_rows - old_size), input_report, head_report,
        source, file_sha256(metadata_path),
    )
    model.save_pretrained(output_dir)
    new.save_pretrained(output_dir)
    shutil.copyfile(metadata_path, output_dir / TOKENIZER_REPORT)
    write_json(output_dir / "extension_init_report.json", report.to_dict())
    return report


def check_initialization(model_dir, input_length=128):
    """Record losses and added/padding softmax mass before any CPT updates."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    from akanlm.data.collator import DataCollatorForT5MLM, compute_input_and_target_lengths
    from akanlm.tokenizer.surgery import CHECK_TEXT

    directory = Path(model_dir)
    metadata = json.loads((directory / TOKENIZER_REPORT).read_text())
    tokenizer = AutoTokenizer.from_pretrained(directory)
    model = AutoModelForSeq2SeqLM.from_pretrained(directory).eval()
    normalize_tying(model)
    raw_length, target_length = compute_input_and_target_lengths(input_length)
    report = {"protocol": "mt5_extension_forward_check_v1", "languages": {},
              "tokenizer_sha256": tokenizer_fingerprint(tokenizer),
              "uniform_nats": math.log(model.config.vocab_size)}
    for language, text in {"akan": CHECK_TEXT, "english": (
        "The quick brown fox jumps over the lazy dog. She calls him every night without fail. "
        "Grammar describes what a language permits and forbids.")}.items():
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        ids = (ids * (raw_length // len(ids) + 1))[:raw_length]
        collator = DataCollatorForT5MLM(tokenizer, input_length=input_length,
                                     target_length=target_length, seed=13)
        batch = collator([{"input_ids": ids}])
        with torch.no_grad():
            output = model(**batch)
            probabilities = output.logits.float().softmax(-1)[batch["labels"] != -100]
        loss = float(output.loss)
        if not math.isfinite(loss) or loss >= report["uniform_nats"]:
            raise ValueError(f"{language} initial loss {loss} fails uniform ln(V) check")
        report["languages"][language] = {
            "loss_nats": loss,
            "added_probability_mass": float(probabilities[:, metadata["old_tokenizer_size"]:len(tokenizer)].sum(-1).mean()),
            "padding_probability_mass": float(probabilities[:, len(tokenizer):].sum(-1).mean()),
            "decoded_input": tokenizer.decode(batch["input_ids"][0], skip_special_tokens=False),
            "decoded_target": tokenizer.decode(batch["labels"][0][batch["labels"][0] >= 0]),
        }
    return write_json(directory / "extension_forward_check.json", report)
