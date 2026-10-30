"""Google's token-span denoising metrics, applied to pinned held-out Twi.

References: T5 data.preprocessors.span_corruption and Mesh transformer.utils
simple_metrics. NumPy samples the masks; identical frozen tasks are reused for
each tokenizer cohort. This does not reproduce TensorFlow's random bit stream
or Google's original mC4 corpus. Native perplexity comparisons require the
same tokenizer AND task fingerprint.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from akanlm.data.collator import DataCollatorForT5MLM, random_spans_noise_mask

PROTOCOL = "mt5_token_span_cpt_v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class DenoisingConfig:
    input_length: int = 1024
    target_length: int = 256
    noise_density: float = 0.15
    mean_noise_span_length: float = 3.0
    seed: int = 20260910
    max_examples: int = 10000
    concatenate_documents: int = 128
    max_document_tokens: int = 65536

    @classmethod
    def from_training(cls, config):
        """Evaluation tasks with the training run's corruption and sequence allocation.

        Validation and test must see the same inputs/targets capacity (Google:
        1024/256), noise density and span length as training, so these are read
        from the training config rather than restated here.
        """
        return cls(input_length=config.input_length, target_length=config.target_length,
                   noise_density=config.noise_density,
                   mean_noise_span_length=config.mean_noise_span_length,
                   seed=config.validation_seed)


DEFAULT_DENOISING_CONFIG = DenoisingConfig()


def tokenizer_fingerprint(tokenizer):
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    # These mutable batching options do not define the vocabulary/normalizer.
    backend.pop("padding", None)
    backend.pop("truncation", None)
    return digest({"backend": backend, "pad": tokenizer.pad_token_id,
                   "eos": tokenizer.eos_token_id, "unk": tokenizer.unk_token_id})


def build_tasks(texts, tokenizer, config=DEFAULT_DENOISING_CONFIG):
    """Select <=64K/document, concatenate 128 documents, split, corrupt, pad.

    Preserve each concatenation group's short final segment, as T5 does.
    EOS is added after corruption, never between source documents. No text is
    repeated to manufacture 10,000 examples from a smaller held-out corpus.
    """
    if config.max_examples < 1 or config.concatenate_documents < 1 or config.max_document_tokens < 2:
        raise ValueError("Invalid evaluation limits")
    collator = DataCollatorForT5MLM(
        tokenizer, input_length=config.input_length, target_length=config.target_length,
        noise_density=config.noise_density, mean_noise_span_length=config.mean_noise_span_length,
        seed=config.seed)
    chunk_rng, mask_rng = [np.random.default_rng(s) for s in np.random.SeedSequence(config.seed).spawn(2)]
    # Read once per call: len(tokenizer) takes ~4 ms on transformers 5.16, and
    # evaluating it per token made the M1 validation split an estimated 53 hours.
    from akanlm.tokenizer.special_tokens import resolve_sentinel_ids

    vocabulary_size, pad = len(tokenizer), tokenizer.pad_token_id
    sentinel_ids = set(resolve_sentinel_ids(tokenizer))
    documents = []
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        if min(ids) < 0 or max(ids) >= vocabulary_size or sentinel_ids.intersection(ids):
            raise ValueError("Source text contains reserved sentinels or invalid token IDs")
        start = int(chunk_rng.integers(math.ceil(len(ids) / config.max_document_tokens))) * config.max_document_tokens
        documents.append([i for i in ids[start:start + config.max_document_tokens] if i != pad])
    rows = []
    for start in range(0, len(documents), config.concatenate_documents):
        combined = [i for doc in documents[start:start + config.concatenate_documents] for i in doc]
        for offset in range(0, len(combined), collator.tokens_length):
            ids = np.array(combined[offset:offset + collator.tokens_length], dtype=np.int64)
            # T5 constructs a mask of at least length two, then slices it back.
            noise = random_spans_noise_mask(max(len(ids), 2), config.noise_density,
                                            config.mean_noise_span_length, mask_rng)[:len(ids)]
            inputs = collator._filter_and_insert(ids[None, :], ~noise[None, :])[0].tolist()
            targets = collator._filter_and_insert(ids[None, :], noise[None, :])[0].tolist()
            if len(inputs) > config.input_length or len(targets) > config.target_length:
                raise ValueError("Reference corruption exceeded its sequence allocation")
            rows.append({"input_ids": inputs + [0] * (config.input_length - len(inputs)),
                         "attention_mask": [1] * len(inputs) + [0] * (config.input_length - len(inputs)),
                         "labels": targets + [-100] * (config.target_length - len(targets))})
            if len(rows) == config.max_examples:
                return rows
    if not rows:
        raise ValueError("No token-denoising evaluation examples")
    return rows


def score(model, rows, *, batch_size=4, device=None):
    """Mesh teacher-forced metrics: float32 CE weighted by positive labels."""
    import torch
    from torch.nn.functional import cross_entropy

    if not rows or batch_size < 1:
        raise ValueError("Require evaluation examples and positive batch_size")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    nats = 0.
    tokens = correct = sequences = sequence_correct = label_sum = maximum = 0
    capacity = 0
    with torch.inference_mode():
        for offset in range(0, len(rows), batch_size):
            part = [rows[i] for i in range(offset, min(offset + batch_size, len(rows)))]
            batch = {k: torch.tensor([r[k] for r in part], device=device)
                     for k in ("input_ids", "attention_mask", "labels")}
            labels = batch["labels"]
            valid = labels.gt(0)
            batch["labels"] = labels.masked_fill(~valid, -100)
            logits = model(**batch).logits.float()
            losses = cross_entropy(logits.flatten(0, 1), batch["labels"].flatten(),
                                   reduction="none", ignore_index=-100).view_as(labels)
            if not torch.isfinite(losses[valid]).all():
                raise ValueError("Non-finite reference evaluation loss")
            matches = logits.argmax(-1).eq(labels) & valid
            counts = valid.sum(-1)
            nats += losses[valid].double().sum().item()
            tokens += valid.sum().item()
            correct += matches.sum().item()
            sequences += counts.gt(0).sum().item()
            sequence_correct += ((matches.sum(-1) == counts) & counts.gt(0)).sum().item()
            label_sum += labels[valid].sum().item()
            maximum = max(maximum, counts.max().item())
            capacity += labels.numel()
    if not tokens:
        raise ValueError("No positive target tokens")
    loss = nats / tokens
    return {"neg_log_perplexity": -loss, "loss_per_token": loss,
            "perplexity": math.exp(loss) if loss < 709 else None,
            "token_accuracy": correct / tokens, "sequence_accuracy": sequence_correct / sequences,
            "mean_label": label_sum / tokens, "num_eval_tokens": tokens,
            "max_targets_length": maximum, "n_examples": sequences, "total_nats": nats,
            "capacity_normalized_loss": nats / capacity}


def compare(candidate, baseline):
    for key in ("protocol", "config", "tokenizer_sha256", "task_sha256", "corpus", "split", "scoring"):
        if candidate[key] != baseline[key]:
            raise ValueError(f"Native perplexity comparison requires identical {key}")
    if candidate["metrics"]["num_eval_tokens"] != baseline["metrics"]["num_eval_tokens"]:
        raise ValueError("Native perplexity comparison requires identical target counts")
    a, b = (r["metrics"]["loss_per_token"] for r in (candidate, baseline))
    delta = a - b
    return {"candidate": candidate["model"], "baseline": baseline["model"],
            "split": candidate["split"], "corpus": candidate["corpus"],
            "lower_perplexity": a < b, "loss_reduction_nats_per_token": b-a,
            "perplexity_reduction_pct": 100 * -math.expm1(delta) if delta < 709 else None}


def weights_identity(checkpoint):
    """Local safetensors checksums, or the Hub revision for a remote id."""
    from akanlm.training.checkpoints import file_sha256

    files = sorted(Path(checkpoint).glob("*.safetensors"))
    if files:
        return {p.name: file_sha256(p) for p in files}
    from transformers import AutoConfig

    return {"hub_revision": getattr(AutoConfig.from_pretrained(checkpoint), "_commit_hash", None)}


def implementation_sha256():
    from akanlm.training.checkpoints import file_sha256

    return file_sha256(Path(__file__))


def load_checkpoint(checkpoint):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, dtype=torch.float32)
    # MT5Config can restore the tied default even for untied saved weights.
    # Forward scaling reads this flag, so recover it from the actual tensors.
    from akanlm.tokenizer.surgery import normalize_tying

    normalize_tying(model)
    return tokenizer, model


def release_memory():
    """Call after the caller has dropped its last model reference."""
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def score_checkpoint(model, tokenizer, checkpoint, corpora, manifest_sha256, *,
                     config=DEFAULT_DENOISING_CONFIG, batch_size=4, tasks=None):
    """Score one loaded model; ``tasks`` caches frozen tasks per tokenizer."""
    import torch
    import transformers

    tasks = {} if tasks is None else tasks
    token_hash = tokenizer_fingerprint(tokenizer)
    identity = weights_identity(checkpoint)
    results = []
    for (label, split), (texts, corpus) in corpora.items():
        key = (token_hash, label, split)
        if key not in tasks:
            tasks[key] = build_tasks(texts, tokenizer, config)
        rows = tasks[key]
        results.append({"model": checkpoint, "weights": identity, "protocol": PROTOCOL,
                        "config": asdict(config), "split": split, "dataset": label,
                        "split_manifest_sha256": manifest_sha256,
                        "corpus": corpus, "tokenizer_sha256": token_hash,
                        "task_sha256": digest(rows), "scoring": {"logits_dtype": "float32",
                            "model_dtype": "float32", "transformers": transformers.__version__,
                            "torch": torch.__version__, "z_loss": 0.0,
                            "implementation_sha256": implementation_sha256()},
                        "metrics": score(model, rows, batch_size=batch_size)})
    return results


def evaluate_checkpoints(checkpoints, split_manifest, *, config=DEFAULT_DENOISING_CONFIG, batch_size=4):
    """Use identical settings for validation/test, all mixtures and model sizes."""
    from akanlm.data.cpt_splits import evaluation_corpora, read_manifest

    manifest = read_manifest(split_manifest)
    corpora = {**evaluation_corpora(manifest, "validation"), **evaluation_corpora(manifest, "test")}
    results, tasks = [], {}
    for checkpoint in dict.fromkeys(checkpoints):
        tokenizer, model = load_checkpoint(checkpoint)
        results.extend(score_checkpoint(model, tokenizer, checkpoint, corpora, manifest["sha256"],
                                        config=config, batch_size=batch_size, tasks=tasks))
        del model
        release_memory()
    return results
