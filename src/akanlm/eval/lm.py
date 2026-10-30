"""Conditional denoising bits per masked character on shared text tasks.

Protocol char_span_v1 constructs document windows and masks in Unicode code
points BEFORE tokenization. Every tokenizer receives identical source/target
strings. All windows, including short tails, are scored without truncation.
The numerator includes target sentinels and EOS; the denominator counts only
original masked characters, never sentinel spellings or visible context.
This is conditional denoising cost, not full-text autoregressive BPC. Compare
only results with identical protocol and task fingerprints. CPT experiments
apply it to corpus holdouts through eval/corpus_bpc.py.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from akanlm.utils.logging import get_logger

logger = get_logger(__name__)

LN2 = math.log(2.0)
PROTOCOL = "char_span_v1"
METRIC_NOTE = (
    "bits_per_char is conditional denoising bits per masked Unicode character, "
    "including sentinel/EOS loss, not full-text BPC. Compare only identical "
    "protocol and task_sha256. Perplexity/loss_per_token require the same tokenizer."
)


@dataclass
class LMResult:
    """Held-out loss for one model on one evaluation set."""

    model: str
    dataset: str
    n_docs: int = 0
    n_chars: int = 0
    n_masked_chars: int = 0
    n_windows: int = 0
    protocol: str = "unspecified"
    task_sha256: str = ""
    protocol_config: dict = field(default_factory=dict)
    n_tokens: int = 0
    n_target_tokens: int = 0
    total_nats: float = 0.0
    provenance: str = "human"
    variety: str | None = None
    document_scores: list[dict] = field(default_factory=list)
    mask_statistics: dict = field(default_factory=dict)
    mask_runs: list[dict] = field(default_factory=list)
    uncertainty: dict = field(default_factory=dict)

    @property
    def loss_per_token(self) -> float:
        """Mean nats per target token. NOT comparable across tokenizers."""
        return (
            self.total_nats / self.n_target_tokens
            if self.n_target_tokens
            else float("nan")
        )

    @property
    def bits_per_char(self) -> float:
        """Conditional target cost, normalized by original masked characters."""
        return bits_per_char(self.total_nats, self.n_masked_chars)

    @property
    def perplexity(self) -> float:
        """Reported only within a single tokenizer, never across."""
        try:
            return math.exp(self.loss_per_token)
        except OverflowError:
            return float("inf")

    @property
    def fertility_tokens_per_char(self) -> float:
        return self.n_tokens / self.n_chars if self.n_chars else float("nan")

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "dataset": self.dataset,
            "provenance": self.provenance,
            "variety": self.variety,
            "n_docs": self.n_docs,
            "n_chars": self.n_chars,
            "n_masked_chars": self.n_masked_chars,
            "n_windows": self.n_windows,
            "protocol": self.protocol,
            "task_sha256": self.task_sha256,
            "protocol_config": self.protocol_config,
            "document_scores": self.document_scores,
            "mask_statistics": self.mask_statistics,
            "mask_runs": self.mask_runs,
            "uncertainty": self.uncertainty,
            "total_nats": self.total_nats,
            "metric_note": METRIC_NOTE,
            "n_tokens": self.n_tokens,
            "n_target_tokens": self.n_target_tokens,
            "bits_per_char": round(self.bits_per_char, 5),
            "loss_per_token": round(self.loss_per_token, 5),
            "perplexity": round(self.perplexity, 3),
            "tokens_per_char": round(self.fertility_tokens_per_char, 5),
            "comparable_across_tokenizers": "bits_per_char only, on identical text tasks",
        }


@dataclass(frozen=True)
class TextTask:
    source: str
    target: str
    n_chars: int
    n_masked_chars: int
    document_index: int = 0


def build_text_tasks(
    texts: Sequence[str],
    max_chars: int = 128,
    noise_density: float = 0.15,
    mean_span_chars: float = 12.0,
    seed: int = 20260827,
) -> list[TextTask]:
    """Create shared masks without consulting any tokenizer.

    Windows never cross documents. Character spans may split words. Whitespace
    is preserved; character counts use Python len (Unicode code points).
    A one-character tail is fully masked, so no source tail is discarded.
    """
    import numpy as np

    from akanlm.data.collator import random_spans_noise_mask

    if max_chars < 2 or not 0 < noise_density <= 0.5 or mean_span_chars < 1:
        raise ValueError(
            "Require max_chars >= 2, 0 < noise_density <= 0.5 and mean_span_chars >= 1"
        )
    rng = np.random.default_rng(seed)
    tasks = []
    for document_index, text in enumerate(texts):
        if re.search(r"<extra_id_\d+>|</s>|<pad>|<unk>", text):
            raise ValueError(
                "Evaluation text contains reserved special-token spellings"
            )
        for start in range(0, len(text), max_chars):
            window = text[start : start + max_chars]
            mask = (
                np.ones(1, dtype=bool)
                if len(window) == 1
                else random_spans_noise_mask(
                    len(window), noise_density, mean_span_chars, rng
                )
            )
            source, target = [], []
            i = sentinel = 0
            while i < len(window):
                if not mask[i]:
                    source.append(window[i])
                    i += 1
                    continue
                if sentinel >= 100:
                    raise ValueError(
                        "Text window requires more than 100 sentinels; reduce max_chars"
                    )
                marker = f"<extra_id_{sentinel}>"
                source.append(marker)
                target.append(marker)
                j = i + 1
                while j < len(window) and mask[j]:
                    j += 1
                target.append(window[i:j])
                sentinel += 1
                i = j
            tasks.append(
                TextTask("".join(source), "".join(target), len(window), int(mask.sum()), document_index)
            )
    return tasks


def evaluate(
    model,
    tokenizer,
    texts: Sequence[str],
    dataset_name: str,
    model_name: str,
    input_length: int = 512,
    noise_density: float = 0.15,
    mean_span_chars: float = 12.0,
    batch_size: int = 8,
    seed: int = 20260827,
    device: str | None = None,
    provenance: str = "human",
    variety: str | None = None,
    max_chars: int = 128,
    document_scores: bool = False,
) -> LMResult:
    """Score shared character-span tasks; reject token overflow, never truncate.

    input_length caps both encoder and target token lengths. If any tokenizer
    exceeds it, reduce max_chars for ALL compared models and rerun. The token
    cap is recorded but does not change the underlying text tasks.
    """
    import torch

    if batch_size < 1 or input_length < 2:
        raise ValueError("batch_size must be positive and input_length at least 2")
    config = {
        "max_chars": max_chars,
        "noise_density": noise_density,
        "mean_span_chars": mean_span_chars,
        "seed": seed,
    }
    tasks = build_text_tasks(texts, **config)
    if not tasks:
        raise ValueError("No nonempty evaluation text")
    # Include original documents as well as exact constructed tasks. Length-
    # prefixed JSON avoids ambiguous concatenation and catches corpus changes.
    digest = hashlib.sha256(
        json.dumps(
            [PROTOCOL, config, list(texts), [(t.source, t.target) for t in tasks]],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    result = LMResult(
        model=model_name,
        dataset=dataset_name,
        provenance=provenance,
        variety=variety,
        n_docs=sum(bool(t) for t in texts),
        n_chars=sum(t.n_chars for t in tasks),
        n_masked_chars=sum(t.n_masked_chars for t in tasks),
        n_windows=len(tasks),
        protocol=PROTOCOL,
        task_sha256=digest,
        protocol_config={**config, "token_length_limit": input_length},
        mask_statistics={
            "one_span_windows": sum(t.target.count("<extra_id_") == 1 for t in tasks),
            "total_spans": sum(t.target.count("<extra_id_") for t in tasks),
            "n_windows": len(tasks),
        },
    )
    per_document = {}
    if document_scores:
        for task in tasks:
            stats = per_document.setdefault(task.document_index, {
                "document_index": task.document_index, "total_nats": 0., "n_masked_chars": 0})
            stats["n_masked_chars"] += task.n_masked_chars
    sentinels = []
    prefixed_sentinels = False
    for i in range(100):
        marker = f"<extra_id_{i}>"
        token_id = tokenizer.convert_tokens_to_ids(marker)
        if token_id is None or token_id == tokenizer.unk_token_id:
            # Stock mT5 (including Afri-mT5) stores these pieces with the
            # SentencePiece word-boundary prefix. Use the pretrained rows;
            # adding unprefixed special tokens would allocate different IDs.
            token_id = tokenizer.convert_tokens_to_ids("▁" + marker)
            prefixed_sentinels = True
        sentinels.append(token_id)
    if (
        None in sentinels
        or len(set(sentinels)) != 100
        or tokenizer.unk_token_id in sentinels
    ):
        raise ValueError(
            f"{model_name}: Evaluation requires 100 distinct sentinel tokens "
            "(<extra_id_0> through <extra_id_99>, optionally prefixed with ▁)"
        )
    eos, pad = tokenizer.eos_token_id, tokenizer.pad_token_id
    if eos is None or pad is None:
        raise ValueError("Evaluation requires EOS and PAD token IDs")

    def encode_task(text: str) -> list[int]:
        if not prefixed_sentinels:
            return tokenizer(text, add_special_tokens=False)["input_ids"] + [eos]
        # Character masks can start inside words. Stock mT5 only recognizes
        # its sentinel spelling at a word boundary, so tokenize text spans
        # separately and insert the resolved marker IDs directly. Do not add
        # whitespace to the shared tasks or let markers become ordinary text.
        ids = []
        for part in re.split(r"(<extra_id_\d+>)", text):
            if not part:
                continue
            marker = re.fullmatch(r"<extra_id_(\d+)>", part)
            if marker:
                ids.append(sentinels[int(marker[1])])
            else:
                ids.extend(tokenizer(part, add_special_tokens=False)["input_ids"])
        return ids + [eos]

    encoded = []
    for task in tasks:
        source = encode_task(task.source)
        target = encode_task(task.target)
        if max(len(source), len(target)) > input_length:
            raise ValueError(
                f"{model_name}: text task exceeds token limit {input_length} "
                f"(encoder={len(source)}, target={len(target)}). "
                "Reduce max_chars for ALL models and rerun; truncation is forbidden."
            )
        if pad in source or pad in target or tokenizer.unk_token_id in source + target:
            raise ValueError(
                f"{model_name}: text task contains PAD/UNK; cannot score the same text faithfully"
            )
        encoded.append((source, target))
    result.n_tokens = sum(
        len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in texts
    )
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    with torch.no_grad():
        for start in range(0, len(encoded), batch_size):
            window = encoded[start : start + batch_size]
            inputs = torch.full(
                (len(window), max(len(x) for x, _ in window)),
                pad,
                dtype=torch.long,
                device=device,
            )
            labels = torch.full(
                (len(window), max(len(y) for _, y in window)),
                -100,
                dtype=torch.long,
                device=device,
            )
            for i, (source, target) in enumerate(window):
                inputs[i, : len(source)] = torch.tensor(source, device=device)
                labels[i, : len(target)] = torch.tensor(target, device=device)
            output = model(
                input_ids=inputs, attention_mask=inputs.ne(pad).long(), labels=labels
            )
            n_target = int(labels.ne(-100).sum())
            if document_scores:
                # Recompute plain cross-entropy from logits. Auxiliary training
                # penalties must never enter likelihood or bootstrap estimates.
                token_losses = torch.nn.functional.cross_entropy(
                    output.logits.float().transpose(1, 2), labels, ignore_index=-100, reduction="none")
                row_nats = token_losses.sum(dim=1).tolist()
                batch_nats = sum(row_nats)
                for offset, nats in enumerate(row_nats):
                    doc = tasks[start + offset].document_index
                    per_document[doc]["total_nats"] += nats
            else:
                batch_nats = float(output.loss) * n_target
            if not math.isfinite(batch_nats):
                raise ValueError(f"{model_name}: non-finite evaluation loss")
            result.total_nats += batch_nats
            result.n_target_tokens += n_target
    result.document_scores = list(per_document.values())
    logger.info(
        "%s on %s: conditional bpc=%.4f over %d masked chars (%s)",
        model_name,
        dataset_name,
        result.bits_per_char,
        result.n_masked_chars,
        PROTOCOL,
    )
    return result


def bits_per_char(total_nats: float, n_chars: int) -> float:
    """Convert total nats over the scored characters into bits per character."""
    return total_nats / (LN2 * n_chars) if n_chars else float("nan")
