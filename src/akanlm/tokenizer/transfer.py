"""Initialize vocabulary rows from pretrained mT5 matrices.

Copy matching pieces/sentinels; otherwise average mT5 subwords or all rows.
Padding also uses pretrained means; zero/random output rows can dominate logits.
Rescale only derived LM-head rows to copied-row mean L2 norm; inputs stay unscaled.
Subword averaging is an initialization heuristic, not contextual/probability equivalence."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import re
from typing import TYPE_CHECKING

from akanlm.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    import torch

logger = get_logger(__name__)

#: SentencePiece's word-boundary marker, used by both mT5 and our tokenizer.
WORD_BOUNDARY = "▁"


@dataclass
class TransferReport:
    """Per-strategy counts and norm statistics, recorded in the model card."""

    matrix: str
    new_vocab_size: int
    strategies: Counter = field(default_factory=Counter)
    mean_norm_copied: float = 0.0
    mean_norm_before_rescale: float = 0.0
    rescale_derived: bool = True
    unmapped_tokens: list[str] = field(default_factory=list)

    @property
    def overlap_rate(self) -> float:
        total = sum(self.strategies.values())
        return self.strategies["overlap"] / total if total else 0.0

    def to_dict(self) -> dict:
        return {
            "matrix": self.matrix,
            "new_vocab_size": self.new_vocab_size,
            "strategies": dict(self.strategies),
            "overlap_rate": round(self.overlap_rate, 6),
            "mean_norm_copied": round(self.mean_norm_copied, 6),
            "mean_norm_before_rescale": round(self.mean_norm_before_rescale, 6),
            "rescale_derived": self.rescale_derived,
            "unmapped_sample": self.unmapped_tokens[:50],
            "unmapped_count": len(self.unmapped_tokens),
        }


def _piece_to_text(piece: str) -> str:
    """Decode a SentencePiece surface form, preserving word-initial leading space."""
    return piece.replace(WORD_BOUNDARY, " ")


def build_matrix(
    old_matrix: "torch.Tensor",
    old_tokenizer,
    new_tokenizer,
    target_rows: int,
    matrix_name: str,
    rescale_derived: bool = True,
) -> tuple["torch.Tensor", TransferReport]:
    """Return ``(matrix, report)`` with source dtype/device and ``target_rows × d`` shape.

    Copy/average rows, fill padding with pretrained means, and optionally rescale
    subword means to copied-row mean L2 norm."""
    import torch

    old_vocab = old_tokenizer.get_vocab()
    new_vocab = new_tokenizer.get_vocab()
    n_tokens = len(new_tokenizer)
    d_model = old_matrix.shape[1]

    report = TransferReport(matrix=matrix_name, new_vocab_size=target_rows, rescale_derived=rescale_derived)
    new_matrix = torch.zeros((target_rows, d_model), dtype=torch.float32)

    source = old_matrix.detach().to(torch.float32).cpu()
    # Rows past the old tokenizer are mT5's own vocabulary padding.
    pretrained_mean = source[: min(len(old_tokenizer), source.shape[0])].mean(dim=0)
    copied_norms: list[float] = []
    derived_rows: list[int] = []
    fallback_rows: list[int] = []

    id_to_piece = {index: piece for piece, index in new_vocab.items()}

    for new_id in range(n_tokens):
        piece = id_to_piece.get(new_id)
        if piece is None:
            report.strategies["missing_piece"] += 1
            fallback_rows.append(new_id)
            continue

        old_id = old_vocab.get(piece)
        if old_id is not None and old_id < source.shape[0]:
            new_matrix[new_id] = source[old_id]
            copied_norms.append(float(source[old_id].norm()))
            report.strategies["overlap"] += 1
            continue

        sentinel = re.fullmatch(r"▁?(<extra_id_\d+>)", piece)
        if sentinel:
            alias = ("" if piece.startswith(WORD_BOUNDARY) else WORD_BOUNDARY) + sentinel[1]
            old_id = old_vocab.get(alias)
            if old_id is None or not 0 <= old_id < source.shape[0]:
                raise ValueError(f"No pretrained sentinel row for {piece}")
            new_matrix[new_id] = source[old_id]
            report.strategies["sentinel_alias"] += 1
            # Sentinel aliases stay out of the literal-overlap norm reference,
            # so ordinary derived rows are scaled by literal overlaps alone.
            continue

        pieces = old_tokenizer(
            _piece_to_text(piece), add_special_tokens=False
        )["input_ids"]
        pieces = [i for i in pieces if i < source.shape[0]]
        if pieces:
            new_matrix[new_id] = source[pieces].mean(dim=0)
            report.strategies["subword_mean"] += 1
            derived_rows.append(new_id)
        else:
            report.strategies["fallback"] += 1
            report.unmapped_tokens.append(piece)
            fallback_rows.append(new_id)

    mean_norm = sum(copied_norms) / len(copied_norms) if copied_norms else 1.0
    report.mean_norm_copied = mean_norm

    if derived_rows:
        index = torch.tensor(derived_rows, dtype=torch.long)
        norms = new_matrix[index].norm(dim=1, keepdim=True)
        report.mean_norm_before_rescale = float(norms.mean())
        if rescale_derived:
            new_matrix[index] = new_matrix[index] * (mean_norm / norms.clamp_min(1e-8))

    # Unsourced and padding rows keep the pretrained logit offset; zero or
    # random rows would outrank every real token in lm_head.
    if fallback_rows:
        new_matrix[torch.tensor(fallback_rows, dtype=torch.long)] = pretrained_mean
    new_matrix[n_tokens:] = pretrained_mean

    logger.info(
        "%s: %d rows | overlap=%d sentinel_alias=%d subword_mean=%d fallback=%d | overlap rate %.1f%% | "
        "rescale_derived=%s",
        matrix_name,
        target_rows,
        report.strategies["overlap"],
        report.strategies["sentinel_alias"],
        report.strategies["subword_mean"],
        report.strategies["fallback"],
        report.overlap_rate * 100,
        rescale_derived,
    )
    return new_matrix.to(old_matrix.dtype), report
