"""Initialize new embedding rows from mT5's pretrained ones.

Replacing mT5's 250k vocabulary with a ~32k Twi/English one throws away the
input embedding matrix and the untied output head — between 42% and 85% of the
checkpoint's parameters depending on size. Reinitializing them randomly would
discard pretrained knowledge the encoder body still expects, so each new row is
derived from the old matrix:

``overlap``
    The new piece string exists verbatim in mT5's vocabulary. Copy its row.
    This covers the special tokens, all 256 byte-fallback pieces, most English
    subwords and the Akan pieces mT5 happened to have (it has ``ɔ`` and ``ɛ``
    as isolated characters, just not as parts of Twi words).

``subword_mean``
    The new piece is new. Tokenize its surface form with mT5's tokenizer and
    average the resulting rows. A new Twi piece like ``▁ɔfrɛ`` becomes the mean
    of mT5's ``▁``, ``ɔ``, ``fr``, ``ɛ``. This is an initialization heuristic,
    not an equivalent contextual representation or sequence probability.

``sentinel_alias``
    Match ``<extra_id_N>`` and ``▁<extra_id_N>`` by identity and copy the
    pretrained row exactly, without rescaling either the input or output row.

``fallback``
    Neither worked. Use the mean of all pretrained rows. Expected to be rare;
    the count is reported so a large number is visible.

Rows past the tokenizer (vocabulary padding) also get the pretrained mean.
Neither these rows nor fallback rows may be zero or random: mT5's output head
gives almost every token a large negative logit (mean about -23 on Twi text
for mt5-small), so a zero row scores 0 and a random row scores around 0 and
both outrank every real token. The first AkanT5 surgery used Gaussian
fallback rows and zero padding rows. Its 8 Gaussian ``lm_head`` rows took all
probability mass, and its span-corruption loss was 27.7 nats against
ln(32128) = 10.4 for a uniform prediction.

With ``rescale_derived``, ``subword_mean`` rows are rescaled to the mean L2
norm of the copied rows, since averaging shrinks vectors toward the origin.
Surgery enables this only for ``lm_head``. On 64 x 512-token span-corruption
batches for mt5-small, disabling it for the input embeddings lowered initial
loss on Twi Wikipedia, Pristine Twi and Pristine English by 0.34-0.57 nats,
while disabling it for ``lm_head`` raised Twi loss. Scaling an output row
scales its logit, and with mostly negative logits a longer row lowers a new
token's probability rather than raising it.
"""

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
    """Surface form of a SentencePiece piece, for re-tokenization.

    ``▁ɔfrɛ`` denotes a word-initial ``ɔfrɛ``, so it becomes ``" ɔfrɛ"`` —
    keeping the leading space matters, since mT5 tokenizes word-initial and
    word-internal occurrences differently.
    """
    return piece.replace(WORD_BOUNDARY, " ")


def build_matrix(
    old_matrix: "torch.Tensor",
    old_tokenizer,
    new_tokenizer,
    target_rows: int,
    matrix_name: str,
    rescale_derived: bool = True,
) -> tuple["torch.Tensor", TransferReport]:
    """Derive a ``target_rows x d`` matrix from *old_matrix*.

    Args:
        old_matrix: mT5's ``V_old x d`` embedding or ``lm_head`` weight.
        old_tokenizer: mT5's tokenizer.
        new_tokenizer: The AkanT5 tokenizer.
        target_rows: Padded model vocab size; rows beyond the tokenizer's
            length get the mean pretrained row (see the module docstring).
        matrix_name: Label for the report.
        rescale_derived: Rescale ``subword_mean`` rows to the copied rows'
            mean norm.
    Returns:
        ``(new_matrix, report)``. ``new_matrix`` matches *old_matrix* in dtype
        and device.
    """
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
