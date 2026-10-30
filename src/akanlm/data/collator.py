"""T5-style span corruption data collator for denoising sequence-to-sequence training.

This module implements the T5-inspired span corruption objective for
encoder-decoder denoising NLP training. The collator samples noise spans,
replaces each dropped span with a unique sentinel token in the encoder
input, and emits a decoder target sequence containing the removed spans in
source order. The implementation follows the deterministic T5 corruption
length relation and expects a tokenizer that reserves sentinel token ids at
the top of the vocabulary.

The primary public interface is :class:`DataCollatorForT5MLM`. It validates
training configuration, derives the expected raw-token and target-token
lengths, and returns padded PyTorch tensors for encoder and decoder inputs.
Helper utilities such as :func:`compute_input_and_target_lengths` and
:func:`random_spans_noise_mask` reproduce the span-layout behavior described
by the original T5 span corruption objective.

Example:
    >>> collator = DataCollatorForT5MLM(tokenizer=tokenizer, input_length=512,
    ...                                 target_length=114)
    >>> batch = collator(examples)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def compute_input_and_target_lengths(
    inputs_length: int,
    noise_density: float = 0.15,
    mean_noise_span_length: float = 3.0,
) -> tuple[int, int]:
    """Return the raw-token and target-token lengths required for T5-style span corruption.

    This helper ports the T5 ``random_spans_helper`` logic and solves for the
    smallest raw token count whose corrupted form fits the requested encoder
    length exactly. The goal is to avoid truncation or padding during packing
    and to keep the encoder-decoder sequence shapes deterministic.

    Args:
        inputs_length: Desired post-corruption encoder length.
        noise_density: Fraction of tokens assigned to corrupted spans.
        mean_noise_span_length: Average corrupted span width.

    Returns:
        A tuple ``(tokens_length, targets_length)`` where ``tokens_length`` is
        the raw token count required to pack the example and
        ``targets_length`` is the resulting decoder target length.
    """

    def _segment_lengths(tokens_length: int) -> tuple[int, int]:
        num_noise_tokens = int(round(tokens_length * noise_density))
        num_nonnoise_tokens = tokens_length - num_noise_tokens
        num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
        num_noise_spans = max(num_noise_spans, 1)
        # Each sequence gets one sentinel per span, plus a trailing EOS.
        return (
            num_nonnoise_tokens + num_noise_spans + 1,
            num_noise_tokens + num_noise_spans + 1,
        )

    tokens_length = inputs_length
    while _segment_lengths(tokens_length + 1)[0] <= inputs_length:
        tokens_length += 1

    inputs, targets = _segment_lengths(tokens_length)

    # Shrinking by one keeps `inputs` at exactly inputs_length in edge cases.
    if inputs > inputs_length and tokens_length > 0:
        tokens_length -= 1
        inputs, targets = _segment_lengths(tokens_length)

    return tokens_length, targets


def random_spans_noise_mask(
    length: int,
    noise_density: float,
    mean_noise_span_length: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create a T5-style boolean noise mask for span corruption.

    This helper generates a binary mask that marks token positions selected
    for corruption. It mirrors the T5 ``random_spans_noise_mask`` procedure by
    sampling noise spans and non-noise spans, then interleaving them so that
    the corrupted and preserved segments alternate in the sequence.

    Args:
        length: Original sequence length before masking.
        noise_density: Fraction of tokens in the sequence to corrupt.
        mean_noise_span_length: Target average length of each noise span.
        rng: NumPy random generator used to sample span boundaries.

    Returns:
        A NumPy boolean mask of length ``length`` where ``True`` indicates a
        corrupted token position.
    """
    orig_length = length
    # Guarantee at least one noise span and one non-noise span.
    length = max(length, 2)

    num_noise_tokens = int(np.round(length * noise_density))
    num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
    num_noise_spans = int(np.round(num_noise_tokens / mean_noise_span_length))
    num_noise_spans = max(num_noise_spans, 1)
    num_nonnoise_tokens = length - num_noise_tokens

    def _random_segmentation(num_items: int, num_segments: int) -> np.ndarray:
        """Split *num_items* into *num_segments* positive-length pieces."""
        bars = np.arange(num_items - 1) < num_segments - 1
        rng.shuffle(bars)
        bars = np.pad(bars, (1, 0))
        segment_id = np.cumsum(bars)
        _, lengths = np.unique(segment_id, return_counts=True)
        return lengths

    noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
    nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

    interleaved = np.reshape(
        np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
        [num_noise_spans * 2],
    )
    span_starts = np.cumsum(interleaved)[:-1]
    span_start_indicator = np.zeros(length, dtype=np.int8)
    span_start_indicator[span_starts] = 1
    span_num = np.cumsum(span_start_indicator)
    is_noise = np.equal(span_num % 2, 1)

    return is_noise[:orig_length]


@dataclass
class DataCollatorForT5MLM:
    """Turn packed token sequences into span-corruption examples.

    Args:
        tokenizer: Provides the sentinel ids and ``eos_token_id``.
        noise_density: Fraction of tokens corrupted.
        mean_noise_span_length: Average corrupted span width.
        input_length: Post-corruption encoder length.
        target_length: Post-corruption decoder length.
        pad_token_id: Padding id.
        decoder_start_token_id: T5 starts the decoder with ``<pad>``.
        seed: Base seed; each call advances a private generator so runs are
            reproducible and independent of global numpy state.
    """

    tokenizer: Any
    noise_density: float = 0.15
    mean_noise_span_length: float = 3.0
    input_length: int = 512
    target_length: int = 114
    pad_token_id: int = 0
    decoder_start_token_id: int = 0
    seed: int = 13

    def __post_init__(self) -> None:
        if not 0 < self.noise_density < 1 or self.mean_noise_span_length < 1:
            raise ValueError("Require 0 < noise_density < 1 and mean_noise_span_length >= 1")
        if self.input_length < 2 or self.target_length < 2:
            raise ValueError("T5 corruption lengths must be at least 2")
        if self.pad_token_id != self.decoder_start_token_id:
            raise ValueError("T5 decoder_start_token_id must equal pad_token_id")
        self.tokens_length, expected_target = compute_input_and_target_lengths(
            self.input_length, self.noise_density, self.mean_noise_span_length
        )
        if self.target_length < expected_target:
            raise ValueError(
                f"target_length={self.target_length}; expected at least {expected_target} "
                f"for input_length={self.input_length}"
            )
        self._unpadded_target_length = expected_target
        self._rng = np.random.default_rng(self.seed)
        self._rng_worker_seed: int | None = None
        size = len(self.tokenizer)
        if size <= 100:
            raise ValueError("T5 span corruption requires 100 sentinel tokens")
        # Extension leaves sentinels at their original IDs; lexical IDs may
        # follow them. Resolve by identity for both vocabulary strategies.
        from akanlm.tokenizer.special_tokens import resolve_sentinel_ids

        self._sentinel_ids = resolve_sentinel_ids(self.tokenizer)
        vocab = self.tokenizer.get_vocab()
        actual = [vocab.get(f"<extra_id_{i}>", vocab.get(f"▁<extra_id_{i}>"))
                  for i in range(100)]
        if actual != self._sentinel_ids:
            raise ValueError("Tokenizer sentinel layout does not match the CPT vocabulary")
        n_noise = round(self.tokens_length * self.noise_density)
        n_spans = max(round(n_noise / self.mean_noise_span_length), 1)
        if n_spans > min(100, n_noise, self.tokens_length - n_noise):
            raise ValueError("Corruption requests more spans than tokens or available sentinels")
        self.eos_token_id = self.tokenizer.eos_token_id
        if self.eos_token_id is None or self.eos_token_id == self.pad_token_id:
            raise ValueError("T5 requires a non-padding EOS token")

    def _mask_rng(self):
        """Return a deterministic, worker-aware NumPy random generator.

        The collator samples corruption spans inside the dataloader worker
        process. When `num_workers > 0`, PyTorch forks data workers and each
        worker receives a distinct `worker_info.seed`. We derive the worker
        generator from the configured collator seed plus that worker seed so
        corruption masks remain reproducible while avoiding identical
        span layouts across workers and epochs.

        When `num_workers == 0`, there is no worker fork and the stored RNG
        advances directly across batches.
        """
        import torch.utils.data

        info = torch.utils.data.get_worker_info()
        if info is None:
            return self._rng
        if self._rng_worker_seed != info.seed:
            self._rng_worker_seed = info.seed
            self._rng = np.random.default_rng([self.seed, info.seed])
        return self._rng

    def sentinel_id(self, index: int) -> int:
        """Id of ``<extra_id_{index}>``."""
        return self._sentinel_ids[index]

    def __call__(self, examples: list[dict]) -> dict:
        import torch

        if not examples or any(len(ex["input_ids"]) != self.tokens_length for ex in examples):
            raise ValueError(f"CPT requires nonempty batches of exactly {self.tokens_length} raw tokens")
        input_ids = np.array([example["input_ids"] for example in examples], dtype=np.int64)
        if np.any(input_ids == self.pad_token_id):
            raise ValueError("Packed CPT inputs must not contain padding tokens")
        if (
            np.any(input_ids < 0)
            or np.any(input_ids >= len(self.tokenizer))
            or np.any(np.isin(input_ids, self._sentinel_ids))
        ):
            raise ValueError("Packed CPT inputs contain invalid IDs or reserved sentinels")
        batch_size, length = input_ids.shape

        rng = self._mask_rng()
        mask = np.stack(
            [
                random_spans_noise_mask(
                    length, self.noise_density, self.mean_noise_span_length, rng
                )
                for _ in range(batch_size)
            ]
        )

        # ``mask`` marks the NOISE positions. The encoder keeps everything else
        # and shows one sentinel per removed span; the decoder target is the
        # mirror image — the removed spans, each preceded by the sentinel that
        # stands for it in the input. Passing ``mask`` here rather than ``~mask``
        # inverts the objective into "see 15%, generate 85%" and truncates most
        # of the target away, which is undetectable from shapes alone.
        inputs = self._filter_and_insert(input_ids, ~mask)
        targets = self._filter_and_insert(input_ids, mask)

        # Verify the exact corruption length before padding to the decoder
        # allocation (Google uses 256 slots for 229 actual target tokens).
        target_array = self._to_rect(targets, self._unpadded_target_length)
        target_array = np.pad(target_array,
                              ((0, 0), (0, self.target_length - self._unpadded_target_length)),
                              constant_values=self.pad_token_id)
        batch = {
            "input_ids": torch.from_numpy(self._to_rect(inputs, self.input_length)),
            "labels": torch.from_numpy(target_array),
        }
        batch["attention_mask"] = (batch["input_ids"] != self.pad_token_id).long()
        # -100 tells the loss to ignore padded target positions.
        batch["labels"] = batch["labels"].masked_fill(
            batch["labels"] == self.pad_token_id, -100
        )
        return batch

    def _filter_and_insert(self, input_ids: np.ndarray, keep: np.ndarray) -> list[np.ndarray]:
        """Keep the marked tokens; replace each dropped run with one sentinel."""
        rows: list[np.ndarray] = []
        for row_ids, row_keep in zip(input_ids, keep):
            out: list[int] = []
            sentinel = 0
            index = 0
            while index < len(row_ids):
                if row_keep[index]:
                    out.append(int(row_ids[index]))
                    index += 1
                    continue
                # One sentinel stands in for the whole dropped run.
                if sentinel < len(self._sentinel_ids):
                    out.append(self._sentinel_ids[sentinel])
                    sentinel += 1
                while index < len(row_ids) and not row_keep[index]:
                    index += 1
            out.append(self.eos_token_id)
            rows.append(np.array(out, dtype=np.int64))
        return rows

    def _to_rect(self, rows: list[np.ndarray], width: int) -> np.ndarray:
        """Require exact lengths; padding/truncation would hide objective bugs."""
        lengths = [len(row) for row in rows]
        if any(length != width for length in lengths):
            raise ValueError(f"Corrupted lengths {lengths} do not match expected width {width}")
        return np.stack(rows)
