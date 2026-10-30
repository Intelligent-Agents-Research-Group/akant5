"""T5 span corruption with identity-resolved sentinels.

Inputs retain unmasked text; targets contain removed spans in source order.
Derive exact raw/target lengths, then pad targets with ignored labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def compute_input_and_target_lengths(
    inputs_length: int,
    noise_density: float = 0.15,
    mean_noise_span_length: float = 3.0,
) -> tuple[int, int]:
    """Return raw/target lengths yielding exactly ``inputs_length`` encoder tokens."""

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
    """Return a boolean corruption mask; True marks noise.

    Random noise/non-noise spans alternate under the supplied NumPy generator."""
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
    """Build sentinel-masked inputs and ordered removed-span targets from packed tokens.

    Keep fixed lengths and a private, advancing RNG."""

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
        """Seed each worker RNG from the collator and PyTorch worker seeds.

        Without workers, advance the stored generator directly."""
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

        # Invert the mask for target sentinels: reconstruct noise, not retained text.
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
