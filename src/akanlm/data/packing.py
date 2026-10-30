"""Concatenate documents into fixed-length chunks; omit document EOS by default.

The collator adds sequence EOS. Each chunk takes its largest-contributor provenance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

from akanlm.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PackedChunk:
    """One fixed-length training sequence."""

    input_ids: list[int]
    provenance: str
    source: str

    def to_dict(self) -> dict:
        return {
            "input_ids": self.input_ids,
            "provenance": self.provenance,
            "source": self.source,
        }


@dataclass
class PackingStats:
    docs: int = 0
    tokens: int = 0
    chunks: int = 0
    dropped_tail_tokens: int = 0

    def to_dict(self) -> dict:
        return {
            "docs": self.docs,
            "tokens": self.tokens,
            "chunks": self.chunks,
            "dropped_tail_tokens": self.dropped_tail_tokens,
            "tokens_per_doc": round(self.tokens / self.docs, 2) if self.docs else 0.0,
        }


def encode(tokenizer, text: str) -> list[int]:
    """Token ids of one document, without special tokens.

    Module-level so ``functools.partial(encode, tokenizer)`` can be pickled into
    worker processes.
    """
    # Documents are split into tokens_length chunks by pack, so exceeding
    # model_max_length here is expected; suppress that warning.
    return tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"]


def pack(
    documents: Iterable[tuple[str | list[int], str, str]],
    tokenizer,
    tokens_length: int,
    stats: PackingStats | None = None,
    append_eos: bool = True,
) -> Iterator[PackedChunk]:
    """Yield exact ``tokens_length`` chunks from text or pre-encoded documents.

    Optional ``append_eos`` inserts document separators; CPT leaves it off.
    Drop and count the final partial chunk."""
    stats = stats if stats is not None else PackingStats()
    eos = tokenizer.eos_token_id

    buffer: list[int] = []
    # Provenance/source of the tokens currently in the buffer, so a chunk can
    # be attributed to whatever contributed most of it.
    tags: list[tuple[str, str]] = []

    for text, source, provenance in documents:
        ids = encode(tokenizer, text) if isinstance(text, str) else list(text)
        if not ids:
            continue
        if append_eos:
            ids.append(eos)

        buffer.extend(ids)
        tags.extend([(source, provenance)] * len(ids))
        stats.docs += 1
        stats.tokens += len(ids)

        while len(buffer) >= tokens_length:
            chunk_ids = buffer[:tokens_length]
            chunk_tags = tags[:tokens_length]
            del buffer[:tokens_length]
            del tags[:tokens_length]

            source_name, provenance_name = _majority(chunk_tags)
            stats.chunks += 1
            yield PackedChunk(
                input_ids=chunk_ids, provenance=provenance_name, source=source_name
            )

    stats.dropped_tail_tokens += len(buffer)
    logger.info(
        "packed %d docs -> %d chunks of %d tokens (%d tail tokens dropped)",
        stats.docs,
        stats.chunks,
        tokens_length,
        stats.dropped_tail_tokens,
    )


def _majority(tags: list[tuple[str, str]]) -> tuple[str, str]:
    """Most common ``(source, provenance)`` pair in a chunk."""
    from collections import Counter

    if not tags:
        return ("unknown", "unknown")
    return Counter(tags).most_common(1)[0][0]
