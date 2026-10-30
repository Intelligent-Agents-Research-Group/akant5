"""Resolve sentinels by identity, including nonfinal extension IDs.

Reject missing, duplicate, out-of-vocabulary, unknown, or padding aliases."""

from __future__ import annotations


def resolve_sentinel_ids(tokenizer, count: int = 100) -> list[int]:
    """Return ordered sentinel IDs, accepting mT5's whitespace-prefixed alias."""
    vocabulary = tokenizer.get_vocab()
    vocabulary_size = len(tokenizer)
    forbidden = {
        token_id
        for token_id in (
            tokenizer.unk_token_id,
            tokenizer.pad_token_id,
            tokenizer.eos_token_id,
        )
        if token_id is not None
    }
    sentinel_ids: list[int] = []
    for index in range(count):
        canonical = f"<extra_id_{index}>"
        alias = f"▁{canonical}"
        token_id = vocabulary.get(canonical, vocabulary.get(alias))
        if (
            not isinstance(token_id, int)
            or token_id < 0
            or token_id >= vocabulary_size
            or token_id in forbidden
            or token_id in sentinel_ids
        ):
            raise ValueError(f"Invalid or missing sentinel token {canonical}: {token_id}")
        sentinel_ids.append(token_id)
    return sentinel_ids
