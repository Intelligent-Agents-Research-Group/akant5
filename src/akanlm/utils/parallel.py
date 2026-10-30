"""Bounded process pool with serial-equivalent input/output ordering."""

from __future__ import annotations

import multiprocessing
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from typing import Any, Callable, Iterable, Iterator

_shared: Any = None


def _install(shared: Any) -> None:
    global _shared
    _shared = shared


def _apply(fn: Callable, batch: list) -> list:
    return [fn(_shared, item) for item in batch]


class OrderedPool:
    """Map ``fn(shared, item)`` in input order; workers must not mutate shared state.

    Use a module-level function or partial with small bound arguments.
    Keep at most ``2 * workers`` batches pending; ``workers <= 1`` runs lazily in-process."""

    def __init__(self, shared: Any = None, workers: int = 1, batch_size: int = 64) -> None:
        self.shared = shared
        self.workers = workers
        self.batch_size = batch_size
        self._executor = None
        if workers > 1:
            # spawn, not fork. A forked worker would inherit the caller's heap
            # (the parsed split manifest), and locks held by its pyarrow threads.
            self._executor = ProcessPoolExecutor(
                workers, mp_context=multiprocessing.get_context("spawn"),
                initializer=_install, initargs=(shared,))

    def map(self, fn: Callable, items: Iterable) -> Iterator:
        items = iter(items)
        if self._executor is None:
            for item in items:
                yield fn(self.shared, item)
            return
        pending: deque = deque()
        while True:
            while len(pending) < 2 * self.workers and (batch := list(islice(items, self.batch_size))):
                pending.append(self._executor.submit(_apply, fn, batch))
            if not pending:
                return
            yield from pending.popleft().result()

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(cancel_futures=True)
            self._executor = None

    def __enter__(self) -> "OrderedPool":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
