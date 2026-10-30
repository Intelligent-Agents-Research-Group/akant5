"""Order-preserving process pool for CPU-bound corpus preparation.

CPython executes Python bytecode in one thread at a time, so text
normalization, the exclusion guard's n-gram scan and per-document tokenizer
calls use more than one core only in separate processes. Results come back in
input order, so the stateful steps applied to them in the calling process
(character budgets, filter counts, first-seen deduplication, packing) see the
same sequence as a serial run.
"""

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
    """Apply ``fn(shared, item)`` to a stream in worker processes, in input order.

    ``shared`` is read-only state, such as an exclusion index and a tokenizer,
    pickled once per worker at start-up. ``fn`` must be a module-level function
    or a ``functools.partial`` of one. It is pickled with every batch, so it
    should bind only small arguments. Workers never return state to the caller,
    so ``fn`` must not rely on mutating ``shared``.

    At most ``2 * workers`` batches are outstanding at a time. That bounds
    memory, and the work wasted when a consumer stops early. With
    ``workers <= 1``, items are processed lazily in the calling process.
    """

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
