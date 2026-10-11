"""Micro-batching: encode concurrent requests together.

Encoding 8 queries costs little more than encoding 1, so under load it pays
to hold each request for a few milliseconds and encode whatever has arrived
as one batch. The trade is a bounded amount of added latency per request
(at most `max_wait_ms` of waiting) for more throughput.

One worker coroutine owns the queue. It blocks for the first request, then
collects more until the batch is full or the window closes, whichever comes
first. The encoder runs in a thread: it is CPU-bound, and running it on the
event loop would stall every other request while it works.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

EncodeBatch = Callable[[list[str]], np.ndarray]


@dataclass
class _Pending:
    text: str
    future: asyncio.Future[np.ndarray]


@dataclass
class BatcherStats:
    batches: int = 0
    items: int = 0
    sizes: list[int] = field(default_factory=list)

    @property
    def mean_batch(self) -> float:
        return self.items / self.batches if self.batches else 0.0


class MicroBatcher:
    def __init__(self, encode: EncodeBatch, max_batch: int = 16, max_wait_ms: float = 5.0) -> None:
        assert max_batch >= 1 and max_wait_ms >= 0
        self._encode = encode
        self._max_batch = max_batch
        self._max_wait = max_wait_ms / 1000
        self._queue: asyncio.Queue[_Pending] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None
        self.stats = BatcherStats()

    async def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker
            self._worker = None

    async def encode(self, text: str) -> np.ndarray:
        """One query in, its vector out; batching is invisible to the caller."""
        future: asyncio.Future[np.ndarray] = asyncio.get_running_loop().create_future()
        await self._queue.put(_Pending(text, future))
        return await future

    async def _run(self) -> None:
        while True:
            batch = [await self._queue.get()]
            deadline = time.perf_counter() + self._max_wait
            while len(batch) < self._max_batch:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self._queue.get(), remaining))
                except TimeoutError:
                    break
            await self._flush(batch)

    async def _flush(self, batch: list[_Pending]) -> None:
        self.stats.batches += 1
        self.stats.items += len(batch)
        self.stats.sizes.append(len(batch))
        try:
            vectors = await asyncio.to_thread(self._encode, [p.text for p in batch])
        except Exception as exc:  # every caller in the batch hears about it
            for p in batch:
                if not p.future.done():
                    p.future.set_exception(exc)
            return
        for p, vector in zip(batch, vectors, strict=True):
            if not p.future.done():
                p.future.set_result(vector)
