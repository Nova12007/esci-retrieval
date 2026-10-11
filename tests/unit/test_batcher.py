from __future__ import annotations

import asyncio

import numpy as np
import pytest

from esci.serving.batcher import MicroBatcher


def _echo(texts: list[str]) -> np.ndarray:
    """Vector i encodes the integer in text i, so routing errors show."""
    return np.array([[float(t)] for t in texts], dtype=np.float32)


def _run(coro):  # type: ignore[no-untyped-def]
    return asyncio.run(coro)


def test_each_caller_gets_its_own_vector() -> None:
    async def go() -> list[float]:
        b = MicroBatcher(_echo, max_batch=8, max_wait_ms=20)
        await b.start()
        out = await asyncio.gather(*(b.encode(str(i)) for i in range(20)))
        await b.stop()
        return [float(v[0]) for v in out]

    assert _run(go()) == [float(i) for i in range(20)]


def test_concurrent_requests_are_batched_up_to_max_batch() -> None:
    async def go() -> list[int]:
        b = MicroBatcher(_echo, max_batch=8, max_wait_ms=50)
        await b.start()
        await asyncio.gather(*(b.encode(str(i)) for i in range(20)))
        await b.stop()
        return b.stats.sizes

    sizes = _run(go())
    assert sum(sizes) == 20 and max(sizes) == 8 and len(sizes) == 3


def test_a_lone_request_is_not_held_past_the_window() -> None:
    async def go() -> float:
        b = MicroBatcher(_echo, max_batch=64, max_wait_ms=10)
        await b.start()
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        await b.encode("1")
        elapsed = loop.time() - t0
        await b.stop()
        return elapsed

    assert _run(go()) < 0.5  # window is 10 ms; generous bound for slow CI


def test_encoder_failure_reaches_every_caller_in_the_batch() -> None:
    def broken(texts: list[str]) -> np.ndarray:
        raise RuntimeError("model fell over")

    async def go() -> list[BaseException | np.ndarray]:
        b = MicroBatcher(broken, max_batch=4, max_wait_ms=20)
        await b.start()
        out = await asyncio.gather(*(b.encode(str(i)) for i in range(4)), return_exceptions=True)
        await b.stop()
        return out

    results = _run(go())
    assert all(isinstance(r, RuntimeError) for r in results)


def test_rejects_nonsense_settings() -> None:
    with pytest.raises(AssertionError):
        MicroBatcher(_echo, max_batch=0)
