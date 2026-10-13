"""The served search path: cache -> encode -> HNSW -> hydrate.

Every stage is timed and returned with the results, so the response itself
says where the milliseconds went (ADR-009, ADR-016 for the measured parts).
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import faiss
import numpy as np

from esci.models.config import EncoderConfig
from esci.serving.batcher import MicroBatcher
from esci.serving.catalog import Catalog
from esci.serving.onnx_encoder import OnnxQueryEncoder

FETCH = 100  # retrieve and cache the top 100 once; any k <= 100 is a slice


@dataclass(frozen=True)
class Hit:
    product_id: str
    title: str
    brand: str
    score: float


@dataclass
class SearchResult:
    hits: list[Hit]
    cached: bool
    timings_ms: dict[str, float] = field(default_factory=dict)


class Engine(Protocol):
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def search(self, query: str, k: int) -> SearchResult: ...
    def info(self) -> dict[str, Any]: ...


class LRUCache:
    """query -> ranked (product_id, score) list, least recently used evicted."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._data: OrderedDict[str, list[tuple[str, float]]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> list[tuple[str, float]] | None:
        value = self._data.get(key)
        if value is None:
            self.misses += 1
            return None
        self._data.move_to_end(key)
        self.hits += 1
        return value

    def put(self, key: str, value: list[tuple[str, float]]) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        if len(self._data) > self.capacity:
            self._data.popitem(last=False)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


def normalise(query: str) -> str:
    """Cache key: case and whitespace do not change the encoder's input much,
    and folding them raises the hit rate on real traffic."""
    return " ".join(query.lower().split())


class RetrievalEngine:
    def __init__(
        self,
        encoder: OnnxQueryEncoder,
        index: faiss.Index,
        ids: list[str],
        catalog: Catalog,
        manifest: dict[str, Any],
        cache_size: int = 10_000,
        max_batch: int = 8,
    ) -> None:
        # FAISS returns row positions; the id list must line up with the index.
        assert index.ntotal == len(ids), f"{index.ntotal} vectors vs {len(ids)} ids"
        self._index = index
        self._ids = ids
        self._catalog = catalog
        self._manifest = manifest
        self.cache = LRUCache(cache_size)
        # Opportunistic batching, no window (ADR-016).
        self._batcher = MicroBatcher(encoder.encode, max_batch=max_batch, max_wait_ms=0)

    @classmethod
    def from_dir(cls, path: Path) -> RetrievalEngine:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        cfg = EncoderConfig.load(path / "encoder_config.json")
        encoder = OnnxQueryEncoder(
            path / "query_fp32.onnx", cfg, threads=2, tokenizer_dir=path / "tokenizer"
        )
        faiss.omp_set_num_threads(1)  # one query per call; threads would only contend
        index = faiss.read_index(str(path / "index.faiss"))
        # Raises if the index type has no efSearch, rather than setting nothing.
        faiss.ParameterSpace().set_index_parameter(index, "efSearch", int(manifest["ef_search"]))
        ids = (path / "ids.txt").read_text(encoding="utf-8").split("\n")
        return cls(encoder, index, ids, Catalog(path / "catalog.sqlite"), manifest)

    async def start(self) -> None:
        await self._batcher.start()

    async def stop(self) -> None:
        await self._batcher.stop()

    async def search(self, query: str, k: int) -> SearchResult:
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        key = normalise(query)
        ranked = self.cache.get(key)
        cached = ranked is not None
        if ranked is None:
            vector = await self._batcher.encode(query)
            t1 = time.perf_counter()
            timings["encode"] = (t1 - t0) * 1000
            scores, rows = await asyncio.to_thread(
                self._index.search, np.ascontiguousarray(vector[None, :], dtype=np.float32), FETCH
            )
            t2 = time.perf_counter()
            timings["search"] = (t2 - t1) * 1000
            ranked = [
                (self._ids[r], float(s)) for r, s in zip(rows[0], scores[0], strict=True) if r >= 0
            ]
            self.cache.put(key, ranked)
        t3 = time.perf_counter()
        top = ranked[:k]
        meta = self._catalog.get_many([pid for pid, _ in top])
        hits = [Hit(pid, *meta.get(pid, ("", "")), round(score, 4)) for pid, score in top]
        t4 = time.perf_counter()
        timings["hydrate"] = (t4 - t3) * 1000
        timings["total"] = (t4 - t0) * 1000
        return SearchResult(hits, cached, {k_: round(v, 3) for k_, v in timings.items()})

    def info(self) -> dict[str, Any]:
        return {
            **self._manifest,
            "products": len(self._ids),
            "cache_entries": len(self.cache._data),
            "cache_hit_rate": round(self.cache.hit_rate, 4),
            "batches": self._batcher.stats.batches,
            "mean_batch": round(self._batcher.stats.mean_batch, 2),
        }
