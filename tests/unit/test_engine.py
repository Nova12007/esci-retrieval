from __future__ import annotations

import asyncio
from pathlib import Path

import faiss
import numpy as np
import pytest

from esci.serving.catalog import Catalog, build_catalog
from esci.serving.engine import LRUCache, RetrievalEngine, normalise

DIM = 8


class _Encoder:
    """Query 'pN' encodes to product N's vector, so the top hit is known."""

    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = vectors
        self.batches: list[int] = []

    def encode(self, texts: list[str]) -> np.ndarray:
        self.batches.append(len(texts))
        return np.stack([self.vectors[int(t.strip().lower()[1:])] for t in texts])


def _engine(tmp_path: Path, n: int = 300) -> tuple[RetrievalEngine, _Encoder]:
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(n, DIM)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    index = faiss.IndexHNSWFlat(DIM, 16, faiss.METRIC_INNER_PRODUCT)
    index.add(vecs)
    index.hnsw.efSearch = 128
    ids = [f"ID{i}" for i in range(n)]
    build_catalog(tmp_path / "c.sqlite", [(f"ID{i}", f"title {i}", "acme") for i in range(n)])
    enc = _Encoder(vecs)
    engine = RetrievalEngine(enc, index, ids, Catalog(tmp_path / "c.sqlite"), {"model": "toy"})  # type: ignore[arg-type]
    return engine, enc


def test_lru_evicts_least_recently_used() -> None:
    c = LRUCache(2)
    c.put("a", [])
    c.put("b", [])
    c.get("a")
    c.put("c", [])
    assert c.get("b") is None and c.get("a") == [] and c.get("c") == []


def test_normalise_folds_case_and_spacing() -> None:
    assert normalise("  USB  C\tCable ") == "usb c cable"


def test_search_finds_the_query_product_and_hydrates_it(tmp_path: Path) -> None:
    engine, _ = _engine(tmp_path)

    async def go():  # type: ignore[no-untyped-def]
        await engine.start()
        r = await engine.search("p42", 5)
        await engine.stop()
        return r

    result = asyncio.run(go())
    top = result.hits[0]
    assert (top.product_id, top.title, top.brand) == ("ID42", "title 42", "acme")
    assert len(result.hits) == 5 and not result.cached
    scores = [h.score for h in result.hits]
    assert scores == sorted(scores, reverse=True)
    assert {"encode", "search", "hydrate", "total"} <= result.timings_ms.keys()


def test_second_identical_query_is_served_from_cache(tmp_path: Path) -> None:
    engine, enc = _engine(tmp_path)

    async def go():  # type: ignore[no-untyped-def]
        await engine.start()
        first = await engine.search("p7", 10)
        second = await engine.search("  P7 ", 3)  # same after normalising, smaller k
        await engine.stop()
        return first, second

    first, second = asyncio.run(go())
    assert second.cached and "encode" not in second.timings_ms
    assert [h.product_id for h in second.hits] == [h.product_id for h in first.hits[:3]]
    assert sum(enc.batches) == 1  # encoder ran once
    assert engine.cache.hit_rate == 0.5


def test_ids_must_line_up_with_the_index(tmp_path: Path) -> None:
    engine, enc = _engine(tmp_path)
    with pytest.raises(AssertionError, match="vectors vs"):
        RetrievalEngine(enc, engine._index, ["only one"], engine._catalog, {})  # type: ignore[arg-type]
