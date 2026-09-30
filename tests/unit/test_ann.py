from __future__ import annotations

import numpy as np
import pytest

from esci.index.ann import (
    build_flat,
    build_hnsw,
    build_ivfpq,
    index_bytes,
    recall_vs_exact,
    set_search_param,
    single_query_latency,
)


@pytest.fixture(scope="module")
def data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(5000, 32)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    queries = emb[:50] + 0.05 * rng.normal(size=(50, 32)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    _, exact = build_flat(emb).search(queries, 10)
    return emb, queries, exact


def test_recall_vs_exact_counts_overlap() -> None:
    exact = np.array([[1, 2, 3, 4]])
    assert recall_vs_exact(np.array([[4, 3, 2, 1]]), exact) == 1.0
    assert recall_vs_exact(np.array([[1, 2, 9, -1]]), exact) == 0.5


def test_hnsw_recall_rises_with_ef_search(data: tuple[np.ndarray, ...]) -> None:
    emb, queries, exact = data
    index = build_hnsw(emb, m=8, ef_construction=40)
    set_search_param(index, 10)
    low = recall_vs_exact(index.search(queries, 10)[1], exact)
    set_search_param(index, 200)
    high = recall_vs_exact(index.search(queries, 10)[1], exact)
    assert high >= low and high > 0.95


def test_ivfpq_compresses_and_recall_rises_with_nprobe(data: tuple[np.ndarray, ...]) -> None:
    emb, queries, exact = data
    index = build_ivfpq(emb, nlist=16, m=8, train_size=5000)
    assert index_bytes(index) < index_bytes(build_flat(emb)) / 2
    set_search_param(index, 1)
    low = recall_vs_exact(index.search(queries, 10)[1], exact)
    set_search_param(index, 16)
    high = recall_vs_exact(index.search(queries, 10)[1], exact)
    assert high > low


def test_ivfpq_rejects_m_that_does_not_divide_dimension(data: tuple[np.ndarray, ...]) -> None:
    with pytest.raises(AssertionError, match="must divide"):
        build_ivfpq(data[0], nlist=16, m=5)


def test_latency_is_positive_and_ordered(data: tuple[np.ndarray, ...]) -> None:
    emb, queries, _ = data
    lat = single_query_latency(build_flat(emb), queries, k=10)
    assert 0 < lat.p50_ms <= lat.p99_ms
