"""Approximate nearest-neighbour indexes and how to measure them.

Every index here is judged against exact search on the same vectors, so the
numbers isolate the index's own error from the encoder's.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import faiss
import numpy as np


def build_flat(emb: np.ndarray) -> faiss.Index:
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index


def build_hnsw(emb: np.ndarray, m: int, ef_construction: int) -> faiss.Index:
    """Graph index: m links per node; ef_construction is the build-time beam."""
    index = faiss.IndexHNSWFlat(emb.shape[1], m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.add(emb)
    return index


def build_ivfpq(
    emb: np.ndarray, nlist: int, m: int, train_size: int = 200_000, seed: int = 0
) -> faiss.Index:
    """nlist coarse clusters; each vector stored as m bytes of product-quantised code.

    FAISS wants roughly 39+ training points per cluster; 200k covers nlist=4096.
    """
    d = emb.shape[1]
    assert d % m == 0, f"m={m} must divide the dimension {d}"
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8, faiss.METRIC_INNER_PRODUCT)
    rng = np.random.default_rng(seed)
    sample = emb[np.sort(rng.choice(emb.shape[0], min(train_size, emb.shape[0]), replace=False))]
    index.train(np.ascontiguousarray(sample))
    index.add(emb)
    return index


def set_search_param(index: faiss.Index, value: int) -> None:
    """efSearch for HNSW, nprobe for IVF: the query-time accuracy/speed knob."""
    if isinstance(index, faiss.IndexHNSWFlat):
        index.hnsw.efSearch = value
    elif isinstance(index, faiss.IndexIVF):
        index.nprobe = value


def index_bytes(index: faiss.Index) -> int:
    """Size of the serialised index: what a container has to load."""
    return int(faiss.serialize_index(index).nbytes)


def recall_vs_exact(approx: np.ndarray, exact: np.ndarray) -> float:
    """Mean fraction of each query's exact top-k that the index also returned."""
    k = exact.shape[1]
    hits = [len(set(a[a >= 0]) & set(e)) / k for a, e in zip(approx, exact, strict=True)]
    return float(np.mean(hits))


@dataclass(frozen=True)
class Latency:
    p50_ms: float
    p99_ms: float


def single_query_latency(index: faiss.Index, queries: np.ndarray, k: int = 100) -> Latency:
    """One query per call on one thread, the way a request handler searches.

    Batched throughput would flatter every index; served traffic arrives one
    query at a time.
    """
    previous = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(1)
    try:
        index.search(queries[:10], k)  # warm up
        times = []
        for q in queries:
            t0 = time.perf_counter()
            index.search(q[None, :], k)
            times.append((time.perf_counter() - t0) * 1000)
    finally:
        faiss.omp_set_num_threads(previous)
    p50, p99 = np.percentile(times, [50, 99])
    return Latency(float(p50), float(p99))
