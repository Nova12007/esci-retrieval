from __future__ import annotations

import numpy as np
import pytest

from esci.index.dense import Backend, DenseRanker

# Four unit vectors at 0, 30, 90 and 180 degrees.
ANGLES = np.deg2rad([0, 30, 90, 180])
DOCS = np.stack([np.cos(ANGLES), np.sin(ANGLES)], axis=1).astype(np.float32)
IDS = ["east", "east_ish", "north", "west"]


def _encode(queries: list[str]) -> np.ndarray:
    """'east' points at 0 degrees, anything else at 90."""
    return np.array([[1.0, 0.0] if q == "east" else [0.0, 1.0] for q in queries], np.float32)


BACKENDS = pytest.mark.parametrize("backend", ["faiss", "torch"])


def _ranker(backend: Backend = "faiss") -> DenseRanker:
    return DenseRanker(IDS, DOCS, _encode, name="toy", backend=backend)


@BACKENDS
def test_retrieve_orders_by_cosine_and_maps_rows_to_ids(backend: Backend) -> None:
    out = _ranker(backend).retrieve({1: "east", 2: "north"}, k=4)
    assert out[1] == ["east", "east_ish", "north", "west"]
    assert out[2][0] == "north"


@BACKENDS
def test_retrieve_respects_k(backend: Backend) -> None:
    assert len(_ranker(backend).retrieve({1: "east"}, k=2)[1]) == 2


def test_rank_candidates_reorders_only_the_judged_set() -> None:
    out = _ranker().rank_candidates({1: "east"}, {1: ["west", "north", "east_ish"]})
    assert out[1] == ["east_ish", "north", "west"]


@BACKENDS
def test_modes_agree_on_scores(backend: Backend) -> None:
    """A product must score the same whether found by search or reranked."""
    r = _ranker(backend)
    retrieved = r.retrieve({1: "east"}, k=4)[1]
    reranked = r.rank_candidates({1: "east"}, {1: list(reversed(IDS))})[1]
    assert retrieved == reranked


def test_mismatched_ids_and_rows_refuse_to_build() -> None:
    with pytest.raises(AssertionError, match="embedding rows"):
        DenseRanker(IDS[:3], DOCS, _encode, name="toy")


def test_backends_return_identical_rankings() -> None:
    rng = np.random.default_rng(0)
    docs = rng.normal(size=(500, 16)).astype(np.float32)
    docs /= np.linalg.norm(docs, axis=1, keepdims=True)
    ids = [f"p{i}" for i in range(500)]
    queries = {i: str(i) for i in range(20)}

    def encode(qs: list[str]) -> np.ndarray:
        return docs[[int(q) * 7 for q in qs]] + 0.01

    f = DenseRanker(ids, docs, encode, "f", backend="faiss").retrieve(queries, 50)
    t = DenseRanker(ids, docs, encode, "t", backend="torch").retrieve(queries, 50)
    assert f == t
