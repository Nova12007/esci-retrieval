from __future__ import annotations

import numpy as np
import pytest

from esci.index.dense import DenseRanker

# Four unit vectors at 0, 30, 90 and 180 degrees.
ANGLES = np.deg2rad([0, 30, 90, 180])
DOCS = np.stack([np.cos(ANGLES), np.sin(ANGLES)], axis=1).astype(np.float32)
IDS = ["east", "east_ish", "north", "west"]


def _encode(queries: list[str]) -> np.ndarray:
    """'east' points at 0 degrees, anything else at 90."""
    return np.array([[1.0, 0.0] if q == "east" else [0.0, 1.0] for q in queries], np.float32)


def _ranker() -> DenseRanker:
    return DenseRanker(IDS, DOCS, _encode, name="toy")


def test_retrieve_orders_by_cosine_and_maps_rows_to_ids() -> None:
    out = _ranker().retrieve({1: "east", 2: "north"}, k=4)
    assert out[1] == ["east", "east_ish", "north", "west"]
    assert out[2][0] == "north"


def test_retrieve_respects_k() -> None:
    assert len(_ranker().retrieve({1: "east"}, k=2)[1]) == 2


def test_rank_candidates_reorders_only_the_judged_set() -> None:
    out = _ranker().rank_candidates({1: "east"}, {1: ["west", "north", "east_ish"]})
    assert out[1] == ["east_ish", "north", "west"]


def test_modes_agree_on_scores() -> None:
    """A product must score the same whether found by search or reranked."""
    r = _ranker()
    retrieved = r.retrieve({1: "east"}, k=4)[1]
    reranked = r.rank_candidates({1: "east"}, {1: list(reversed(IDS))})[1]
    assert retrieved == reranked


def test_mismatched_ids_and_rows_refuse_to_build() -> None:
    with pytest.raises(AssertionError, match="embedding rows"):
        DenseRanker(IDS[:3], DOCS, _encode, name="toy")
