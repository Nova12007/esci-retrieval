from __future__ import annotations

import pytest

from esci.index.fusion import FusionRanker, rrf


def test_hand_worked_two_list_example() -> None:
    """k=60. a: 1/61 + 1/63; b: 1/62 + 1/61; c: 1/63; d: 1/62.

    b = 0.032522 beats a = 0.032266: second in one list and first in the
    other outranks first and third.
    """
    fused = rrf([["a", "b", "c"], ["b", "d", "a"]], k=60)
    assert fused == ["b", "a", "d", "c"]


def test_scores_match_the_formula() -> None:
    a = 1 / 61 + 1 / 63
    b = 1 / 62 + 1 / 61
    assert b > a
    assert b - a == pytest.approx(0.000256, abs=1e-6)


def test_single_list_is_unchanged() -> None:
    assert rrf([["x", "y", "z"]]) == ["x", "y", "z"]


def test_agreement_beats_a_single_top_rank() -> None:
    """Ranked 2nd by both of two rankers beats 1st by only one of them."""
    fused = rrf([["solo", "both"], ["other", "both"]])
    assert fused[0] == "both"


class _Fixed:
    def __init__(self, name: str, order: list[str]) -> None:
        self.name, self._order = name, order

    def retrieve(self, queries: dict[int, str], k: int) -> dict[int, list[str]]:
        return {q: self._order[:k] for q in queries}

    def rank_candidates(
        self, queries: dict[int, str], candidates: dict[int, list[str]]
    ) -> dict[int, list[str]]:
        return {q: [p for p in self._order if p in candidates[q]] for q in queries}


def test_fusion_ranker_retrieves_from_the_union_and_cuts_to_k() -> None:
    r = FusionRanker([_Fixed("x", ["a", "b", "c"]), _Fixed("y", ["d", "b", "e"])], "rrf", depth=3)
    out = r.retrieve({1: "q"}, k=2)[1]
    assert out[0] == "b" and len(out) == 2


def test_fusion_ranker_reranks_only_the_judged_set() -> None:
    r = FusionRanker([_Fixed("x", ["a", "b", "c"]), _Fixed("y", ["c", "b", "a"])], "rrf")
    out = r.rank_candidates({1: "q"}, {1: ["c", "a"]})[1]
    assert sorted(out) == ["a", "c"]
