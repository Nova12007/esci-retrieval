"""Metric tests against values computed by hand
"""

from __future__ import annotations

import pytest

from esci.eval.metrics import dcg, ndcg_at_k, recall_at_k, reciprocal_rank

E, S, C, I = 1.0, 0.1, 0.01, 0.0


def test_position_one_is_undiscounted() -> None:
    assert dcg([E], k=1) == pytest.approx(1.0)


def test_dcg_matches_hand_computation() -> None:
    # [E, E, S, I] = 1/log2(2) + 1/log2(3) + 0.1/log2(4) + 0/log2(5)
    #              = 1.0 + 0.6309298 + 0.05 + 0 = 1.6809298
    assert dcg([E, E, S, I], k=4) == pytest.approx(1.6809298, abs=1e-6)


def test_k_truncates() -> None:
    assert dcg([E, E, E], k=1) == pytest.approx(1.0)


def test_perfect_ranking_scores_one() -> None:
    gains = [E, E, S, I]
    assert ndcg_at_k(gains, gains, k=4) == pytest.approx(1.0)


def test_reversed_ranking() -> None:
    # DCG 0.9937695 / IDCG 1.6809298 = 0.5912023
    assert ndcg_at_k([I, S, E, E], [E, E, S, I], k=4) == pytest.approx(
        0.5912, abs=1e-4
    )


def test_single_hit_at_rank_three() -> None:
    # DCG = 1/log2(4) = 0.5 ; IDCG = 1.0
    assert ndcg_at_k([I, I, E, I], [E, I, I, I], k=4) == pytest.approx(0.5)


def test_unscoreable_query_returns_none() -> None:
    assert ndcg_at_k([I, I], [I, I], k=2) is None


def test_pool_not_ranking_defines_the_ideal() -> None:
    """Mode B: the system returned two items, but the query has three
    relevant ones. IDCG must reflect all three, so the score is < 1.0.
    """
    returned = [E, E]
    pool = [E, E, E]
    value = ndcg_at_k(returned, pool, k=3)
    assert value is not None
    assert value < 1.0


@pytest.mark.parametrize("gains", [[E, S, I], [S, E], [E], [C, C]])
def test_ndcg_never_exceeds_one(gains: list[float]) -> None:
    value = ndcg_at_k(gains, gains, k=10)
    assert value is not None
    assert value <= 1.0 + 1e-9


def test_recall_counts_only_judged_relevant() -> None:
    assert recall_at_k(["a", "b", "x"], {"a", "b", "c"}, k=3) == pytest.approx(2 / 3)


def test_recall_respects_k() -> None:
    assert recall_at_k(["a", "b", "c"], {"a", "b", "c"}, k=1) == pytest.approx(1 / 3)


def test_recall_is_none_without_relevants() -> None:
    assert recall_at_k(["a"], set(), k=10) is None


def test_reciprocal_rank_finds_first_exact() -> None:
    assert reciprocal_rank([S, S, E, E]) == pytest.approx(1 / 3)
    assert reciprocal_rank([E, I]) == pytest.approx(1.0)
    assert reciprocal_rank([S, C, I]) == 0.0