from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from esci.eval.stats import bootstrap_ci, paired_bootstrap_diff


def test_constant_scores_give_zero_width_interval() -> None:
    ci = bootstrap_ci(np.full(500, 0.7))
    assert ci.estimate == ci.low == ci.high == pytest.approx(0.7)


def test_interval_matches_normal_theory_for_bernoulli() -> None:
    """p=0.5, n=10,000: standard error 0.005, so the 95% CI is about +/-0.0098."""
    scores = np.random.default_rng(1).binomial(1, 0.5, size=10_000).astype(float)
    ci = bootstrap_ci(scores, n_resamples=2000)
    assert ci.low < ci.estimate < ci.high
    assert (ci.high - ci.low) / 2 == pytest.approx(0.0098, rel=0.1)


def test_seeded_bootstrap_is_reproducible() -> None:
    scores = np.random.default_rng(2).random(1000)
    assert bootstrap_ci(scores, seed=5) == bootstrap_ci(scores, seed=5)


def _frame(values: list[float | None]) -> pl.DataFrame:
    return pl.DataFrame({"query_id": list(range(len(values))), "m": values})


def test_identical_systems_have_zero_difference() -> None:
    a = _frame([0.1, 0.5, 0.9, 0.3])
    diff = paired_bootstrap_diff(a, a, "m")
    assert diff.estimate == diff.low == diff.high == 0.0


def test_pairing_cancels_query_difficulty() -> None:
    """b beats a by exactly 0.01 on every query; difficulty varies wildly.

    Unpaired CIs would overlap heavily. The paired difference is exact.
    """
    base = np.random.default_rng(3).random(2000)
    a, b = _frame(list(base)), _frame(list(base + 0.01))
    diff = paired_bootstrap_diff(a, b, "m")
    assert diff.low == pytest.approx(0.01) and diff.high == pytest.approx(0.01)


def test_queries_skipped_by_either_system_are_dropped_from_both() -> None:
    a = _frame([0.0, 0.0, None])
    b = _frame([1.0, 1.0, 1.0])
    assert paired_bootstrap_diff(a, b, "m").estimate == pytest.approx(1.0)
