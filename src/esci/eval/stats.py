"""Uncertainty for per-query metrics: bootstrap confidence intervals.

Resampling is over queries, the unit the test fold was sampled in. A paired
bootstrap resamples queries once and takes both systems' scores for the same
draw, so per-query difficulty cancels; comparing two separate CIs ignores that
and is badly conservative.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float

    def __str__(self) -> str:
        return f"{self.estimate:.4f} [{self.low:.4f}, {self.high:.4f}]"


def _bootstrap_means(
    values: np.ndarray, n_resamples: int, seed: int, chunk: int = 100
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = values.shape[0]
    means = []
    for start in range(0, n_resamples, chunk):  # chunked: 1000 x 22k indices is 180 MB
        size = min(chunk, n_resamples - start)
        means.append(values[rng.integers(0, n, size=(size, n))].mean(axis=1))
    return np.concatenate(means)


def bootstrap_ci(
    scores: np.ndarray, n_resamples: int = 1000, alpha: float = 0.05, seed: int = 0
) -> Interval:
    """Percentile bootstrap CI for the mean of per-query scores."""
    values = np.asarray(scores, dtype=np.float64)
    means = _bootstrap_means(values, n_resamples, seed)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return Interval(float(values.mean()), float(low), float(high))


def paired_bootstrap_diff(
    a: pl.DataFrame,
    b: pl.DataFrame,
    metric: str,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Interval:
    """CI for mean(b - a) over queries scored by both systems.

    Queries either system skipped (null metric) are dropped from both, so the
    two means are always over the same queries.
    """
    return bootstrap_ci(paired_differences(a, b, metric), n_resamples, alpha, seed)


def paired_differences(a: pl.DataFrame, b: pl.DataFrame, metric: str) -> np.ndarray:
    """Per-query b - a over queries both systems scored."""
    joined = a.select("query_id", pl.col(metric).alias("a")).join(
        b.select("query_id", pl.col(metric).alias("b")), on="query_id", how="inner"
    )
    joined = joined.drop_nulls()
    return (joined["b"] - joined["a"]).to_numpy()


def paired_permutation_test(
    differences: np.ndarray, n_permutations: int = 10_000, seed: int = 0, chunk: int = 500
) -> float:
    """Two-sided p-value for mean(differences) == 0 by random sign flips.

    Under the null the two systems are exchangeable on every query, so each
    per-query difference is as likely to be +d as -d. Flip signs at random and
    count how often the flipped mean is at least as extreme as the observed.

    Chosen over a paired t-test because nDCG and recall are bounded and
    heavily skewed (30% of queries score exactly 1.0), so the t-test's
    normality assumption does not hold. The +1 in numerator and denominator
    counts the observed assignment itself, so p is never exactly 0.
    """
    d = np.asarray(differences, dtype=np.float64)
    observed = abs(d.mean())
    rng = np.random.default_rng(seed)
    extreme = 0
    for start in range(0, n_permutations, chunk):
        size = min(chunk, n_permutations - start)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(size, d.shape[0]))
        extreme += int((np.abs((signs * d).mean(axis=1)) >= observed - 1e-15).sum())
    return (extreme + 1) / (n_permutations + 1)
