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
    joined = a.select("query_id", pl.col(metric).alias("a")).join(
        b.select("query_id", pl.col(metric).alias("b")), on="query_id", how="inner"
    )
    joined = joined.drop_nulls()
    return bootstrap_ci((joined["b"] - joined["a"]).to_numpy(), n_resamples, alpha, seed)
