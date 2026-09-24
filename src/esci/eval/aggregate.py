"""Turn ranked results into per-query scores, then into a summary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import polars as pl

from esci.eval.metrics import ndcg_at_k, recall_at_k, reciprocal_rank


@dataclass(frozen=True)
class Summary:
    system: str
    mode: str
    n_scored: int
    n_skipped: int
    metrics: dict[str, float]

    def row(self) -> dict[str, object]:
        return {
            "system": self.system,
            "mode": self.mode,
            "n_scored": self.n_scored,
            "n_skipped": self.n_skipped,
            **{k: round(v, 4) for k, v in self.metrics.items()},
        }


def summarise(per_query: pl.DataFrame, system: str, mode: str) -> Summary:
    """Mean each metric column, ignoring nulls, and count what was skipped."""
    metric_cols = [c for c in per_query.columns if c not in ("query_id",)]

    # A query is 'skipped' if its primary metric could not be computed.
    primary = metric_cols[0]
    n_skipped = int(per_query[primary].null_count())
    n_scored = per_query.height - n_skipped

    metrics = {col: cast(float, per_query[col].mean() or 0.0) for col in metric_cols}

    return Summary(
        system=system,
        mode=mode,
        n_scored=n_scored,
        n_skipped=n_skipped,
        metrics=metrics,
    )


def score_rerank(
    ranked: dict[int, list[str]],
    judgements: pl.DataFrame,
    k: int = 10,
) -> pl.DataFrame:
    """Mode A: the system reordered each query's judged candidate set.

    Returns one row per query with its individual scores.
    """
    gain_by_pair = {
        (qid, pid): g
        for qid, pid, g in zip(
            judgements["query_id"],
            judgements["product_id"],
            judgements["gain"],
            strict=True,
        )
    }
    pool_by_query: dict[int, list[float]] = {}
    for qid, gain in zip(judgements["query_id"], judgements["gain"], strict=True):
        pool_by_query.setdefault(qid, []).append(gain)

    rows = []
    for qid, product_ids in ranked.items():
        gains = [gain_by_pair.get((qid, pid), 0.0) for pid in product_ids]
        pool = pool_by_query.get(qid, [])
        rows.append(
            {
                "query_id": qid,
                f"ndcg@{k}": ndcg_at_k(gains, pool, k),
                "mrr": reciprocal_rank(gains),
            }
        )
    return pl.DataFrame(rows)


def score_retrieval(
    retrieved: dict[int, list[str]],
    judgements: pl.DataFrame,
    k: int = 100,
    gain_threshold: float = 1.0,
) -> pl.DataFrame:
    """Mode B: the system retrieved from the full corpus.

    Recall is computed over judged-Exact products only and is a LOWER
    BOUND -- unjudged retrieved products are unknown, not wrong.
    """
    relevant_by_query: dict[int, set[str]] = {}
    for qid, pid, gain in zip(
        judgements["query_id"],
        judgements["product_id"],
        judgements["gain"],
        strict=True,
    ):
        if gain >= gain_threshold:
            relevant_by_query.setdefault(qid, set()).add(pid)

    rows = []
    for qid, product_ids in retrieved.items():
        relevant = relevant_by_query.get(qid, set())
        rows.append(
            {
                "query_id": qid,
                f"recall@{k}": recall_at_k(product_ids, relevant, k),
            }
        )
    return pl.DataFrame(rows)
