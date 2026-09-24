"""Ranking metrics"""

from __future__ import annotations

import math
from collections.abc import Sequence


def dcg(gains: Sequence[float], k: int) -> float:
    """Discounted cumulative gain at k.

    DCG@k = sum_{i=1..k} gain_i / log2(i + 1)
    """
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg_at_k(
    ranked_gains: Sequence[float],
    pool_gains: Sequence[float],
    k: int,
) -> float | None:
    """Normalised DCG at k.

    Args:
        ranked_gains: gains of the items the system returned, in its order.
        pool_gains:   gains of EVERY judged item for this query.

    Returns:
        A value in [0, 1], or None when the query has no relevant item at
        all (IDCG == 0).
    """
    idcg = dcg(sorted(pool_gains, reverse=True), k)
    if idcg == 0.0:
        return None
    return dcg(ranked_gains, k) / idcg


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
    k: int,
) -> float | None:
    """Fraction of judged-relevant products appearing in the top k.

    This is a LOWER BOUND on true recall, not true recall. ESCI judges a
    few dozen products per query, so a retrieved product carrying no
    judgement is unknown, not irrelevant.

    Returns None when the query has no relevant items.
    """
    if not relevant_ids:
        return None
    hits = sum(1 for pid in retrieved_ids[:k] if pid in relevant_ids)
    return hits / len(relevant_ids)


def reciprocal_rank(ranked_gains: Sequence[float], threshold: float = 1.0) -> float:
    """1 / rank of the first item at or above `threshold` gain, else 0.0.
    """
    for rank, gain in enumerate(ranked_gains, start=1):
        if gain >= threshold:
            return 1.0 / rank
    return 0.0