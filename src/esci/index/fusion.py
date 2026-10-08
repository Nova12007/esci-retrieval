"""Reciprocal rank fusion: combine rankers by rank, not by score.

score(d) = sum over rankers of 1 / (k + rank_i(d)), ranks starting at 1.

Ranks, not scores, because BM25 scores and cosine similarities live on
unrelated scales; normalising them is fragile, and RRF needs no tuning beyond
k. k=60 is the value from Cormack et al. (2009), used here untuned: it damps
the top ranks so one ranker's #1 cannot outvote broad agreement.
"""

from __future__ import annotations

from esci.eval.runner import Ranker

RRF_K = 60


def rrf(rankings: list[list[str]], k: int = RRF_K) -> list[str]:
    """Fuse ranked lists, best first. Ties keep first-seen order."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, pid in enumerate(ranking, start=1):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda p: -scores[p])  # sorted() is stable


class FusionRanker:
    """Implements the Ranker protocol over any rankers."""

    def __init__(self, rankers: list[Ranker], name: str, depth: int = 100, k: int = RRF_K) -> None:
        self.name = name
        self._rankers = rankers
        self._depth = depth
        self._k = k

    def retrieve(self, queries: dict[int, str], k: int) -> dict[int, list[str]]:
        """Each ranker retrieves `depth`, fused, cut to k. The fused pool can
        hold up to len(rankers) x depth products; only the top k come back."""
        runs = [r.retrieve(queries, max(k, self._depth)) for r in self._rankers]
        return {q: rrf([run[q] for run in runs], self._k)[:k] for q in queries}

    def rank_candidates(
        self, queries: dict[int, str], candidates: dict[int, list[str]]
    ) -> dict[int, list[str]]:
        runs = [r.rank_candidates(queries, candidates) for r in self._rankers]
        return {q: rrf([run[q] for run in runs], self._k) for q in queries}
