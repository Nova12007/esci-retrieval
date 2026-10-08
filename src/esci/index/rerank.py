"""Second stage: rescore candidates with a pairwise scorer (the cross-encoder)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from esci.eval.runner import Ranker

ScorePairs = Callable[[list[str], list[str]], np.ndarray]


class RerankRanker:
    """Implements the Ranker protocol.

    Mode A: score each query's judged candidates and sort them.
    Mode B: take the first stage's top `depth` and sort those, so the
    candidate set (and therefore Recall@depth) is exactly the first stage's.
    """

    def __init__(
        self,
        score_pairs: ScorePairs,
        text_by_id: dict[str, str],
        name: str,
        first_stage: Ranker | None = None,
        depth: int = 100,
    ) -> None:
        self.name = name
        self._score = score_pairs
        self._text = text_by_id
        self._first_stage = first_stage
        self._depth = depth

    def rank_candidates(
        self, queries: dict[int, str], candidates: dict[int, list[str]]
    ) -> dict[int, list[str]]:
        qids = list(queries)
        # One flat batch for the whole chunk: the scorer batches far better
        # than per-query calls of ~16 pairs.
        flat_q = [queries[q] for q in qids for _ in candidates[q]]
        flat_d = [self._text.get(p, "") for q in qids for p in candidates[q]]
        scores = self._score(flat_q, flat_d)
        out: dict[int, list[str]] = {}
        start = 0
        for q in qids:
            pool = candidates[q]
            s = scores[start : start + len(pool)]
            out[q] = [pool[i] for i in np.argsort(-s, kind="stable")]
            start += len(pool)
        return out

    def retrieve(self, queries: dict[int, str], k: int) -> dict[int, list[str]]:
        assert self._first_stage is not None, "Mode B needs a first-stage retriever"
        candidates = self._first_stage.retrieve(queries, max(k, self._depth))
        reranked = self.rank_candidates(queries, candidates)
        return {q: ranked[:k] for q, ranked in reranked.items()}
