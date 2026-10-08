from __future__ import annotations

import numpy as np

from esci.index.rerank import RerankRanker

TEXT = {f"p{i}": f"doc {i}" for i in range(10)}


def _score_by_number(queries: list[str], docs: list[str]) -> np.ndarray:
    """Higher product number scores higher, for every query."""
    return np.array([float(d.split()[1]) for d in docs], dtype=np.float32)


class _FirstStage:
    name = "first"

    def retrieve(self, queries: dict[int, str], k: int) -> dict[int, list[str]]:
        return {q: [f"p{i}" for i in range(k)] for q in queries}

    def rank_candidates(
        self, queries: dict[int, str], candidates: dict[int, list[str]]
    ) -> dict[int, list[str]]:
        return candidates


def test_rank_candidates_sorts_each_query_by_score() -> None:
    r = RerankRanker(_score_by_number, TEXT, "ce")
    out = r.rank_candidates({1: "a", 2: "b"}, {1: ["p1", "p7", "p3"], 2: ["p9", "p0"]})
    assert out == {1: ["p7", "p3", "p1"], 2: ["p9", "p0"]}


def test_two_stage_reorders_but_keeps_the_first_stage_set() -> None:
    """Recall@depth must equal the first stage's: reranking only reorders."""
    r = RerankRanker(_score_by_number, TEXT, "ce", first_stage=_FirstStage(), depth=5)
    out = r.retrieve({1: "a"}, k=5)[1]
    assert out == ["p4", "p3", "p2", "p1", "p0"]
    assert set(out) == set(_FirstStage().retrieve({1: "a"}, 5)[1])


def test_scores_are_split_back_to_the_right_query() -> None:
    """Flat batching across queries must not shift scores between them."""
    calls: list[int] = []

    def score(queries: list[str], docs: list[str]) -> np.ndarray:
        calls.append(len(docs))
        return np.array(
            [1.0 if q == "b" else -float(d.split()[1]) for q, d in zip(queries, docs, strict=True)]
        )

    out = RerankRanker(score, TEXT, "ce").rank_candidates(
        {1: "a", 2: "b"}, {1: ["p1", "p2"], 2: ["p5", "p6"]}
    )
    assert calls == [4]  # one batched call for the whole chunk
    assert out[1] == ["p1", "p2"] and out[2] == ["p5", "p6"]
