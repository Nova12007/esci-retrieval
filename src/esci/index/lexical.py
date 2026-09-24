"""BM25 baseline"""

from __future__ import annotations

from typing import Any

import bm25s
import Stemmer  # package name is PyStemmer

K1 = 1.2  # term-frequency saturation, standard default
B = 0.75  # length normalisation, standard default


class BM25Ranker:
    """Implements the Ranker protocol from eval.runner."""

    def __init__(self, product_ids: list[str], texts: list[str], name: str = "bm25_tbb") -> None:
        self.name = name
        self._stemmer = Stemmer.Stemmer("english")
        self._product_ids = product_ids
        self._position = {pid: i for i, pid in enumerate(product_ids)}

        tokens = bm25s.tokenize(texts, stopwords="en", stemmer=self._stemmer, show_progress=True)
        self._model = bm25s.BM25(k1=K1, b=B)
        self._model.index(tokens, show_progress=True)

    def _tokenize(self, queries: list[str]) -> Any:
        return bm25s.tokenize(queries, stopwords="en", stemmer=self._stemmer, show_progress=False)

    def retrieve(self, queries: dict[int, str], k: int) -> dict[int, list[str]]:
        """Mode B: top-k from the full corpus."""
        out: dict[int, list[str]] = {}
        for qid, text in queries.items():
            tokens = self._tokenize([text])
            if not tokens[0][0]:
                out[qid] = []
                continue
            indices, _ = self._model.retrieve(tokens, k=k, show_progress=False)
            out[qid] = [self._product_ids[i] for i in indices[0]]
        return out

    def rank_candidates(
        self, queries: dict[int, str], candidates: dict[int, list[str]]
    ) -> dict[int, list[str]]:
        """Mode A: reorder each query's judged set.

        Scores come from the FULL-corpus model, because BM25's IDF term
        depends on corpus-wide document frequencies.
        """
        out: dict[int, list[str]] = {}
        for qid, text in queries.items():
            tokens = self._tokenize([text])
            if not tokens[0][0]:
                out[qid] = candidates[qid]
                continue
            scores = self._model.get_scores(tokens[0][0])
            out[qid] = sorted(
                candidates[qid], key=lambda pid: scores[self._position[pid]], reverse=True
            )
        return out
