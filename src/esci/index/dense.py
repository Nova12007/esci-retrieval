"""Dense retrieval over precomputed product embeddings."""

from __future__ import annotations

from collections.abc import Callable

import faiss
import numpy as np


class DenseRanker:
    """Implements the Ranker protocol from eval.runner.

    Takes an encoding function rather than a model, so the index logic is
    testable without a GPU and the same class serves zero-shot and fine-tuned
    encoders.
    """

    def __init__(
        self,
        product_ids: list[str],
        embeddings: np.ndarray,  # (N, D) float32, L2-normalised, row i <-> product_ids[i]
        encode_queries: Callable[[list[str]], np.ndarray],
        name: str,
    ) -> None:
        # FAISS returns row positions, not ids. If these ever disagree, every
        # result is silently wrong, so refuse to build rather than drift.
        assert embeddings.shape[0] == len(product_ids), (
            f"{embeddings.shape[0]} embedding rows vs {len(product_ids)} product ids"
        )
        self.name = name
        self._product_ids = product_ids
        self._position = {pid: i for i, pid in enumerate(product_ids)}
        self._embeddings = embeddings
        self._encode = encode_queries

        # Exact search: inner product on unit vectors is cosine. This is the
        # recall ceiling that the approximate indexes in week 4 are judged against.
        self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
        assert self._index.ntotal == len(product_ids)

    def retrieve(self, queries: dict[int, str], k: int) -> dict[int, list[str]]:
        """Mode B: top-k from the full corpus."""
        qids = list(queries)
        q = self._encode([queries[i] for i in qids])
        _, rows = self._index.search(np.ascontiguousarray(q, dtype=np.float32), k)
        return {
            qid: [self._product_ids[r] for r in row if r >= 0]
            for qid, row in zip(qids, rows, strict=True)
        }

    def rank_candidates(
        self, queries: dict[int, str], candidates: dict[int, list[str]]
    ) -> dict[int, list[str]]:
        """Mode A: reorder each query's judged set by cosine to the query.

        Candidate vectors are looked up from the corpus embeddings, so a
        product scores identically in both modes.
        """
        qids = list(queries)
        q = self._encode([queries[i] for i in qids])
        out: dict[int, list[str]] = {}
        for qid, q_vec in zip(qids, q, strict=True):
            pool = candidates[qid]
            docs = self._embeddings[[self._position[p] for p in pool]]
            order = np.argsort(-(docs @ q_vec), kind="stable")
            out[qid] = [pool[i] for i in order]
        return out
