"""Collation: sample positives and negatives, and build the mask."""

from __future__ import annotations

import random
from typing import TypedDict

import torch

from esci.data.pairs import QueryExample


class Batch(TypedDict):
    query_ids: list[int]
    queries: list[str]
    positive_texts: list[str]  # (M,) flat, M = sum of positives per row
    positive_products: list[str]  # (M,)
    positive_owner: torch.Tensor  # (M,) long, row index of each positive
    negative_texts: list[list[str]]  # (B, n) padded with "[pad]"
    negative_valid: torch.Tensor  # (B, n) bool, False = padding


class QueryCollator:
    def __init__(
        self,
        text_by_id: dict[str, str],
        relevant_by_query: dict[int, frozenset[str]],
        n_positives: int = 1,
        n_negatives: int = 4,
        seed: int = 13,
    ) -> None:
        self.text_by_id = text_by_id
        self.relevant_by_query = relevant_by_query
        self.n_positives = n_positives
        self.n_negatives = n_negatives
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Call before each epoch so a different positive is drawn."""
        self.epoch = epoch

    def _rng(self, query_id: int) -> random.Random:
        """Deterministic per (seed, epoch, query)"""
        return random.Random((self.seed * 1_000_003) ^ (self.epoch * 7919) ^ query_id)

    def __call__(self, batch: list[QueryExample]) -> Batch:
        query_ids = [e.query_id for e in batch]
        assert len(set(query_ids)) == len(query_ids), "duplicate query in batch"

        PAD = "[pad]"  # encoded then masked out; a handful of wasted forward passes
        queries: list[str] = []
        pos_texts: list[str] = []
        pos_owner: list[int] = []
        neg_texts: list[list[str]] = []
        pos_products: list[str] = []
        neg_valid: list[list[bool]] = []

        for row, ex in enumerate(batch):
            rng = self._rng(ex.query_id)
            queries.append(ex.query)

            chosen_pos = rng.sample(
                list(ex.positive_ids), min(self.n_positives, len(ex.positive_ids))
            )
            for pid in chosen_pos:
                pos_texts.append(self.text_by_id.get(pid, ""))
                pos_products.append(pid)
                pos_owner.append(row)

            # Anything at or above the mask threshold is not a negative. At the
            # default (1.0, Exact only) this is a no-op; at 0.1 it removes S, which
            # is what makes the --mask-threshold ablation actually mask Substitutes.
            relevant = self.relevant_by_query.get(ex.query_id, frozenset())
            candidates = [n for n in ex.negative_ids if n not in relevant]

            # Negatives keep their hardest-first order; sample from the
            # front so S dominates. Pad by repetition only if non-empty.
            pool = candidates[: max(self.n_negatives * 3, self.n_negatives)]
            if pool:
                rng.shuffle(pool)
                chosen_neg = (pool * self.n_negatives)[: self.n_negatives]
                valid = [True] * self.n_negatives
            else:
                # finding 3: the query is KEPT and trains on in-batch negatives only
                chosen_neg = []
                valid = [False] * self.n_negatives

            texts = [self.text_by_id.get(n, "") for n in chosen_neg]
            texts += [PAD] * (self.n_negatives - len(texts))
            neg_texts.append(texts)
            neg_valid.append(valid)

        return {
            "query_ids": query_ids,
            "queries": queries,
            "positive_texts": pos_texts,
            "positive_products": pos_products,
            "positive_owner": torch.tensor(pos_owner, dtype=torch.long),
            "negative_texts": neg_texts,
            "negative_valid": torch.tensor(neg_valid, dtype=torch.bool),
        }


def build_relevance_mask(
    query_ids: list[int],
    positive_products: list[str],
    positive_owner: torch.Tensor,
    relevant_by_query: dict[int, frozenset[str]],
    device: str,
) -> torch.Tensor:
    """(B, M) bool. True where column c is known-relevant to row i and
    c is not one of row i's own positive columns."""
    b, m = len(query_ids), len(positive_products)
    mask = torch.zeros(b, m, dtype=torch.bool)

    for i, qid in enumerate(query_ids):
        relevant = relevant_by_query.get(qid, frozenset())
        if not relevant:
            continue
        for c, pid in enumerate(positive_products):
            if positive_owner[c].item() != i and pid in relevant:
                mask[i, c] = True

    return mask.to(device)
