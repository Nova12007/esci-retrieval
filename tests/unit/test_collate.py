from __future__ import annotations

import torch

from esci.data.collate import QueryCollator, build_relevance_mask
from esci.data.pairs import QueryExample

TEXT = {pid: f"text {pid}" for pid in ["e1", "e2", "s1", "s2", "c1", "i1", "x1"]}


def _example() -> QueryExample:
    return QueryExample(
        query_id=7,
        query="usb c cable",
        positive_ids=("e1", "e2"),
        negative_ids=("s1", "s2", "c1", "i1"),  # hardest first
    )


def _negatives_drawn(relevant: dict[int, frozenset[str]]) -> set[str]:
    collator = QueryCollator(TEXT, relevant, n_negatives=4)
    batch = collator([_example()])
    return {t.removeprefix("text ") for t in batch["negative_texts"][0]}


def test_default_threshold_keeps_substitutes_as_negatives() -> None:
    """Threshold 1.0: only Exact is relevant, so S stays a negative."""
    drawn = _negatives_drawn({7: frozenset({"e1", "e2"})})
    assert {"s1", "s2"} <= drawn


def test_masking_substitutes_removes_them_from_private_negatives() -> None:
    """Threshold 0.1: S is relevant, so it must never be drawn as a negative.

    Before this was enforced, --mask-threshold 0.1 only touched the in-batch
    block and the S-masking ablation silently changed nothing.
    """
    drawn = _negatives_drawn({7: frozenset({"e1", "e2", "s1", "s2"})})
    assert drawn.isdisjoint({"s1", "s2"})
    assert drawn <= {"c1", "i1"}


def test_all_negatives_masked_falls_back_to_in_batch_only() -> None:
    collator = QueryCollator(TEXT, {7: frozenset({"e1", "e2", "s1", "s2", "c1", "i1"})})
    batch = collator([_example()])
    assert not batch["negative_valid"].any()


def test_relevance_mask_flags_other_rows_relevant_positives_only() -> None:
    # row 0 owns column 0, row 1 owns column 1; x1 is also relevant to query 1
    mask = build_relevance_mask(
        query_ids=[1, 2],
        positive_products=["x1", "e2"],
        positive_owner=torch.tensor([0, 1]),
        relevant_by_query={1: frozenset({"x1"}), 2: frozenset({"e2", "x1"})},
        device="cpu",
    )
    # row 1 must not treat row 0's x1 as a negative; own columns never flagged
    assert mask.tolist() == [[False, False], [True, False]]
