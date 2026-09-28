"""Query-centric training examples."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class QueryExample:
    query_id: int
    query: str
    positive_ids: tuple[str, ...]  # all Exact
    negative_ids: tuple[str, ...]  # judged S, then C, then I


def build_query_examples(
    judgements: pl.DataFrame,
    label_priority: tuple[str, ...] = ("S", "C", "I"),
) -> tuple[list[QueryExample], dict[str, int]]:
    """Returns the examples plus a stats dict for the ADR."""
    examples: list[QueryExample] = []
    n_no_positive = 0
    n_no_negative = 0

    for (qid,), group in judgements.group_by(["query_id"]):
        positives = group.filter(pl.col("esci_label") == "E")["product_id"].to_list()
        if not positives:
            n_no_positive += 1
            continue  # no correct answer exists

        negatives: list[str] = []
        for label in label_priority:  # hardest first
            negatives.extend(group.filter(pl.col("esci_label") == label)["product_id"].to_list())
        if not negatives:
            n_no_negative += 1  # KEPT -- in-batch negatives only

        examples.append(
            QueryExample(
                query_id=int(qid),
                query=group["query"][0],
                positive_ids=tuple(positives),
                negative_ids=tuple(negatives),
            )
        )

    stats = {
        "queries_kept": len(examples),
        "dropped_no_exact": n_no_positive,
        "kept_without_explicit_negatives": n_no_negative,
        "multi_positive_queries": sum(1 for e in examples if len(e.positive_ids) > 1),
    }
    return examples, stats


def build_relevance_index(
    judgements: pl.DataFrame, threshold: float = 1.0
) -> dict[int, frozenset[str]]:
    """query_id -> products known relevant at or above `threshold` gain."""
    relevant = (
        judgements.filter(pl.col("gain") >= threshold)
        .group_by("query_id")
        .agg(pl.col("product_id"))
    )
    return {
        int(q): frozenset(p)
        for q, p in zip(relevant["query_id"], relevant["product_id"], strict=True)
    }
