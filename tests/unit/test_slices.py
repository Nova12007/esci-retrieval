from __future__ import annotations

import polars as pl

from esci.eval.slices import unsaturated_queries


def test_keeps_only_queries_below_the_exact_threshold() -> None:
    judgements = pl.DataFrame(
        {
            "query_id": [1] * 10 + [2] * 9 + [2],
            "esci_label": ["E"] * 10 + ["E"] * 9 + ["S"],
        }
    )
    # query 1: 10 Exact, saturated. query 2: 9 Exact + 1 S, unsaturated.
    assert unsaturated_queries(judgements).to_list() == [2]


def test_queries_with_no_exact_are_unsaturated() -> None:
    judgements = pl.DataFrame({"query_id": [5, 5], "esci_label": ["S", "I"]})
    assert unsaturated_queries(judgements).to_list() == [5]
