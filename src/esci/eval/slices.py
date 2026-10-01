"""Query slices for reporting metrics where they can discriminate."""

from __future__ import annotations

import polars as pl

# With 10 or more Exact products in the judged set, any ranking that fills the
# top 10 with Exacts scores ~1.0, so Mode A cannot separate good from great.
SATURATION_EXACT = 10


def unsaturated_queries(judgements: pl.DataFrame, min_exact: int = SATURATION_EXACT) -> pl.Series:
    """query_ids with fewer than `min_exact` Exact judgements (35.8% of test)."""
    exact = judgements.group_by("query_id").agg(
        (pl.col("esci_label") == "E").sum().alias("n_exact")
    )
    return exact.filter(pl.col("n_exact") < min_exact)["query_id"]
