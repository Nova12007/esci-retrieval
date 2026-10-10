"""Query slices for reporting metrics where they can discriminate.

Every slice is defined from the query text and its judgements alone, never
from any system's results, so no slice can be tuned to flatter a system.
"""

from __future__ import annotations

import re

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


def _length_bucket(n_words: int) -> str:
    if n_words <= 2:
        return "1-2 words"
    if n_words <= 4:
        return "3-4 words"
    if n_words <= 7:
        return "5-7 words"
    return "8+ words"


def _mentions_brand(query: str, brands: list[str | None]) -> bool:
    """Whole-word, case-insensitive match of any Exact product's brand.

    Single characters and very generic placeholders are skipped; they would
    match by accident.
    """
    q = query.lower()
    for brand in brands:
        b = (brand or "").strip().lower()
        if len(b) < 2 or b in {"generic", "unknown", "n/a", "na"}:
            continue
        if re.search(rf"(?<!\w){re.escape(b)}(?!\w)", q):
            return True
    return False


def query_slices(judgements: pl.DataFrame) -> pl.DataFrame:
    """One row per query: length bucket, has_number, mentions_brand, saturation."""
    per_query = judgements.group_by("query_id").agg(
        pl.col("query").first(),
        (pl.col("esci_label") == "E").sum().alias("n_exact"),
        pl.col("product_brand").filter(pl.col("esci_label") == "E").alias("exact_brands"),
    )
    rows = []
    for qid, query, n_exact, brands in per_query.iter_rows():
        rows.append(
            {
                "query_id": qid,
                "length": _length_bucket(len(query.split())),
                "has_number": "has a number" if any(c.isdigit() for c in query) else "no number",
                "brand": "names a brand" if _mentions_brand(query, brands) else "names no brand",
                "saturation": "<10 Exact" if n_exact < SATURATION_EXACT else ">=10 Exact",
            }
        )
    return pl.DataFrame(rows)
