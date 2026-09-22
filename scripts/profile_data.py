"""Profile the US slice of ESCI"""

from __future__ import annotations

from pathlib import Path

import polars as pl

RAW = Path("data/raw")
EXAMPLES = RAW / "shopping_queries_dataset_examples.parquet"
PRODUCTS = RAW / "shopping_queries_dataset_products.parquet"


def main() -> None:
    ex = pl.read_parquet(EXAMPLES)
    pr = pl.read_parquet(PRODUCTS)

    print(f"all examples: {ex.height:,}")
    print(f"all products: {pr.height:,}\n")

    print("--- locale distribution ---")
    print(ex["product_locale"].value_counts(sort=True))

    # ADR-001: US locale, large_version.
    us = ex.filter(
        (pl.col("product_locale") == "us") & (pl.col("large_version") == 1)
    )
    print(f"\nUS + large_version rows: {us.height:,}")
    print(f"unique queries:  {us['query_id'].n_unique():,}")
    print(f"unique products: {us['product_id'].n_unique():,}")

    print("\n--- official train/test split ---")
    print(us["split"].value_counts(sort=True))

    print("\n--- ESCI label distribution ---")
    labels = us["esci_label"].value_counts(sort=True)
    print(labels.with_columns(
        (pl.col("count") / us.height * 100).round(2).alias("pct")
    ))

    print("\n--- judged products per query ---")
    per_query = us.group_by("query_id").len().rename({"len": "n_judged"})
    print(per_query["n_judged"].describe())
    print("\npercentiles:")
    for q in (0.01, 0.25, 0.50, 0.75, 0.95, 0.99):
        val = per_query["n_judged"].quantile(q)
        print(f"    p{int(q * 100):<3} {val:.0f}")

    # Exact-labelled positives per query
    print("\n--- EXACT products per query ---")
    exact = (
        us.filter(pl.col("esci_label") == "E")
        .group_by("query_id")
        .len()
        .rename({"len": "n_exact"})
    )
    print(exact["n_exact"].describe())

    print("\n--- query length in whitespace tokens ---")
    qlen = us.select(
        pl.col("query").str.split(" ").list.len().alias("tokens")
    )
    print(qlen["tokens"].describe())

    # Query frequency
    print("\n--- query frequency (rows per query) ---")
    freq = per_query["n_judged"].sort(descending=True)
    print(f"    top 1%    of queries hold {freq.head(max(1, len(freq) // 100)).sum():,} rows")
    print(f"    bottom 50% of queries hold {freq.tail(len(freq) // 2).sum():,} rows")

    print("\n--- product text coverage (US products) ---")
    us_pr = pr.filter(pl.col("product_locale") == "us")
    print(f"US products: {us_pr.height:,}")
    for col in ("product_title", "product_description",
                "product_bullet_point", "product_brand"):
        missing = us_pr.select(
            (pl.col(col).is_null() | (pl.col(col).str.strip_chars() == ""))
            .mean()
            .alias("frac")
        ).item()
        print(f"    {col:<24} missing: {missing * 100:.1f}%")


if __name__ == "__main__":
    main()