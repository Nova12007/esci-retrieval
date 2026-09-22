"""Scope is fixed by ADR-001: US locale, large_version subset."""

from __future__ import annotations

from pathlib import Path

import polars as pl

RAW = Path("data/raw")
PROCESSED = Path("data/processed")

EXAMPLES = RAW / "shopping_queries_dataset_examples.parquet"
PRODUCTS = RAW / "shopping_queries_dataset_products.parquet"

LOCALE = "us"

# ADR-002: the KDD Cup 2022 Task 1 gain mapping
ESCI_GAIN: dict[str, float] = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}


def load_judgements() -> pl.LazyFrame:
    """
    Judged (query, product) pairs joined to product text.
    Returns a LazyFrame - nothing is read until .collect()
    """
    examples = (
        pl.scan_parquet(EXAMPLES)
        .filter(
            (pl.col("product_locale") == LOCALE)
            & (pl.col("large_version") == 1)
            & ~((pl.col("query_id") == 79706) & (pl.col("split") == "train"))
        )
        .select("query_id", "query", "product_id", "esci_label", "split")
    )

    products = (
        pl.scan_parquet(PRODUCTS)
        .filter(pl.col("product_locale") == LOCALE)
        .select(
            "product_id",
            "product_title",
            "product_brand",
            "product_bullet_point",
        )
    )

    return examples.join(products, on="product_id", how="left").with_columns(
        pl.col("esci_label")
        .replace_strict(ESCI_GAIN, default=0.0, return_dtype=pl.Float64)
        .alias("gain")
    )


def build_corpus() -> pl.DataFrame:
    """
    One row per unique US product
    """
    return (
        pl.scan_parquet(PRODUCTS)
        .filter(pl.col("product_locale") == LOCALE)
        .select("product_id", "product_title", "product_brand", "product_bullet_point")
        .with_columns(
            pl.concat_str(
                [
                    pl.col("product_title").fill_null(""),
                    pl.col("product_brand").fill_null(""),
                    pl.col("product_bullet_point").fill_null(""),
                ],
                separator=" ",
            )
            .str.strip_chars()
            .alias("text")
        )
        .select("product_id", "text")
        .collect()
    )


def materialise() -> None:
    """Write the joined view to data/processed/"""

    PROCESSED.mkdir(parents=True, exist_ok=True)
    load_judgements().collect().write_parquet(PROCESSED / "judgements.parquet")
    build_corpus().write_parquet(PROCESSED / "corpus.parquet")


if __name__ == "__main__":
    materialise()
