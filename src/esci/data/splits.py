"""Deterministic train / dev / test folds.

The official ESCI `split` column gives train and test. Dev is carved out
of train by hashing query_id, so the assignment is stable across runs,
machines and reorderings of the data.
"""

from __future__ import annotations

import hashlib

import polars as pl

DEV_PERCENT = 10  # 10% of TRAIN queries become dev


def query_bucket(query_id: int) -> int:
    """Map a query id to a stable bucket in [0, 100)."""
    digest = hashlib.md5(str(query_id).encode("utf-8")).hexdigest()
    return int(digest, 16) % 100


def add_folds(df: pl.DataFrame) -> pl.DataFrame:
    """Add a `fold` column with values train / dev / test."""
    buckets = (
        df.select("query_id")
        .unique()
        .with_columns(
            pl.col("query_id").map_elements(query_bucket, return_dtype=pl.Int64).alias("bucket")
        )
    )

    return (
        df.join(buckets, on="query_id", how="left")
        .with_columns(
            pl.when(pl.col("split") == "test")
            .then(pl.lit("test"))
            .when(pl.col("bucket") < DEV_PERCENT)
            .then(pl.lit("dev"))
            .otherwise(pl.lit("train"))
            .alias("fold")
        )
        .drop("bucket")
    )
