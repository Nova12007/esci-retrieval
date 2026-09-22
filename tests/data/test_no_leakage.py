from __future__ import annotations

import polars as pl
import pytest

from esci.data.splits import add_folds, query_bucket


@pytest.fixture(scope="module")
def folded() -> pl.DataFrame:
    df = pl.read_parquet("data/processed/judgements.parquet")
    return add_folds(df)


def test_every_row_has_a_fold(folded: pl.DataFrame) -> None:
    assert folded["fold"].null_count() == 0
    assert set(folded["fold"].unique()) <= {"train", "dev", "test"}


def test_no_query_appears_in_two_folds(folded: pl.DataFrame) -> None:
    per_query = folded.group_by("query_id").agg(pl.col("fold").n_unique().alias("n_folds"))
    offenders = per_query.filter(pl.col("n_folds") > 1)
    assert offenders.height == 0, f"{offenders.height} queries span multiple folds"


def test_no_query_overlap_between_train_and_test(folded: pl.DataFrame) -> None:
    train_queries = set(folded.filter(pl.col("split") == "train")["query_id"].to_list())
    test_queries = set(folded.filter(pl.col("split") == "test")["query_id"].to_list())
    overlap = train_queries & test_queries
    assert not overlap, (
        f"{len(overlap)} queries appear in both train and test: {sorted(overlap)[:10]}"
    )


def test_all_three_folds_are_populated(folded: pl.DataFrame) -> None:
    counts = folded.group_by("fold").len().to_dict(as_series=False)
    sizes = dict(zip(counts["fold"], counts["len"], strict=True))
    for fold in ("train", "dev", "test"):
        assert sizes.get(fold, 0) > 0, f"fold {fold} is empty"


def test_bucketing_is_deterministic() -> None:
    """Same input, same bucket"""
    assert query_bucket(12345) == query_bucket(12345)
    assert 0 <= query_bucket(999) < 100


def test_bucketing_spreads_across_the_range() -> None:
    buckets = {query_bucket(i) for i in range(2000)}
    assert len(buckets) > 90
