"""Build the committed regression fixture: a seeded sample of unsaturated test queries with
every judged product and its text, small enough for CI.

    uv run python scripts/make_regression_fixture.py

ESCI is Apache-2.0, so a ~1 MB extract can live in the repo, which lets CI run
the encoder and metric end to end without the dataset.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from esci.data.splits import add_folds
from esci.eval.slices import unsaturated_queries

OUT = Path("tests/fixtures/regression_queries.parquet")
N_QUERIES = 20  # ~400 products: about two minutes of CPU encoding on a CI runner
SEED = 7


def main() -> None:
    judgements = add_folds(pl.read_parquet("data/processed/judgements.parquet"))
    test = judgements.filter(pl.col("fold") == "test")
    # Unsaturated queries only: on a judged set of mostly Exact products any
    # ranking scores ~1.0, which would hide exactly the bugs this test exists
    # to catch (ADR-005).
    qids = unsaturated_queries(test).sort().sample(N_QUERIES, seed=SEED)
    corpus = pl.read_parquet("data/processed/corpus_tbb.parquet").select("product_id", "text")
    fixture = (
        test.filter(pl.col("query_id").is_in(qids.implode()))
        .select("query_id", "query", "product_id", "esci_label", "gain")
        .join(corpus, on="product_id", how="left")
        .sort("query_id", "product_id")
    )
    assert fixture["text"].null_count() == 0, "judged product missing from corpus"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fixture.write_parquet(OUT, compression="zstd")
    print(
        f"{fixture.height} rows, {N_QUERIES} queries -> {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)"
    )


if __name__ == "__main__":
    main()
