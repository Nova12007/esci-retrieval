"""Mode B Recall@100 per query slice, and where dense wins or loses to BM25.

    uv run python scripts/slice_report.py

Writes benchmarks/slices.md. Each cell is the mean over the slice's queries;
the last column is the paired difference v3 - BM25 with its 95% bootstrap CI.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from esci.data.splits import add_folds
from esci.eval.runner import PER_QUERY
from esci.eval.slices import query_slices
from esci.eval.stats import paired_bootstrap_diff

SYSTEMS = ["bm25_tbb", "bge_small_zeroshot", "biencoder_v3", "rrf_bm25_biencoder_v3"]
DENSE, LEXICAL = "biencoder_v3", "bm25_tbb"
METRIC = "recall@100"
OUT = Path("benchmarks/slices.md")


def main() -> None:
    test = add_folds(pl.read_parquet("data/processed/judgements.parquet")).filter(
        pl.col("fold") == "test"
    )
    slices = query_slices(test)
    scores = {
        s: pl.read_parquet(PER_QUERY / f"{s}__retrieval.parquet").select("query_id", METRIC)
        for s in SYSTEMS
    }

    header = ["slice", "queries", *SYSTEMS, f"{DENSE} - {LEXICAL} [95% CI]"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for dimension in ["length", "has_number", "brand", "saturation"]:
        for value in sorted(slices[dimension].unique().to_list()):
            ids = slices.filter(pl.col(dimension) == value)["query_id"].implode()
            sub = {s: df.filter(pl.col("query_id").is_in(ids)) for s, df in scores.items()}
            means = [f"{sub[s][METRIC].drop_nulls().mean():.4f}" for s in SYSTEMS]
            d = paired_bootstrap_diff(sub[LEXICAL], sub[DENSE], METRIC)
            cell = f"{d.estimate:+.4f} [{d.low:+.4f}, {d.high:+.4f}]"
            lines.append(
                f"| {value} | {sub[DENSE].height:,} | " + " | ".join([*means, cell]) + " |"
            )

    table = "\n".join(lines)
    OUT.write_text(f"Mode B {METRIC} by query slice, test fold.\n\n{table}\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
