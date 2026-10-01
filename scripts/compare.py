"""Systems with 95% bootstrap CIs, and paired differences against a baseline.

    uv run python scripts/compare.py --baseline bm25_tbb \
        --systems bge_small_zeroshot biencoder_v2

Reads the per-query scores every evaluation writes to artifacts/per_query/.
Mode A is also reported on the unsaturated slice (queries with fewer than 10
Exact), where it can still discriminate (ADR-005).
"""

from __future__ import annotations

import argparse

import polars as pl

from esci.data.splits import add_folds
from esci.eval.runner import PER_QUERY
from esci.eval.slices import unsaturated_queries
from esci.eval.stats import bootstrap_ci, paired_bootstrap_diff

# (label, mode, metric, restrict to unsaturated queries)
REPORTS = [
    ("Mode B retrieval", "retrieval", "recall@100", False),
    ("Mode A rerank, all queries", "rerank", "ndcg@10", False),
    ("Mode A rerank, unsaturated (<10 Exact)", "rerank", "ndcg@10", True),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True)
    p.add_argument("--systems", nargs="+", required=True)
    p.add_argument("--resamples", type=int, default=1000)
    p.add_argument("--fold", default="test", help="fold the per-query files were scored on")
    args = p.parse_args()

    judgements = add_folds(pl.read_parquet("data/processed/judgements.parquet"))
    unsaturated = unsaturated_queries(judgements.filter(pl.col("fold") == args.fold))

    def load(system: str, mode: str, sliced: bool) -> pl.DataFrame:
        df = pl.read_parquet(PER_QUERY / f"{system}__{mode}.parquet")
        return df.filter(pl.col("query_id").is_in(unsaturated.implode())) if sliced else df

    for label, mode, metric, sliced in REPORTS:
        base = load(args.baseline, mode, sliced)
        print(f"\n{label} -- {metric} over {base.height:,} queries, 95% CI")
        for system in [args.baseline, *args.systems]:
            df = load(system, mode, sliced)
            ci = bootstrap_ci(df[metric].drop_nulls().to_numpy(), args.resamples)
            line = f"  {system:<22} {ci}"
            if system != args.baseline:
                diff = paired_bootstrap_diff(base, df, metric, args.resamples)
                sig = "significant" if diff.low > 0 or diff.high < 0 else "not significant"
                line += (
                    f"   vs {args.baseline}: {diff.estimate:+.4f} "
                    f"[{diff.low:+.4f}, {diff.high:+.4f}] {sig}"
                )
            print(line)


if __name__ == "__main__":
    main()
