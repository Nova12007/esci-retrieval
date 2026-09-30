"""Systems with 95% bootstrap CIs, and paired differences against a baseline.

    uv run python scripts/compare.py --baseline bm25_tbb \
        --systems bge_small_zeroshot biencoder_v2

Reads the per-query scores every evaluation writes to artifacts/per_query/.
"""

from __future__ import annotations

import argparse

import polars as pl

from esci.eval.runner import PER_QUERY
from esci.eval.stats import bootstrap_ci, paired_bootstrap_diff

METRICS = {"retrieval": "recall@100", "rerank": "ndcg@10"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True)
    p.add_argument("--systems", nargs="+", required=True)
    p.add_argument("--resamples", type=int, default=1000)
    args = p.parse_args()

    for mode, metric in METRICS.items():
        print(f"\n{mode} -- {metric}, 95% CI over queries, {args.resamples} resamples")
        base = pl.read_parquet(PER_QUERY / f"{args.baseline}__{mode}.parquet")
        for system in [args.baseline, *args.systems]:
            df = pl.read_parquet(PER_QUERY / f"{system}__{mode}.parquet")
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
