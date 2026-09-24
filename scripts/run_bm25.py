"""Build the BM25 index and evaluate it in both modes."""

from __future__ import annotations

import argparse
import time

import polars as pl

from esci.data.splits import add_folds
from esci.eval.runner import evaluate_rerank, evaluate_retrieval
from esci.index.lexical import BM25Ranker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", default="test")
    parser.add_argument(
        "--limit", type=int, default=None, help="evaluate on N queries only, for fast iteration"
    )
    parser.add_argument("--corpus", choices=["t", "tbb"], default="tbb")
    args = parser.parse_args()

    corpus_path = f"data/processed/corpus_{args.corpus}.parquet"
    corpus = pl.read_parquet(corpus_path)
    print(f"corpus: {corpus.height:,} products")

    judgements = add_folds(pl.read_parquet("data/processed/judgements.parquet"))
    evalset = judgements.filter(pl.col("fold") == args.fold)

    if args.limit:
        keep = evalset["query_id"].unique().head(args.limit)
        evalset = evalset.filter(pl.col("query_id").is_in(keep))

    print(
        f"eval fold '{args.fold}': "
        f"{evalset['query_id'].n_unique():,} queries, {evalset.height:,} rows"
    )

    t0 = time.perf_counter()
    ranker = BM25Ranker(
        product_ids=corpus["product_id"].to_list(),
        texts=corpus["text"].to_list(),
        name=f"bm25_{args.corpus}",
    )
    print(f"index built in {time.perf_counter() - t0:.1f}s")

    print("\n--- Mode A: rerank judged set ---")
    print(evaluate_rerank(ranker, evalset, k=10))

    print("\n--- Mode B: full-corpus retrieval ---")
    print(evaluate_retrieval(ranker, evalset, k=100))


if __name__ == "__main__":
    main()
