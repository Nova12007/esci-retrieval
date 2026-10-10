"""Evaluate a cross-encoder in both modes, and time it.

    uv run python scripts/run_rerank.py --name ce_v1 \
        --ce-checkpoint artifacts/crossencoder_v1/final.pt \
        --first-stage biencoder_v3 --first-stage-checkpoint artifacts/biencoder_v3/final.pt

Mode A: the cross-encoder ranks each query's judged set (the quality ceiling).
Mode B: two-stage. The bi-encoder retrieves the top 100 from 1.2M products by
exact search, the cross-encoder reorders them. Recall@100 must equal the first
stage's; nDCG@10 over the list is what reranking can move.

Latency is the cross-encoder scoring one query's 100 candidates, on the GPU and
on the CPU, since the served path has no GPU.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

from esci.data.splits import add_folds
from esci.eval.runner import Ranker, evaluate_rerank, evaluate_retrieval
from esci.index.dense import DenseRanker
from esci.index.fusion import FusionRanker
from esci.index.lexical import BM25Ranker
from esci.index.rerank import RerankRanker
from esci.models.biencoder import BiEncoder, EncoderConfig
from esci.models.crossencoder import CrossEncoder, CrossEncoderConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="system name in results.jsonl")
    p.add_argument("--ce-checkpoint", default=None, help="omit for the MS MARCO model as-is")
    p.add_argument("--first-stage", default="biencoder_v3", help="cached embeddings name")
    p.add_argument("--first-stage-checkpoint", default="artifacts/biencoder_v3/final.pt")
    p.add_argument("--corpus", choices=["t", "tbb"], default="tbb")
    p.add_argument("--depth", type=int, default=100)
    p.add_argument("--fold", default="test")
    p.add_argument("--limit", type=int, default=None, help="evaluate on N queries only")
    p.add_argument("--latency-queries", type=int, default=50)
    p.add_argument(
        "--fusion",
        action="store_true",
        help="first stage = RRF of BM25 and the dense model, instead of dense alone",
    )
    return p.parse_args()


def load_ce(checkpoint: str | None, device: str) -> CrossEncoder:
    if checkpoint is None:
        return CrossEncoder(CrossEncoderConfig(), device=device)
    ckpt = Path(checkpoint)
    ce = CrossEncoder(CrossEncoderConfig.load(ckpt.parent / "ce_config.json"), device=device)
    ce.model.load_state_dict(torch.load(ckpt, map_location=device))
    return ce


def time_per_query(ce: CrossEncoder, queries: list[str], pools: list[list[str]]) -> str:
    """p50/p99 ms to score one query's candidate pool, one call per query."""
    ce.score([queries[0]] * len(pools[0]), pools[0])  # warm up
    times = []
    for q, pool in zip(queries, pools, strict=True):
        if ce.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        ce.score([q] * len(pool), pool, batch_size=len(pool))
        if ce.device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    p50, p99 = np.percentile(times, [50, 99])
    return f"p50 {p50:.1f} ms, p99 {p99:.1f} ms"


def main() -> None:
    args = parse_args()
    corpus = pl.read_parquet(f"data/processed/corpus_{args.corpus}.parquet")
    text = dict(zip(corpus["product_id"], corpus["text"], strict=True))
    judgements = add_folds(pl.read_parquet("data/processed/judgements.parquet"))
    evalset = judgements.filter(pl.col("fold") == args.fold)
    if args.limit:
        keep = evalset["query_id"].unique().sort().head(args.limit)
        evalset = evalset.filter(pl.col("query_id").is_in(keep.implode()))
    print(f"eval fold '{args.fold}': {evalset['query_id'].n_unique():,} queries")

    ce = load_ce(args.ce_checkpoint, "cuda")

    fs_ckpt = Path(args.first_stage_checkpoint)
    encoder = BiEncoder(EncoderConfig.load(fs_ckpt.parent / "encoder_config.json"), "cuda")
    encoder.model.load_state_dict(torch.load(fs_ckpt, map_location="cuda"))
    stem = f"artifacts/embeddings/{args.first_stage}_{args.corpus}"
    ids = Path(f"{stem}.ids.txt").read_text(encoding="utf-8").split("\n")
    assert ids == corpus["product_id"].to_list(), "first-stage embeddings do not match corpus"
    first = DenseRanker(
        ids,
        np.load(f"{stem}.npy"),
        lambda qs: encoder.encode(qs, is_query=True),
        args.first_stage,
        backend="torch",
        device="cuda",
    )
    first_stage: Ranker = first
    if args.fusion:
        bm25 = BM25Ranker(corpus["product_id"].to_list(), corpus["text"].to_list(), "bm25")
        first_stage = FusionRanker([bm25, first], name=f"rrf_bm25_{args.first_stage}")
    ranker = RerankRanker(ce.score, text, args.name, first_stage=first_stage, depth=args.depth)

    print("\n--- Mode A: cross-encoder over the judged set ---")
    print(evaluate_rerank(ranker, evalset, k=10))
    print(f"\n--- Mode B: {first_stage.name} top-{args.depth} -> cross-encoder ---")
    print(evaluate_retrieval(ranker, evalset, k=args.depth))

    # Latency on one query's real candidate pool from the first stage.
    sample = evalset["query_id"].unique().sort().head(args.latency_queries).to_list()
    qtext = dict(zip(evalset["query_id"], evalset["query"], strict=True))
    pools = first.retrieve({q: qtext[q] for q in sample}, args.depth)
    queries = [qtext[q] for q in sample]
    docs = [[text.get(p, "") for p in pools[q]] for q in sample]
    print(f"\nlatency, {args.depth} candidates per query:")
    print(f"  GPU (RTX 4050, bf16): {time_per_query(ce, queries, docs)}")
    del ce, first  # free the 1.9 GB first-stage matrix; the 130 MB encoder can stay
    torch.cuda.empty_cache()
    cpu = load_ce(args.ce_checkpoint, "cpu")
    cpu_lat = time_per_query(cpu, queries[:20], docs[:20])
    print(f"  CPU ({torch.get_num_threads()} threads, fp32): {cpu_lat}")


if __name__ == "__main__":
    main()
