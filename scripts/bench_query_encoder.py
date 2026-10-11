"""Query encoder variants on the served path: quality and CPU latency.

    uv run python scripts/bench_query_encoder.py --run biencoder_v3           # quality
    uv run python scripts/bench_query_encoder.py --run biencoder_v3 --latency # + timing

Variants: PyTorch fp32, ONNX fp32, ONNX INT8 (scripts/export_onnx.py).
Quality: a seeded sample of test queries is encoded by each variant and
searched against the same offline product vectors by exact search, so the
only thing that differs is the query vector. Latency: one query per call, at
1 and 2 threads (a Cloud Run instance has 2 vCPUs). Measure latency on an
otherwise idle machine; a training run on the same CPU inflates it.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path

import faiss
import numpy as np
import polars as pl
import torch

from esci.data.splits import add_folds
from esci.eval.aggregate import score_retrieval
from esci.eval.stats import bootstrap_ci
from esci.models.biencoder import BiEncoder, EncoderConfig
from esci.serving.onnx_encoder import OnnxQueryEncoder

Encode = Callable[[list[str]], np.ndarray]


def latency(encode: Encode, queries: list[str]) -> str:
    for q in queries[:10]:
        encode([q])  # warm up
    times = []
    for q in queries:
        t0 = time.perf_counter()
        encode([q])
        times.append((time.perf_counter() - t0) * 1000)
    p50, p99 = np.percentile(times, [50, 99])
    return f"p50 {p50:5.1f} ms  p99 {p99:5.1f} ms"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="biencoder_v3")
    p.add_argument("--queries", type=int, default=3000)
    p.add_argument("--latency", action="store_true")
    p.add_argument("--latency-queries", type=int, default=500)
    args = p.parse_args()

    run = Path("artifacts") / args.run
    cfg = EncoderConfig.load(run / "encoder_config.json")
    torch_enc = BiEncoder(cfg, device="cpu")
    torch_enc.model.load_state_dict(torch.load(run / "final.pt", map_location="cpu"))
    onnx_dir = Path("artifacts/onnx") / args.run

    test = add_folds(pl.read_parquet("data/processed/judgements.parquet")).filter(
        pl.col("fold") == "test"
    )
    qids = test["query_id"].unique().sort().sample(args.queries, seed=0)
    sample = test.filter(pl.col("query_id").is_in(qids.implode()))
    qtext = dict(zip(sample["query_id"], sample["query"], strict=True))
    order = list(qtext)
    texts = [qtext[q] for q in order]

    stem = f"artifacts/embeddings/{args.run}_tbb"
    ids = Path(f"{stem}.ids.txt").read_text(encoding="utf-8").split("\n")
    assert Path(f"{stem}.source.txt").read_text(encoding="utf-8") == (run / "final.pt").as_posix()
    index = faiss.IndexFlatIP(cfg_dim := torch_enc.dim)
    index.add(np.ascontiguousarray(np.load(f"{stem}.npy")))
    assert index.ntotal == len(ids) and cfg_dim == 384

    onnx_fp32 = OnnxQueryEncoder(onnx_dir / "query_fp32.onnx", cfg, threads=4)
    onnx_int8 = OnnxQueryEncoder(onnx_dir / "query_int8.onnx", cfg, threads=4)
    variants: dict[str, Encode] = {
        "torch fp32": lambda qs: torch_enc.encode(qs, is_query=True),
        "onnx fp32": onnx_fp32.encode,
        "onnx int8": onnx_int8.encode,
    }

    per_query: dict[str, pl.DataFrame] = {}
    print(f"quality on {len(order):,} test queries (exact search, same product vectors):")
    for name, encode in variants.items():
        q = np.concatenate([encode(texts[s : s + 256]) for s in range(0, len(texts), 256)])
        _, rows = index.search(np.ascontiguousarray(q, dtype=np.float32), 100)
        retrieved = {qid: [ids[r] for r in row] for qid, row in zip(order, rows, strict=True)}
        per_query[name] = score_retrieval(retrieved, sample, k=100)
        recall = bootstrap_ci(per_query[name]["recall@100"].drop_nulls().to_numpy())
        ndcg = per_query[name]["ndcg@10"].drop_nulls().mean()
        line = f"  {name:11} Recall@100 {recall}  nDCG@10 {ndcg:.4f}"
        if name != "torch fp32":
            ref = per_query["torch fp32"]
            d = (per_query[name]["recall@100"] - ref["recall@100"]).drop_nulls().mean()
            line += f"   recall vs torch {d:+.4f}"
        print(line, flush=True)

    if args.latency:
        lat_q = texts[: args.latency_queries]
        for threads in (1, 2):
            torch.set_num_threads(threads)
            timed: dict[str, Encode] = {
                "torch fp32": variants["torch fp32"],
                "onnx fp32": OnnxQueryEncoder(onnx_dir / "query_fp32.onnx", cfg, threads).encode,
                "onnx int8": OnnxQueryEncoder(onnx_dir / "query_int8.onnx", cfg, threads).encode,
            }
            print(f"latency, one query per call, {threads} thread(s):")
            for name, encode in timed.items():
                print(f"  {name:11} {latency(encode, lat_q)}", flush=True)


if __name__ == "__main__":
    main()
