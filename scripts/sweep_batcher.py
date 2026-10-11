"""Throughput and latency of micro-batched query encoding under load.

    uv run python scripts/sweep_batcher.py --run biencoder_v3

Closed-loop load: C concurrent clients each send a query, wait for its
vector, and send the next, for a fixed duration. The ONNX fp32 encoder gets
2 intra-op threads, a 2-vCPU instance. max_batch=1 is no batching, the
baseline. Run on an idle machine; anything else on the CPU skews it.
Writes benchmarks/batcher_sweep.md.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

import numpy as np
import polars as pl

from esci.models.biencoder import EncoderConfig
from esci.serving.batcher import MicroBatcher
from esci.serving.onnx_encoder import OnnxQueryEncoder

OUT = Path("benchmarks/batcher_sweep.md")


async def load(
    batcher: MicroBatcher, queries: list[str], clients: int, seconds: float
) -> tuple[float, float, float]:
    """Returns (queries/s, p50 ms, p99 ms)."""
    latencies: list[float] = []
    stop = time.perf_counter() + seconds

    async def client(offset: int) -> None:
        i = offset
        while time.perf_counter() < stop:
            t0 = time.perf_counter()
            await batcher.encode(queries[i % len(queries)])
            latencies.append((time.perf_counter() - t0) * 1000)
            i += clients

    started = time.perf_counter()
    await asyncio.gather(*(client(c) for c in range(clients)))
    elapsed = time.perf_counter() - started
    p50, p99 = np.percentile(latencies, [50, 99])
    return len(latencies) / elapsed, float(p50), float(p99)


async def sweep(args: argparse.Namespace) -> list[str]:
    run = Path("artifacts") / args.run
    cfg = EncoderConfig.load(run / "encoder_config.json")
    encoder = OnnxQueryEncoder(Path("artifacts/onnx") / args.run / "query_fp32.onnx", cfg, 2)
    queries = (
        pl.read_parquet("data/processed/judgements.parquet")["query"]
        .unique()
        .sort()
        .sample(2000, seed=0)
        .to_list()
    )
    rows = []
    for clients in args.clients:
        for max_batch, wait in [(1, 0.0), (8, 0.0), (16, 0.0), (8, 2.0), (16, 5.0)]:
            batcher = MicroBatcher(encoder.encode, max_batch=max_batch, max_wait_ms=wait)
            await batcher.start()
            await load(batcher, queries, clients, 2.0)  # warm up
            batcher.stats.sizes.clear()
            batcher.stats.batches = batcher.stats.items = 0
            qps, p50, p99 = await load(batcher, queries, clients, args.seconds)
            await batcher.stop()
            row = (
                f"| {clients} | {max_batch} | {wait:g} | {qps:,.0f} | {p50:.1f} | {p99:.1f} "
                f"| {batcher.stats.mean_batch:.1f} |"
            )
            print(row, flush=True)
            rows.append(row)
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="biencoder_v3")
    p.add_argument("--clients", type=int, nargs="+", default=[1, 8, 32])
    p.add_argument("--seconds", type=float, default=10.0)
    args = p.parse_args()
    header = [
        "Micro-batched ONNX fp32 query encoding, 2 intra-op threads, closed-loop load.\n",
        "| clients | max batch | window ms | queries/s | p50 ms | p99 ms | mean batch |",
        "|---|---|---|---|---|---|---|",
    ]
    print("\n".join(header[1:]))
    rows = asyncio.run(sweep(args))
    OUT.write_text("\n".join(header + rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
