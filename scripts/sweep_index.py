"""Sweep ANN indexes over one model's corpus embeddings and draw the Pareto chart.

    uv run python scripts/sweep_index.py --name biencoder_v3 \
        --checkpoint artifacts/biencoder_v3/final.pt

Needs the embeddings run_dense.py cached. Every point reports, on a seeded
sample of test queries:
  recall_vs_exact   share of exact search's top-100 the index also returns
  judged_recall     Mode B Recall@100 on judged-Exact products (the headline)
  size_mb           serialised index size, what the container must load
  p50/p99_ms        one query per call, one thread, as a request handler searches
Writes benchmarks/index_sweep.jsonl, .md and benchmarks/figures/pareto.png.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np
import polars as pl
import torch

from esci.data.splits import add_folds
from esci.eval.aggregate import score_retrieval
from esci.index.ann import (
    build_flat,
    build_hnsw,
    build_ivfpq,
    index_bytes,
    recall_vs_exact,
    set_search_param,
    single_query_latency,
)
from esci.models.biencoder import BiEncoder, EncoderConfig

OUT = Path("benchmarks")
HNSW_M = [16, 32, 64]
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = [16, 32, 64, 128, 256]
IVF_NLIST = [1024, 4096]
IVF_M = [32, 48, 96]
IVF_NPROBE = [8, 16, 32, 64]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="embeddings to index, as cached by run_dense")
    p.add_argument("--checkpoint", required=True, help="the same model, to encode queries")
    p.add_argument("--corpus", choices=["t", "tbb"], default="tbb")
    p.add_argument("--queries", type=int, default=2000, help="test queries for recall")
    p.add_argument("--latency-queries", type=int, default=500)
    p.add_argument("--device", default="cuda", help="query encoding and exact ground truth")
    p.add_argument("--plot-only", action="store_true", help="redraw from index_sweep.jsonl")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="50k vectors, tiny grid, writes to the scratch dir: exercises every path in ~1 min",
    )
    return p.parse_args()


def exact_top_k(emb: np.ndarray, q: np.ndarray, k: int, device: str) -> np.ndarray:
    matrix = torch.from_numpy(np.ascontiguousarray(emb)).to(device)
    rows = []
    for s in range(0, q.shape[0], 256):
        scores = torch.from_numpy(q[s : s + 256]).to(device) @ matrix.T
        rows.append(torch.topk(scores, k, dim=1).indices.cpu())
    return torch.cat(rows).numpy()


def sweep(args: argparse.Namespace) -> list[dict[str, object]]:
    corpus_ids = Path(f"artifacts/embeddings/{args.name}_{args.corpus}.ids.txt")
    ids = corpus_ids.read_text(encoding="utf-8").split("\n")
    emb = np.ascontiguousarray(
        np.load(f"artifacts/embeddings/{args.name}_{args.corpus}.npy")  # into RAM, writable
    )
    assert emb.shape[0] == len(ids)
    hnsw_m, ef_search, nlists, pq_m, nprobes = HNSW_M, HNSW_EF_SEARCH, IVF_NLIST, IVF_M, IVF_NPROBE
    if args.smoke:
        emb, ids = emb[:50_000], ids[:50_000]
        hnsw_m, ef_search, nlists, pq_m, nprobes = [16], [32], [256], [48], [8]

    test = add_folds(pl.read_parquet("data/processed/judgements.parquet")).filter(
        pl.col("fold") == "test"
    )
    qids = test["query_id"].unique().sort().sample(args.queries, seed=0).to_list()
    sample = test.filter(pl.col("query_id").is_in(qids))
    text = dict(zip(sample["query_id"], sample["query"], strict=True))

    ckpt = Path(args.checkpoint)
    encoder = BiEncoder(EncoderConfig.load(ckpt.parent / "encoder_config.json"), args.device)
    encoder.model.load_state_dict(torch.load(ckpt, map_location=args.device))
    q = np.ascontiguousarray(encoder.encode([text[i] for i in qids], is_query=True))
    del encoder
    exact = exact_top_k(emb, q, 100, args.device)
    lat_q = q[: args.latency_queries]

    rows: list[dict[str, object]] = []

    def measure(index: faiss.Index, family: str, config: str, build_s: float, param: int) -> None:
        set_search_param(index, param)
        _, found = index.search(q, 100)
        retrieved = {
            qid: [ids[r] for r in row if r >= 0] for qid, row in zip(qids, found, strict=True)
        }
        judged = score_retrieval(retrieved, sample, k=100)["recall@100"].drop_nulls().mean()
        lat = single_query_latency(index, lat_q)
        row: dict[str, object] = {
            "family": family,
            "config": config,
            "search_param": param,
            "recall_vs_exact": round(recall_vs_exact(found, exact), 4),
            "judged_recall@100": round(float(judged or 0.0), 4),
            "size_mb": round(index_bytes(index) / 1e6, 1),
            "build_s": round(build_s, 1),
            "p50_ms": round(lat.p50_ms, 2),
            "p99_ms": round(lat.p99_ms, 2),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    t0 = time.perf_counter()
    flat = build_flat(emb)
    measure(flat, "flat", "IndexFlatIP", time.perf_counter() - t0, 0)
    del flat

    for m in hnsw_m:
        t0 = time.perf_counter()
        index = build_hnsw(emb, m, HNSW_EF_CONSTRUCTION)
        built = time.perf_counter() - t0
        for ef in ef_search:
            measure(index, "hnsw", f"M={m},efC={HNSW_EF_CONSTRUCTION}", built, ef)
        del index

    for nlist in nlists:
        for m in pq_m:
            t0 = time.perf_counter()
            index = build_ivfpq(emb, nlist, m)
            built = time.perf_counter() - t0
            for nprobe in nprobes:
                measure(index, "ivfpq", f"nlist={nlist},m={m}", built, nprobe)
            del index
    return rows


def write_table(rows: list[dict[str, object]]) -> None:
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    (OUT / "index_sweep.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(rows: list[dict[str, object]], name: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullFormatter

    # Categorical slots 1-3 in fixed order, plus a marker per family so identity
    # never rests on colour alone (aqua is under 3:1 contrast on white).
    style = {
        "flat": ("#2a78d6", "s", "Flat (exact)"),
        "hnsw": ("#eb6834", "o", "HNSW"),
        "ivfpq": ("#1baf7a", "^", "IVF-PQ"),
    }
    ink, muted, grid = "#0b0b0b", "#52514e", "#e6e5e0"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True, facecolor="#fcfcfb")
    for ax, x, xlabel in [
        (axes[0], "size_mb", "Index size (MB, log scale)"),
        (axes[1], "p99_ms", "p99 latency, one query, one thread (ms, log scale)"),
    ]:
        ax.set_facecolor("#fcfcfb")
        for family, (colour, marker, label) in style.items():
            pts = [r for r in rows if r["family"] == family]
            # One thin line per built index: the search-time knob (efSearch or
            # nprobe) traces a recall/cost curve without rebuilding.
            for config in dict.fromkeys(str(r["config"]) for r in pts):
                curve = [r for r in pts if r["config"] == config]
                ax.plot(
                    [float(str(r[x])) for r in curve],
                    [float(str(r["recall_vs_exact"])) for r in curve],
                    color=colour,
                    linewidth=1,
                    alpha=0.6,
                    zorder=2,
                )
            ax.scatter(
                [float(str(r[x])) for r in pts],
                [float(str(r["recall_vs_exact"])) for r in pts],
                s=64,
                c=colour,
                marker=marker,
                label=label,
                edgecolors="#fcfcfb",
                linewidths=1.5,
                zorder=3,
            )
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(xlabel, color=muted)
        ax.grid(True, color=grid, linewidth=0.8, zorder=0)
        ax.tick_params(colors=muted)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_ylabel("Recall@100 vs exact search", color=muted)
    axes[0].legend(frameon=False, labelcolor=ink)
    fig.suptitle(
        f"ANN indexes over 1.2M products ({name} embeddings): recall vs memory and latency",
        color=ink,
        fontsize=12,
        x=0.02,
        ha="left",
    )
    fig.tight_layout()
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figures" / "pareto.png", dpi=150, facecolor=fig.get_facecolor())


def main() -> None:
    args = parse_args()
    global OUT
    if args.smoke:
        OUT = Path("artifacts/sweep_smoke")
        OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "index_sweep.jsonl"
    if args.plot_only:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    else:
        rows = sweep(args)
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        write_table(rows)
    plot(rows, args.name)
    print(f"wrote {path}, index_sweep.md, figures/pareto.png")


if __name__ == "__main__":
    main()
