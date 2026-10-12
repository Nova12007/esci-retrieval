"""Build the folder the search service loads: index, ids, encoder, catalog.

    uv run python scripts/build_serving_artifacts.py --run biencoder_v3

Needs the run's cached corpus embeddings (run_dense.py) and its ONNX export
(export_onnx.py). Writes artifacts/serving/<run>/ with a manifest recording
where every piece came from.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

import faiss
import numpy as np
import polars as pl
from transformers import AutoTokenizer

from esci.models.biencoder import EncoderConfig
from esci.serving.catalog import build_catalog

HNSW_M, EF_CONSTRUCTION, EF_SEARCH = 16, 200, 256  # ADR-009


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="biencoder_v3")
    p.add_argument("--corpus", choices=["t", "tbb"], default="tbb")
    args = p.parse_args()

    run = Path("artifacts") / args.run
    out = Path("artifacts/serving") / args.run
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(f"artifacts/embeddings/{args.run}_{args.corpus}")
    checkpoint = run / "final.pt"
    built_by = stem.with_suffix(".source.txt").read_text(encoding="utf-8")
    assert built_by == checkpoint.as_posix(), f"embeddings built by {built_by}, not {checkpoint}"

    ids = stem.with_suffix(".ids.txt").read_text(encoding="utf-8").split("\n")
    emb = np.ascontiguousarray(np.load(stem.with_suffix(".npy")), dtype=np.float32)
    assert emb.shape[0] == len(ids)

    t0 = time.perf_counter()
    faiss.omp_set_num_threads(max(1, faiss.omp_get_max_threads()))
    index = faiss.IndexHNSWFlat(emb.shape[1], HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = EF_CONSTRUCTION
    index.add(emb)
    build_s = time.perf_counter() - t0
    assert index.ntotal == len(ids)
    faiss.write_index(index, str(out / "index.faiss"))
    (out / "ids.txt").write_text("\n".join(ids), encoding="utf-8")
    print(f"HNSW over {len(ids):,} vectors in {build_s:.0f}s")

    cfg = EncoderConfig.load(run / "encoder_config.json")
    shutil.copy(run / "encoder_config.json", out / "encoder_config.json")
    shutil.copy(Path("artifacts/onnx") / args.run / "query_fp32.onnx", out / "query_fp32.onnx")
    AutoTokenizer.from_pretrained(cfg.model_name).save_pretrained(out / "tokenizer")

    meta = (
        pl.read_parquet("data/processed/judgements.parquet")
        .select("product_id", "product_title", "product_brand")
        .unique("product_id", keep="first")
    )
    lookup = {pid: (t or "", b or "") for pid, t, b in meta.iter_rows()}
    missing = sum(1 for pid in ids if pid not in lookup)
    build_catalog(out / "catalog.sqlite", [(pid, *lookup.get(pid, ("", ""))) for pid in ids])
    print(f"catalog: {len(ids):,} products, {missing:,} without a title")

    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    manifest = {
        "model": args.run,
        "checkpoint_sha256": sha256(checkpoint),
        "onnx_sha256": sha256(out / "query_fp32.onnx"),
        "corpus": args.corpus,
        "index": "HNSWFlat",
        "hnsw_m": HNSW_M,
        "ef_construction": EF_CONSTRUCTION,
        "ef_search": EF_SEARCH,
        "git_commit": commit,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    sizes = {f.name: f"{f.stat().st_size / 1e6:.0f} MB" for f in out.iterdir() if f.is_file()}
    print(json.dumps(manifest, indent=2))
    print("sizes:", sizes)


if __name__ == "__main__":
    main()
