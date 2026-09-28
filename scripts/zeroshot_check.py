"""Sanity check: off-the-shelf bge-small over a 50k-product subset."""

from __future__ import annotations

import faiss
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DEVICE = "cuda"


def encode(texts, tok, model, prefix="", max_len=128, bs=128):
    out = []
    for i in range(0, len(texts), bs):
        chunk = [prefix + t for t in texts[i : i + bs]]
        batch = tok(
            chunk, padding=True, truncation=True, max_length=max_len, return_tensors="pt"
        ).to(DEVICE)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = model(**batch).last_hidden_state[:, 0]  # CLS for bge
        out.append(F.normalize(hidden.float(), p=2, dim=-1).cpu().numpy())
    return np.vstack(out).astype(np.float32)


def main() -> None:
    corpus = pl.read_parquet("data/processed/corpus_tbb.parquet").head(50_000)
    corpus["product_id"].to_list()
    texts = corpus["text"].to_list()

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(DEVICE).eval()

    print("encoding 50k products...")
    doc_emb = encode(texts, tok, model)  # NO prefix on docs

    index = faiss.IndexFlatIP(doc_emb.shape[1])  # IP on normalised = cosine
    index.add(doc_emb)

    probes = [
        "wireless bluetooth earbuds",
        "stainless steel water bottle 1 litre",
        "usb c to hdmi adapter 4k",
    ]
    q_emb = encode(probes, tok, model, prefix=QUERY_PREFIX, max_len=48)
    scores, idx = index.search(q_emb, 5)

    for qi, q in enumerate(probes):
        print(f"\n=== {q}")
        for rank in range(5):
            print(f"  {scores[qi][rank]:.3f}  {texts[idx[qi][rank]][:90]}")


if __name__ == "__main__":
    main()
