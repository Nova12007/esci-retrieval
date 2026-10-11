"""Export the query encoder to ONNX (fp32 and INT8) and check parity first.

    uv run python scripts/export_onnx.py --checkpoint artifacts/biencoder_v3/final.pt

A silently wrong export produces fast, meaningless numbers, so nothing is
timed until the ONNX outputs match PyTorch on real test queries. Writes
artifacts/onnx/<run>/query_fp32.onnx and query_int8.onnx.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from onnxruntime.quantization import QuantType, quantize_dynamic

from esci.data.splits import add_folds
from esci.models.biencoder import BiEncoder, EncoderConfig
from esci.serving.onnx_encoder import OnnxQueryEncoder


class _Pooled(torch.nn.Module):
    """input_ids, attention_mask -> L2-normalised CLS vector, as one graph."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return F.normalize(hidden[:, 0], p=2, dim=-1)


def parity(name: str, reference: np.ndarray, candidate: np.ndarray) -> float:
    cos = (reference * candidate).sum(1)
    print(
        f"  {name:5} max |diff| {np.abs(reference - candidate).max():.2e}   "
        f"cosine to PyTorch: min {cos.min():.6f}, mean {cos.mean():.6f}"
    )
    return float(cos.min())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--queries", type=int, default=1000, help="test queries for parity")
    args = p.parse_args()

    ckpt = Path(args.checkpoint)
    cfg = EncoderConfig.load(ckpt.parent / "encoder_config.json")
    assert cfg.pooling == "cls", "export graph implements CLS pooling only"
    encoder = BiEncoder(cfg, device="cpu")
    encoder.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    encoder.model.eval()

    out = Path("artifacts/onnx") / ckpt.parent.name
    out.mkdir(parents=True, exist_ok=True)
    fp32, int8 = out / "query_fp32.onnx", out / "query_int8.onnx"

    sample = encoder.tokenizer(
        [cfg.query_prefix + "usb c cable"], return_tensors="pt", padding=True
    )
    torch.onnx.export(
        _Pooled(encoder.model),
        (sample["input_ids"], sample["attention_mask"]),
        str(fp32),
        input_names=["input_ids", "attention_mask"],
        output_names=["embedding"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "embedding": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    # Dynamic INT8: weights quantised ahead of time, activations at run time.
    # No calibration set needed, and it targets the matmuls that dominate.
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
    print(f"fp32 {fp32.stat().st_size / 1e6:.0f} MB, int8 {int8.stat().st_size / 1e6:.0f} MB")

    test = add_folds(pl.read_parquet("data/processed/judgements.parquet")).filter(
        pl.col("fold") == "test"
    )
    queries = test["query"].unique().sort().head(args.queries).to_list()
    reference = encoder.encode(queries, is_query=True)

    print(f"parity on {len(queries)} test queries:")
    worst_fp32 = parity("fp32", reference, OnnxQueryEncoder(fp32, cfg).encode(queries))
    parity("int8", reference, OnnxQueryEncoder(int8, cfg).encode(queries))
    # fp32 must agree to rounding; int8 is judged by retrieval quality, not parity
    assert worst_fp32 > 0.9999, "fp32 ONNX export does not match PyTorch"
    print("fp32 parity OK")


if __name__ == "__main__":
    main()
