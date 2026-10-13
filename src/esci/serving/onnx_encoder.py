"""Query encoding for the served path: ONNX Runtime on CPU.

Only queries are encoded per request; product vectors are computed offline by
the PyTorch model. The exported graph includes CLS pooling and L2
normalisation, and the prefix comes from the same EncoderConfig the model
was trained with, so the server cannot drift from training on either.

Tokenisation uses the `tokenizers` library on the model's own tokenizer.json,
the same pipeline the transformers fast tokenizer runs, without importing
transformers or torch: the serving image carries no training stack.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from esci.models.config import EncoderConfig


def load_tokenizer(cfg: EncoderConfig, tokenizer_dir: Path | None = None) -> Tokenizer:
    """The model's tokenizer, truncating and padding as training did."""
    if tokenizer_dir is not None:
        tok = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
    else:
        tok = Tokenizer.from_pretrained(cfg.model_name)
    tok.enable_truncation(max_length=cfg.max_len_query)
    pad_id = tok.token_to_id("[PAD]")
    assert pad_id is not None, "tokenizer has no [PAD] token"
    tok.enable_padding(pad_id=pad_id, pad_token="[PAD]")
    return tok


class OnnxQueryEncoder:
    def __init__(
        self,
        model_path: Path,
        cfg: EncoderConfig,
        threads: int = 1,
        tokenizer_dir: Path | None = None,
    ) -> None:
        self.cfg = cfg
        # A bundled tokenizer means the server never reaches the Hub at startup.
        self.tokenizer = load_tokenizer(cfg, tokenizer_dir)
        options = ort.SessionOptions()
        # One request, one thread: on a 2-vCPU instance, concurrent requests
        # share cores, and intra-op threads would just contend with each other.
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        """(N, D) float32 unit vectors, prefixed exactly as in training."""
        batch = self.tokenizer.encode_batch([self.cfg.query_prefix + t for t in texts])
        (emb,) = self.session.run(
            ["embedding"],
            {
                "input_ids": np.array([e.ids for e in batch], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in batch], dtype=np.int64),
            },
        )
        return np.asarray(emb, dtype=np.float32)
