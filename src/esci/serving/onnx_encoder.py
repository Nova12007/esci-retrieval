"""Query encoding for the served path: ONNX Runtime on CPU.

Only queries are encoded per request; product vectors are computed offline by
the PyTorch model. The exported graph includes CLS pooling and L2
normalisation, and the prefix comes from the same EncoderConfig the model
was trained with, so the server cannot drift from training on either.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from esci.models.biencoder import EncoderConfig


class OnnxQueryEncoder:
    def __init__(self, model_path: Path, cfg: EncoderConfig, threads: int = 1) -> None:
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
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
        batch = self.tokenizer(
            [self.cfg.query_prefix + t for t in texts],
            padding=True,
            truncation=True,
            max_length=self.cfg.max_len_query,
            return_tensors="np",
        )
        (emb,) = self.session.run(
            ["embedding"],
            {
                "input_ids": batch["input_ids"].astype(np.int64),
                "attention_mask": batch["attention_mask"].astype(np.int64),
            },
        )
        return np.asarray(emb, dtype=np.float32)
