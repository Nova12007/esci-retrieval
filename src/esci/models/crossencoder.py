"""The cross-encoder: query and product read together, one relevance logit out."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass(frozen=True)
class CrossEncoderConfig:
    # Trained on MS MARCO passage ranking: a relevance model before we touch it.
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_len: int = 256  # query + product together; median product is 131 tokens

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> CrossEncoderConfig:
        return cls(**json.loads(path.read_text(encoding="utf-8")))


class CrossEncoder(torch.nn.Module):
    def __init__(self, cfg: CrossEncoderConfig, device: str = "cuda") -> None:
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name, num_labels=1
        ).to(device)

    def forward(self, queries: list[str], docs: list[str]) -> torch.Tensor:
        """(B,) relevance logits. Only the product side is truncated."""
        batch = self.tokenizer(
            queries,
            docs,
            padding=True,
            truncation="only_second",
            max_length=self.cfg.max_len,
            return_tensors="pt",
        ).to(self.device)
        logits: torch.Tensor = self.model(**batch).logits
        return logits.squeeze(-1)

    @torch.no_grad()
    def score(self, queries: list[str], docs: list[str], batch_size: int = 256) -> np.ndarray:
        """Inference over many pairs: float32 logits, in input order."""
        self.model.eval()
        out = []
        for s in range(0, len(queries), batch_size):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = self.forward(queries[s : s + batch_size], docs[s : s + batch_size])
            out.append(logits.float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)
