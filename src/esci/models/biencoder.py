"""The query and document encoders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Re-exported: training code imports these from here.
from esci.models.config import BGE_QUERY_PREFIX, EncoderConfig

__all__ = ["BGE_QUERY_PREFIX", "BiEncoder", "EncoderConfig"]


class BiEncoder(torch.nn.Module):
    def __init__(self, cfg: EncoderConfig, device: str = "cuda") -> None:
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = AutoModel.from_pretrained(cfg.model_name).to(device)

    @property
    def dim(self) -> int:
        return int(self.model.config.hidden_size)

    def _pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.cfg.pooling == "cls":
            return hidden[:, 0]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        summed = (hidden * mask).sum(dim=1)
        return summed / mask.sum(dim=1).clamp(min=1e-9)

    def forward(self, texts: list[str], is_query: bool) -> torch.Tensor:
        prefix = self.cfg.query_prefix if is_query else self.cfg.doc_prefix
        max_len = self.cfg.max_len_query if is_query else self.cfg.max_len_doc

        batch = self.tokenizer(
            [prefix + t for t in texts],
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(self.device)

        emb = self._pool(self.model(**batch).last_hidden_state, batch["attention_mask"])
        return F.normalize(emb, p=2, dim=-1) if self.cfg.normalize else emb

    @torch.no_grad()
    def encode(self, texts: list[str], is_query: bool, batch_size: int = 256) -> np.ndarray:
        """Batched inference to a float32 (N, D) array, for queries and small sets."""
        self.model.eval()
        out = []
        for start in range(0, len(texts), batch_size):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"):
                emb = self.forward(texts[start : start + batch_size], is_query=is_query)
            out.append(emb.float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.dim), dtype=np.float32)

    @torch.no_grad()
    def encode_to_memmap(
        self, texts: list[str], out_path: Path, is_query: bool, batch_size: int = 256
    ) -> np.ndarray:
        """Encode a large collection straight into a memory-mapped array.

        Row i of the output is always texts[i]. Batches are formed in length
        order, longest first: each batch pads to its longest member, so mixing a
        20-token title with a 256-token listing wastes most of the compute.
        Longest first also means an OOM shows up in the first batch, not the last.
        """
        self.model.eval()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        arr = np.lib.format.open_memmap(
            out_path, mode="w+", dtype=np.float32, shape=(len(texts), self.dim)
        )
        order = np.argsort([-len(t) for t in texts], kind="stable")
        for start in tqdm(range(0, len(texts), batch_size), desc="encode", unit="batch"):
            rows = order[start : start + batch_size]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"):
                emb = self.forward([texts[i] for i in rows], is_query=is_query)
            arr[rows] = emb.float().cpu().numpy()  # scatter back to original positions
        arr.flush()
        return arr
