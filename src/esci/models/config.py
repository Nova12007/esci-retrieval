"""Encoder settings shared by training and serving.

Kept free of torch so the serving image does not install the training stack:
the server needs these settings (prefix, pooling, lengths) but not the model.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

# bge: instruction prefix on the QUERY side only, CLS pooling.
# e5:  "query: " / "passage: " on both sides, MEAN pooling.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class EncoderConfig:
    model_name: str = "BAAI/bge-small-en-v1.5"
    query_prefix: str = BGE_QUERY_PREFIX
    doc_prefix: str = ""
    pooling: str = "cls"  # "cls" for bge, "mean" for e5
    max_len_query: int = 48
    max_len_doc: int = 256  # 21.5% of docs truncated; 512 needs 9.4 GB to train at batch 16
    normalize: bool = True

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> EncoderConfig:
        return cls(**json.loads(path.read_text(encoding="utf-8")))
