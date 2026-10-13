"""The serving tokenizer must produce exactly the ids the model was trained on.

Serving uses the `tokenizers` library to avoid importing transformers; any
difference from the transformers tokenizer used in training would shift every
query vector silently, so it is checked id for id.
"""

from __future__ import annotations

import pytest

from esci.models.config import EncoderConfig
from esci.serving.onnx_encoder import load_tokenizer

transformers = pytest.importorskip("transformers")

QUERIES = [
    "usb c cable 6ft",
    "Arteckin OBD2 J1939",
    "10 succulent plants without plastic containers",
    "autocollant moto",
    "chocolate banana flavored peanut spread, 16 oz (pack of 2)",
    "a" * 400,  # longer than max_len_query: truncation must match
]


@pytest.fixture(scope="module")
def tokenizers() -> tuple[object, object]:
    cfg = EncoderConfig()
    try:
        reference = transformers.AutoTokenizer.from_pretrained(cfg.model_name)
        serving = load_tokenizer(cfg)
    except OSError as e:  # offline machine with no cached tokenizer
        pytest.skip(f"tokenizer not available: {e}")
    return reference, serving


def test_ids_and_masks_match_transformers(tokenizers: tuple[object, object]) -> None:
    reference, serving = tokenizers
    cfg = EncoderConfig()
    texts = [cfg.query_prefix + q for q in QUERIES]
    ref = reference(  # type: ignore[operator]
        texts, padding=True, truncation=True, max_length=cfg.max_len_query
    )
    got = serving.encode_batch(texts)  # type: ignore[attr-defined]
    assert [e.ids for e in got] == ref["input_ids"]
    assert [e.attention_mask for e in got] == ref["attention_mask"]
