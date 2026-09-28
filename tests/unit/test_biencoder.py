from __future__ import annotations

import pytest
import torch

from esci.models.biencoder import BiEncoder, EncoderConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture(scope="module")
def enc() -> BiEncoder:
    return BiEncoder(EncoderConfig(), device="cuda")


def test_output_is_unit_norm(enc: BiEncoder) -> None:
    v = enc(["wireless earbuds"], is_query=True)
    assert torch.allclose(v.norm(dim=-1), torch.ones(1, device=v.device), atol=1e-4)


def test_encoding_is_deterministic(enc: BiEncoder) -> None:
    """The same text must give the same vector."""
    enc.model.eval()
    with torch.no_grad():
        a = enc(["usb c cable"], is_query=False)
        b = enc(["usb c cable"], is_query=False)
    assert torch.allclose(a, b, atol=1e-5)


def test_query_and_doc_paths_differ(enc: BiEncoder) -> None:
    """Same text, two roles: the query prefix must actually change the vector."""
    with torch.no_grad():
        q = enc(["running shoes"], is_query=True)
        d = enc(["running shoes"], is_query=False)
    assert not torch.allclose(q, d, atol=1e-3)
