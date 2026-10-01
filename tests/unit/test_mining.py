from __future__ import annotations

from esci.data.pairs import QueryExample
from esci.training.mining import apply_negatives, filter_mined

RANKED = [f"p{i}" for i in range(100)]


def test_skips_top_ranks_even_when_unjudged() -> None:
    out = filter_mined(RANKED, frozenset(), query_id=1, skip_top=3, keep=200)
    assert out[0] == "p3" and "p0" not in out


def test_drops_judged_relevant_products() -> None:
    out = filter_mined(RANKED, frozenset({"p5", "p6"}), query_id=1, keep=200)
    assert "p5" not in out and "p6" not in out


def test_samples_keep_in_rank_order_and_reproducibly() -> None:
    a = filter_mined(RANKED, frozenset(), query_id=7, keep=12)
    b = filter_mined(RANKED, frozenset(), query_id=7, keep=12)
    assert a == b and len(a) == 12
    assert a == sorted(a, key=lambda p: int(p[1:]))  # rank order kept
    assert int(a[-1][1:]) > 15  # reaches past the hardest few, guard 3


def test_different_queries_sample_differently() -> None:
    a = filter_mined(RANKED, frozenset(), query_id=1, keep=12)
    b = filter_mined(RANKED, frozenset(), query_id=2, keep=12)
    assert a != b


EX = QueryExample(query_id=1, query="q", positive_ids=("e1",), negative_ids=("s1", "c1"))


def test_esci_source_leaves_examples_unchanged() -> None:
    assert apply_negatives([EX], {1: ["m1"]}, "esci") == [EX]


def test_mined_source_replaces_negatives() -> None:
    assert apply_negatives([EX], {1: ["m1", "m2"]}, "mined")[0].negative_ids == ("m1", "m2")
    assert apply_negatives([EX], {}, "mined")[0].negative_ids == ()


def test_mixed_source_interleaves_without_duplicates() -> None:
    out = apply_negatives([EX], {1: ["m1", "s1", "m2"]}, "mixed")[0].negative_ids
    assert out == ("s1", "m1", "c1", "m2")
