from __future__ import annotations

import polars as pl

from esci.eval.slices import query_slices, unsaturated_queries


def test_keeps_only_queries_below_the_exact_threshold() -> None:
    judgements = pl.DataFrame(
        {
            "query_id": [1] * 10 + [2] * 9 + [2],
            "esci_label": ["E"] * 10 + ["E"] * 9 + ["S"],
        }
    )
    # query 1: 10 Exact, saturated. query 2: 9 Exact + 1 S, unsaturated.
    assert unsaturated_queries(judgements).to_list() == [2]


def test_queries_with_no_exact_are_unsaturated() -> None:
    judgements = pl.DataFrame({"query_id": [5, 5], "esci_label": ["S", "I"]})
    assert unsaturated_queries(judgements).to_list() == [5]


def _judged(query: str, brand: str | None = "Anker") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "query_id": [1, 1],
            "query": [query, query],
            "esci_label": ["E", "S"],
            "product_brand": [brand, "OtherBrand"],
        }
    )


def test_length_and_number_slices() -> None:
    row = query_slices(_judged("usb c cable 6ft anker")).row(0, named=True)
    assert row["length"] == "5-7 words" and row["has_number"] == "has a number"


def test_brand_must_be_a_whole_word_of_an_exact_products_brand() -> None:
    assert query_slices(_judged("anker charger"))["brand"][0] == "names a brand"
    assert query_slices(_judged("ankerite rock"))["brand"][0] == "names no brand"
    # OtherBrand belongs to a Substitute, not an Exact product
    assert query_slices(_judged("otherbrand charger"))["brand"][0] == "names no brand"


def test_missing_and_generic_brands_never_match() -> None:
    assert query_slices(_judged("generic cable", brand="Generic"))["brand"][0] == "names no brand"
    assert query_slices(_judged("cable", brand=None))["brand"][0] == "names no brand"
