"""Print the real shape of the ESCI files."""

from __future__ import annotations

from pathlib import Path

import polars as pl

RAW = Path("data/raw")


def main() -> None:
    for path in sorted(RAW.glob("*.parquet")):
        print("=" * 70)
        print(path.name)
        print("=" * 70)

        # scan_parquet is lazy -- reads metadata without loading the file.
        lazy = pl.scan_parquet(path)
        schema = lazy.collect_schema()

        print(f"columns ({len(schema)}):")
        for name, dtype in schema.items():
            print(f"    {name:<28} {dtype}")

        n_rows = lazy.select(pl.len()).collect().item()
        print(f"\nrows: {n_rows:,}")
        print("\nfirst 3 rows:")
        print(lazy.head(3).collect())
        print()


if __name__ == "__main__":
    main()