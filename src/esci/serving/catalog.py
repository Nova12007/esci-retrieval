"""Product metadata for the served results, read from SQLite by id.

Titles for 1.2M products would cost a few hundred MB as a Python dict and be
paid again in every worker. SQLite keeps them on disk, fetched ten at a time
by primary key, which takes well under a millisecond.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


class Catalog:
    def __init__(self, path: Path) -> None:
        # Read-only URI: the service can never modify its own data.
        self._db = sqlite3.connect(
            f"file:{path.as_posix()}?mode=ro", uri=True, check_same_thread=False
        )

    def get_many(self, product_ids: list[str]) -> dict[str, tuple[str, str]]:
        """product_id -> (title, brand) for the ids that exist."""
        if not product_ids:
            return {}
        marks = ",".join("?" * len(product_ids))
        rows = self._db.execute(
            f"SELECT product_id, title, brand FROM products WHERE product_id IN ({marks})",
            product_ids,
        ).fetchall()
        return {pid: (title or "", brand or "") for pid, title, brand in rows}

    def __len__(self) -> int:
        (n,) = self._db.execute("SELECT COUNT(*) FROM products").fetchone()
        return int(n)


def build_catalog(path: Path, rows: list[tuple[str, str, str]]) -> None:
    """Write (product_id, title, brand) rows into a fresh SQLite file."""
    path.unlink(missing_ok=True)
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE products (product_id TEXT PRIMARY KEY, title TEXT, brand TEXT)")
    db.executemany("INSERT INTO products VALUES (?, ?, ?)", rows)
    db.commit()
    db.execute("VACUUM")
    db.close()
