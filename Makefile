# Tabs, not spaces, for the indented lines. Make requires it.
.PHONY: setup lint fmt type test check serve clean

setup:
	uv sync

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

type:
	uv run mypy src

test:
	uv run pytest

check: lint type test

serve:
	uv run uvicorn esci.serving.app:app --reload --port 8000

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache

.PHONY: data bm25 table

data:
	uv run python -m esci.data.loaders

bm25:
	uv run python scripts/run_bm25.py --fold test --corpus $(corpus)

table:
	uv run python -c "from esci.eval.runner import render_table; print(render_table())"