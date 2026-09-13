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