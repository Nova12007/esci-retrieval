# Tabs, not spaces, for the indented lines. Make requires it.
.PHONY: setup lint fmt type test check serve docker loadtest clean

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

serve:  # needs artifacts/serving/biencoder_v3 (scripts/build_serving_artifacts.py)
	ESCI_SERVING_DIR=artifacts/serving/biencoder_v3 uv run uvicorn esci.serving.app:create_app --factory --port 8000

docker:  # self-contained image from the published artifacts, 2 CPUs / 4 GB
	docker compose -f docker/compose.yaml up --build

loadtest:  # k6, open loop, 50 rps against a running service on :8000
	docker run --rm -v "$$(pwd)/loadtest:/lt" -e BASE_URL=http://host.docker.internal:8000 grafana/k6 run /lt/search.js

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache

.PHONY: data bm25 table

data:
	uv run python -m esci.data.loaders

bm25:
	uv run python scripts/run_bm25.py --fold test --corpus $(corpus)

table:
	uv run python -c "from esci.eval.runner import render_table; print(render_table())"