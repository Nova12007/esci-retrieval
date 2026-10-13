# Search service image: ONNX query encoder, HNSW index, SQLite catalog.
#
#   Self-contained (downloads the serving artifacts, no dataset needed):
#     docker build -f docker/Dockerfile \
#       --build-arg ARTIFACT_REPO=N0va12007/esci-retrieval-artifacts -t esci-search .
#
#   From a local folder built by scripts/build_serving_artifacts.py:
#     docker build -f docker/Dockerfile -t esci-search .
#     docker run -p 8000:8000 -v <serving dir>:/home/user/serving:ro esci-search

FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv
RUN useradd --create-home --uid 1000 user
USER user
WORKDIR /home/user/app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/home/user/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1

# Dependencies first, so code changes do not reinstall them. Only the core
# (serving) dependencies: no torch, polars or training stack (pyproject.toml).
COPY --chown=user pyproject.toml uv.lock ./
RUN uv sync --frozen --no-default-groups --no-install-project

COPY --chown=user src/ src/
RUN uv sync --frozen --no-default-groups

# With ARTIFACT_REPO set, the 2.3 GB of serving artifacts are baked in at
# build time; without it, mount the folder at run time instead.
ARG ARTIFACT_REPO
# The cache is removed in the same layer: the hub client keeps its own copy of
# every downloaded chunk, which would put the 2.3 GB in the image twice.
RUN if [ -n "${ARTIFACT_REPO}" ]; then \
      hf download "${ARTIFACT_REPO}" --repo-type model --local-dir /home/user/serving \
      && rm -rf /home/user/.cache/huggingface /home/user/serving/.cache; \
    fi

ENV ESCI_SERVING_DIR=/home/user/serving
EXPOSE 8000

# start-period covers loading the 2 GB index before the first check counts.
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "esci.serving.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
