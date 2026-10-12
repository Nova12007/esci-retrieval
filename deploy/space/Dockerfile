# Hugging Face Space image for the search service (scripts/deploy_space.py).
# Spaces run the container as uid 1000 and route traffic to port 7860.

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

# Dependencies first, so code changes do not reinstall them. On Linux the
# lockfile resolves torch from the CPU-only index.
COPY --chown=user pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY --chown=user src/ src/
RUN uv sync --frozen --no-dev

# Serving artifacts (HNSW index, ONNX encoder, tokenizer, catalog) are baked
# in at build time, so a Space waking from sleep does not re-download 2.3 GB.
# Without ARTIFACT_REPO (a local build) the download is skipped and the
# folder is mounted at run time instead: -v <serving dir>:/home/user/serving
ARG ARTIFACT_REPO
RUN if [ -n "${ARTIFACT_REPO}" ]; then \
      hf download "${ARTIFACT_REPO}" --repo-type model --local-dir /home/user/serving; \
    fi

ENV ESCI_SERVING_DIR=/home/user/serving
EXPOSE 7860
CMD ["uvicorn", "esci.serving.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "7860"]
