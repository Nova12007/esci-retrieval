# ESCI Retrieval

[![ci](https://github.com/Nova12007/esci-retrieval/actions/workflows/ci.yaml/badge.svg)](https://github.com/Nova12007/esci-retrieval/actions/workflows/ci.yaml)

Two-stage neural retrieval over the Amazon Shopping Queries (ESCI) dataset.

**Status:** week 1 of 9 — engineering foundations. Results land from week 2.

## Quickstart

```bash
uv sync
make serve
curl "http://localhost:8000/search?q=running+shoes&k=5"
```