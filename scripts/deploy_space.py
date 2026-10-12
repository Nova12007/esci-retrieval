"""Publish the serving artifacts and the service to Hugging Face.

    uv run hf auth login                     # once, with a write token
    uv run python scripts/deploy_space.py --user <hf-username>

1. artifacts/serving/<run>/ -> model repo <user>/esci-retrieval-artifacts
2. Dockerfile, Space card, pyproject, lockfile and src/ -> Docker Space
   <user>/esci-search, which downloads (1) at build time.

Re-running is safe: repos are created if missing and files are replaced.
Use --skip-artifacts when only the code changed.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

SPACE_CARD = """---
title: ESCI Product Search
emoji: 🔎
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---

Dense retrieval over 1.2M Amazon Shopping Queries (ESCI) products: a fine-tuned
bge-small bi-encoder (ONNX, CPU) and an HNSW index. Search at `/`, API at
`/search?q=...&k=10`, model and index details at `/info`.

Code, evaluation and design decisions: {github}
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--user", required=True, help="Hugging Face username")
    p.add_argument("--run", default="biencoder_v3")
    p.add_argument("--space", default="esci-search")
    p.add_argument("--artifact-repo", default="esci-retrieval-artifacts")
    p.add_argument("--github", default="https://github.com/Nova12007/esci-retrieval")
    p.add_argument("--skip-artifacts", action="store_true")
    args = p.parse_args()

    api = HfApi()
    artifacts = f"{args.user}/{args.artifact_repo}"
    space = f"{args.user}/{args.space}"

    if not args.skip_artifacts:
        folder = Path("artifacts/serving") / args.run
        assert (folder / "manifest.json").exists(), f"build {folder} first"
        api.create_repo(artifacts, repo_type="model", exist_ok=True)
        api.upload_folder(
            repo_id=artifacts,
            repo_type="model",
            folder_path=str(folder),
            commit_message=f"Serving artifacts for {args.run}",
        )
        print(f"artifacts -> https://huggingface.co/{artifacts}")

    api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        dockerfile = Path("deploy/space/Dockerfile").read_text(encoding="utf-8")
        # Spaces pass no build args, so the artifact repo is written into the file.
        dockerfile = dockerfile.replace("ARG ARTIFACT_REPO", f"ARG ARTIFACT_REPO={artifacts}")
        # LF endings explicitly: write_text would emit CRLF on Windows, and a
        # backslash continuation followed by CRLF breaks a RUN step on Linux.
        dockerfile = dockerfile.replace("
", "
")
        (stage / "Dockerfile").write_text(dockerfile, encoding="utf-8", newline="
")
        card = SPACE_CARD.format(github=args.github)
        (stage / "README.md").write_text(card, encoding="utf-8", newline="
")
        # Not README.MD: on Windows it is the same file as the Space card README.md.
        for name in ["pyproject.toml", "uv.lock"]:
            (stage / name).write_bytes(Path(name).read_bytes())
        api.upload_folder(
            repo_id=space,
            repo_type="space",
            folder_path=str(stage),
            commit_message="Deploy service",
        )
    api.upload_folder(
        repo_id=space,
        repo_type="space",
        folder_path="src",
        path_in_repo="src",
        ignore_patterns=["**/__pycache__/**", "*.pyc"],
        commit_message="Deploy service code",
    )
    print(f"space -> https://huggingface.co/spaces/{space}")
    print(f"app   -> https://{args.user}-{args.space}.hf.space")


if __name__ == "__main__":
    main()
