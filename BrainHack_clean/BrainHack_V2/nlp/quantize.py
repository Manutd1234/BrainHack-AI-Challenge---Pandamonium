"""Populate src/qwen-quantized with an official Qwen3 AWQ checkpoint."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_ID = os.getenv("QWEN_AWQ_MODEL_ID", "Qwen/Qwen3-8B-AWQ")
OUTPUT_DIR = Path(os.getenv("QWEN_QUANTIZED_DIR", "src/qwen-quantized"))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(OUTPUT_DIR),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.md", "*.png", "*.pdf"],
    )
    print(f"Downloaded {MODEL_ID} to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
