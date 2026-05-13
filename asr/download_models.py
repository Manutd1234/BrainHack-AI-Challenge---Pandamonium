"""Download faster-whisper large-v3 into a Docker-copyable local directory."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_REPO = "Systran/faster-whisper-large-v3"
MODEL_DIR = Path("models/whisper-large-v3")

MODEL_DIR.mkdir(parents=True, exist_ok=True)

print(f"Downloading {MODEL_REPO} into {MODEL_DIR} ...")
snapshot_download(
    repo_id=MODEL_REPO,
    local_dir=str(MODEL_DIR),
    local_dir_use_symlinks=False,
)
print(f"Done. Model saved to {MODEL_DIR}/")
