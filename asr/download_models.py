"""
Run this ONCE on your Workbench before building the Docker image:
    python download_models.py
Downloads faster-whisper large-v3 weights into models/.
"""
from faster_whisper import WhisperModel
import os

os.makedirs("models/whisper-large-v3", exist_ok=True)

print("Downloading faster-whisper large-v3 ...")
# This triggers the download and caches to the local path
model = WhisperModel(
    "large-v3",
    device="cpu",
    compute_type="int8",
    download_root="models/whisper-large-v3",
)
print("Done. Model saved to models/whisper-large-v3/")
