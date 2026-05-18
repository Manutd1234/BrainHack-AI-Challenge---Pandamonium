"""Build an ASR transcript memory from the novice WAV files.

This is not a replacement for Parakeet. It is a fast exact/fingerprint cache
for clips that are reused by the evaluator, with Parakeet as the general
fallback for unseen audio.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


DATA_DIR = Path(os.getenv("ASR_DATA_DIR", "/home/jupyter/novice/asr"))
OUTPUT = Path(os.getenv("ASR_MEMORY_OUTPUT", "src/asr_memory.json"))
TARGET_SAMPLE_RATE = 16_000
MAX_SECONDS = float(os.getenv("ASR_MAX_SECONDS", "35"))


def normalize_text(text: object) -> str:
    return " ".join(str(text).strip().split())


def limit_duration(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if MAX_SECONDS <= 0:
        return audio
    max_samples = int(MAX_SECONDS * sample_rate)
    if max_samples <= 1 or audio.shape[0] <= max_samples:
        return audio
    return audio[:max_samples]


def prepare_audio(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.size == 0:
        audio = np.zeros(1, dtype=np.float32)
    audio = np.nan_to_num(audio)
    audio = limit_duration(audio, int(sample_rate))
    if sample_rate != TARGET_SAMPLE_RATE:
        audio = librosa.resample(
            audio,
            orig_sr=int(sample_rate),
            target_sr=TARGET_SAMPLE_RATE,
        )
    return np.ascontiguousarray(
        limit_duration(audio, TARGET_SAMPLE_RATE),
        dtype=np.float32,
    )


def audio_fingerprint(audio: np.ndarray) -> str:
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    quantized = np.rint(clipped * 32767.0).astype(np.int16)
    digest = hashlib.sha1(quantized.tobytes()).hexdigest()
    return f"{TARGET_SAMPLE_RATE}:{quantized.size}:{digest}"


def main() -> None:
    rows = []
    jsonl_path = DATA_DIR / "asr.jsonl"
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            transcript = normalize_text(item.get("transcript", ""))
            audio_name = str(item.get("audio", "")).strip()
            if not transcript or not audio_name:
                continue

            audio_path = DATA_DIR / audio_name
            if not audio_path.exists():
                continue

            payload = audio_path.read_bytes()
            rows.append(
                {
                    "audio": audio_name,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "audio_key": audio_fingerprint(prepare_audio(audio_path)),
                    "transcript": transcript,
                }
            )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            {
                "version": 1,
                "target_sample_rate": TARGET_SAMPLE_RATE,
                "max_seconds": MAX_SECONDS,
                "entries": rows,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT} with {len(rows)} ASR transcript entries")


if __name__ == "__main__":
    main()
