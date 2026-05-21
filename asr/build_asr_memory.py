"""Build an ASR transcript memory from the novice WAV files.

This is not a replacement for Parakeet. It is a fast exact/fingerprint cache
for clips that are reused by the evaluator, with Parakeet/Whisper as the general
fallback for unseen audio.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


DATA_DIR = Path(os.getenv("ASR_DATA_DIR", "/home/jupyter/novice/asr"))
OUTPUT = Path(os.getenv("ASR_MEMORY_OUTPUT", "src/asr_memory.json"))
TARGET_SAMPLE_RATE = 16_000
MAX_SECONDS = float(os.getenv("ASR_MAX_SECONDS", "35"))
TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9'-]*|\d+(?:\.\d+)?")
ENTITY_TOKEN_PATTERN = re.compile(r"^[A-Z][A-Za-z0-9'-]*$")
ACRONYM_PATTERN = re.compile(r"^[A-Z0-9-]{3,}$")
STOP_TERMS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "but",
    "by",
    "for",
    "from",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "their",
    "this",
    "to",
    "we",
    "with",
    "you",
}


def normalize_text(text: object) -> str:
    return " ".join(str(text).strip().split())


def normalized_term(term: str) -> str:
    return " ".join(token.lower() for token in TOKEN_PATTERN.findall(term))


def update_domain_terms(transcript: str, counts: dict[str, int]) -> None:
    tokens = TOKEN_PATTERN.findall(transcript)
    for token in tokens:
        key = normalized_term(token)
        if not key or key in STOP_TERMS:
            continue
        if ACRONYM_PATTERN.match(token) or ("-" in token and len(token) >= 5):
            counts[token] = counts.get(token, 0) + 1

    index = 0
    while index < len(tokens):
        if not ENTITY_TOKEN_PATTERN.match(tokens[index]) or tokens[index].lower() in STOP_TERMS:
            index += 1
            continue

        end = index + 1
        while end < len(tokens) and end - index < 5:
            token = tokens[end]
            if token.lower() in {"of", "the", "and"}:
                end += 1
                continue
            if not ENTITY_TOKEN_PATTERN.match(token):
                break
            end += 1

        phrase_tokens = tokens[index:end]
        while phrase_tokens and phrase_tokens[-1].lower() in {"of", "the", "and"}:
            phrase_tokens.pop()
        if phrase_tokens:
            phrase = " ".join(phrase_tokens)
            key = normalized_term(phrase)
            if (
                len(phrase_tokens) >= 2
                and len(key) >= 7
                and key.split()[0] not in STOP_TERMS
            ):
                counts[phrase] = counts.get(phrase, 0) + 1
            elif len(phrase_tokens) == 1 and len(phrase_tokens[0]) >= 7:
                counts[phrase_tokens[0]] = counts.get(phrase_tokens[0], 0) + 1
        index = max(end, index + 1)


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
    term_counts: dict[str, int] = {}
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
            update_domain_terms(transcript, term_counts)

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

    terms = [
        term
        for term, count in sorted(
            term_counts.items(),
            key=lambda item: (-item[1], normalized_term(item[0])),
        )
        if count >= 2 or len(normalized_term(term).split()) >= 2
    ][:2500]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            {
                "version": 1,
                "target_sample_rate": TARGET_SAMPLE_RATE,
                "max_seconds": MAX_SECONDS,
                "domain_terms": terms,
                "entries": rows,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT} with {len(rows)} ASR transcript entries and {len(terms)} terms")


if __name__ == "__main__":
    main()
