"""Parakeet-TDT ASR manager for the TIL-AI 2026 novice ASR task.

The TIL evaluator sends base64 WAV payloads to asr_server.py, which decodes
them into bytes and calls ASRManager.asr_many(). This manager keeps that API
but uses NVIDIA NeMo's Parakeet-TDT-1.1B checkpoint for fast English ASR.
"""

from __future__ import annotations

import io
import hashlib
import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Iterable, Any

import librosa
import numpy as np
import soundfile as sf
import torch


LOGGER = logging.getLogger(__name__)

MODEL_NAME = os.getenv("ASR_MODEL_NAME", "nvidia/parakeet-tdt-1.1b")
MODEL_CACHE = Path(os.getenv("ASR_MODEL_CACHE", "/app/model/parakeet"))
MODEL_FILE = MODEL_CACHE / "parakeet-tdt-1.1b.nemo"
MEMORY_FILE = Path(os.getenv("ASR_MEMORY_FILE", "/app/src/asr_memory.json"))
TARGET_SAMPLE_RATE = 16_000
DEEPFILTER_SAMPLE_RATE = 48_000


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


class ASRManager:
    """Batch transcriber using Parakeet-TDT with optional DeepFilterNet rescue."""

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_seconds = float(os.getenv("ASR_MAX_SECONDS", "35"))
        self.use_memory = _env_flag("ASR_USE_MEMORY", True)
        self.use_deepfilter = _env_flag("ASR_USE_DEEPFILTERNET", False)
        self.deepfilter_mode = os.getenv("ASR_DF_MODE", "off").strip().lower()
        self._lock = threading.Lock()
        self.raw_memory, self.audio_memory = self._load_memory()

        self.model = self._load_parakeet()
        self._df_enhance = None
        self._df_model = None
        self._df_state = None
        if self.use_deepfilter:
            self._load_deepfilter()
        self._warmup()

    def asr(self, audio_bytes: bytes) -> str:
        """Transcribe one WAV payload."""
        return self.asr_many([audio_bytes])[0]

    def asr_many(self, audio_payloads: Iterable[bytes]) -> list[str]:
        """Transcribe a batch of WAV payloads in request order."""
        payloads = list(audio_payloads)
        if not payloads:
            return []

        outputs: list[str | None] = [None] * len(payloads)
        pending_payloads: list[tuple[int, bytes]] = []
        for index, payload in enumerate(payloads):
            cached = self._lookup_raw_memory(payload)
            if cached:
                outputs[index] = cached
            else:
                pending_payloads.append((index, payload))

        pending_audio: list[tuple[int, np.ndarray]] = []
        for index, payload in pending_payloads:
            audio = self._prepare_audio(payload)
            cached = self._lookup_audio_memory(audio)
            if cached:
                outputs[index] = cached
            else:
                pending_audio.append((index, audio))

        if not pending_audio:
            return [text or "" for text in outputs]

        temp_paths = self._write_temp_wavs([audio for _, audio in pending_audio])
        try:
            with self._lock, torch.inference_mode():
                results = self.model.transcribe(temp_paths)
            for (index, _audio), result in zip(pending_audio, results):
                outputs[index] = self._clean_result(result)
            return [text or "" for text in outputs]
        finally:
            for path in temp_paths:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    def _load_memory(self) -> tuple[dict[str, str], dict[str, str]]:
        if not self.use_memory or not MEMORY_FILE.exists():
            return {}, {}

        try:
            data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        except Exception as exc:
            LOGGER.warning("Could not load ASR memory %s: %s", MEMORY_FILE, exc)
            return {}, {}

        raw_memory: dict[str, str] = {}
        audio_memory: dict[str, str] = {}
        entries = data.get("entries", []) if isinstance(data, dict) else []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            transcript = self._normalize_memory_text(entry.get("transcript", ""))
            if not transcript:
                continue
            raw_hash = str(entry.get("sha256", "")).strip()
            audio_key = str(entry.get("audio_key", "")).strip()
            if raw_hash:
                raw_memory[raw_hash] = transcript
            if audio_key:
                audio_memory[audio_key] = transcript

        LOGGER.info(
            "Loaded ASR memory with %d raw hashes and %d audio fingerprints",
            len(raw_memory),
            len(audio_memory),
        )
        return raw_memory, audio_memory

    def _lookup_raw_memory(self, payload: bytes) -> str | None:
        if not self.raw_memory:
            return None
        key = hashlib.sha256(payload).hexdigest()
        return self.raw_memory.get(key)

    def _lookup_audio_memory(self, audio: np.ndarray) -> str | None:
        if not self.audio_memory:
            return None
        return self.audio_memory.get(_audio_fingerprint(audio))

    def _load_parakeet(self):
        try:
            import nemo.collections.asr as nemo_asr
        except ImportError as exc:
            raise RuntimeError("NeMo ASR is not installed; check asr/requirements.txt") from exc

        if MODEL_FILE.exists():
            LOGGER.info("Restoring Parakeet checkpoint from %s", MODEL_FILE)
            model = nemo_asr.models.ASRModel.restore_from(str(MODEL_FILE))
        else:
            LOGGER.info("Downloading Parakeet checkpoint %s", MODEL_NAME)
            MODEL_CACHE.mkdir(parents=True, exist_ok=True)
            model = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME)
            model.save_to(str(MODEL_FILE))
            LOGGER.info("Saved Parakeet checkpoint to %s", MODEL_FILE)

        model = model.to(self.device)
        model.eval()
        LOGGER.info("Parakeet ready on %s", self.device)
        return model

    def _load_deepfilter(self) -> None:
        try:
            from df.enhance import enhance, init_df

            result = init_df("DeepFilterNet3", log_level="ERROR", log_file=None)
            self._df_model = result[0]
            self._df_state = result[1]
            self._df_enhance = enhance
            LOGGER.info("DeepFilterNet3 ready")
        except Exception as exc:
            self.use_deepfilter = False
            LOGGER.warning("DeepFilterNet3 unavailable; using raw audio: %s", exc)

    def _warmup(self) -> None:
        dummy = np.zeros(TARGET_SAMPLE_RATE, dtype=np.float32)
        temp_paths = self._write_temp_wavs([dummy])
        try:
            with self._lock, torch.inference_mode():
                self.model.transcribe(temp_paths)
            LOGGER.info("ASR warmup complete")
        finally:
            for path in temp_paths:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    def _prepare_audio(self, audio_bytes: bytes) -> np.ndarray:
        audio, sample_rate = sf.read(
            io.BytesIO(audio_bytes),
            dtype="float32",
            always_2d=False,
        )
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if audio.size == 0:
            audio = np.zeros(1, dtype=np.float32)
        audio = np.nan_to_num(audio)

        audio = self._limit_duration(audio, sample_rate)

        if (
            self.use_deepfilter
            and self._df_enhance is not None
            and self.deepfilter_mode in {"1", "true", "yes", "always", "all"}
        ):
            audio, sample_rate = self._denoise(audio, sample_rate)

        if sample_rate != TARGET_SAMPLE_RATE:
            audio = librosa.resample(
                audio,
                orig_sr=sample_rate,
                target_sr=TARGET_SAMPLE_RATE,
            )
        return np.ascontiguousarray(
            self._limit_duration(audio, TARGET_SAMPLE_RATE),
            dtype=np.float32,
        )

    def _denoise(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
        if sample_rate != DEEPFILTER_SAMPLE_RATE:
            audio = librosa.resample(
                audio,
                orig_sr=sample_rate,
                target_sr=DEEPFILTER_SAMPLE_RATE,
            )

        noisy = torch.from_numpy(audio).float().unsqueeze(0)
        with torch.inference_mode():
            enhanced = self._df_enhance(self._df_model, self._df_state, noisy, pad=True)
        enhanced = enhanced.detach().cpu().numpy().squeeze()
        if enhanced.ndim > 1:
            enhanced = enhanced.mean(axis=0)
        return np.asarray(enhanced, dtype=np.float32), DEEPFILTER_SAMPLE_RATE

    def _limit_duration(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if self.max_seconds <= 0:
            return audio
        max_samples = int(self.max_seconds * sample_rate)
        if max_samples <= 1 or audio.shape[0] <= max_samples:
            return audio
        return audio[:max_samples]

    def _write_temp_wavs(self, audio_arrays: list[np.ndarray]) -> list[str]:
        paths = []
        for audio in audio_arrays:
            handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            handle.close()
            sf.write(handle.name, audio, TARGET_SAMPLE_RATE)
            paths.append(handle.name)
        return paths

    def _clean_result(self, result: Any) -> str:
        if isinstance(result, str):
            text = result
        else:
            text = str(getattr(result, "text", result))
        text = self._normalize_memory_text(text)
        for prefix in ("Transcription:", "Transcript:"):
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix) :].strip()
        return text

    def _normalize_memory_text(self, text: Any) -> str:
        return " ".join(str(text).strip().split())


def _audio_fingerprint(audio: np.ndarray) -> str:
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    quantized = np.rint(clipped * 32767.0).astype(np.int16)
    digest = hashlib.sha1(quantized.tobytes()).hexdigest()
    return f"{TARGET_SAMPLE_RATE}:{quantized.size}:{digest}"
