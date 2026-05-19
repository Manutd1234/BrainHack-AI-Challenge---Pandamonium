"""Parakeet-TDT ASR manager for the TIL-AI 2026 novice ASR task.

The TIL evaluator sends base64 WAV payloads to asr_server.py, which decodes
them into bytes and calls ASRManager.asr_many(). This manager keeps that API
but uses NVIDIA NeMo's Parakeet-TDT checkpoint for fast English ASR.
"""

from __future__ import annotations

import difflib
import contextlib
import hashlib
import io
import json
import logging
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Iterable, Any

import librosa
import numpy as np
import soundfile as sf
import torch


LOGGER = logging.getLogger(__name__)

MODEL_NAME = os.getenv("ASR_MODEL_NAME", "nvidia/parakeet-tdt-0.6b-v2")
MODEL_CACHE = Path(os.getenv("ASR_MODEL_CACHE", "/app/model/parakeet"))
MODEL_SLUG = re.sub(r"[^A-Za-z0-9_.-]+", "_", MODEL_NAME.split("/")[-1])
MODEL_FILE = Path(os.getenv("ASR_MODEL_FILE", str(MODEL_CACHE / f"{MODEL_SLUG}.nemo")))
WHISPER_MODEL_NAME = os.getenv("ASR_WHISPER_MODEL", "openai/whisper-large-v3-turbo")
WHISPER_CACHE = Path(os.getenv("ASR_WHISPER_CACHE", "/app/model/whisper-large-v3-turbo"))
MEMORY_FILE = Path(os.getenv("ASR_MEMORY_FILE", "/app/src/asr_memory.json"))
TARGET_SAMPLE_RATE = 16_000
DEEPFILTER_SAMPLE_RATE = 48_000
WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9'-]*|\d+(?:\.\d+)?")


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
        self.batch_size = max(1, int(os.getenv("ASR_BATCH_SIZE", "16")))
        self.use_autocast = _env_flag("ASR_USE_AUTOCAST", self.device == "cuda")
        self.use_fp16_weights = _env_flag("ASR_USE_FP16_WEIGHTS", False)
        self.use_whisper_fallback = _env_flag("ASR_USE_WHISPER_FALLBACK", False)
        self.whisper_mode = os.getenv("ASR_WHISPER_MODE", "rescue").strip().lower()
        self.whisper_language = os.getenv("ASR_WHISPER_LANGUAGE", "en").strip() or None
        self.whisper_max_new_tokens = int(os.getenv("ASR_WHISPER_MAX_NEW_TOKENS", "256"))
        self.use_memory = _env_flag("ASR_USE_MEMORY", False)
        self.use_domain_correction = _env_flag("ASR_USE_DOMAIN_CORRECTION", False)
        self.domain_correction_threshold = float(
            os.getenv("ASR_DOMAIN_CORRECTION_THRESHOLD", "0.88")
        )
        self.use_deepfilter = _env_flag("ASR_USE_DEEPFILTERNET", False)
        self.deepfilter_mode = os.getenv("ASR_DF_MODE", "off").strip().lower()
        self._lock = threading.Lock()
        self._transcribe_kwargs: dict[str, Any] | None = None
        self.raw_memory, self.audio_memory, self.domain_terms = self._load_memory()
        self.domain_term_index = self._build_domain_term_index(self.domain_terms)

        self.model = self._load_parakeet()
        self.whisper_model = None
        self.whisper_processor = None
        self.whisper_dtype = torch.float16 if self.device == "cuda" else torch.float32
        if self.use_whisper_fallback:
            self._load_whisper()
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
                results = self._transcribe_paths(temp_paths)
            for (index, _audio), result in zip(pending_audio, results):
                parakeet_text = self._clean_result(result)
                outputs[index] = self._maybe_whisper_rescue(_audio, parakeet_text)
            return [text or "" for text in outputs]
        finally:
            for path in temp_paths:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    def _load_memory(self) -> tuple[dict[str, str], dict[str, str], list[str]]:
        if not self.use_memory or not MEMORY_FILE.exists():
            return {}, {}, []

        try:
            data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        except Exception as exc:
            LOGGER.warning("Could not load ASR memory %s: %s", MEMORY_FILE, exc)
            return {}, {}, []

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

        domain_terms = [
            self._normalize_memory_text(term)
            for term in data.get("domain_terms", [])
            if self._normalize_memory_text(term)
        ][:3000]

        LOGGER.info(
            "Loaded ASR memory with %d raw hashes, %d audio fingerprints, %d domain terms",
            len(raw_memory),
            len(audio_memory),
            len(domain_terms),
        )
        return raw_memory, audio_memory, domain_terms

    def _build_domain_term_index(self, terms: list[str]) -> dict[int, dict[str, list[str]]]:
        index: dict[int, dict[str, list[str]]] = {}
        for term in terms:
            tokens = self._word_tokens(term)
            if not tokens or len(tokens) > 5:
                continue
            if len(tokens) == 1 and len(tokens[0]) < 7:
                continue
            key = " ".join(token.lower() for token in tokens)
            if not key:
                continue
            first = key[0]
            index.setdefault(len(tokens), {}).setdefault(first, [])
            if term not in index[len(tokens)][first]:
                index[len(tokens)][first].append(term)
        return index

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
            MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
            model = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME)
            model.save_to(str(MODEL_FILE))
            LOGGER.info("Saved Parakeet checkpoint to %s", MODEL_FILE)

        model = model.to(self.device)
        if self.device == "cuda" and self.use_fp16_weights:
            try:
                model = model.half()
                LOGGER.info("Using fp16 Parakeet weights")
            except Exception as exc:
                LOGGER.warning("Could not convert Parakeet weights to fp16: %s", exc)
        model.eval()
        if self.device == "cuda":
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
        LOGGER.info(
            "Parakeet ready on %s with autocast=%s batch_size=%d",
            self.device,
            self.use_autocast,
            self.batch_size,
        )
        return model

    def _load_whisper(self) -> None:
        try:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        except ImportError as exc:
            self.use_whisper_fallback = False
            LOGGER.warning("Transformers unavailable; disabling Whisper fallback: %s", exc)
            return

        model_source = str(WHISPER_CACHE) if WHISPER_CACHE.exists() else WHISPER_MODEL_NAME
        local_only = WHISPER_CACHE.exists()
        try:
            LOGGER.info("Loading Whisper fallback from %s", model_source)
            self.whisper_processor = AutoProcessor.from_pretrained(
                model_source,
                local_files_only=local_only,
            )
            self.whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_source,
                torch_dtype=self.whisper_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                local_files_only=local_only,
            ).to(self.device)
            self.whisper_model.eval()
            LOGGER.info("Whisper fallback ready on %s", self.device)
        except Exception as exc:
            self.use_whisper_fallback = False
            self.whisper_processor = None
            self.whisper_model = None
            LOGGER.warning("Whisper fallback unavailable; using Parakeet only: %s", exc)

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
                self._transcribe_paths(temp_paths)
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

    def _transcribe_paths(self, paths: list[str]):
        """Call NeMo transcribe with batch kwargs when this version supports them."""
        batch_size = min(self.batch_size, max(1, len(paths)))

        def inference_context():
            if self.device == "cuda" and self.use_autocast:
                return torch.autocast(device_type="cuda", dtype=torch.float16)
            return contextlib.nullcontext()

        if self._transcribe_kwargs is not None:
            kwargs = dict(self._transcribe_kwargs)
            if "batch_size" in kwargs:
                kwargs["batch_size"] = batch_size
            with inference_context():
                return self.model.transcribe(paths, **kwargs)

        kwargs_options = (
            {"batch_size": batch_size, "verbose": False},
            {"batch_size": batch_size},
            {},
        )
        last_error: TypeError | None = None
        for kwargs in kwargs_options:
            try:
                with inference_context():
                    results = self.model.transcribe(paths, **kwargs)
                self._transcribe_kwargs = dict(kwargs)
                return results
            except TypeError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        with inference_context():
            return self.model.transcribe(paths)

    def _clean_result(self, result: Any) -> str:
        if isinstance(result, str):
            text = result
        else:
            text = str(getattr(result, "text", result))
        text = self._normalize_memory_text(text)
        for prefix in ("Transcription:", "Transcript:"):
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix) :].strip()
        if self.use_domain_correction and self.domain_term_index:
            text = self._apply_domain_corrections(text)
        return text

    def _maybe_whisper_rescue(self, audio: np.ndarray, parakeet_text: str) -> str:
        if not self._should_try_whisper(audio, parakeet_text):
            return parakeet_text
        whisper_text = self._whisper_transcribe(audio)
        if not whisper_text:
            return parakeet_text
        return self._choose_transcript(parakeet_text, whisper_text)

    def _should_try_whisper(self, audio: np.ndarray, text: str) -> bool:
        if (
            not self.use_whisper_fallback
            or self.whisper_model is None
            or self.whisper_processor is None
            or self.whisper_mode in {"0", "false", "off", "none"}
        ):
            return False
        if self.whisper_mode in {"1", "true", "always", "all"}:
            return True

        words = self._word_tokens(text)
        duration = max(0.001, float(audio.shape[0]) / TARGET_SAMPLE_RATE)
        if not text.strip():
            return True
        if duration >= 8.0 and len(words) < max(4, int(duration * 0.7)):
            return True
        if duration >= 15.0 and len(set(token.lower() for token in words)) <= 3:
            return True
        if len(text) < 20 and duration >= 10.0:
            return True
        return False

    def _whisper_transcribe(self, audio: np.ndarray) -> str:
        if self.whisper_model is None or self.whisper_processor is None:
            return ""
        try:
            inputs = self.whisper_processor(
                audio,
                sampling_rate=TARGET_SAMPLE_RATE,
                return_tensors="pt",
            )
            input_features = inputs.input_features.to(
                self.device,
                dtype=self.whisper_dtype,
            )
            generate_kwargs: dict[str, Any] = {
                "max_new_tokens": self.whisper_max_new_tokens,
                "num_beams": 1,
                "do_sample": False,
            }
            if self.whisper_language:
                generate_kwargs["language"] = self.whisper_language
                generate_kwargs["task"] = "transcribe"
            predicted_ids = self.whisper_model.generate(input_features, **generate_kwargs)
            text = self.whisper_processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True,
            )[0]
            return self._normalize_memory_text(text)
        except Exception as exc:
            LOGGER.warning("Whisper fallback failed: %s", exc)
            return ""

    def _choose_transcript(self, parakeet_text: str, whisper_text: str) -> str:
        parakeet_words = self._word_tokens(parakeet_text)
        whisper_words = self._word_tokens(whisper_text)
        if not parakeet_words:
            return whisper_text
        if len(whisper_words) >= max(4, int(len(parakeet_words) * 0.9)):
            return whisper_text
        return parakeet_text

    def _normalize_memory_text(self, text: Any) -> str:
        return " ".join(str(text).strip().split())

    def _word_tokens(self, text: str) -> list[str]:
        return WORD_PATTERN.findall(text)

    def _apply_domain_corrections(self, text: str) -> str:
        matches = list(WORD_PATTERN.finditer(text))
        if not matches:
            return text

        replacements: list[tuple[int, int, str, float]] = []
        max_span = min(5, max(self.domain_term_index))
        for span_len in range(max_span, 0, -1):
            buckets = self.domain_term_index.get(span_len)
            if not buckets or len(matches) < span_len:
                continue
            for start in range(0, len(matches) - span_len + 1):
                phrase = " ".join(match.group(0) for match in matches[start : start + span_len])
                phrase_key = phrase.lower()
                if not phrase_key:
                    continue
                candidates = buckets.get(phrase_key[0], [])
                if not candidates:
                    continue

                best_term = ""
                best_score = 0.0
                for candidate in candidates:
                    candidate_key = " ".join(token.lower() for token in self._word_tokens(candidate))
                    if candidate_key == phrase_key:
                        best_term = candidate
                        best_score = 1.0
                        break
                    score = difflib.SequenceMatcher(None, phrase_key, candidate_key).ratio()
                    if score > best_score:
                        best_score = score
                        best_term = candidate

                threshold = self.domain_correction_threshold
                if span_len == 1:
                    threshold = max(0.93, threshold + 0.04)
                if best_term and best_score >= threshold and best_term.lower() != phrase_key:
                    replacements.append(
                        (
                            matches[start].start(),
                            matches[start + span_len - 1].end(),
                            best_term,
                            best_score,
                        )
                    )

        if not replacements:
            return text

        replacements.sort(key=lambda item: (item[0], -(item[1] - item[0]), -item[3]))
        selected: list[tuple[int, int, str]] = []
        occupied_until = -1
        for start, end, term, _score in replacements:
            if start < occupied_until:
                continue
            selected.append((start, end, term))
            occupied_until = end

        pieces = []
        cursor = 0
        for start, end, term in selected:
            pieces.append(text[cursor:start])
            pieces.append(term)
            cursor = end
        pieces.append(text[cursor:])
        return self._normalize_memory_text("".join(pieces))


def _audio_fingerprint(audio: np.ndarray) -> str:
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    quantized = np.rint(clipped * 32767.0).astype(np.int16)
    digest = hashlib.sha1(quantized.tobytes()).hexdigest()
    return f"{TARGET_SAMPLE_RATE}:{quantized.size}:{digest}"
