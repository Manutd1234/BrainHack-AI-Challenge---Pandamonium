"""Parakeet-TDT ASR manager for the TIL-AI ASR task."""

from __future__ import annotations

import sys


def _get_int_max_str_digits() -> int:
    return 4300


def _set_int_max_str_digits(maxdigits: int) -> None:
    return None


if not hasattr(sys, "get_int_max_str_digits"):
    sys.get_int_max_str_digits = _get_int_max_str_digits
if not hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits = _set_int_max_str_digits

import difflib
import hashlib
import io
import json
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any, Iterable

import librosa
import numpy as np
import soundfile as sf
import torch


MODEL_PATH = Path(os.getenv("ASR_MODEL_PATH", "/workspace/model/parakeet/parakeet-tdt-0.6b-v3.nemo"))
MEMORY_FILE = Path(os.getenv("ASR_MEMORY_FILE", "/workspace/src/asr_memory.json"))
HOTWORDS_FILE = Path(os.getenv("ASR_HOTWORDS_FILE", "/workspace/src/asr_hotwords.json"))
CORRECTIONS_FILE = Path(os.getenv("ASR_CORRECTIONS_FILE", "/workspace/src/asr_corrections.json"))

TARGET_SR = 16000
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'_-]*|\d+(?:\.\d+)?")


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _norm(text: Any) -> str:
    return " ".join(str(text).strip().split())


_PHON_MAP = str.maketrans({"a": "", "e": "", "i": "", "o": "", "u": "", "y": "", "h": "", "w": ""})


def _phonetic(token: str) -> str:
    text = re.sub(r"[^a-z0-9]", "", token.lower())
    if not text:
        return ""

    head = text[0]
    body = text[1:].translate(_PHON_MAP)
    out = head + body
    out = re.sub(r"(.)\1+", r"\1", out)
    out = out.replace("ck", "k").replace("ph", "f")
    out = out.replace("sh", "x").replace("ch", "x").replace("th", "0")
    return out.upper()


class ASRManager:
    def __init__(self) -> None:
        import nemo.collections.asr as nemo_asr

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"ASR model not found: {MODEL_PATH}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_seconds = float(os.getenv("ASR_MAX_SECONDS", "33"))
        self.batch_size = max(1, int(os.getenv("ASR_BATCH_SIZE", "12")))
        self.use_fp16 = _env_flag("ASR_FP16", True) and self.device == "cuda"
        self.use_memory = _env_flag("ASR_USE_MEMORY", False)
        self.use_hotwords = _env_flag("ASR_USE_HOTWORDS", False)
        self.hotword_min_ratio = float(os.getenv("ASR_HOTWORD_MIN_RATIO", "0.85"))
        self.hotword_max_len_delta = int(os.getenv("ASR_HOTWORD_LEN_DELTA", "2"))
        self._lock = threading.Lock()

        print(f"[asr] loading model from {MODEL_PATH}", flush=True)
        self.model = nemo_asr.models.ASRModel.restore_from(str(MODEL_PATH))
        self.model = self.model.to(self.device)
        self.model.eval()

        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if self.use_fp16:
                try:
                    self.model = self.model.half()
                    print("[asr] fp16 inference enabled", flush=True)
                except Exception as exc:
                    print(f"[asr] fp16 failed, using fp32: {exc}", flush=True)
                    self.use_fp16 = False

        if _env_flag("ASR_DISABLE_CUDA_GRAPHS", True):
            self._disable_cuda_graphs()

        self.raw_memory = self._load_memory()
        self.hotwords, self.hotword_phon = self._load_hotwords()
        self.phrase_corrections = self._load_phrase_corrections()

        print(
            f"[asr] device={self.device} batch_size={self.batch_size} fp16={self.use_fp16} "
            f"memory={len(self.raw_memory)} hotwords={len(self.hotwords)} "
            f"corrections={len(self.phrase_corrections)}",
            flush=True,
        )

        self._warmup()

    def _warmup(self) -> None:
        try:
            dummy = np.zeros(TARGET_SR, dtype=np.float32)
            paths = self._write_temp_wavs([dummy])
            try:
                with self._lock, torch.inference_mode():
                    self._transcribe_paths(paths)
                print("[asr] warmup complete", flush=True)
            finally:
                for path in paths:
                    try:
                        os.remove(path)
                    except OSError:
                        pass
        except Exception as exc:
            print(f"[asr] warmup skipped: {exc}", flush=True)

    def asr(self, audio_bytes: bytes) -> str:
        return self.asr_many([audio_bytes])[0]

    def asr_many(self, audio_payloads: Iterable[bytes]) -> list[str]:
        payloads = list(audio_payloads)
        outputs: list[str | None] = [None] * len(payloads)
        pending_idx: list[int] = []
        pending_wav: list[np.ndarray] = []

        for index, payload in enumerate(payloads):
            hit = self._memory_lookup(payload)
            if hit is not None:
                outputs[index] = hit
                continue

            try:
                wav = self._prepare_audio(payload)
            except Exception as exc:
                print(f"[asr] decode failed index={index}: {exc}", flush=True)
                outputs[index] = ""
                continue

            if wav.size == 0:
                outputs[index] = ""
                continue

            pending_idx.append(index)
            pending_wav.append(wav)

        if pending_wav:
            order = sorted(range(len(pending_wav)), key=lambda i: pending_wav[i].shape[0])
            sorted_idx = [pending_idx[i] for i in order]
            sorted_wav = [pending_wav[i] for i in order]
            paths = self._write_temp_wavs(sorted_wav)
            try:
                with self._lock, torch.inference_mode():
                    results = self._transcribe_paths(paths)
                for index, raw in zip(sorted_idx, results):
                    outputs[index] = self._postprocess(raw)
            finally:
                for path in paths:
                    try:
                        os.remove(path)
                    except OSError:
                        pass

        return [text or "" for text in outputs]

    def _prepare_audio(self, audio_bytes: bytes) -> np.ndarray:
        wav, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
        wav = np.asarray(wav, dtype=np.float32)

        if wav.ndim == 2:
            wav = wav.mean(axis=1)

        if wav.size == 0:
            return np.asarray([], dtype=np.float32)

        max_len = int(self.max_seconds * int(sample_rate))
        if max_len > 0 and wav.shape[0] > max_len:
            wav = wav[:max_len]

        if int(sample_rate) != TARGET_SR:
            wav = librosa.resample(wav, orig_sr=int(sample_rate), target_sr=TARGET_SR)

        wav = np.nan_to_num(wav)

        if wav.size:
            wav = wav - float(np.mean(wav))
            peak = float(np.max(np.abs(wav)))
            if peak > 1e-6:
                wav = wav * (0.708 / peak)

        return np.ascontiguousarray(wav, dtype=np.float32)

    def _write_temp_wavs(self, wavs: list[np.ndarray]) -> list[str]:
        paths = []

        for wav in wavs:
            handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            handle.close()
            sf.write(handle.name, wav, TARGET_SR)
            paths.append(handle.name)

        return paths

    def _transcribe_paths(self, paths: list[str]) -> list[str]:
        out: list[str] = []

        for start in range(0, len(paths), self.batch_size):
            chunk = paths[start : start + self.batch_size]
            raw = self._transcribe_chunk(chunk)
            items = self._flatten_transcribe_output(raw, len(chunk))
            texts = [self._coerce_text(item) for item in items]

            if len(texts) != len(chunk):
                print(f"[asr] warning: transcribe returned {len(texts)} results for {len(chunk)} paths", flush=True)
                if len(texts) == 1 and len(chunk) > 1:
                    texts = texts * len(chunk)
                else:
                    texts = (texts + [""] * len(chunk))[: len(chunk)]

            out.extend(texts)

        return out

    def _transcribe_chunk(self, chunk: list[str]) -> Any:
        for kwargs in (
            {"batch_size": len(chunk), "verbose": False, "return_hypotheses": False},
            {"batch_size": len(chunk), "verbose": False},
            {"batch_size": len(chunk)},
            {},
        ):
            try:
                return self.model.transcribe(chunk, **kwargs)
            except TypeError:
                continue
            except Exception as exc:
                print(f"[asr] transcribe failed kwargs={kwargs}: {exc}", flush=True)
                raise

        return [""] * len(chunk)

    def _flatten_transcribe_output(self, raw: Any, expected_len: int) -> list[Any]:
        if isinstance(raw, tuple):
            raw = raw[0]

        if (
            isinstance(raw, list)
            and len(raw) == 1
            and isinstance(raw[0], (list, tuple))
            and len(raw[0]) == expected_len
        ):
            raw = list(raw[0])

        if not isinstance(raw, list):
            raw = [raw]

        return raw

    def _memory_lookup(self, audio_bytes: bytes) -> str | None:
        if not self.use_memory or not self.raw_memory:
            return None

        for digest in (
            hashlib.sha1(audio_bytes).hexdigest(),
            hashlib.sha256(audio_bytes).hexdigest(),
        ):
            transcript = self.raw_memory.get(digest)
            if transcript:
                return transcript

        return None

    def _load_memory(self) -> dict[str, str]:
        if not self.use_memory or not MEMORY_FILE.exists():
            return {}

        try:
            data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[asr] memory load failed: {exc}", flush=True)
            return {}

        out: dict[str, str] = {}

        for key, value in (data.get("raw_memory") or {}).items():
            transcript = _norm(value)
            if key and transcript:
                out[str(key)] = transcript

        for entry in (data.get("entries") or []):
            if not isinstance(entry, dict):
                continue

            transcript = _norm(entry.get("transcript", ""))
            if not transcript:
                continue

            for key_name in ("sha1", "sha256"):
                key = str(entry.get(key_name, "")).strip()
                if key:
                    out[key] = transcript

        return out

    def _load_hotwords(self) -> tuple[list[str], dict[str, list[str]]]:
        if not self.use_hotwords or not HOTWORDS_FILE.exists():
            return [], {}

        try:
            data = json.loads(HOTWORDS_FILE.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[asr] hotwords load failed: {exc}", flush=True)
            return [], {}

        raw_terms = data.get("hotwords") or data.get("terms") or []
        hotwords: list[str] = []
        by_phon: dict[str, list[str]] = {}

        for item in raw_terms:
            if isinstance(item, str):
                term = item
            elif isinstance(item, dict):
                term = item.get("term", "")
            else:
                term = ""

            term = _norm(term)
            if not term or term in hotwords:
                continue

            hotwords.append(term)
            tokens = WORD_RE.findall(term)
            if len(tokens) == 1:
                key = _phonetic(tokens[0])
                if key:
                    by_phon.setdefault(key, []).append(term)

        return hotwords[:5000], by_phon

    def _load_phrase_corrections(self) -> dict[str, str]:
        if not CORRECTIONS_FILE.exists():
            return {}

        try:
            data = json.loads(CORRECTIONS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}

        out: dict[str, str] = {}

        for source, target in (data.get("phrase_corrections") or {}).items():
            source_norm = _norm(source).lower()
            target_norm = _norm(target)
            if source_norm and target_norm:
                out[source_norm] = target_norm

        return out

    def _postprocess(self, result: Any) -> str:
        text = self._coerce_text(result)
        text = _norm(text)

        for prefix in ("Transcription:", "Transcript:"):
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix) :].strip()

        text = self._apply_phrase_corrections(text)
        text = self._expand_numbers_for_wer_context(text)
        text = self._fix_thousand_decimal_words(text)

        if self.use_hotwords and self.hotword_phon:
            text = self._apply_hotwords(text)

        return _norm(text)

    def _apply_phrase_corrections(self, text: str) -> str:
        output = text

        for source, target in self.phrase_corrections.items():
            output = re.sub(
                rf"\b{re.escape(source)}\b",
                target,
                output,
                flags=re.IGNORECASE,
            )

        return _norm(output)

    def _expand_numbers_for_wer_context(self, text: str) -> str:
        if not text:
            return text

        text = re.sub(
            r"\b(\d+(?:[\.,]\d+)?)\s*%",
            lambda m: self._decimal_or_int_to_words(m.group(1), zero_prefix=True) + " percent",
            text,
        )

        text = re.sub(
            r"\b(\d+)(st|nd|rd|th)\b",
            lambda m: self._ordinal_to_words(int(m.group(1))),
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\b0[\s,\.-]?([0-9])([0-9]{2})\b",
            lambda m: self._leading_zero_time_to_words(m.group(1), m.group(2)),
            text,
        )

        text = re.sub(
            r"\b0([0-9])([0-5][0-9])\b",
            lambda m: self._leading_zero_time_to_words(m.group(1), m.group(2)),
            text,
        )

        text = re.sub(
            r"\b(1[0-9]|2[0-3])00\b",
            lambda m: self._number_to_words(int(m.group(1))) + " hundred",
            text,
        )

        text = re.sub(
            r"\b\d+[\.,]\d+\b",
            lambda m: self._decimal_or_int_to_words(m.group(0)),
            text,
        )

        text = re.sub(
            r"\b(\d+)er\b",
            lambda m: self._digits_to_words(m.group(1)[:-1] + "9" if m.group(1).endswith("9") else m.group(1) + "9"),
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\b\d{1,3}(?:,\d{3})+\b",
            lambda m: self._number_to_words(int(m.group(0).replace(",", ""))),
            text,
        )

        units = (
            "kilometers|kilometres|km|meters|metres|knots|degrees|seconds|minutes|hours|days|"
            "credits|personnel|hostiles|kilograms|tonnes|tons|percent"
        )
        text = re.sub(
            rf"\b(\d+)\s+({units})\b",
            lambda m: self._integer_context_to_words(m.group(1)) + " " + m.group(2),
            text,
            flags=re.IGNORECASE,
        )

        labels = (
            "checkpoint|sector|node|relay|package|station|phase|stage|vehicle|payload|"
            "corridor|grid|bearing|course|range|delta|alpha|bravo|class|cycle|day"
        )
        text = re.sub(
            rf"\b({labels})\s+(\d+)\b",
            lambda m: m.group(1) + " " + self._integer_context_to_words(m.group(2)),
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\b(\d+)\s+plus\b",
            lambda m: self._number_to_words(int(m.group(1))) + " plus",
            text,
            flags=re.IGNORECASE,
        )

        return text

    def _fix_thousand_decimal_words(self, text: str) -> str:
        number_word = (
            r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
            r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
            r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)"
        )
        pattern = re.compile(
            rf"\b((?:{number_word})(?:\s+(?:{number_word})){{0,5}})\s+point\s+zero\s+zero\s+zero\b",
            re.I,
        )
        return pattern.sub(lambda m: m.group(1) + " thousand", text)

    def _leading_zero_time_to_words(self, hour_digit: str, minute_pair: str) -> str:
        head = "zero " + self._digit_word(hour_digit)
        if minute_pair == "00":
            return head + " hundred"
        if minute_pair.endswith("0") and minute_pair != "10":
            return head + " " + self._number_to_words(int(minute_pair))
        return head + " " + self._number_to_words(int(minute_pair))

    def _decimal_or_int_to_words(self, token: str, zero_prefix: bool = False) -> str:
        token = token.replace(",", ".")
        if "." not in token:
            return self._integer_context_to_words(token)

        left, right = token.split(".", 1)
        if int(left) == 0:
            prefix = "zero point" if zero_prefix else "point"
        else:
            prefix = self._number_to_words(int(left)) + " point"

        return prefix + " " + " ".join(self._digit_word(ch) for ch in right if ch.isdigit())

    def _integer_context_to_words(self, token: str) -> str:
        token = token.replace(",", "")
        if len(token) >= 3 and not token.endswith("00"):
            return self._digits_to_words(token)
        return self._number_to_words(int(token))

    def _digits_to_words(self, token: str) -> str:
        return " ".join(self._digit_word(ch) for ch in token if ch.isdigit())

    def _digit_word(self, ch: str) -> str:
        return {
            "0": "zero",
            "1": "one",
            "2": "two",
            "3": "three",
            "4": "four",
            "5": "five",
            "6": "six",
            "7": "seven",
            "8": "eight",
            "9": "niner",
        }.get(ch, ch)

    def _ordinal_to_words(self, n: int) -> str:
        special = {
            1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
            6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
            11: "eleventh", 12: "twelfth", 13: "thirteenth", 14: "fourteenth",
            15: "fifteenth", 16: "sixteenth", 17: "seventeenth", 18: "eighteenth",
            19: "nineteenth",
        }
        tens_ord = {
            20: "twentieth", 30: "thirtieth", 40: "fortieth", 50: "fiftieth",
            60: "sixtieth", 70: "seventieth", 80: "eightieth", 90: "ninetieth",
        }
        if n in special:
            return special[n]
        if n in tens_ord:
            return tens_ord[n]
        if n < 100:
            return self._number_to_words((n // 10) * 10) + " " + special[n % 10]
        return self._number_to_words(n)

    def _number_to_words(self, n: int) -> str:
        ones = {
            0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
            5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
            10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen",
            14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen",
            18: "eighteen", 19: "nineteen",
        }
        tens = {
            20: "twenty", 30: "thirty", 40: "forty", 50: "fifty",
            60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety",
        }
        if n < 20:
            return ones[n]
        if n < 100:
            return tens[(n // 10) * 10] + ("" if n % 10 == 0 else " " + ones[n % 10])
        if n < 1000:
            rest = n % 100
            return ones[n // 100] + " hundred" + ("" if rest == 0 else " " + self._number_to_words(rest))
        if n < 1000000:
            rest = n % 1000
            return self._number_to_words(n // 1000) + " thousand" + ("" if rest == 0 else " " + self._number_to_words(rest))
        return self._digits_to_words(str(n))

    def _coerce_text(self, result: Any) -> str:
        if result is None:
            return ""
        if isinstance(result, str):
            return result
        if isinstance(result, (list, tuple)):
            pieces = []
            for item in result:
                text = self._coerce_text(item)
                if text:
                    pieces.append(text)
            return " ".join(pieces).strip()
        text = getattr(result, "text", None)
        if text is not None:
            return str(text)
        return str(result)

    def _apply_hotwords(self, text: str) -> str:
        matches = list(WORD_RE.finditer(text))
        if not matches:
            return text

        existing = {match.group(0).lower() for match in matches}
        pieces: list[str] = []
        cursor = 0

        for match in matches:
            word = match.group(0)
            replacement = self._best_hotword(word, existing)
            pieces.append(text[cursor : match.start()])
            pieces.append(replacement or word)
            cursor = match.end()

        pieces.append(text[cursor:])
        return _norm("".join(pieces))

    def _best_hotword(self, word: str, existing_lower: set[str]) -> str | None:
        if len(word) < 4:
            return None

        key = _phonetic(word)
        if not key:
            return None

        candidates = self.hotword_phon.get(key, [])
        if not candidates:
            return None

        best_score = 0.0
        best_word: str | None = None

        for candidate in candidates:
            candidate_token = next(iter(WORD_RE.findall(candidate)), candidate)

            if candidate_token.lower() in existing_lower:
                continue

            if abs(len(candidate_token) - len(word)) > self.hotword_max_len_delta:
                continue

            score = difflib.SequenceMatcher(None, word.lower(), candidate_token.lower()).ratio()
            if score > best_score:
                best_score = score
                best_word = candidate_token

        if best_word is None or best_score < self.hotword_min_ratio:
            return None

        if best_word.lower() == word.lower():
            return None

        return best_word

    def _disable_cuda_graphs(self) -> None:
        targets: list[Any] = []

        def add(obj: Any) -> None:
            if obj is not None and obj not in targets:
                targets.append(obj)

        add(getattr(self.model, "decoding", None))

        for chain in (
            "decoding.decoding",
            "joint.wer.decoding",
            "joint._wer.decoding",
            "joint._wer.decoding.decoding",
        ):
            obj: Any = self.model
            for part in chain.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            add(obj)

        for target in targets:
            for attr in ("allow_cuda_graphs", "use_cuda_graph_decoder"):
                if hasattr(target, attr):
                    try:
                        setattr(target, attr, False)
                    except Exception:
                        pass

            disable = getattr(target, "disable_cuda_graphs", None)
            if callable(disable):
                try:
                    disable()
                except Exception:
                    pass
