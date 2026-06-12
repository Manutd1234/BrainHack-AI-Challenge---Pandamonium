from __future__ import annotations

import io
import json
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any, Iterable
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf
import torch

MODEL_PATH = Path(os.getenv("ASR_MODEL_PATH", "/workspace/model/parakeet/parakeet_v2_decoder_e1_hard120_replay.nemo"))
CORRECTIONS_FILE = Path(os.getenv("ASR_CORRECTIONS_FILE", "/workspace/src/asr_corrections.json"))
VOCABULARY_FILE = Path(os.getenv("ASR_VOCABULARY_FILE", "/workspace/src/asr_vocabulary.json"))

TARGET_SR = 16000


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _norm(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


class ASRManager:
    def __init__(self) -> None:
        import nemo.collections.asr as nemo_asr

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"ASR model not found: {MODEL_PATH}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_seconds = float(os.getenv("ASR_MAX_SECONDS", "33"))
        self.batch_size = max(1, int(os.getenv("ASR_BATCH_SIZE", "8")))
        self.prep_workers = max(1, int(os.getenv("ASR_PREP_WORKERS", "4")))
        self.use_fp16 = _env_flag("ASR_FP16", True) and self.device == "cuda"
        self.disable_cuda_graphs = _env_flag("ASR_DISABLE_CUDA_GRAPHS", True)

        self._lock = threading.Lock()
        self._tmp_dir = tempfile.mkdtemp(prefix="asr_")

        print(f"[asr] loading model from {MODEL_PATH}", flush=True)
        self.model = nemo_asr.models.ASRModel.restore_from(str(MODEL_PATH)).to(self.device).eval()

        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

            if self.use_fp16:
                try:
                    self.model = self.model.half()
                except Exception:
                    self.use_fp16 = False

        if self.disable_cuda_graphs:
            self._disable_cuda_graphs()

        self.phrase_corrections = self._load_phrase_corrections()
        self.vocabulary = self._load_vocabulary()
        self._compiled_vocabulary = self._compile_vocabulary_patterns()
        self._compiled_phrase_corrections = [
            (re.compile(rf"(?<!\w){re.escape(src)}(?!\w)", re.IGNORECASE), dst)
            for src, dst in self.phrase_corrections
        ]

        print(
            f"[asr] device={self.device} bs={self.batch_size} fp16={self.use_fp16} "
            f"tf32={self.device == 'cuda'} graphs_disabled={self.disable_cuda_graphs} "
            f"max_s={self.max_seconds} corr={len(self.phrase_corrections)} vocab={len(self.vocabulary)}",
            flush=True,
        )

        self._warmup()

    def _warmup(self) -> None:
        try:
            dummy = np.zeros(TARGET_SR, dtype=np.float32)
            paths = self._write_temp_wavs([dummy])
            with self._lock, torch.inference_mode():
                self._transcribe_paths(paths)
            print("[asr] warmup complete", flush=True)
        except Exception as e:
            print(f"[asr] warmup skipped: {e}", flush=True)

    def asr(self, audio_bytes: bytes) -> str:
        return self.asr_many([audio_bytes])[0]

    def asr_many(self, audio_payloads: Iterable[bytes]) -> list[str]:
        payloads = list(audio_payloads)
        outputs: list[str | None] = [None] * len(payloads)
        pending_idx: list[int] = []
        pending_wav: list[np.ndarray] = []

        def prep_one(item: tuple[int, bytes]) -> tuple[int, np.ndarray | None]:
            index, payload = item
            try:
                wav = self._prepare_audio(payload)
                return index, wav
            except Exception:
                return index, None

        if self.prep_workers > 1 and len(payloads) > 1:
            with ThreadPoolExecutor(max_workers=self.prep_workers) as ex:
                prepared = list(ex.map(prep_one, enumerate(payloads)))
        else:
            prepared = [prep_one(item) for item in enumerate(payloads)]

        for index, wav in prepared:
            if wav is None or wav.size == 0:
                outputs[index] = ""
                continue

            pending_idx.append(index)
            pending_wav.append(wav)

        if pending_wav:
            order = sorted(range(len(pending_wav)), key=lambda i: len(pending_wav[i]))
            sorted_idx = [pending_idx[i] for i in order]
            sorted_wav = [pending_wav[i] for i in order]

            paths = self._write_temp_wavs(sorted_wav)
            with self._lock, torch.inference_mode():
                results = self._transcribe_paths(paths)

            for index, raw in zip(sorted_idx, results):
                outputs[index] = self._postprocess(raw)

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
            try:
                import librosa
                wav = librosa.resample(wav, orig_sr=int(sample_rate), target_sr=TARGET_SR).astype(np.float32)
            except Exception:
                from scipy import signal
                wav = signal.resample_poly(wav, TARGET_SR, int(sample_rate)).astype(np.float32)

        wav = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)

        if wav.size:
            wav -= float(np.mean(wav))
            peak = float(np.max(np.abs(wav)))
            if peak > 1e-6:
                wav *= 0.708 / peak

        return np.ascontiguousarray(wav, dtype=np.float32)

    def _write_temp_wavs(self, wavs: list[np.ndarray]) -> list[str]:
        paths: list[str] = []
        for i, wav in enumerate(wavs):
            path = os.path.join(self._tmp_dir, f"chunk_{i}.wav")
            sf.write(path, wav, TARGET_SR)
            paths.append(path)
        return paths

    def _transcribe_paths(self, paths: list[str]) -> list[str]:
        out: list[str] = []

        for start in range(0, len(paths), self.batch_size):
            chunk = paths[start:start + self.batch_size]
            raw = self._transcribe_chunk(chunk)
            items = self._flatten_transcribe_output(raw, len(chunk))
            texts = [self._coerce_text(item) for item in items]

            if len(texts) != len(chunk):
                if len(texts) == 1 and len(chunk) > 1:
                    texts = texts * len(chunk)
                else:
                    texts = (texts + [""] * len(chunk))[:len(chunk)]

            out.extend(texts)

        return out

    def _transcribe_chunk(self, chunk: list[str]) -> Any:
        for kwargs in (
            {"batch_size": len(chunk), "verbose": False},
            {"batch_size": len(chunk)},
            {},
        ):
            try:
                return self.model.transcribe(chunk, **kwargs)
            except TypeError:
                continue
            except Exception as e:
                print(f"[asr] transcribe failed: {e}", flush=True)
                continue

        return [""] * len(chunk)

    def _flatten_transcribe_output(self, raw: Any, expected_len: int) -> list[Any]:
        if isinstance(raw, tuple):
            raw = raw[0]

        if isinstance(raw, list) and len(raw) == 1:
            inner = raw[0]
            if isinstance(inner, (list, tuple)) and len(inner) == expected_len:
                raw = list(inner)

        if not isinstance(raw, list):
            raw = [raw]

        return raw

    def _coerce_text(self, result: Any) -> str:
        if result is None:
            return ""

        if isinstance(result, str):
            return result

        if isinstance(result, (list, tuple)):
            return " ".join(x for x in (self._coerce_text(i) for i in result) if x).strip()

        text = getattr(result, "text", None)
        if text is not None:
            return str(text)

        return str(result)

    def _postprocess(self, result: Any) -> str:
        text = _norm(self._coerce_text(result))

        for prefix in ("Transcription:", "Transcript:"):
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()

        text = self._apply_phrase_corrections(text)

        text = re.sub(r"\s+([,.!?;:])", r"\1", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _load_phrase_corrections(self) -> list[tuple[str, str]]:
        if not CORRECTIONS_FILE.exists():
            return []

        try:
            data = json.loads(CORRECTIONS_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[asr] failed to load corrections: {e}", flush=True)
            return []

        if isinstance(data, dict) and isinstance(data.get("phrase_corrections"), dict):
            raw = data["phrase_corrections"]
        elif isinstance(data, dict):
            raw = data
        else:
            raw = {}

        pairs: list[tuple[str, str]] = []
        for src, dst in raw.items():
            src_n = _norm(src)
            dst_n = _norm(dst)
            if src_n and dst_n and src_n.lower() != dst_n.lower():
                pairs.append((src_n, dst_n))

        pairs.sort(key=lambda kv: len(kv[0]), reverse=True)
        return pairs

    def _apply_phrase_corrections(self, text: str) -> str:
        if not text or not self.phrase_corrections:
            return _norm(text)

        out = text
        for pattern, dst in self._compiled_phrase_corrections:
            out = pattern.sub(dst, out)

        return _norm(out)

    def _load_vocabulary(self) -> list[str]:
        if not VOCABULARY_FILE.exists():
            return []

        try:
            data = json.loads(VOCABULARY_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[asr] failed to load vocabulary: {e}", flush=True)
            return []

        words = data.get("hotwords") or data.get("vocabulary") or []
        if not isinstance(words, list):
            return []

        out: list[str] = []
        seen: set[str] = set()

        for word in words:
            word = _norm(word)
            if not word:
                continue
            key = word.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(word)

        out.sort(key=len, reverse=True)
        return out

    def _compile_vocabulary_patterns(self) -> list[tuple[re.Pattern[str], str]]:
        risky_singletons = {
            "class", "credit", "alpha", "delta", "science", "sciences", "abstract",
            "articles", "technologies", "foreign", "front", "commander", "secretary",
            "grade", "archives", "thursday", "september", "november", "manager",
            "studio", "plant", "casino", "birth", "deputy", "municipal", "tertiary",
            "emerging", "institute", "firms", "centre", "green", "counsel",
            "exceptional", "association", "analytics", "unengaged", "unguided",
            "detachment", "counterterrorism", "containerized",
        }

        compiled: list[tuple[re.Pattern[str], str]] = []
        for hotword in self.vocabulary:
            hw = hotword.strip()
            low = hw.lower().strip("-")
            if " " not in low and "-" not in low and low in risky_singletons:
                continue

            if "-" in hw:
                parts = [re.escape(x) for x in hw.split("-") if x]
                if not parts:
                    continue
                pattern = r"(?<!\w)" + r"[\s-]+".join(parts) + r"(?!\w)"
            else:
                pattern = rf"(?<!\w){re.escape(hw)}(?!\w)"

            compiled.append((re.compile(pattern, re.IGNORECASE), hw))
        return compiled

    def _apply_vocabulary(self, text: str) -> str:
        if not text or not self.vocabulary:
            return _norm(text)

        out = text
        for pattern, hotword in self._compiled_vocabulary:
            out = pattern.sub(hotword, out)

        return _norm(out)

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
            obj = self.model
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
