"""Whisper ASR with optional MERaLiON fallback for the TIL-AI ASR task."""

from __future__ import annotations

import io
import os
import threading
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


MERALION_MODEL_ID = "MERaLiON/MERaLiON-2-3B"
TARGET_SAMPLE_RATE = 16_000
DEEPFILTER_SAMPLE_RATE = 48_000
PROMPT_TEMPLATE = (
    "Instruction: {query} \n"
    "Follow the text instruction based on the following audio: <SpeechHere>"
)


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


class ASRManager:
    """Loads the ASR stack once and serves transcription requests."""

    def __init__(self) -> None:
        self.model_path = os.getenv(
            "MERALION_MODEL_PATH",
            os.getenv("MERALION_MODEL_ID", MERALION_MODEL_ID),
        )
        self.engine = os.getenv("ASR_ENGINE", "whisper").strip().lower()
        self.max_new_tokens = int(os.getenv("ASR_MAX_NEW_TOKENS", "128"))
        self.max_seconds = float(os.getenv("ASR_MAX_SECONDS", "30"))
        self.hybrid_min_chars = int(os.getenv("ASR_HYBRID_MIN_CHARS", "24"))
        self.hybrid_min_words = int(os.getenv("ASR_HYBRID_MIN_WORDS", "5"))
        self.hybrid_max_repeat_ratio = float(
            os.getenv("ASR_HYBRID_MAX_REPEAT_RATIO", "0.55")
        )
        self.hybrid_min_chars_per_second = float(
            os.getenv("ASR_HYBRID_MIN_CHARS_PER_SECOND", "4.0")
        )
        self.hybrid_min_words_per_second = float(
            os.getenv("ASR_HYBRID_MIN_WORDS_PER_SECOND", "0.45")
        )
        self.use_deepfilter = _env_flag("ASR_USE_DEEPFILTERNET", False)
        self.deepfilter_mode = os.getenv("ASR_DF_MODE", "selective").strip().lower()
        self.use_df_rescue = _env_flag("ASR_USE_DF_RESCUE", True)
        self.use_whisper_fallback = _env_flag("ASR_USE_WHISPER_FALLBACK", True)
        self.whisper_path = os.getenv(
            "WHISPER_MODEL_PATH",
            os.getenv("WHISPER_MODEL_ID", "/workspace/model/whisper-small-til26"),
        )
        self.whisper_base_path = os.getenv(
            "WHISPER_BASE_MODEL_PATH",
            "/workspace/models/whisper-small",
        )
        self.prompt = PROMPT_TEMPLATE.format(
            query=os.getenv("ASR_PROMPT", "Please transcribe this speech.")
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = (
            torch.bfloat16 if self.device.type == "cuda" else torch.float32
        )
        self._lock = threading.Lock()

        self.processor = None
        self.model = None
        if self.engine != "whisper":
            self._load_meralion()

        self._df_enhance = None
        self._df_model = None
        self._df_state = None
        self._df_atten_lim_db = os.getenv("ASR_DF_ATTEN_LIM_DB")
        self.whisper_processor = None
        self.whisper_model = None
        if self.use_deepfilter:
            self._load_deepfilter()
        if self.engine == "whisper" or self.use_whisper_fallback:
            self._load_whisper()

    def asr(self, audio_bytes: bytes) -> str:
        """Transcribe one WAV byte payload."""
        return self.asr_many([audio_bytes])[0]

    def asr_many(self, audio_payloads: Iterable[bytes]) -> list[str]:
        """Transcribe a batch of WAV byte payloads in request order."""
        audio_arrays = [self._prepare_audio(payload) for payload in audio_payloads]
        if not audio_arrays:
            return []

        if self.engine == "whisper" or self.model is None or self.processor is None:
            return self._whisper_transcribe(audio_arrays)

        if self.engine == "hybrid":
            durations = [len(audio) / TARGET_SAMPLE_RATE for audio in audio_arrays]
            return self._hybrid_transcribe(audio_arrays, durations)

        return self._meralion_transcribe(audio_arrays)

    def _meralion_transcribe(self, audio_arrays: list[np.ndarray]) -> list[str]:
        conversation = [
            [{"role": "user", "content": self.prompt}] for _ in audio_arrays
        ]
        chat_prompt = self.processor.tokenizer.apply_chat_template(
            conversation=conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=chat_prompt,
            audios=audio_arrays,
            sampling_rate=TARGET_SAMPLE_RATE,
            padding=True,
        )
        inputs = self._move_inputs_to_device(inputs)

        with self._lock, torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        prompt_length = inputs["input_ids"].size(1)
        generated_ids = outputs[:, prompt_length:]
        predictions = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        cleaned = [self._clean_prediction(prediction) for prediction in predictions]
        if self.whisper_model is not None and any(not prediction for prediction in cleaned):
            fallback = self._whisper_transcribe(audio_arrays)
            cleaned = [
                prediction if prediction else fallback[index]
                for index, prediction in enumerate(cleaned)
            ]
        return cleaned

    def _hybrid_transcribe(
        self,
        audio_arrays: list[np.ndarray],
        durations: list[float],
    ) -> list[str]:
        """Use fast Whisper first, then MERaLiON only for suspicious outputs."""
        whisper_predictions = self._whisper_transcribe(audio_arrays)
        fallback_indexes = [
            index
            for index, prediction in enumerate(whisper_predictions)
            if self._needs_meralion_fallback(prediction, durations[index])
        ]

        if not fallback_indexes:
            return whisper_predictions

        merged = list(whisper_predictions)

        if (
            self.use_df_rescue
            and self.use_deepfilter
            and self._df_enhance is not None
        ):
            denoised_audio = [
                self._denoise_to_target(audio_arrays[index])
                for index in fallback_indexes
            ]
            denoised_predictions = self._whisper_transcribe(denoised_audio)
            remaining_indexes = []
            remaining_audio = []
            for index, audio, prediction in zip(
                fallback_indexes,
                denoised_audio,
                denoised_predictions,
            ):
                if prediction and not self._needs_meralion_fallback(
                    prediction,
                    len(audio) / TARGET_SAMPLE_RATE,
                ):
                    merged[index] = prediction
                else:
                    remaining_indexes.append(index)
                    remaining_audio.append(audio)
            fallback_indexes = remaining_indexes
            fallback_audio = remaining_audio
        else:
            fallback_audio = [audio_arrays[index] for index in fallback_indexes]

        if not fallback_indexes:
            return merged

        meralion_predictions = self._meralion_transcribe(fallback_audio)
        for index, prediction in zip(fallback_indexes, meralion_predictions):
            if prediction:
                merged[index] = prediction
        return merged

    def _load_whisper(self) -> None:
        try:
            whisper_path = self._resolve_whisper_path()
            print(f"Loading Whisper from {whisper_path}", flush=True)
            self.whisper_processor = AutoProcessor.from_pretrained(whisper_path)
            self.whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                whisper_path,
                torch_dtype=self.torch_dtype,
            ).to(self.device)
            self.whisper_model.eval()
        except Exception as exc:
            self.use_whisper_fallback = False
            self.whisper_processor = None
            self.whisper_model = None
            print(f"Whisper fallback unavailable. Error: {exc}", flush=True)

    def _resolve_whisper_path(self) -> str:
        if (Path(self.whisper_path) / "config.json").exists():
            return self.whisper_path
        if (Path(self.whisper_base_path) / "config.json").exists():
            return self.whisper_base_path
        return self.whisper_path

    def _load_meralion(self) -> None:
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_path,
            use_safetensors=True,
            trust_remote_code=True,
            torch_dtype=self.torch_dtype,
        ).to(self.device)
        self.model.eval()

    def _load_deepfilter(self) -> None:
        try:
            from df.enhance import enhance, init_df

            result = init_df("DeepFilterNet3", log_level="ERROR", log_file=None)
            self._df_model = result[0]
            self._df_state = result[1]
            self._df_enhance = enhance
        except Exception as exc:
            self.use_deepfilter = False
            print(
                f"DeepFilterNet3 unavailable; using raw audio. Error: {exc}",
                flush=True,
            )

    def _whisper_transcribe(self, audio_arrays: list[np.ndarray]) -> list[str]:
        if self.whisper_model is None or self.whisper_processor is None:
            return ["" for _ in audio_arrays]

        forced_decoder_ids = self.whisper_processor.get_decoder_prompt_ids(
            task="transcribe",
        )
        inputs = self.whisper_processor(
            audio_arrays,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        input_features = inputs.input_features.to(self.device)
        if self.device.type == "cuda":
            input_features = input_features.to(self.torch_dtype)

        with self._lock, torch.inference_mode():
            predicted_ids = self.whisper_model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        predictions = self.whisper_processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
        )
        return [self._clean_prediction(prediction) for prediction in predictions]

    def _prepare_audio(self, audio_bytes: bytes) -> np.ndarray:
        audio, sample_rate = self._read_wav(audio_bytes)
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

        audio = self._limit_duration(audio, TARGET_SAMPLE_RATE)
        return np.ascontiguousarray(audio, dtype=np.float32)

    def _denoise_to_target(self, audio: np.ndarray) -> np.ndarray:
        enhanced, sample_rate = self._denoise(audio, TARGET_SAMPLE_RATE)
        if sample_rate != TARGET_SAMPLE_RATE:
            enhanced = librosa.resample(
                enhanced,
                orig_sr=sample_rate,
                target_sr=TARGET_SAMPLE_RATE,
            )
        enhanced = self._limit_duration(enhanced, TARGET_SAMPLE_RATE)
        return np.ascontiguousarray(enhanced, dtype=np.float32)

    def _read_wav(self, audio_bytes: bytes) -> tuple[np.ndarray, int]:
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
        return np.nan_to_num(audio), int(sample_rate)

    def _denoise(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> tuple[np.ndarray, int]:
        if sample_rate != DEEPFILTER_SAMPLE_RATE:
            audio = librosa.resample(
                audio,
                orig_sr=sample_rate,
                target_sr=DEEPFILTER_SAMPLE_RATE,
            )

        noisy_audio = torch.from_numpy(audio).float().unsqueeze(0)
        kwargs = {"pad": True}
        if self._df_atten_lim_db:
            kwargs["atten_lim_db"] = int(self._df_atten_lim_db)

        with torch.inference_mode():
            enhanced = self._df_enhance(
                self._df_model,
                self._df_state,
                noisy_audio,
                **kwargs,
            )

        enhanced = enhanced.detach().cpu().numpy().squeeze()
        if enhanced.ndim > 1:
            enhanced = enhanced.mean(axis=0)
        return np.asarray(enhanced, dtype=np.float32), DEEPFILTER_SAMPLE_RATE

    def _limit_duration(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if self.max_seconds <= 0:
            return audio

        max_samples = int(sample_rate * self.max_seconds)
        if max_samples <= 1 or audio.shape[0] < max_samples:
            return audio
        return audio[: max_samples - 1]

    def _move_inputs_to_device(self, inputs):
        for key, value in list(inputs.items()):
            if not isinstance(value, torch.Tensor):
                continue
            value = value.to(self.device)
            if value.dtype == torch.float32 and self.device.type == "cuda":
                value = value.to(self.torch_dtype)
            inputs[key] = value
        return inputs

    def _clean_prediction(self, prediction: str) -> str:
        prediction = " ".join(prediction.strip().split())
        for prefix in ("Transcription:", "Transcript:"):
            if prediction.lower().startswith(prefix.lower()):
                return prediction[len(prefix) :].strip()
        return prediction

    def _needs_meralion_fallback(self, prediction: str, duration: float) -> bool:
        if len(prediction) < self.hybrid_min_chars:
            return True

        words = prediction.split()
        if len(words) < self.hybrid_min_words:
            return True
        if duration >= 8.0:
            if len(prediction) / duration < self.hybrid_min_chars_per_second:
                return True
            if len(words) / duration < self.hybrid_min_words_per_second:
                return True
        if len(words) >= 8:
            counts = {}
            for word in words:
                key = word.lower().strip(".,!?;:")
                counts[key] = counts.get(key, 0) + 1
            if max(counts.values()) / len(words) > self.hybrid_max_repeat_ratio:
                return True
        return False
