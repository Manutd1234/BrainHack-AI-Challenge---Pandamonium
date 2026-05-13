"""Manages the ASR model."""

import io
import logging
import os
import tempfile

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# Domain-specific vocabulary from the TIL-26 fictional world (Clairos).
# Providing this as initial_prompt helps Whisper correctly transcribe
# uncommon terms that appear in the NLP RAG corpus.
INITIAL_PROMPT = (
    "Haven, Clairos, the Cascade, megacorporations, cyberpunk, "
    "NovaCorp, SynthWave, NetRunners, ChromeGuard, DataVault, "
    "HoloMesh, BioForge, NeuroLink, SkyForge, AquaPlex, "
    "reconnaissance, wargame, deployment, tactical"
)


class ASRManager:

    def __init__(self):
        logger.info("Loading Whisper model...")

        model_name = self._resolve_model_name()
        device_index = int(os.getenv("WHISPER_DEVICE_INDEX", "0"))
        cpu_threads = int(os.getenv("WHISPER_CPU_THREADS", "4"))
        num_workers = int(os.getenv("WHISPER_NUM_WORKERS", "1"))

        try:
            self.model = WhisperModel(
                model_name,
                device="cuda",
                device_index=device_index,
                compute_type="float16",
                cpu_threads=cpu_threads,
                num_workers=num_workers,
            )
            self.device = "cuda"
            logger.info(f"Whisper {model_name} loaded on GPU (float16).")
        except Exception as e:
            logger.warning(f"GPU not available ({e}), falling back to CPU.")
            try:
                self.model = WhisperModel(
                    model_name,
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=cpu_threads,
                    num_workers=num_workers,
                )
            except Exception:
                # If large-v3 isn't available, fall back to small
                logger.warning("large-v3 not found, falling back to 'small'.")
                self.model = WhisperModel(
                    "small",
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=cpu_threads,
                    num_workers=num_workers,
                )
            self.device = "cpu"
            logger.info("Whisper model loaded on CPU (int8).")

        self.language_hint = os.getenv("WHISPER_LANGUAGE")
        if not self.language_hint and os.getenv("TEAM_TRACK", "").lower() == "novice":
            self.language_hint = "en"

    @staticmethod
    def _resolve_model_name() -> str:
        """Prefer bundled/fine-tuned CTranslate2 weights before remote IDs."""
        candidates = [
            os.getenv("WHISPER_MODEL"),
            "models/whisper-large-v3-finetuned",
            "models/whisper-large-v3",
            os.getenv("WHISPER_FALLBACK_MODEL", "large-v3"),
        ]
        for candidate in candidates:
            if not candidate:
                continue
            if candidate.startswith("models/") and not os.path.exists(candidate):
                continue
            return candidate
        return "large-v3"

    def _transcribe_once(
        self,
        temp_path: str,
        *,
        vad_filter: bool,
        language: str | None,
    ) -> tuple[str, object]:
        vad_parameters = None
        if vad_filter:
            vad_parameters = {
                "min_silence_duration_ms": int(os.getenv("WHISPER_VAD_SILENCE_MS", "250")),
                "speech_pad_ms": int(os.getenv("WHISPER_VAD_PAD_MS", "300")),
            }

        segments, info = self.model.transcribe(
            temp_path,
            beam_size=int(os.getenv("WHISPER_BEAM_SIZE", "7")),
            language=language,
            initial_prompt=INITIAL_PROMPT,
            task="transcribe",
            temperature=0.0,
            vad_filter=vad_filter,
            vad_parameters=vad_parameters,
            condition_on_previous_text=False,
            word_timestamps=False,
            no_speech_threshold=float(os.getenv("WHISPER_NO_SPEECH_THRESHOLD", "0.95")),
            log_prob_threshold=float(os.getenv("WHISPER_LOG_PROB_THRESHOLD", "-1.5")),
            compression_ratio_threshold=float(os.getenv("WHISPER_COMPRESSION_RATIO_THRESHOLD", "2.6")),
        )
        transcription = " ".join(segment.text.strip() for segment in segments)
        transcription = " ".join(transcription.split())
        return transcription, info

    def asr(self, audio_bytes: bytes) -> str:
        """Performs ASR transcription on an audio file.

        Args:
            audio_bytes: The audio file in bytes (WAV format).

        Returns:
            A string containing the transcription of the audio.
        """
        temp_path = None
        try:
            # faster-whisper requires a file path or file-like object.
            # Using a temp file is more reliable across formats.
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_bytes)
                temp_path = f.name

            # Auto-detect language by default for Advanced, but force English
            # when WHISPER_LANGUAGE=en or TEAM_TRACK=novice is present.
            transcription, info = self._transcribe_once(
                temp_path,
                vad_filter=True,  # Enable VAD with default safe parameters
                language=self.language_hint,
            )
            if not transcription:
                logger.info("Empty ASR result with VAD; retrying without VAD.")
                transcription, info = self._transcribe_once(
                    temp_path,
                    vad_filter=False,
                    language=self.language_hint,
                )
            logger.info(
                f"Detected language: {info.language} "
                f"(prob={info.language_probability:.2f}), "
                f"transcription length: {len(transcription)}"
            )
            return transcription

        except Exception as e:
            logger.error(f"ASR transcription failed: {e}")
            return ""
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
