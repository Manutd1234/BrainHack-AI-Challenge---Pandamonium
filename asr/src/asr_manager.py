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

        # Determine model: prefer large-v3 if pre-downloaded, else fall back
        model_name = "large-v3"

        try:
            self.model = WhisperModel(
                model_name,
                device="cuda",
                compute_type="float16",
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
                )
            except Exception:
                # If large-v3 isn't available, fall back to small
                logger.warning("large-v3 not found, falling back to 'small'.")
                self.model = WhisperModel(
                    "small",
                    device="cpu",
                    compute_type="int8",
                )
            self.device = "cpu"
            logger.info("Whisper model loaded on CPU (int8).")

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

            # Auto-detect language to support multilingual Advanced track
            # (English, Malay, Tamil, Chinese). Setting language=None
            # enables auto-detection. For Novice (English-only), this
            # still works fine since it will detect English.
            segments, info = self.model.transcribe(
                temp_path,
                beam_size=5,
                language=None,  # auto-detect for multilingual support
                initial_prompt=INITIAL_PROMPT,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=300,
                    speech_pad_ms=200,
                ),
                condition_on_previous_text=False,  # prevent hallucination cascading
                no_speech_threshold=0.5,
                log_prob_threshold=-0.8,
            )

            transcription = " ".join(
                segment.text.strip() for segment in segments
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
