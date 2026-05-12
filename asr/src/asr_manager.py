"""Manages the ASR model."""

import io
import logging

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class ASRManager:

    def __init__(self):
        # Load the Whisper model. The model was pre-downloaded during
        # Docker build and cached in /workspace/models/whisper.
        logger.info("Loading Whisper model...")
        try:
            self.model = WhisperModel(
                "/workspace/whisper_model",
                device="cuda",
                compute_type="float16",
            )
            logger.info("Whisper model loaded on GPU.")
        except Exception:
            logger.warning("GPU not available, falling back to CPU.")
            self.model = WhisperModel(
                "/workspace/whisper_model",
                device="cpu",
                compute_type="int8",
            )
            logger.info("Whisper model loaded on CPU.")

    def asr(self, audio_bytes: bytes) -> str:
        """Performs ASR transcription on an audio file.

        Args:
            audio_bytes: The audio file in bytes (WAV format).

        Returns:
            A string containing the transcription of the audio.
        """

        audio_stream = io.BytesIO(audio_bytes)

        segments, info = self.model.transcribe(
            audio_stream,
            beam_size=5,
            language=None,  # Auto-detect language
            vad_filter=True,
        )

        transcription = " ".join(segment.text.strip() for segment in segments)
        return transcription
