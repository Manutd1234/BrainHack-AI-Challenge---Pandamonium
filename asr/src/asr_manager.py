"""Manages the ASR model."""

import logging
import os
import tempfile

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class ASRManager:

    def __init__(self):
        # Load the Whisper model. The model was pre-downloaded and cached
        # during Docker build. Using model name 'small' will find it from cache.
        logger.info("Loading Whisper model...")
        try:
            self.model = WhisperModel(
                "small",
                device="cuda",
                compute_type="float16",
            )
            logger.info("Whisper model loaded on GPU.")
        except Exception as e:
            logger.warning(f"GPU not available ({e}), falling back to CPU.")
            self.model = WhisperModel(
                "small",
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
        temp_path = None
        try:
            # Write audio bytes to a temp file (faster-whisper needs a file path)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_bytes)
                temp_path = f.name

            segments, info = self.model.transcribe(
                temp_path,
                beam_size=5,
                language=None,
                vad_filter=True,
            )

            transcription = " ".join(segment.text.strip() for segment in segments)
            return transcription

        except Exception as e:
            logger.error(f"ASR transcription failed: {e}")
            return ""
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
