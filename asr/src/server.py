"""
ASR Server — port 5001
POST /asr
  Input:  {"instances": [{"key": int, "b64": "<base64 WAV>"}]}
  Output: {"predictions": ["transcript 1", ...]}
"""

import base64
import io
import logging
import os
import tempfile
from typing import Any

import numpy as np
from fastapi import FastAPI
from faster_whisper import WhisperModel
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model loading — done once at startup
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "whisper-large-v3")

logger.info("Loading faster-whisper large-v3 ...")
_model = WhisperModel(
    MODEL_DIR,
    device="cuda",
    compute_type="float16",      # float16 on GPU — fastest for quality
    num_workers=4,
)
logger.info("ASR model ready.")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI()


class ASRRequest(BaseModel):
    instances: list[dict[str, Any]]


def _transcribe_one(b64_audio: str) -> str:
    """Decode base64 WAV and transcribe with faster-whisper."""
    raw = base64.b64decode(b64_audio)
    # Write to a temp file — faster-whisper needs a file path or numpy array
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(raw)
        tmp_path = f.name
    try:
        segments, info = _model.transcribe(
            tmp_path,
            language=None,          # auto-detect (EN/MY/TA/ZH for Advanced)
            beam_size=5,
            best_of=5,
            temperature=0.0,
            vad_filter=True,        # strip silence
            vad_parameters={"min_silence_duration_ms": 300},
        )
        text = " ".join(seg.text.strip() for seg in segments)
        return text.strip()
    finally:
        os.unlink(tmp_path)


@app.post("/asr")
def asr(request: ASRRequest):
    predictions = []
    for instance in request.instances:
        transcript = _transcribe_one(instance["b64"])
        predictions.append(transcript)
        logger.info("Transcribed: %s", transcript[:80])
    return {"predictions": predictions}


@app.get("/health")
def health():
    return {"status": "ok"}
