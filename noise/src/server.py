import base64
import io
import logging
from typing import Any, List, Dict
from fastapi import FastAPI
from pydantic import BaseModel
import noisereduce as nr
import numpy as np
import scipy.io.wavfile as wavfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class NoiseRequest(BaseModel):
    instances: List[Dict[str, Any]]

def _clean_audio(b64_audio: str) -> str:
    raw = base64.b64decode(b64_audio)
    
    # Read the wav file
    rate, data = wavfile.read(io.BytesIO(raw))
    
    # Apply noise reduction
    # prop_decrease controls the amount of noise reduction (0 to 1)
    reduced_noise = nr.reduce_noise(y=data, sr=rate, prop_decrease=0.8)
    
    # Write back to bytes
    out_io = io.BytesIO()
    wavfile.write(out_io, rate, reduced_noise)
    out_io.seek(0)
    
    return base64.b64encode(out_io.read()).decode('utf-8')

@app.post("/noise")
def noise_endpoint(request: NoiseRequest):
    predictions = []
    for instance in request.instances:
        try:
            cleaned_b64 = _clean_audio(instance["b64"])
            predictions.append(cleaned_b64)
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            predictions.append("")
            
    return {"predictions": predictions}

@app.get("/health")
def health():
    return {"status": "ok"}
