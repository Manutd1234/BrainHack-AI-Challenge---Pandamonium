"""Runs the adversarial noising server."""

from __future__ import annotations

import base64
import logging

from fastapi import FastAPI, HTTPException, Request
from noise_manager import NoiseManager
from pydantic import BaseModel


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

app = FastAPI(title="TIL-AI 2026 Noise")
manager = NoiseManager()


class NoiseRequest(BaseModel):
    b64_img: str


class NoiseResponse(BaseModel):
    b64_img: str


@app.post("/noise")
async def noise(request: Request) -> dict[str, list[str]]:
    """Apply adversarial noising to TIL-formatted image instances."""
    inputs_json = await request.json()

    predictions = []
    for instance in inputs_json["instances"]:
        image_bytes = base64.b64decode(instance["b64"])
        predictions.append(manager.noise(image_bytes))

    return {"predictions": predictions}


@app.post("/", response_model=NoiseResponse)
def noise_compat(req: NoiseRequest) -> NoiseResponse:
    """Compatibility route for the standalone draft API."""
    if not req.b64_img:
        raise HTTPException(status_code=400, detail="b64_img is empty")
    try:
        return NoiseResponse(b64_img=manager.add_noise(req.b64_img))
    except Exception as exc:
        LOGGER.exception("Noise error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"message": "health ok"}
