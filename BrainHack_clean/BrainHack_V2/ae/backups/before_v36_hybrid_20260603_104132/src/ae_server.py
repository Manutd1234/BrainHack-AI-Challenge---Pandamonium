from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ae_manager import AEManager

logging.basicConfig(
    level=os.environ.get("AE_LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ae_server")


class AEInstance(BaseModel):
    observation: Dict[str, Any]


class AERequest(BaseModel):
    instances: List[AEInstance]


class AEPrediction(BaseModel):
    action: int


class AEResponse(BaseModel):
    predictions: List[AEPrediction]


app = FastAPI()
manager: AEManager | None = None


@app.on_event("startup")
def _load_model() -> None:
    global manager
    logger.info("Loading AEManager...")
    manager = AEManager()
    logger.info("AEManager ready.")


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "health ok"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"message": "health ok"}


@app.post("/ae", response_model=AEResponse)
def ae(req: AERequest) -> AEResponse:
    if manager is None:
        raise HTTPException(status_code=503, detail="model not loaded yet")

    preds: List[AEPrediction] = []
    for inst in req.instances:
        try:
            action = manager.ae(inst.observation)
        except Exception as e:
            logger.exception("inference failure: %s", e)
            action = 4
        preds.append(AEPrediction(action=action))

    return AEResponse(predictions=preds)