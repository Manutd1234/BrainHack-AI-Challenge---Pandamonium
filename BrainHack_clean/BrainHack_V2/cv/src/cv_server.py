from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import FastAPI, Request

from cv_manager import CVManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TIL-AI 2026 CV")
manager: Optional[CVManager] = None


@app.on_event("startup")
def _startup() -> None:
    global manager
    manager = CVManager()
    logger.info("CV manager initialized")


def _get_manager() -> CVManager:
    if manager is None:
        raise RuntimeError("CV manager not initialized")
    return manager


@app.get("/health")
def health():
    return {"message": "health ok"}


@app.post("/cv")
async def cv(request: Request) -> dict[str, list[list[dict[str, Any]]]]:
    body = await request.json()
    instances = body.get("instances", [])
    if not instances:
        return {"predictions": []}

    preds = _get_manager().predict_b64_batch(instances)
    return {"predictions": preds}
