from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import FastAPI, Request
from ae_manager import AEManager

logging.basicConfig(
    level=os.environ.get("AE_LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ae_server")

app = FastAPI()
manager: AEManager | None = None


@app.on_event("startup")
def _startup() -> None:
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


@app.post("/reset")
def reset() -> dict[str, str]:
    if manager is not None and hasattr(manager, "reset"):
        manager.reset()
    return {"message": "reset ok"}


@app.post("/ae")
async def ae(request: Request) -> dict[str, list[dict[str, int]]]:
    global manager
    if manager is None:
        manager = AEManager()

    try:
        payload: dict[str, Any] = await request.json()
    except Exception:
        payload = {}

    instances = payload.get("instances") or []
    if not instances:
        if hasattr(manager, "reset"):
            manager.reset()
        return {"predictions": []}

    predictions: list[dict[str, int]] = []
    for inst in instances:
        obs = inst.get("observation", {}) if isinstance(inst, dict) else {}
        try:
            action = int(manager.ae(obs))
        except Exception as exc:
            logger.exception("AE inference failed: %s", exc)
            action = 4
        predictions.append({"action": action})

    return {"predictions": predictions}
