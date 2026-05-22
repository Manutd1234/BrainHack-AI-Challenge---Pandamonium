"""AE Server — port 5005
POST /ae      Vertex AI compatible endpoint (Act / Reset)
GET  /health  → {"status": "ok"}
GET  /reset   → {"status": "ok"}
"""
import logging
from typing import Any, Optional
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from ae_manager import AEManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app     = FastAPI(title="TIL-AI 2026 AE")
manager = AEManager()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/reset")
def reset_get():
    manager.reset()
    return {"status": "ok"}

@app.post("/reset")
def reset_post():
    manager.reset()
    return {"status": "ok"}

@app.post("/ae")
async def ae_endpoint(request: Request):
    """
    Unified endpoint matching test_ae.py's expected API signature.
    Handles both resets (empty POST) and action predictions.
    """
    try:
        body = await request.json()
    except Exception:
        # Empty body or non-JSON is treated as a reset in test_ae.py
        logger.info("Empty POST or non-JSON body received at /ae -> Resetting environment")
        manager.reset()
        return {"status": "reset"}

    instances = body.get("instances", [])
    if not instances:
        logger.info("No instances provided in /ae POST body -> Resetting environment")
        manager.reset()
        return {"status": "reset"}

    try:
        first_instance = instances[0]
        obs = first_instance.get("observation")
        if not obs:
            logger.warning("No observation found in instance -> Resetting environment")
            manager.reset()
            return {"status": "reset"}
            
        action = manager.act(obs)
        return {
            "predictions": [
                {"action": action}
            ]
        }
    except Exception as exc:
        logger.exception("AE act error")
        return {"predictions": [{"action": 0}], "error": str(exc)}

# Also maintain root fallback for backward compatibility
@app.post("/")
async def root_endpoint(request: Request):
    return await ae_endpoint(request)
