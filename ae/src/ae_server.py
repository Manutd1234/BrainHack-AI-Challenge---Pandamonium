"""AE Server — port 5005
GET  /reset → {}
POST /       {observation} → {"action": int}
GET  /health → {"status": "ok"}
"""
import logging
from typing import Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ae_manager import AEManager

logging.basicConfig(level=logging.INFO)
app     = FastAPI(title="TIL-AI 2026 AE")
manager = AEManager()

class ObsRequest(BaseModel):
    viewcone:  Any
    direction: int
    location:  list[int]
    scout:     bool = False
    step:      int  = 0

class ActionResponse(BaseModel):
    action: int

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/reset")
def reset():
    manager.reset()
    return {}

@app.post("/", response_model=ActionResponse)
def act(req: ObsRequest):
    try:
        action = manager.act(req.model_dump())
        return ActionResponse(action=action)
    except Exception as exc:
        logging.exception("Act error")
        raise HTTPException(500, str(exc))
