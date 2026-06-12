"""Runs the AE server."""

from __future__ import annotations

import json
import logging

from ae_manager import AEManager
from fastapi import FastAPI, HTTPException, Request


logging.basicConfig(level=logging.INFO)

app = FastAPI(title="TIL-AI 2026 AE")
manager = AEManager()


@app.post("/ae")
async def ae(request: Request) -> dict[str, list[dict[str, int]]]:
    """Feeds one TIL observation into the AE policy."""
    body = await request.body()
    if not body:
        manager.reset()
        return {"predictions": [{"action": 4}]}

    try:
        input_json = json.loads(body)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON body") from exc

    predictions = []
    for instance in input_json.get("instances", []):
        observation = instance.get("observation", instance)
        predictions.append({"action": manager.ae(observation)})

    return {"predictions": predictions}


@app.post("/")
async def act(request: Request) -> dict[str, int]:
    """Compatibility route for direct observation payloads."""
    observation = await request.json()
    if "observation" in observation:
        observation = observation["observation"]
    return {"action": manager.ae(observation)}


@app.get("/reset")
@app.post("/reset")
def reset() -> dict:
    """Reset rule-agent memory between manual matches."""
    manager.reset()
    return {}


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"message": "health ok"}
