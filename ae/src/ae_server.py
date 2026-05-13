"""Runs the AE server."""

from __future__ import annotations

from ae_manager import AEManager
from fastapi import FastAPI, Request

app = FastAPI()
manager = AEManager()


def _reset_manager() -> None:
    global manager
    manager = AEManager()


@app.post("/ae")
async def ae(request: Request) -> dict[str, list[dict[str, int]]]:
    """Feeds an observation into the model and returns the selected action."""
    try:
        input_json = await request.json()
    except Exception:
        _reset_manager()
        return {"predictions": []}

    instances = input_json.get("instances", [])
    if not instances:
        _reset_manager()
        return {"predictions": []}

    predictions = []
    for instance in instances:
        observation = instance["observation"]
        if observation.get("step", 0) == 0:
            _reset_manager()
        predictions.append({"action": manager.ae(observation)})
    return {"predictions": predictions}


@app.get("/reset")
@app.post("/reset")
async def reset() -> dict[str, str]:
    """Resets the AE manager for a new round."""
    _reset_manager()
    return {"message": "reset ok"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"message": "health ok"}
