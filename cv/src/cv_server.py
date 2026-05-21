"""Runs the CV server."""

import base64
from typing import Any

from cv_manager import CVManager
from fastapi import FastAPI, Request

app = FastAPI()
manager = CVManager()


@app.post("/cv")
async def cv(request: Request) -> dict[str, list[list[dict[str, Any]]]]:
    """Perform CV object detection on image frames."""
    inputs_json = await request.json()

    image_payloads = [
        base64.b64decode(instance["b64"]) for instance in inputs_json["instances"]
    ]
    predictions = manager.cv_many(image_payloads)

    return {"predictions": predictions}


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint for the model."""
    return {"message": "health ok"}
