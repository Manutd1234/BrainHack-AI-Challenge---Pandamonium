"""ASR FastAPI server."""

from __future__ import annotations

import asyncio
import base64
import logging
import time

from fastapi import FastAPI, Request

from asr_manager import ASRManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("asr")

app = FastAPI()
manager = ASRManager()


@app.post("/asr")
async def asr(request: Request) -> dict[str, list[str]]:
    body = await request.json()
    instances = body.get("instances") or []

    payloads = []
    for inst in instances:
        b64 = inst.get("b64") if isinstance(inst, dict) else None
        payloads.append(base64.b64decode(b64) if b64 else b"")

    start = time.perf_counter()
    predictions = await asyncio.to_thread(manager.asr_many, payloads)
    elapsed = time.perf_counter() - start
    log.info("asr n=%d total=%.2fs avg=%.2fs", len(payloads), elapsed, elapsed / max(1, len(payloads)))
    return {"predictions": predictions}


@app.get("/health")
def health() -> dict[str, str]:
    return {"message": "health ok"}
