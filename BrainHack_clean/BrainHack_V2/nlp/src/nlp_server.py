"""NLP server — synchronous corpus load, plain-string predictions."""
from __future__ import annotations

import asyncio
import logging

from fastapi import FastAPI, Request

from nlp_manager import NLPManager

app     = FastAPI()
manager = NLPManager()
logger  = logging.getLogger(__name__)


@app.post("/nlp")
async def nlp(request: Request) -> dict:
    body:      dict = await request.json()
    instances: list = body.get("instances", [])

    if not instances:
        return {"predictions": []}

    first = instances[0]

    # ------------------------------------------------------------------
    # Corpus load — block until done, then return "loaded" as a string.
    # test_nlp.py checks: predictions[0] == "loaded"
    # ------------------------------------------------------------------
    if "documents" in first:
        await asyncio.to_thread(manager.load_corpus, first["documents"])
        return {"predictions": ["loaded"]}

    # ------------------------------------------------------------------
    # QA — run all questions in parallel on the thread pool.
    # Returns plain strings, not dicts.
    # ------------------------------------------------------------------
    answers: list[str] = list(
        await asyncio.gather(
            *[
                asyncio.to_thread(manager.qa, inst.get("question", ""))
                for inst in instances
            ]
        )
    )
    return {"predictions": answers}


@app.get("/health")
def health() -> dict[str, str]:
    return {"message": "health ok"}