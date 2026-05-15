"""Runs the NLP server."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from nlp_manager import NLPManager
from pydantic import BaseModel


app = FastAPI(title="TIL-AI 2026 NLP")
manager = NLPManager()
logger = logging.getLogger(__name__)


class LoadRequest(BaseModel):
    corpus: list[str | dict[str, str]]


class QueryRequest(BaseModel):
    query: str


class _LoadState:
    """Tracks corpus-loading state for async, pollable behavior."""

    def __init__(self) -> None:
        self.status = "idle"
        self.error: str | None = None
        self.task: asyncio.Task | None = None
        self.lock = asyncio.Lock()


load_state = _LoadState()


def _normalise_load_documents(raw_documents: list[Any]) -> list[dict[str, str]]:
    documents = []
    for index, document in enumerate(raw_documents):
        if isinstance(document, str):
            documents.append({"id": f"DOC-{index:04d}", "document": document})
        else:
            documents.append(
                {
                    "id": str(document.get("id") or f"DOC-{index:04d}"),
                    "document": str(
                        document.get("document") or document.get("text") or ""
                    ),
                }
            )
    return documents


def _do_load(documents: list[dict[str, str]]) -> bool:
    manager.load_corpus(documents)
    return manager.loaded


async def _load_task(documents: list[dict[str, str]]) -> None:
    try:
        ok = await asyncio.to_thread(_do_load, documents)
        load_state.status = "loaded" if ok else "failed"
        load_state.error = None if ok else "Corpus load returned false."
    except Exception as exc:
        logger.exception("Corpus load failed")
        load_state.status = "failed"
        load_state.error = str(exc)


async def _start_load(raw_documents: list[Any]) -> dict[str, str]:
    documents = _normalise_load_documents(raw_documents)
    if not documents:
        raise HTTPException(status_code=400, detail="Corpus cannot be empty.")

    async with load_state.lock:
        if load_state.status == "loading":
            return {"status": "loading"}
        load_state.status = "loading"
        load_state.error = None
        load_state.task = asyncio.create_task(_load_task(documents))
        return {"status": load_state.status}


@app.post("/nlp")
async def nlp(request: Request) -> dict[str, list[dict[str, Any]]]:
    """Load the corpus or answer TIL-formatted NLP questions."""
    inputs_json = await request.json()
    instances = inputs_json["instances"]
    first = instances[0]

    if first.get("documents") is not None:
        status = await _start_load(first["documents"])
        return {"predictions": [status]}

    if first.get("poll") is not None:
        result = {"status": load_state.status}
        if load_state.error:
            result["error"] = load_state.error
        return {"predictions": [result]}

    if load_state.status != "loaded":
        raise HTTPException(status_code=400, detail=f"Corpus status: {load_state.status}")

    predictions = [
        await asyncio.to_thread(manager.qa, instance["question"])
        for instance in instances
    ]
    return {"predictions": predictions}


@app.post("/load")
async def load_corpus(req: LoadRequest) -> dict[str, str]:
    """Compatibility route for manually loading a corpus."""
    return await _start_load(req.corpus)


@app.post("/")
async def query_model(req: QueryRequest) -> dict[str, Any]:
    """Compatibility route matching the standalone Gemini draft."""
    if load_state.status != "loaded":
        raise HTTPException(status_code=400, detail=f"Corpus status: {load_state.status}")

    prediction = await asyncio.to_thread(manager.qa, req.query)
    return {
        "response": prediction["answer"],
        "doc_ids": prediction["documents"],
    }


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"message": "health ok"}
