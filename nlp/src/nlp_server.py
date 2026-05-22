"""NLP Server — port 5004
POST /load   {"documents": [...]}        → {"status": "loaded", "chunks": N}
POST /       {"question": "...", ...}    → {"answer": "...", "documents": [...], "path": "..."}
GET  /health → {"status": "ok", "chunks": N, "qa_cache": N}
"""
import logging, time
from typing import Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from nlp_manager import NLPManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app     = FastAPI(title="TIL-AI 2026 NLP")
manager = NLPManager()

class LoadRequest(BaseModel):
    documents: list[dict[str, Any]]

class QueryRequest(BaseModel):
    question:  str
    query_id:  Optional[str] = None

class QueryResponse(BaseModel):
    answer:    str
    documents: list[str]
    path:      str = "unknown"

@app.get("/health")
def health():
    return {
        "status":   "ok",
        "chunks":   len(manager.chunks),
        "qa_cache": manager.qa_index.ntotal if manager.qa_index else 0,
    }

@app.post("/load")
@app.post("/corpus")
def load(req: LoadRequest):
    if not req.documents:
        raise HTTPException(400, "documents list is empty")
    t0 = time.perf_counter()
    manager.load_corpus(req.documents)
    logger.info(f"Corpus loaded in {time.perf_counter()-t0:.1f}s")
    return {"status": "loaded", "chunks": len(manager.chunks)}

@app.post("/", response_model=QueryResponse)
@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question:
        raise HTTPException(400, "question is empty")
    try:
        t0 = time.perf_counter()
        answer, documents, path = manager.answer(req.question)
        logger.info(
            f"[{req.query_id or '?'}] path={path} "
            f"elapsed={1000*(time.perf_counter()-t0):.0f}ms"
        )
        return QueryResponse(answer=answer, documents=documents, path=path)
    except Exception as exc:
        logger.exception("Query error")
        raise HTTPException(500, str(exc))
