"""NLP Server — port 5004
POST /nlp      Vertex AI compatible endpoint (Load / Query / Poll)
POST /load     {"documents": [...]}        → {"status": "loaded", "chunks": N}
POST /       {"question": "...", ...}    → {"answer": "...", "documents": [...], "path": "..."}
GET  /health   → {"status": "ok", "chunks": N, "qa_cache": N}
"""
import logging, time
from typing import Any, Optional
from fastapi import FastAPI, HTTPException, Request
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

@app.post("/nlp")
async def nlp_endpoint(request: Request):
    """
    Unified endpoint matching test_nlp.py's expected API signature.
    Handles load, poll, and batch queries.
    """
    try:
        body = await request.json()
    except Exception as e:
        logger.error(f"Failed to parse JSON: {e}")
        return {"predictions": ["error"]}

    instances = body.get("instances", [])
    if not instances:
        logger.error("No instances provided in request body")
        return {"predictions": ["error"]}

    first_instance = instances[0]

    # 1. Check for corpus loading
    if "documents" in first_instance:
        try:
            doc_contents = first_instance["documents"]
            logger.info(f"Received corpus load request with {len(doc_contents)} documents")
            t0 = time.perf_counter()
            # Prepare doc dicts for nlp_manager
            documents = [{"id": f"doc_{i}", "text": text} for i, text in enumerate(doc_contents)]
            manager.load_corpus(documents)
            logger.info(f"Corpus loaded successfully in {time.perf_counter()-t0:.1f}s")
            return {"predictions": ["loaded"]}
        except Exception as e:
            logger.exception("Error loading corpus")
            return {"predictions": ["error"]}

    # 2. Check for corpus load polling
    if "poll" in first_instance:
        if len(manager.chunks) > 0:
            return {"predictions": ["loaded"]}
        else:
            return {"predictions": ["loading"]}

    # 3. Batch Query handling
    predictions = []
    for inst in instances:
        question = inst.get("question")
        if not question:
            predictions.append({"answer": "", "documents": [], "path": "error"})
            continue
        try:
            t0 = time.perf_counter()
            answer, documents, path = manager.answer(question)
            logger.info(
                f"Batch query path={path} "
                f"elapsed={1000*(time.perf_counter()-t0):.0f}ms"
            )
            predictions.append({
                "answer": answer,
                "documents": documents,
                "path": path
            })
        except Exception as e:
            logger.exception(f"Error answering question: {question}")
            predictions.append({"answer": "", "documents": [], "path": "error"})

    return {"predictions": predictions}
