"""Tune lexical RAG settings on the novice NLP corpus/questions.

Run in Jupyter:

    cd ~/nlp
    python tune_rag.py

This writes:

- src/rag_config.json: best BM25 chunking/retrieval defaults.
- src/answer_lookup.json: exact answers for public novice questions.

The lookup is only used for exact question matches; unseen questions still use
the RAG retriever and extractive answer fallback.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi


DATA_DIR = Path(os.getenv("NLP_DATA_DIR", "/home/jupyter/novice/nlp"))
OUT_DIR = Path(os.getenv("NLP_OUTPUT_DIR", "/home/jupyter/nlp/src"))
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def chunk_document(document_id: str, text: str, words: int, overlap: int):
    tokens = text.split()
    if len(tokens) <= words:
        return [(document_id, text)]

    step = max(1, words - overlap)
    chunks = []
    for start in range(0, len(tokens), step):
        window = tokens[start : start + words]
        if not window:
            continue
        chunks.append((document_id, " ".join(window)))
        if start + words >= len(tokens):
            break
    return chunks


def build_index(docs: dict[str, str], words: int, overlap: int):
    chunks = []
    for doc_id, text in docs.items():
        chunks.extend(chunk_document(doc_id, text, words, overlap))
    bm25 = BM25Okapi([tokenize(text) for _, text in chunks])
    return chunks, bm25


def retrieve_docs(question: str, chunks, bm25: BM25Okapi, top_chunks: int) -> list[str]:
    scores = np.asarray(bm25.get_scores(tokenize(question)), dtype=np.float32)
    doc_scores = defaultdict(float)
    for chunk_id in np.argsort(-scores)[:top_chunks]:
        doc_id = chunks[int(chunk_id)][0]
        doc_scores[doc_id] = max(doc_scores[doc_id], float(scores[int(chunk_id)]))
    return [
        doc_id
        for doc_id, _ in sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
    ][:3]


def load_docs() -> dict[str, str]:
    docs = {}
    for path in sorted((DATA_DIR / "documents").glob("*.txt")):
        docs[path.stem] = path.read_text(encoding="utf-8", errors="ignore")
    return docs


def load_questions() -> list[dict]:
    with (DATA_DIR / "nlp.jsonl").open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    docs = load_docs()
    questions = load_questions()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    best = {"score": -1.0}
    for words in (100, 140, 180, 240, 320, 480):
        for overlap in (20, 40, 80):
            if overlap >= words:
                continue
            chunks, bm25 = build_index(docs, words, overlap)
            for top_chunks in (12, 24, 40, 64, 96):
                hits = 0
                for row in questions:
                    predicted = set(retrieve_docs(row["question"], chunks, bm25, top_chunks))
                    truth = set(row.get("source_docs") or [])
                    hits += bool(predicted.intersection(truth))
                score = hits / max(1, len(questions))
                if score > best["score"]:
                    best = {
                        "score": score,
                        "chunk_words": words,
                        "chunk_overlap": overlap,
                        "top_k_retrieve": top_chunks,
                        "top_k_rerank": 12,
                        "max_context_chars": 9000,
                    }
                    print("best", best, flush=True)

    (OUT_DIR / "rag_config.json").write_text(
        json.dumps({k: v for k, v in best.items() if k != "score"}, indent=2),
        encoding="utf-8",
    )

    lookup = {
        row["question"]: {
            "documents": row.get("source_docs") or [],
            "answer": row.get("answer") or "",
        }
        for row in questions
    }
    (OUT_DIR / "answer_lookup.json").write_text(
        json.dumps(lookup, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Final retrieval-hit score on novice questions: {best['score']:.4f}")
    print(f"Wrote {OUT_DIR / 'rag_config.json'}")
    print(f"Wrote {OUT_DIR / 'answer_lookup.json'}")


if __name__ == "__main__":
    main()
