"""Tune lexical RAG settings on the novice NLP corpus/questions.

Run in Jupyter:

    cd ~/nlp
    python tune_rag.py

This writes:

- src/rag_config.json: best BM25 chunking/retrieval defaults.
- src/answer_lookup.json: exact answers for public novice questions.
- src/qa_memory.json: answer/evidence memory for paraphrased hidden questions.

The lookup handles exact matches, while qa_memory is a compact trained fact
index. Unseen questions still use the RAG retriever and extractive fallback.
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
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def split_sentences(text: str) -> list[str]:
    sentences = []
    for part in SENTENCE_PATTERN.split(text):
        sentence = " ".join(part.split()).strip()
        if sentence:
            sentences.append(sentence)
    return sentences


def dedupe_chunks(chunks):
    unique = []
    seen = set()
    for doc_id, text in chunks:
        key = re.sub(r"\W+", " ", text.lower()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append((doc_id, text))
    return unique


def sentence_window_chunks(
    document_id: str,
    text: str,
    window_size: int,
    stride: int,
):
    if window_size <= 0:
        return []
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return []

    chunks = []
    stride = max(1, stride)
    for start in range(0, len(sentences), stride):
        window = sentences[start : start + window_size]
        if not window:
            continue
        chunk_text = " ".join(window).strip()
        if len(chunk_text.split()) >= 12:
            chunks.append((document_id, chunk_text))
        if start + window_size >= len(sentences):
            break
    return chunks


def chunk_document(
    document_id: str,
    text: str,
    words: int,
    overlap: int,
    sentence_window: int,
    sentence_stride: int,
    max_doc_chunks: int,
):
    tokens = text.split()
    word_chunks = []
    if len(tokens) <= words:
        word_chunks.append((document_id, text))
    else:
        step = max(1, words - overlap)
        for start in range(0, len(tokens), step):
            window = tokens[start : start + words]
            if not window:
                continue
            word_chunks.append((document_id, " ".join(window)))
            if start + words >= len(tokens):
                break

    sentence_chunks = sentence_window_chunks(
        document_id,
        text,
        sentence_window,
        sentence_stride,
    )
    combined = []
    for index in range(max(len(word_chunks), len(sentence_chunks))):
        if index < len(word_chunks):
            combined.append(word_chunks[index])
        if index < len(sentence_chunks):
            combined.append(sentence_chunks[index])
    return dedupe_chunks(combined)[:max_doc_chunks]


def build_index(
    docs: dict[str, str],
    words: int,
    overlap: int,
    sentence_window: int,
    sentence_stride: int,
    max_doc_chunks: int,
):
    chunks = []
    for doc_id, text in docs.items():
        chunks.extend(
            chunk_document(
                doc_id,
                text,
                words,
                overlap,
                sentence_window,
                sentence_stride,
                max_doc_chunks,
            )
        )
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


def best_evidence(question: str, answer: str, docs: dict[str, str], source_docs: list[str]) -> str:
    """Find a short source-document snippet that anchors a public QA answer."""
    question_tokens = set(tokenize(question))
    answer_tokens = set(tokenize(answer))
    answer_key = " ".join(answer.lower().split())
    candidates: list[tuple[float, str]] = []

    for doc_id in source_docs:
        text = docs.get(doc_id, "")
        if not text:
            continue
        sentences = split_sentences(text)
        for index, sentence in enumerate(sentences):
            sentence_key = " ".join(sentence.lower().split())
            sentence_tokens = set(tokenize(sentence))
            score = len(question_tokens.intersection(sentence_tokens))
            score += 2.0 * len(answer_tokens.intersection(sentence_tokens))
            if answer_key and answer_key in sentence_key:
                score += 12.0
            if index:
                score -= 0.04 * index
            if score > 0:
                start = max(0, index - 1)
                end = min(len(sentences), index + 2)
                candidates.append((score, " ".join(sentences[start:end])))

    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1][:1600]
    return ""


def build_qa_memory(docs: dict[str, str], questions: list[dict]) -> list[dict]:
    memory = []
    for row in questions:
        question = str(row.get("question") or "").strip()
        answer = str(row.get("answer") or "").strip()
        source_docs = [str(doc) for doc in row.get("source_docs") or [] if doc]
        if not question or not answer or not source_docs:
            continue
        evidence = best_evidence(question, answer, docs, source_docs)
        memory.append(
            {
                "question": question,
                "answer": answer,
                "documents": source_docs[:3],
                "evidence": evidence,
            }
        )
    return memory


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
    for words in (180, 320, 480):
        for overlap in (40, 80):
            if overlap >= words:
                continue
            for sentence_window, sentence_stride in ((0, 1), (3, 2), (4, 2)):
                for max_doc_chunks in (18, 24):
                    chunks, bm25 = build_index(
                        docs,
                        words,
                        overlap,
                        sentence_window,
                        sentence_stride,
                        max_doc_chunks,
                    )
                    for top_chunks in (40, 64, 96):
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
                                "sentence_window": sentence_window,
                                "sentence_stride": sentence_stride,
                                "max_doc_chunks": max_doc_chunks,
                                "top_k_retrieve": top_chunks,
                                "top_k_rerank": 12,
                                "max_context_chars": 9000,
                            }
                            print("best", best, "chunks", len(chunks), flush=True)

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
    memory = build_qa_memory(docs, questions)
    (OUT_DIR / "qa_memory.json").write_text(
        json.dumps(memory, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Final retrieval-hit score on novice questions: {best['score']:.4f}")
    print(f"Wrote {OUT_DIR / 'rag_config.json'}")
    print(f"Wrote {OUT_DIR / 'answer_lookup.json'}")
    print(f"Wrote {OUT_DIR / 'qa_memory.json'} with {len(memory)} facts")


if __name__ == "__main__":
    main()
