"""Build RAFT-style supervised examples from novice NLP docs/questions.

This is the NLP equivalent of "training on the dataset": each public question
is paired with gold source documents, short evidence snippets, distractor docs,
and the exact target answer. The output can be used for LoRA SFT.

Run on Jupyter:

    cd ~/nlp
    python build_raft_data.py
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi


DATA_DIR = Path(os.getenv("NLP_DATA_DIR", "/home/jupyter/novice/nlp"))
OUT_DIR = Path(os.getenv("RAFT_OUT_DIR", "/home/jupyter/nlp/raft_data"))
TRAIN_SIZE = int(os.getenv("RAFT_TRAIN_SIZE", "733"))
SEED = int(os.getenv("RAFT_SEED", "26"))
MAX_CONTEXT_CHARS = int(os.getenv("RAFT_MAX_CONTEXT_CHARS", "3600"))
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def split_sentences(text: str) -> list[str]:
    return [" ".join(part.split()).strip() for part in SENTENCE_PATTERN.split(text) if part.strip()]


def load_docs() -> dict[str, str]:
    docs = {}
    for path in sorted((DATA_DIR / "documents").glob("DOC-*.txt")):
        docs[path.stem] = path.read_text(encoding="utf-8", errors="ignore").strip()
    return docs


def load_questions() -> list[dict]:
    with (DATA_DIR / "nlp.jsonl").open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def best_snippet(question: str, answer: str, doc_text: str) -> str:
    q_tokens = set(tokenize(question))
    a_tokens = set(tokenize(answer))
    answer_key = " ".join(answer.lower().split())
    scored = []
    sentences = split_sentences(doc_text)
    for index, sentence in enumerate(sentences):
        sentence_key = " ".join(sentence.lower().split())
        tokens = set(tokenize(sentence))
        score = len(q_tokens & tokens) + 2.5 * len(a_tokens & tokens)
        if answer_key and answer_key in sentence_key:
            score += 12.0
        if score > 0:
            start = max(0, index - 1)
            end = min(len(sentences), index + 2)
            scored.append((score - 0.03 * index, " ".join(sentences[start:end])))
    if scored:
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]
    return doc_text[:900]


def build_doc_bm25(docs: dict[str, str]) -> tuple[list[str], BM25Okapi]:
    doc_ids = list(docs)
    return doc_ids, BM25Okapi([tokenize(docs[doc_id]) for doc_id in doc_ids])


def distractor_docs(question: str, truth: set[str], doc_ids: list[str], bm25: BM25Okapi, limit: int = 2) -> list[str]:
    scores = np.asarray(bm25.get_scores(tokenize(question)), dtype=np.float32)
    picks = []
    for index in np.argsort(-scores):
        doc_id = doc_ids[int(index)]
        if doc_id not in truth:
            picks.append(doc_id)
        if len(picks) >= limit:
            break
    return picks


def make_prompt(question: str, context: str) -> str:
    return (
        "Answer using only the context. Return only the final short answer. "
        "If the answer is a number, date, name, amount, score, percentage, or phrase, "
        "copy it exactly from the context when possible.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )


def make_example(row: dict, docs: dict[str, str], doc_ids: list[str], bm25: BM25Okapi) -> dict:
    question = str(row.get("question") or "").strip()
    answer = str(row.get("answer") or "").strip()
    source_docs = [str(doc) for doc in row.get("source_docs") or [] if str(doc) in docs]
    truth = set(source_docs)
    selected_docs = source_docs + distractor_docs(question, truth, doc_ids, bm25)
    parts = []
    current = 0
    for doc_id in selected_docs:
        snippet = best_snippet(question, answer, docs[doc_id])
        part = f"[{doc_id}]\n{snippet}"
        if parts and current + len(part) > MAX_CONTEXT_CHARS:
            break
        parts.append(part)
        current += len(part)
    context = "\n\n".join(parts)
    prompt = make_prompt(question, context)
    return {
        "question": question,
        "answer": answer,
        "documents": source_docs[:3],
        "prompt": prompt,
        "text": f"{prompt} {answer}",
    }


def main() -> None:
    docs = load_docs()
    rows = [row for row in load_questions() if row.get("question") and row.get("answer")]
    doc_ids, bm25 = build_doc_bm25(docs)
    examples = [make_example(row, docs, doc_ids, bm25) for row in rows]
    random.Random(SEED).shuffle(examples)
    train = examples[: min(TRAIN_SIZE, len(examples))]
    valid = examples[len(train) :]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", train), ("eval", valid)):
        with (OUT_DIR / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for example in split:
                handle.write(json.dumps(example, ensure_ascii=False) + "\n")
    print(f"docs={len(docs)} examples={len(examples)} train={len(train)} eval={len(valid)}")
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
