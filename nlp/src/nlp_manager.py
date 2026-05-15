"""Qwen3-AWQ RAG manager for the TIL-AI 2026 NLP challenge."""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from rank_bm25 import BM25Okapi
from transformers import AutoModelForCausalLM, AutoTokenizer


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class Chunk:
    document_id: str
    text: str


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


class NLPManager:
    """Hybrid BM25 + BGE-M3 + reranker retrieval with Qwen3 generation."""

    loaded = False

    def __init__(self) -> None:
        self.model_path = os.getenv("QWEN_MODEL_PATH", "./qwen-quantized")
        self.embedding_model_id = os.getenv("NLP_EMBEDDING_MODEL", "BAAI/bge-m3")
        self.reranker_model_id = os.getenv(
            "NLP_RERANKER_MODEL",
            "BAAI/bge-reranker-large",
        )
        self.chunk_words = int(os.getenv("NLP_CHUNK_WORDS", "360"))
        self.chunk_overlap = int(os.getenv("NLP_CHUNK_OVERLAP", "60"))
        self.top_k_retrieve = int(os.getenv("NLP_TOP_K_RETRIEVE", "12"))
        self.top_k_rerank = int(os.getenv("NLP_TOP_K_RERANK", "4"))
        self.max_context_chars = int(os.getenv("NLP_MAX_CONTEXT_CHARS", "7000"))
        self.max_new_tokens = int(os.getenv("NLP_MAX_NEW_TOKENS", "256"))
        self.enable_thinking = _env_flag("QWEN_ENABLE_THINKING", False)
        self.do_sample = _env_flag("QWEN_DO_SAMPLE", False)
        self.lock = threading.Lock()

        use_fp16 = torch.cuda.is_available()
        print(f"Loading embedding model: {self.embedding_model_id}", flush=True)
        self.embedding_model = BGEM3FlagModel(self.embedding_model_id, use_fp16=use_fp16)

        print(f"Loading reranker: {self.reranker_model_id}", flush=True)
        self.reranker = FlagReranker(self.reranker_model_id, use_fp16=use_fp16)

        print(f"Loading quantized Qwen3 model from {self.model_path}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self.llm.eval()

        self.documents: dict[str, str] = {}
        self.chunks: list[Chunk] = []
        self.bm25: BM25Okapi | None = None
        self.dense_embeddings: np.ndarray | None = None

    def load_corpus(self, documents: list[dict[str, str]]) -> None:
        """Load challenge documents and build sparse and dense retrieval indexes."""
        self.documents = self._normalise_documents(documents)
        self.chunks = self._chunk_documents(self.documents)

        tokenized_chunks = [self._tokenize(chunk.text) for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)

        dense_embeddings = self.embedding_model.encode(
            [chunk.text for chunk in self.chunks],
            batch_size=int(os.getenv("NLP_EMBED_BATCH_SIZE", "12")),
        )["dense_vecs"]
        self.dense_embeddings = self._normalise_matrix(np.asarray(dense_embeddings))
        self.loaded = True
        print(
            f"Loaded {len(self.documents)} documents into {len(self.chunks)} chunks.",
            flush=True,
        )

    def qa(self, question: str) -> dict[str, list[str] | str]:
        """Answer one question and return relevant document IDs."""
        if not self.loaded or self.bm25 is None or self.dense_embeddings is None:
            return {"documents": [], "answer": ""}

        candidate_chunk_ids = self._retrieve(question)
        reranked_chunk_ids = self._rerank(question, candidate_chunk_ids)
        context_chunks = [self.chunks[index] for index in reranked_chunk_ids]
        document_ids = self._unique_document_ids(context_chunks, limit=3)
        answer = self._generate(question, context_chunks)
        return {"documents": document_ids, "answer": answer}

    def _normalise_documents(self, documents: list[dict[str, str]]) -> dict[str, str]:
        normalised = {}
        for index, document in enumerate(documents):
            document_id = str(document.get("id") or f"DOC-{index:04d}")
            text = str(document.get("document") or document.get("text") or "")
            if text.strip():
                normalised[document_id] = text.strip()
        return normalised

    def _chunk_documents(self, documents: dict[str, str]) -> list[Chunk]:
        chunks: list[Chunk] = []
        step = max(1, self.chunk_words - self.chunk_overlap)

        for document_id, text in documents.items():
            words = text.split()
            if not words:
                continue
            if len(words) <= self.chunk_words:
                chunks.append(Chunk(document_id=document_id, text=text))
                continue

            for start in range(0, len(words), step):
                window = words[start : start + self.chunk_words]
                if not window:
                    continue
                chunks.append(
                    Chunk(document_id=document_id, text=" ".join(window).strip())
                )
                if start + self.chunk_words >= len(words):
                    break

        return chunks or [Chunk(document_id="DOC-0000", text="")]

    def _retrieve(self, question: str) -> list[int]:
        tokenized_query = self._tokenize(question)
        bm25_scores = np.asarray(self.bm25.get_scores(tokenized_query), dtype=np.float32)

        query_embedding = self.embedding_model.encode([question])["dense_vecs"]
        query_embedding = self._normalise_matrix(np.asarray(query_embedding))[0]
        dense_scores = self.dense_embeddings @ query_embedding

        return self._rrf(bm25_scores, dense_scores)[: self.top_k_retrieve]

    def _rrf(
        self,
        bm25_scores: np.ndarray,
        dense_scores: np.ndarray,
        k: int = 60,
    ) -> list[int]:
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        for rank, chunk_id in enumerate(np.argsort(-bm25_scores)):
            scores[chunk_id] += 1.0 / (k + rank + 1)
        for rank, chunk_id in enumerate(np.argsort(-dense_scores)):
            scores[chunk_id] += 1.0 / (k + rank + 1)
        return np.argsort(-scores).tolist()

    def _rerank(self, question: str, candidate_chunk_ids: list[int]) -> list[int]:
        if not candidate_chunk_ids:
            return []

        pairs = [[question, self.chunks[index].text] for index in candidate_chunk_ids]
        scores = self.reranker.compute_score(pairs)
        if isinstance(scores, (float, int)):
            scores = [float(scores)]

        scored = sorted(
            zip(candidate_chunk_ids, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return [chunk_id for chunk_id, _ in scored[: self.top_k_rerank]]

    def _generate(self, question: str, context_chunks: list[Chunk]) -> str:
        context = self._format_context(context_chunks)
        prompt = (
            "Answer the question using only the context below. "
            "If the context is insufficient, say you do not know. "
            "Keep the answer concise.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        messages = [{"role": "user", "content": prompt}]
        text = self._apply_chat_template(messages)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.llm.device)

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if self.do_sample:
            generation_kwargs.update(
                {
                    "temperature": float(os.getenv("QWEN_TEMPERATURE", "0.7")),
                    "top_p": float(os.getenv("QWEN_TOP_P", "0.8")),
                    "top_k": int(os.getenv("QWEN_TOP_K", "20")),
                }
            )

        with self.lock, torch.inference_mode():
            outputs = self.llm.generate(**inputs, **generation_kwargs)

        generated_ids = outputs[0][inputs["input_ids"].shape[-1] :]
        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return self._strip_thinking(answer)

    def _apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def _format_context(self, chunks: list[Chunk]) -> str:
        parts = []
        current_length = 0
        for chunk in chunks:
            part = f"[{chunk.document_id}]\n{chunk.text}"
            if parts and current_length + len(part) > self.max_context_chars:
                break
            parts.append(part)
            current_length += len(part)
        return "\n\n".join(parts)

    def _unique_document_ids(self, chunks: list[Chunk], limit: int) -> list[str]:
        document_ids = []
        for chunk in chunks:
            if chunk.document_id not in document_ids:
                document_ids.append(chunk.document_id)
            if len(document_ids) >= limit:
                break
        return document_ids

    def _tokenize(self, text: str) -> list[str]:
        return TOKEN_PATTERN.findall(text.lower())

    def _normalise_matrix(self, matrix: np.ndarray) -> np.ndarray:
        matrix = matrix.astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.clip(norms, 1e-12, None)

    def _strip_thinking(self, answer: str) -> str:
        answer = answer.strip()
        if "</think>" in answer:
            answer = answer.split("</think>", 1)[1]
        answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL)
        return " ".join(answer.strip().split())
