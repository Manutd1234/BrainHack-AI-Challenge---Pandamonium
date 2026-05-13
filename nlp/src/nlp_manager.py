"""Manages the NLP model: hybrid RAG with Qwen3.6 generation.

The qualifier endpoint receives a corpus at runtime, so this manager builds a
fresh dense+sparse index per corpus. At answer time it sends the fused context
to a local OpenAI-compatible vLLM server running Qwen3.6. If that server is not
available in a lightweight local test environment, it falls back to extractive
answer selection instead of failing the endpoint.
"""

from __future__ import annotations

from collections import defaultdict
import logging
import os
import re
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_sentence_model = None
_llm_client = None
_llm_unavailable_logged = False


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _get_sentence_model(model_name: str):
    """Lazy-load the BGE-M3 embedding model."""
    global _sentence_model
    if _sentence_model is None:
        from sentence_transformers import SentenceTransformer

        try:
            _sentence_model = SentenceTransformer(
                model_name,
                trust_remote_code=True,
            )
        except TypeError:
            _sentence_model = SentenceTransformer(model_name)
        logger.info("Loaded embedding model: %s", model_name)
    return _sentence_model


def _get_llm_client(base_url: str):
    """Lazy-load an OpenAI-compatible client for the local vLLM server."""
    global _llm_client
    if _llm_client is None:
        from openai import OpenAI

        _llm_client = OpenAI(
            base_url=base_url,
            api_key=os.getenv("QWEN_API_KEY", "EMPTY"),
            timeout=float(os.getenv("QWEN_TIMEOUT_SECONDS", "45")),
        )
        logger.info("Configured Qwen client at %s", base_url)
    return _llm_client


class NLPManager:
    loaded = False

    def __init__(self):
        self.embedding_model_name = os.getenv("NLP_EMBEDDING_MODEL", "BAAI/bge-m3")
        self.qwen_model = os.getenv("QWEN_MODEL", "Qwen/Qwen3.6-27B")
        self.qwen_base_url = os.getenv("QWEN_BASE_URL", "http://127.0.0.1:8000/v1")
        self.use_llm = _env_bool("NLP_USE_LLM", True)

        self.chunks: list[str] = []
        self.chunk_sources: list[int] = []
        self.embeddings: Optional[np.ndarray] = None
        self.bm25 = None

        # Larger chunks are intentional for Qwen3.6's long-context reasoning.
        # This is an approximate word-token budget; the prompt builder also
        # enforces a character budget before generation.
        self.chunk_size_tokens = int(os.getenv("NLP_CHUNK_TOKENS", "1024"))
        self.chunk_overlap_tokens = int(os.getenv("NLP_CHUNK_OVERLAP_TOKENS", "160"))
        self.top_k_retrieve = int(os.getenv("NLP_TOP_K_RETRIEVE", "18"))
        self.top_k_context = int(os.getenv("NLP_TOP_K_CONTEXT", "8"))
        self.max_context_chars = int(os.getenv("NLP_MAX_CONTEXT_CHARS", "28000"))

    def _chunk_text(self, text: str, doc_id: int) -> list[tuple[str, int]]:
        """Splits text into overlapping, retrieval-sized chunks."""
        words = text.split()
        if not words:
            return []

        step = max(1, self.chunk_size_tokens - self.chunk_overlap_tokens)
        chunks: list[tuple[str, int]] = []
        for start in range(0, len(words), step):
            end = min(len(words), start + self.chunk_size_tokens)
            chunk = " ".join(words[start:end]).strip()
            if len(chunk) > 20:
                chunks.append((chunk, doc_id))
            if end == len(words):
                break
        return chunks

    def load_corpus(self, documents: list[str]) -> None:
        """Loads and indexes the corpus of documents for RAG QA."""
        logger.info("Loading corpus of %d documents...", len(documents))

        self.chunks = []
        self.chunk_sources = []
        for doc_id, doc in enumerate(documents):
            for chunk_text, source_id in self._chunk_text(doc, doc_id):
                self.chunks.append(chunk_text)
                self.chunk_sources.append(source_id)

        logger.info(
            "Created %d chunks from %d documents.",
            len(self.chunks),
            len(documents),
        )

        if not self.chunks:
            self.embeddings = np.empty((0, 0), dtype=np.float32)
            self.loaded = True
            return

        model = _get_sentence_model(self.embedding_model_name)
        logger.info("Encoding chunks with %s...", self.embedding_model_name)
        embeddings = model.encode(
            self.chunks,
            show_progress_bar=False,
            batch_size=int(os.getenv("NLP_EMBED_BATCH_SIZE", "32")),
            normalize_embeddings=True,
        )
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        logger.info("Embeddings shape: %s", self.embeddings.shape)

        try:
            from rank_bm25 import BM25Okapi

            tokenized_chunks = [self._bm25_tokens(chunk) for chunk in self.chunks]
            self.bm25 = BM25Okapi(tokenized_chunks)
            logger.info("BM25 index built successfully.")
        except ImportError:
            logger.warning("rank_bm25 not installed, using dense-only retrieval.")
            self.bm25 = None

        self.loaded = True

    @staticmethod
    def _bm25_tokens(text: str) -> list[str]:
        return re.findall(r"[A-Za-z0-9_]+", text.lower())

    def _retrieve_dense(self, question: str, top_k: int) -> list[tuple[int, float]]:
        """Retrieve top-k chunks using dense embedding similarity."""
        if self.embeddings is None or len(self.embeddings) == 0:
            return []

        model = _get_sentence_model(self.embedding_model_name)
        q_emb = model.encode([question], normalize_embeddings=True)
        q_emb = np.asarray(q_emb, dtype=np.float32)
        similarities = np.dot(self.embeddings, q_emb.T).flatten()
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [
            (int(idx), float(similarities[idx]))
            for idx in top_indices
            if similarities[idx] > 0.0
        ]

    def _retrieve_bm25(self, question: str, top_k: int) -> list[tuple[int, float]]:
        """Retrieve top-k chunks using BM25 keyword matching."""
        if self.bm25 is None:
            return []

        scores = self.bm25.get_scores(self._bm25_tokens(question))
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [
            (int(idx), float(scores[idx]))
            for idx in top_indices
            if scores[idx] > 0.0
        ]

    def _fuse_results(
        self,
        dense_results: list[tuple[int, float]],
        bm25_results: list[tuple[int, float]],
    ) -> list[int]:
        """Fuse dense and sparse retrieval with reciprocal-rank fusion."""
        scores: defaultdict[int, float] = defaultdict(float)
        for results, weight in ((dense_results, 1.0), (bm25_results, 0.85)):
            for rank, (idx, _score) in enumerate(results, start=1):
                scores[idx] += weight / (60.0 + rank)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [idx for idx, _score in ranked]

    def _build_context(self, candidate_indices: list[int]) -> str:
        selected: list[str] = []
        total_chars = 0

        for rank, idx in enumerate(candidate_indices[: self.top_k_context], start=1):
            chunk = self.chunks[idx]
            block = f"[{rank}] {chunk}"
            if selected and total_chars + len(block) > self.max_context_chars:
                break
            selected.append(block)
            total_chars += len(block)

        return "\n\n".join(selected)

    def _build_prompt(self, question: str, context: str) -> list[dict[str, str]]:
        system_prompt = (
            "You answer questions about the Clairos corpus. Use only the "
            "provided context. If the answer is not present, or the question "
            "contains a false premise, return an empty string. Return only the "
            "final answer, with no citations or explanation."
        )
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _call_llm(self, question: str, context: str) -> Optional[str]:
        """Calls Qwen3.6 through vLLM. Returns None when unavailable."""
        global _llm_unavailable_logged
        if not self.use_llm:
            return None

        try:
            client = _get_llm_client(self.qwen_base_url)
            response = client.chat.completions.create(
                model=self.qwen_model,
                messages=self._build_prompt(question, context),
                max_tokens=int(os.getenv("QWEN_MAX_NEW_TOKENS", "512")),
                temperature=float(os.getenv("QWEN_TEMPERATURE", "0.2")),
                top_p=float(os.getenv("QWEN_TOP_P", "0.8")),
                presence_penalty=float(os.getenv("QWEN_PRESENCE_PENALTY", "1.5")),
                extra_body={
                    "top_k": int(os.getenv("QWEN_TOP_K", "20")),
                    "chat_template_kwargs": {
                        "enable_thinking": False,
                        "preserve_thinking": True,
                    },
                },
            )
            content = response.choices[0].message.content or ""
            return self._clean_llm_answer(content)
        except Exception as exc:
            if not _llm_unavailable_logged:
                logger.warning(
                    "Qwen generation unavailable, using extractive fallback: %s",
                    exc,
                )
                _llm_unavailable_logged = True
            return None

    @staticmethod
    def _clean_llm_answer(answer: str) -> str:
        answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL)
        answer = answer.strip()
        answer = re.sub(r"^(final\s+answer|answer)\s*:\s*", "", answer, flags=re.I)
        if answer in {'""', "''", "N/A", "n/a", "None", "none"}:
            return ""
        if (
            (answer.startswith('"') and answer.endswith('"'))
            or (answer.startswith("'") and answer.endswith("'"))
        ):
            answer = answer[1:-1].strip()
        return answer[:1200]

    def _extract_answer(self, question: str, context: str) -> str:
        """Extract a likely answer span when the LLM runtime is unavailable."""
        sentences = re.split(r"(?<=[.!?])\s+", context)
        sentences = [
            re.sub(r"^\[\d+\]\s*", "", sentence).strip()
            for sentence in sentences
            if len(sentence.strip()) > 10
        ]

        if not sentences:
            return ""
        if len(sentences) == 1:
            return sentences[0][:500]

        model = _get_sentence_model(self.embedding_model_name)
        q_emb = model.encode([question], normalize_embeddings=True)
        s_embs = model.encode(sentences, normalize_embeddings=True)
        sims = np.dot(np.asarray(s_embs), np.asarray(q_emb).T).flatten()

        max_sim = float(np.max(sims))
        if max_sim < float(os.getenv("NLP_UNANSWERABLE_SIM_THRESHOLD", "0.28")):
            return ""

        top_indices = np.argsort(sims)[-3:][::-1]
        best_sentences = [sentences[i] for i in top_indices if sims[i] > 0.15]
        if not best_sentences:
            return ""

        answer = " ".join(best_sentences[:2]).strip()
        answer = re.sub(
            r"^(Additionally|However|Therefore|Furthermore|Indeed|"
            r"Specifically|In addition|Moreover),\s*",
            "",
            answer,
            flags=re.IGNORECASE,
        )
        return re.sub(r"\s+", " ", answer).strip()[:800]

    def qa(self, question: str) -> str:
        """Performs question answering using hybrid retrieval + Qwen3.6."""
        if not self.loaded:
            return ""

        dense_results = self._retrieve_dense(question, self.top_k_retrieve)
        bm25_results = self._retrieve_bm25(question, self.top_k_retrieve)
        candidate_indices = self._fuse_results(dense_results, bm25_results)

        if not candidate_indices:
            return ""

        context = self._build_context(candidate_indices)
        answer = self._call_llm(question, context)
        if answer is not None:
            return answer
        return self._extract_answer(question, context)
