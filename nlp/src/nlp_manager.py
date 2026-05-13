"""Manages the NLP model — Speed-Optimized RAG Question Answering.

Pipeline (speed-first):
  1. Dense embeddings via sentence-transformers (all-MiniLM-L6-v2, 22MB)
  2. BM25 sparse retrieval (zero GPU, instant)
  3. Cross-encoder reranking for precision (MiniLM, 22MB)
  4. Extractive answer selection (no LLM — saves ~30s startup + GPU memory)

Strategy: The 75/25 accuracy/speed ratio heavily rewards fast inference.
Dropping the 3B LLM eliminates:
  - ~30s container startup time
  - ~2GB GPU memory
  - ~500ms per-question generation latency
The cross-encoder + extractive approach is ~10x faster per query.
"""

import logging
import re
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Lazy-loaded models
_sentence_model = None
_cross_encoder = None


def _get_sentence_model():
    """Lazy-load the sentence-transformer embedding model."""
    global _sentence_model
    if _sentence_model is None:
        from sentence_transformers import SentenceTransformer
        _sentence_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        logger.info("Loaded sentence-transformer embedding model.")
    return _sentence_model


def _get_cross_encoder():
    """Lazy-load the cross-encoder reranker model."""
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        logger.info("Loaded cross-encoder reranker model.")
    return _cross_encoder


class NLPManager:
    loaded = False

    def __init__(self):
        self.chunks: list[str] = []
        self.chunk_sources: list[int] = []
        self.embeddings: Optional[np.ndarray] = None
        self.bm25 = None
        self.chunk_size = 400       # Smaller chunks = more precise retrieval
        self.chunk_overlap = 80
        self.top_k_retrieve = 10    # Fewer candidates = faster
        self.top_k_rerank = 3       # Focus on top 3 for answer extraction

    def _chunk_text(self, text: str, doc_id: int) -> list[tuple[str, int]]:
        """Splits text into overlapping chunks for retrieval."""
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_chunk = []
        current_len = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(para) > self.chunk_size:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue
                    if current_len + len(sent) > self.chunk_size and current_chunk:
                        chunk_text = " ".join(current_chunk)
                        chunks.append((chunk_text, doc_id))
                        overlap_words = chunk_text.split()
                        keep = max(1, len(overlap_words) // 5)
                        overlap = " ".join(overlap_words[-keep:])
                        current_chunk = [overlap]
                        current_len = len(overlap)
                    current_chunk.append(sent)
                    current_len += len(sent) + 1
            else:
                if current_len + len(para) > self.chunk_size and current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append((chunk_text, doc_id))
                    overlap_words = chunk_text.split()
                    keep = max(1, len(overlap_words) // 5)
                    overlap = " ".join(overlap_words[-keep:])
                    current_chunk = [overlap]
                    current_len = len(overlap)
                current_chunk.append(para)
                current_len += len(para) + 1

        if current_chunk:
            chunks.append((" ".join(current_chunk), doc_id))

        return [(c, d) for c, d in chunks if len(c.strip()) > 20]

    def load_corpus(self, documents: list[str]) -> None:
        """Loads and indexes the corpus of documents for RAG QA."""
        logger.info(f"Loading corpus of {len(documents)} documents...")

        self.chunks = []
        self.chunk_sources = []
        for doc_id, doc in enumerate(documents):
            doc_chunks = self._chunk_text(doc, doc_id)
            for chunk_text, did in doc_chunks:
                self.chunks.append(chunk_text)
                self.chunk_sources.append(did)

        logger.info(
            f"Created {len(self.chunks)} chunks from "
            f"{len(documents)} documents."
        )

        # Build dense embeddings index
        model = _get_sentence_model()
        logger.info("Encoding chunks with sentence-transformer...")
        self.embeddings = model.encode(
            self.chunks,
            show_progress_bar=False,
            batch_size=256,           # Larger batch = faster encoding
            normalize_embeddings=True,
        )
        logger.info(f"Embeddings shape: {self.embeddings.shape}")

        # Build BM25 sparse index
        try:
            from rank_bm25 import BM25Okapi
            tokenized_chunks = [
                chunk.lower().split() for chunk in self.chunks
            ]
            self.bm25 = BM25Okapi(tokenized_chunks)
            logger.info("BM25 index built successfully.")
        except ImportError:
            logger.warning("rank_bm25 not installed, using dense-only retrieval.")
            self.bm25 = None

        # Pre-load reranker (warm up during corpus load, not during QA)
        _get_cross_encoder()

        logger.info("Corpus loading complete.")
        self.loaded = True

    def _retrieve_dense(self, question: str, top_k: int) -> list[tuple[int, float]]:
        """Retrieve top-k chunks using dense embedding similarity."""
        if self.embeddings is None:
            return []

        model = _get_sentence_model()
        q_emb = model.encode([question], normalize_embeddings=True)
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

        tokenized_query = question.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [
            (int(idx), float(scores[idx]))
            for idx in top_indices
            if scores[idx] > 0.0
        ]

    def _rerank(
        self, question: str, candidate_indices: list[int]
    ) -> list[tuple[int, float]]:
        """Re-score candidates using cross-encoder for precision."""
        if not candidate_indices:
            return []

        cross_encoder = _get_cross_encoder()
        pairs = [
            [question, self.chunks[idx]] for idx in candidate_indices
        ]
        scores = cross_encoder.predict(pairs)
        ranked = sorted(
            zip(candidate_indices, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(int(idx), float(score)) for idx, score in ranked]

    def _extract_answer(self, question: str, context: str) -> str:
        """Extract the best answer span from context using sentence scoring.

        Uses the sentence-transformer to find the most relevant sentence(s)
        to the question. This is ~10x faster than LLM generation.
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', context)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if not sentences:
            return context[:300]

        if len(sentences) == 1:
            return sentences[0]

        # Score each sentence against the question
        model = _get_sentence_model()
        q_emb = model.encode([question], normalize_embeddings=True)
        s_embs = model.encode(sentences, normalize_embeddings=True)
        sims = np.dot(s_embs, q_emb.T).flatten()

        # Take top 2 most relevant sentences
        top_indices = np.argsort(sims)[-2:][::-1]
        best_sentences = [sentences[i] for i in top_indices if sims[i] > 0.15]

        if not best_sentences:
            return sentences[int(np.argmax(sims))]

        # If top sentence is very short, concatenate top 2
        answer = best_sentences[0]
        if len(answer) < 50 and len(best_sentences) > 1:
            answer = " ".join(best_sentences[:2])

        return answer

    def qa(self, question: str) -> str:
        """Performs question answering using hybrid retrieval + extractive QA.

        Args:
            question: The question to answer.

        Returns:
            A string containing the answer to the question.
        """
        if not self.loaded:
            return ""

        # Step 1: Hybrid retrieval — dense + BM25
        dense_results = self._retrieve_dense(question, self.top_k_retrieve)
        bm25_results = self._retrieve_bm25(question, self.top_k_retrieve)

        # Merge candidates (deduplicated union)
        seen = set()
        candidate_indices = []
        for idx, _score in dense_results + bm25_results:
            if idx not in seen:
                seen.add(idx)
                candidate_indices.append(idx)

        if not candidate_indices:
            return ""

        # Step 2: Rerank with cross-encoder (on limited candidates for speed)
        reranked = self._rerank(question, candidate_indices[:15])

        if not reranked:
            if dense_results:
                return self.chunks[dense_results[0][0]][:300]
            return ""

        # Step 3: Build context from top reranked chunks
        top_chunks = [
            self.chunks[idx]
            for idx, score in reranked[:self.top_k_rerank]
        ]
        context = "\n\n".join(top_chunks)

        # Step 4: Extractive answer (no LLM — pure speed)
        return self._extract_answer(question, context)
