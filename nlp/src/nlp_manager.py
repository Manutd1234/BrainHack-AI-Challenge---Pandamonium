"""Manages the NLP model — RAG-based Question Answering with LLM generation.

Pipeline:
  1. Dense embeddings via sentence-transformers (semantic similarity)
  2. BM25 sparse retrieval (keyword coverage)
  3. Cross-encoder reranking for precision
  4. Quantized LLM (Qwen2.5-3B-Instruct AWQ) for answer generation via vLLM
  
Falls back to extractive QA if LLM is not available.
"""

import logging
import re
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Lazy-loaded models
_sentence_model = None
_cross_encoder = None
_llm = None
_llm_tokenizer = None
_llm_available = False


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


def _get_llm():
    """Lazy-load the quantized LLM for answer generation.

    Uses Qwen2.5-3B-Instruct-AWQ — small enough to be fast,
    large enough for good QA accuracy, AWQ quantized for speed.
    Falls back gracefully if not available.
    """
    global _llm, _llm_tokenizer, _llm_available
    if _llm is not None:
        return _llm, _llm_tokenizer

    # Try vLLM first (highest throughput)
    try:
        from vllm import LLM, SamplingParams
        _llm = LLM(
            model="Qwen/Qwen2.5-3B-Instruct-AWQ",
            quantization="awq",
            max_model_len=2048,
            gpu_memory_utilization=0.5,  # Leave room for other models
            dtype="float16",
        )
        _llm_tokenizer = "vllm"  # Sentinel to indicate vLLM mode
        _llm_available = True
        logger.info("Loaded Qwen2.5-3B-Instruct-AWQ via vLLM.")
        return _llm, _llm_tokenizer
    except Exception as e:
        logger.warning(f"vLLM not available: {e}")

    # Fallback: transformers with auto quantization
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        model_id = "Qwen/Qwen2.5-3B-Instruct-AWQ"
        _llm_tokenizer = AutoTokenizer.from_pretrained(model_id)
        _llm = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        _llm_available = True
        logger.info("Loaded Qwen2.5-3B-Instruct-AWQ via transformers.")
        return _llm, _llm_tokenizer
    except Exception as e:
        logger.warning(f"LLM not available ({e}), using extractive QA only.")
        _llm_available = False

    return None, None


class NLPManager:
    loaded = False

    def __init__(self):
        self.chunks: list[str] = []
        self.chunk_sources: list[int] = []
        self.embeddings: Optional[np.ndarray] = None
        self.bm25 = None
        self.chunk_size = 512
        self.chunk_overlap = 128
        self.top_k_retrieve = 15
        self.top_k_rerank = 5

    def _chunk_text(self, text: str, doc_id: int) -> list[tuple[str, int]]:
        """Splits text into overlapping chunks, respecting paragraph/sentence
        boundaries where possible.
        """
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
                        keep = max(1, len(overlap_words) // 4)
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
                    keep = max(1, len(overlap_words) // 4)
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
            batch_size=128,
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

        # Pre-load reranker + LLM (warm up during corpus load)
        _get_cross_encoder()
        _get_llm()

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

    def _generate_answer_llm(self, question: str, context: str) -> str:
        """Generate an answer using the quantized LLM.

        Uses a concise prompt to keep generation fast.
        """
        llm, tokenizer = _get_llm()
        if llm is None:
            return ""

        prompt = (
            "Answer the question using ONLY the provided context. "
            "Give a concise, direct answer. If the answer is not in the "
            "context, say exactly: ''\n\n"
            f"Context:\n{context[:1500]}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        try:
            if tokenizer == "vllm":
                # vLLM inference
                from vllm import SamplingParams
                params = SamplingParams(
                    max_tokens=150,
                    temperature=0.1,
                    top_p=0.9,
                    stop=["\n\n", "Question:", "Context:"],
                )
                outputs = llm.generate([prompt], params)
                answer = outputs[0].outputs[0].text.strip()
            else:
                # transformers inference
                import torch
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                ).to(llm.device)

                with torch.no_grad():
                    output_ids = llm.generate(
                        **inputs,
                        max_new_tokens=150,
                        temperature=0.1,
                        top_p=0.9,
                        do_sample=True,
                    )

                # Decode only the generated tokens
                answer = tokenizer.decode(
                    output_ids[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True,
                ).strip()

            # Clean up answer
            answer = answer.split("\n")[0].strip()
            if answer in ("''", '""', "N/A", "None", "I don't know"):
                return ""
            return answer

        except Exception as e:
            logger.warning(f"LLM generation failed: {e}")
            return ""

    def _extract_answer_fallback(self, question: str, context: str) -> str:
        """Extractive answer fallback when LLM is not available.

        Uses sentence-level embedding scoring against the question.
        """
        sentences = re.split(r'(?<=[.!?])\s+', context)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if not sentences:
            return context[:300]

        if len(sentences) == 1:
            return sentences[0]

        model = _get_sentence_model()
        q_emb = model.encode([question], normalize_embeddings=True)
        s_embs = model.encode(sentences, normalize_embeddings=True)
        sims = np.dot(s_embs, q_emb.T).flatten()

        top_indices = np.argsort(sims)[-3:][::-1]
        best_sentences = [sentences[i] for i in top_indices if sims[i] > 0.1]

        if not best_sentences:
            return sentences[0]

        answer = best_sentences[0]
        if len(answer) < 40 and len(best_sentences) > 1:
            answer = " ".join(best_sentences[:2])

        return answer

    def qa(self, question: str) -> str:
        """Performs question answering using hybrid retrieval + LLM generation.

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

        # Step 2: Rerank with cross-encoder
        reranked = self._rerank(question, candidate_indices[:25])

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

        # Step 4: Generate answer with LLM (or extractive fallback)
        if _llm_available:
            answer = self._generate_answer_llm(question, context)
            if answer:
                return answer
            # LLM returned empty — fall through to extractive

        return self._extract_answer_fallback(question, context)
