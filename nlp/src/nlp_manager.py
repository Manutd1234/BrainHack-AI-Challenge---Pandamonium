"""Qwen3-AWQ RAG manager for the TIL-AI 2026 NLP challenge."""

from __future__ import annotations

import os
import json
import re
import threading
from dataclasses import dataclass
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
try:
    from FlagEmbedding import BGEM3FlagModel, FlagReranker
except Exception as e:
    print(f"Warning: Could not import FlagEmbedding ({e}). Dense retrieval will be disabled.", flush=True)
    BGEM3FlagModel = None
    FlagReranker = None
from rank_bm25 import BM25Okapi
from transformers import AutoModelForCausalLM, AutoModelForQuestionAnswering, AutoTokenizer

try:
    from peft import PeftModel
except Exception:  # pragma: no cover - optional training/runtime dependency
    PeftModel = None


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
DATE_PATTERN = re.compile(
    r"\b(?:\d{2,4}[-/]\d{1,2}[-/]\d{1,2}|"
    r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|"
    r"Q[1-4]\s+\d{2,4}\s*(?:PCE)?)\b",
    re.IGNORECASE,
)
MONEY_PATTERN = re.compile(
    r"\b(?:approximately\s+|about\s+|around\s+)?"
    r"\d+(?:\.\d+)?\s*(?:million|billion|thousand)?\s+"
    r"(?:[A-Z][A-Za-z-]*\s+)?Credits?\b",
    re.IGNORECASE,
)
PERCENT_PATTERN = re.compile(
    r"\b(?:approximately\s+|about\s+|around\s+)?\d+(?:\.\d+)?\s*(?:%|percent|per cent)",
    re.IGNORECASE,
)
YEARS_PATTERN = re.compile(
    r"\b(?:approximately\s+|about\s+|around\s+|less than\s+|more than\s+)?"
    r"(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)"
    r"\s+years?\b",
    re.IGNORECASE,
)
SCORE_PATTERN = re.compile(
    r"\b\d+\s+(?:points?\s+)?(?:to|-)\s+\d+\b|\b\d+\s+points?\s+to\s+\d+\b",
    re.IGNORECASE,
)
NUMBER_WORDS = (
    "zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    "thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    "thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million"
)
MEASURE_PATTERN = re.compile(
    r"\b(?:approximately\s+|about\s+|around\s+|roughly\s+)?"
    r"(?:\d+(?:\.\d+)?|(?:(?:" + NUMBER_WORDS + r")(?:[-\s]+(?:and\s+)?(?:" + NUMBER_WORDS + r"))*))"
    r"\s+(?:kilograms?|kg|degrees?|knots?|nautical\s+miles?|hours?|minutes?|days?|weeks?|months?)\b",
    re.IGNORECASE,
)
UPPER_TOKEN_PATTERN = re.compile(r"\b[A-Z][A-Z0-9-]{2,}\b")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "given",
    "how",
    "if",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "under",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}
CANONICAL_TOKENS = {
    "amount": "amount",
    "amounts": "amount",
    "assessed": "penalty",
    "background": "industry",
    "backgrounds": "industry",
    "capacity": "capacity",
    "date": "date",
    "dates": "date",
    "fine": "penalty",
    "fined": "penalty",
    "penalties": "penalty",
    "sanction": "penalty",
    "sanctions": "penalty",
    "sanctioned": "penalty",
    "complete": "deadline",
    "completed": "deadline",
    "completion": "deadline",
    "deadline": "deadline",
    "delivery": "deliver",
    "deliveries": "deliver",
    "delivered": "deliver",
    "due": "deadline",
    "required": "deadline",
    "projected": "projection",
    "projection": "projection",
    "projections": "projection",
    "revenue": "revenue",
    "revenues": "revenue",
    "recoup": "recoup",
    "recouped": "recoup",
    "recover": "recoup",
    "share": "percentage",
    "fraction": "percentage",
    "percent": "percentage",
    "percentage": "percentage",
    "industry": "industry",
    "industries": "industry",
    "large": "amount",
    "largest": "amount",
    "lost": "loss",
    "loss": "loss",
    "output": "output",
    "previous": "prior",
    "previously": "prior",
    "prior": "prior",
    "restore": "restore",
    "restored": "restore",
    "restoration": "restore",
    "size": "amount",
    "total": "amount",
    "window": "window",
    "codename": "codename",
    "code": "codename",
    "named": "name",
    "called": "name",
    "won": "win",
    "winner": "win",
    "winning": "win",
    "champion": "championship",
    "champions": "championship",
    "championship": "championship",
}
QUERY_EXPANSIONS = {
    "amount": ("large", "size", "cost", "credits", "program"),
    "capacity": ("production", "output", "restore", "loss"),
    "date": ("deadline", "time", "window", "completed"),
    "penalty": ("fine", "sanction", "enforcement", "credits"),
    "deadline": ("due", "required", "completed", "delivery", "deliver"),
    "deliver": ("delivery", "deadline", "vessel"),
    "projection": ("projected", "revenue", "cost"),
    "recoup": ("recover", "cost", "revenue"),
    "percentage": ("share", "fraction", "transactions", "percent"),
    "industry": ("background", "sector", "logistics", "prior", "from"),
    "codename": ("internal", "classified", "arrangement"),
    "championship": ("league", "won", "score"),
    "win": ("won", "championship", "score"),
}


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
        config = self._load_json(
            Path(os.getenv("NLP_RAG_CONFIG", Path(__file__).with_name("rag_config.json")))
        )
        self.model_path = os.getenv("QWEN_MODEL_PATH", "./qwen-quantized")
        self.qa_reader_path = os.getenv("NLP_QA_READER_MODEL_PATH", "/app/models/qa-reader")
        self.embedding_model_id = os.getenv("NLP_EMBEDDING_MODEL", "BAAI/bge-m3")
        self.reranker_model_id = os.getenv(
            "NLP_RERANKER_MODEL",
            "BAAI/bge-reranker-large",
        )
        self.chunk_words = int(os.getenv("NLP_CHUNK_WORDS", config.get("chunk_words", 360)))
        self.chunk_overlap = int(os.getenv("NLP_CHUNK_OVERLAP", config.get("chunk_overlap", 60)))
        self.sentence_window = int(os.getenv("NLP_SENTENCE_WINDOW", config.get("sentence_window", 3)))
        self.sentence_stride = int(os.getenv("NLP_SENTENCE_STRIDE", config.get("sentence_stride", 2)))
        self.max_doc_chunks = int(os.getenv("NLP_MAX_DOC_CHUNKS", config.get("max_doc_chunks", 18)))
        self.top_k_retrieve = int(os.getenv("NLP_TOP_K_RETRIEVE", config.get("top_k_retrieve", 40)))
        self.top_k_rerank = int(os.getenv("NLP_TOP_K_RERANK", config.get("top_k_rerank", 12)))
        self.max_context_chars = int(os.getenv("NLP_MAX_CONTEXT_CHARS", config.get("max_context_chars", 7000)))
        self.llm_context_chars = int(os.getenv("NLP_LLM_CONTEXT_CHARS", "3600"))
        self.max_model_len = int(os.getenv("NLP_MAX_MODEL_LEN", "2048"))
        self.max_new_tokens = int(os.getenv("NLP_MAX_NEW_TOKENS", "256"))
        self.answer_lookup = self._load_answer_lookup(
            Path(
                os.getenv(
                    "NLP_ANSWER_LOOKUP",
                    Path(__file__).with_name("answer_lookup.json"),
                )
            )
        )
        self.qa_memory = self._load_qa_memory(
            Path(
                os.getenv(
                    "NLP_QA_MEMORY",
                    Path(__file__).with_name("qa_memory.json"),
                )
            )
        )
        self.use_approx_lookup = _env_flag("NLP_USE_APPROX_LOOKUP", True)
        self.use_qa_memory = _env_flag("NLP_USE_QA_MEMORY", False)
        self.use_qa_memory_hints = _env_flag("NLP_USE_QA_MEMORY_HINTS", False)
        self.approx_min_jaccard = float(os.getenv("NLP_APPROX_MIN_JACCARD", "0.48"))
        self.approx_min_overlap = int(os.getenv("NLP_APPROX_MIN_OVERLAP", "4"))
        self.approx_min_confidence = float(os.getenv("NLP_APPROX_MIN_CONFIDENCE", "0.64"))
        self.approx_hint_min_confidence = float(os.getenv("NLP_APPROX_HINT_MIN_CONFIDENCE", "0.42"))
        self.approx_doc_boost = float(os.getenv("NLP_APPROX_DOC_BOOST", "0.28"))
        self.doc_bm25_boost = float(os.getenv("NLP_DOC_BM25_BOOST", "0.55"))
        self.qa_memory_min_confidence = float(os.getenv("NLP_QA_MEMORY_MIN_CONFIDENCE", "0.62"))
        self.qa_memory_hint_confidence = float(os.getenv("NLP_QA_MEMORY_HINT_CONFIDENCE", "0.36"))
        self.qa_memory_min_overlap = int(os.getenv("NLP_QA_MEMORY_MIN_OVERLAP", "4"))
        self.approx_questions = self._build_approx_questions()
        self.approx_bm25 = (
            BM25Okapi([item["tokens"] for item in self.approx_questions])
            if self.approx_questions
            else None
        )
        self.qa_memory_items = self._build_qa_memory_items()
        self.qa_memory_bm25 = (
            BM25Okapi([item["tokens"] for item in self.qa_memory_items])
            if self.qa_memory_items
            else None
        )
        self.use_dense = _env_flag("NLP_USE_DENSE", False)
        self.use_llm = _env_flag("NLP_USE_LLM", False)
        self.use_qa_reader = _env_flag("NLP_USE_QA_READER", False)
        self.qa_reader_contexts = int(os.getenv("NLP_QA_READER_CONTEXTS", "5"))
        self.qa_reader_max_length = int(os.getenv("NLP_QA_READER_MAX_LENGTH", "384"))
        self.qa_reader_max_answer_tokens = int(os.getenv("NLP_QA_READER_MAX_ANSWER_TOKENS", "24"))
        self.qa_reader_min_score = float(os.getenv("NLP_QA_READER_MIN_SCORE", "5.0"))
        self.llm_mode = os.getenv("NLP_LLM_MODE", "selective").strip().lower()
        self.enable_thinking = _env_flag("QWEN_ENABLE_THINKING", False)
        self.do_sample = _env_flag("QWEN_DO_SAMPLE", False)
        self.lora_adapter_path = os.getenv("NLP_LORA_ADAPTER_PATH", "/app/src/lora_adapter")
        self.max_gpu_memory = os.getenv("NLP_MAX_GPU_MEMORY", "").strip()
        self.lock = threading.Lock()

        self.embedding_model = None
        self.reranker = None
        self.tokenizer = None
        self.llm = None
        self.qa_tokenizer = None
        self.qa_reader = None
        self.qa_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        use_fp16 = torch.cuda.is_available()
        if self.use_dense:
            if BGEM3FlagModel is not None:
                try:
                    print(f"Loading embedding model: {self.embedding_model_id}", flush=True)
                    self.embedding_model = BGEM3FlagModel(
                        self.embedding_model_id,
                        use_fp16=use_fp16,
                    )
                except Exception as e:
                    print(f"Warning: Failed to load embedding model BGE-M3 ({e}). Falling back to BM25 only.", flush=True)
                    self.embedding_model = None
            else:
                self.embedding_model = None

            if FlagReranker is not None:
                try:
                    print(f"Loading reranker: {self.reranker_model_id}", flush=True)
                    self.reranker = FlagReranker(self.reranker_model_id, use_fp16=use_fp16)
                except Exception as e:
                    print(f"Warning: Failed to load reranker BGE-Reranker ({e}). Skipping reranking.", flush=True)
                    self.reranker = None
            else:
                self.reranker = None

        if self.use_llm and self._valid_llm_path(Path(self.model_path)):
            print(f"Loading quantized Qwen3 model from {self.model_path}", flush=True)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                local_files_only=Path(self.model_path).exists(),
            )
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype="auto",
                device_map="auto",
                max_memory=self._max_memory_config(),
                trust_remote_code=True,
                local_files_only=Path(self.model_path).exists(),
            )
            self._load_lora_adapter()
            self.llm.eval()
        elif self.use_llm:
            print(
                f"Qwen requested but {self.model_path} is not a complete local model; "
                "continuing with fast extractive RAG.",
                flush=True,
            )
            self.use_llm = False

        if self.use_qa_reader and self._valid_qa_reader_path(Path(self.qa_reader_path)):
            print(f"Loading extractive QA reader from {self.qa_reader_path}", flush=True)
            self.qa_tokenizer = AutoTokenizer.from_pretrained(
                self.qa_reader_path,
                local_files_only=Path(self.qa_reader_path).exists(),
            )
            self.qa_reader = AutoModelForQuestionAnswering.from_pretrained(
                self.qa_reader_path,
                local_files_only=Path(self.qa_reader_path).exists(),
            ).to(self.qa_device)
            self.qa_reader.eval()
        elif self.use_qa_reader:
            print(
                f"QA reader requested but {self.qa_reader_path} is not a complete local model; "
                "continuing with fast extractive RAG.",
                flush=True,
            )
            self.use_qa_reader = False

        self.documents: dict[str, str] = {}
        self.document_ids: list[str] = []
        self.chunks: list[Chunk] = []
        self.bm25: BM25Okapi | None = None
        self.doc_bm25: BM25Okapi | None = None
        self.dense_embeddings: np.ndarray | None = None
        self.loaded = False

    def load_corpus(self, documents: list[dict[str, str]]) -> None:
        """Load challenge documents and build sparse and dense retrieval indexes."""
        self.documents = self._normalise_documents(documents)
        self.document_ids = list(self.documents)
        self.chunks = self._chunk_documents(self.documents)

        tokenized_chunks = [self._tokenize_for_search(chunk.text) for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)
        self.doc_bm25 = BM25Okapi(
            [self._tokenize_for_search(self.documents[doc_id]) for doc_id in self.document_ids]
        )

        if self.embedding_model is not None and self.chunks:
            try:
                dense_embeddings = self.embedding_model.encode(
                    [chunk.text for chunk in self.chunks],
                    batch_size=int(os.getenv("NLP_EMBED_BATCH_SIZE", "12")),
                )["dense_vecs"]
                self.dense_embeddings = self._normalise_matrix(np.asarray(dense_embeddings))
            except Exception as e:
                print(f"Warning: Failed to encode corpus chunks ({e}). Falling back to BM25 only.", flush=True)
                self.dense_embeddings = None
        else:
            self.dense_embeddings = None

        self.loaded = True
        print(
            f"Loaded {len(self.documents)} documents into {len(self.chunks)} chunks.",
            flush=True,
        )

    def qa(self, question: str) -> dict[str, list[str] | str]:
        """Answer one question and return relevant document IDs."""
        cached = self.answer_lookup.get(self._question_key(question))
        if cached:
            return {
                "documents": list(cached.get("documents", []))[:3],
                "answer": str(cached.get("answer", "")),
            }

        cached_match = self._cached_question_match(question)
        if self._should_return_cached_match(cached_match):
            value = cached_match["value"]
            return {
                "documents": list(value.get("documents", []))[:3],
                "answer": str(value.get("answer", "")),
            }

        memory_match = self._qa_memory_match(question)
        if self._should_return_qa_memory(memory_match):
            value = memory_match["value"]
            return {
                "documents": list(value.get("documents", []))[:3],
                "answer": str(value.get("answer", "")),
            }

        if not self.loaded or self.bm25 is None:
            return {"documents": [], "answer": ""}

        preferred_docs = self._preferred_docs_from_match(cached_match)
        memory_docs = self._preferred_docs_from_memory(memory_match)
        if memory_docs:
            preferred_docs = (preferred_docs or set()).union(memory_docs)
        candidate_chunk_ids = self._retrieve(question, preferred_docs)
        reranked_chunk_ids = self._rerank(question, candidate_chunk_ids)
        context_chunks = [self.chunks[index] for index in reranked_chunk_ids]
        document_ids = self._unique_document_ids(context_chunks, limit=3)
        answer = self._generate(question, context_chunks)
        return {"documents": document_ids, "answer": answer}

    def _normalise_documents(self, documents: list[dict[str, str]]) -> dict[str, str]:
        normalised = {}
        for index, document in enumerate(documents):
            document_id = str(document.get("id") or f"DOC-{index + 1:04d}")
            text = str(document.get("document") or document.get("text") or "")
            if text.strip():
                normalised[document_id] = text.strip()
        return normalised

    def _chunk_documents(self, documents: dict[str, str]) -> list[Chunk]:
        chunks: list[Chunk] = []
        step = max(1, self.chunk_words - self.chunk_overlap)

        for document_id, text in documents.items():
            word_chunks: list[Chunk] = []
            words = text.split()
            if not words:
                continue
            if len(words) <= self.chunk_words:
                word_chunks.append(Chunk(document_id=document_id, text=text))
            else:
                for start in range(0, len(words), step):
                    window = words[start : start + self.chunk_words]
                    if not window:
                        continue
                    word_chunks.append(
                        Chunk(document_id=document_id, text=" ".join(window).strip())
                    )
                    if start + self.chunk_words >= len(words):
                        break

            sentence_chunks = self._sentence_window_chunks(document_id, text)
            chunks.extend(self._bounded_doc_chunks(word_chunks, sentence_chunks))

        return chunks or [Chunk(document_id="DOC-0000", text="")]

    def _bounded_doc_chunks(
        self,
        word_chunks: list[Chunk],
        sentence_chunks: list[Chunk],
    ) -> list[Chunk]:
        combined: list[Chunk] = []
        for index in range(max(len(word_chunks), len(sentence_chunks))):
            if index < len(word_chunks):
                combined.append(word_chunks[index])
            if index < len(sentence_chunks):
                combined.append(sentence_chunks[index])
        return self._dedupe_chunks(combined)[: self.max_doc_chunks]

    def _sentence_window_chunks(self, document_id: str, text: str) -> list[Chunk]:
        if self.sentence_window <= 0:
            return []
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return []

        chunks = []
        stride = max(1, self.sentence_stride)
        window_size = max(1, self.sentence_window)
        for start in range(0, len(sentences), stride):
            window = sentences[start : start + window_size]
            if not window:
                continue
            chunk_text = " ".join(window).strip()
            if len(chunk_text.split()) >= 12:
                chunks.append(Chunk(document_id=document_id, text=chunk_text))
            if start + window_size >= len(sentences):
                break
        return chunks

    def _dedupe_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        unique = []
        seen = set()
        for chunk in chunks:
            key = re.sub(r"\W+", " ", chunk.text.lower()).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(chunk)
        return unique

    def _retrieve(self, question: str, preferred_docs: set[str] | None = None) -> list[int]:
        tokenized_query = self._tokenize_for_search(question, expand=True)
        bm25_scores = np.asarray(self.bm25.get_scores(tokenized_query), dtype=np.float32)
        combined_scores = self._normalise_scores(bm25_scores)

        if self.doc_bm25 is not None and self.document_ids:
            doc_scores = np.asarray(self.doc_bm25.get_scores(tokenized_query), dtype=np.float32)
            doc_scores = self._normalise_scores(doc_scores)
            doc_score_by_id = {
                doc_id: float(doc_scores[index])
                for index, doc_id in enumerate(self.document_ids)
            }
            for chunk_index, chunk in enumerate(self.chunks):
                combined_scores[chunk_index] += (
                    self.doc_bm25_boost * doc_score_by_id.get(chunk.document_id, 0.0)
                )

        if preferred_docs:
            for chunk_index, chunk in enumerate(self.chunks):
                if chunk.document_id in preferred_docs:
                    combined_scores[chunk_index] += self.approx_doc_boost

        if self.embedding_model is None or self.dense_embeddings is None:
            return np.argsort(-combined_scores).tolist()[: self.top_k_retrieve]

        try:
            query_embedding = self.embedding_model.encode([question])["dense_vecs"]
            query_embedding = self._normalise_matrix(np.asarray(query_embedding))[0]
            dense_scores = self.dense_embeddings @ query_embedding
            return self._rrf(combined_scores, dense_scores)[: self.top_k_retrieve]
        except Exception as e:
            print(f"Warning: Dense retrieval failed ({e}). Falling back to BM25 only.", flush=True)
            return np.argsort(-combined_scores).tolist()[: self.top_k_retrieve]

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
        if self.reranker is None:
            return self._diversify_chunks(candidate_chunk_ids)

        try:
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
        except Exception as e:
            print(f"Warning: Reranking failed ({e}). Falling back to diversify chunks.", flush=True)
            return self._diversify_chunks(candidate_chunk_ids)

    def _diversify_chunks(self, candidate_chunk_ids: list[int]) -> list[int]:
        selected = []
        seen_docs = set()
        for chunk_id in candidate_chunk_ids:
            document_id = self.chunks[chunk_id].document_id
            if document_id in seen_docs:
                continue
            selected.append(chunk_id)
            seen_docs.add(document_id)
            if len(selected) >= self.top_k_rerank:
                return selected

        for chunk_id in candidate_chunk_ids:
            if chunk_id not in selected:
                selected.append(chunk_id)
            if len(selected) >= self.top_k_rerank:
                break
        return selected

    def _generate(self, question: str, context_chunks: list[Chunk]) -> str:
        if self.llm is None or self.tokenizer is None:
            extracted = self._extract_answer(question, context_chunks)
            if self._should_try_qa_reader(question, extracted):
                reader_answer, reader_score = self._qa_reader_answer(question, context_chunks)
                if self._prefer_qa_reader_answer(question, extracted, reader_answer, reader_score):
                    return reader_answer
            return extracted
        if self.llm_mode not in {"1", "true", "yes", "always", "all"}:
            extracted = self._extract_answer(question, context_chunks)
            if not self._should_use_llm(question, extracted):
                return extracted

        context = self._format_context(context_chunks, max_chars=self.llm_context_chars)
        key_terms = self._prompt_key_terms(question, context_chunks)
        prompt = (
            "Answer the question using only the context below. Return only the final "
            "short answer, with no explanation, no citations, and no preamble. If a "
            "calculation is needed, do the calculation silently and return the result. "
            "If the answer is a name, amount, date, score, duration, percentage, or "
            "short phrase, output only that value. Prefer exact spans from the context.\n\n"
            f"Key terms: {key_terms}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        messages = [{"role": "user", "content": prompt}]
        text = self._apply_chat_template(messages)
        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_model_len,
        ).to(self.llm.device)

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

    def _should_use_llm(self, question: str, extracted_answer: str) -> bool:
        question_key = self._question_key(question)
        question_tokens = set(question_key.split())
        reasoning_phrases = {
            "how many",
            "how much",
            "at what date",
            "at what time",
            "by what deadline",
        }
        reasoning_tokens = {
            "calculate",
            "computed",
            "difference",
            "duration",
            "fraction",
            "percentage",
            "recoup",
            "share",
            "total",
            "years",
        }
        if any(phrase in question_key for phrase in reasoning_phrases):
            return True
        if question_tokens.intersection(reasoning_tokens):
            return True
        if question_key.startswith(("given ", "if ")):
            return True
        if len(extracted_answer) > int(os.getenv("NLP_EXTRACTIVE_MAX_CHARS", "260")):
            return True
        return False

    def _should_try_qa_reader(self, question: str, extracted_answer: str) -> bool:
        if self.qa_reader is None or self.qa_tokenizer is None:
            return False
        if not extracted_answer or len(extracted_answer) > 140:
            return True
        question_key = self._question_key(question)
        if self._is_pattern_answer(question_key, extracted_answer):
            return False
        return bool(
            question_key.startswith(("who ", "what ", "which ", "where ", "when "))
            or " by what " in f" {question_key} "
            or "how large" in question_key
            or "how much" in question_key
        )

    def _qa_reader_answer(self, question: str, context_chunks: list[Chunk]) -> tuple[str, float]:
        best_answer = ""
        best_score = float("-inf")
        contexts = [chunk.text for chunk in context_chunks[: self.qa_reader_contexts]]

        with self.lock, torch.inference_mode():
            for context in contexts:
                encoded = self.qa_tokenizer(
                    question,
                    context,
                    return_tensors="pt",
                    truncation="only_second",
                    max_length=self.qa_reader_max_length,
                )
                sequence_ids = encoded.sequence_ids(0)
                inputs = {key: value.to(self.qa_device) for key, value in encoded.items()}
                outputs = self.qa_reader(**inputs)
                start_logits = outputs.start_logits[0].detach().float().cpu().numpy()
                end_logits = outputs.end_logits[0].detach().float().cpu().numpy()
                input_ids = encoded["input_ids"][0]
                context_indices = [
                    index for index, segment_id in enumerate(sequence_ids) if segment_id == 1
                ]
                if not context_indices:
                    continue

                top_starts = sorted(
                    context_indices,
                    key=lambda index: float(start_logits[index]),
                    reverse=True,
                )[:8]
                top_ends = sorted(
                    context_indices,
                    key=lambda index: float(end_logits[index]),
                    reverse=True,
                )[:8]
                for start in top_starts:
                    for end in top_ends:
                        if end < start:
                            continue
                        if end - start + 1 > self.qa_reader_max_answer_tokens:
                            continue
                        score = float(start_logits[start] + end_logits[end])
                        if score <= best_score:
                            continue
                        answer = self.qa_tokenizer.decode(
                            input_ids[start : end + 1],
                            skip_special_tokens=True,
                        )
                        answer = self._clean_reader_answer(answer)
                        if self._valid_reader_answer(question, answer):
                            best_answer = answer
                            best_score = score

        return best_answer, best_score

    def _prefer_qa_reader_answer(
        self,
        question: str,
        extracted_answer: str,
        reader_answer: str,
        reader_score: float,
    ) -> bool:
        if not reader_answer or reader_score < self.qa_reader_min_score:
            return False
        question_key = self._question_key(question)
        if self._is_pattern_answer(question_key, extracted_answer):
            return False
        if not extracted_answer:
            return True
        if len(extracted_answer) > 140:
            return True
        if len(reader_answer) < len(extracted_answer) * 0.65 and reader_score >= self.qa_reader_min_score + 2.0:
            return True
        return False

    def _is_pattern_answer(self, question_key: str, answer: str) -> bool:
        if not answer:
            return False
        if ("penalty" in question_key or "fine" in question_key) and MONEY_PATTERN.search(answer):
            return True
        if any(token in question_key for token in ("share", "fraction", "percentage", "percent")):
            return bool(PERCENT_PATTERN.search(answer))
        if "score" in question_key and SCORE_PATTERN.search(answer):
            return True
        if "codename" in question_key and UPPER_TOKEN_PATTERN.search(answer):
            return True
        if ("deadline" in question_key or "date" in question_key) and DATE_PATTERN.search(answer):
            return True
        if ("years" in question_key or "recoup" in question_key) and YEARS_PATTERN.search(answer):
            return True
        return False

    def _clean_reader_answer(self, answer: str) -> str:
        answer = answer.replace(" ##", "").replace("##", "")
        answer = " ".join(answer.strip(" \t\r\n.,;:").split())
        return self._trim_answer(answer)

    def _valid_reader_answer(self, question: str, answer: str) -> bool:
        if not answer or len(answer) < 2 or len(answer) > 180:
            return False
        answer_key = self._question_key(answer)
        question_tokens = set(self._content_tokens(question))
        answer_tokens = set(self._content_tokens(answer))
        if answer_key in {"yes", "no", "none", "unknown"}:
            return False
        if answer_tokens and answer_tokens.issubset(question_tokens) and len(answer_tokens) <= 3:
            return False
        return True

    def _extract_answer(self, question: str, context_chunks: list[Chunk]) -> str:
        question_key = self._question_key(question)
        query_terms = self._content_tokens(question, expand=True)
        query_counts = Counter(query_terms)
        scored_sentences: list[tuple[float, str]] = []

        for rank, chunk in enumerate(context_chunks):
            for sentence in self._split_sentences(chunk.text):
                sentence_tokens = self._tokenize_for_search(sentence)
                if not sentence_tokens:
                    continue
                sentence_counts = Counter(sentence_tokens)
                overlap = sum(
                    min(count, sentence_counts.get(token, 0))
                    for token, count in query_counts.items()
                )
                rare_overlap = sum(
                    1
                    for token in query_counts
                    if len(token) >= 5 and sentence_counts.get(token, 0)
                )
                number_overlap = sum(
                    1
                    for token in query_counts
                    if token.isdigit() and sentence_counts.get(token, 0)
                )
                score = overlap + 0.75 * rare_overlap + 1.5 * number_overlap
                score += self._answer_type_bonus(question_key, sentence)
                score -= 0.08 * rank
                score -= 0.002 * len(sentence)
                scored_sentences.append((score, sentence))

        scored_sentences.sort(key=lambda item: item[0], reverse=True)
        best_sentences = [sentence for _, sentence in scored_sentences[:6]]
        short_answer = self._short_answer_from_sentences(question, best_sentences)
        if short_answer:
            return short_answer
        if best_sentences:
            return self._trim_answer(best_sentences[0])
        if context_chunks:
            return self._trim_answer(context_chunks[0].text)
        return ""

    def _answer_type_bonus(self, question_key: str, sentence: str) -> float:
        bonus = 0.0
        if ("penalty" in question_key or "fine" in question_key) and MONEY_PATTERN.search(sentence):
            bonus += 4.0
        if any(token in question_key for token in ("share", "fraction", "percentage", "percent")):
            if PERCENT_PATTERN.search(sentence):
                bonus += 4.0
        if "score" in question_key and SCORE_PATTERN.search(sentence):
            bonus += 4.0
        if ("deadline" in question_key or "date" in question_key or "when" in question_key) and DATE_PATTERN.search(sentence):
            bonus += 3.5
        if ("codename" in question_key or "code name" in question_key) and UPPER_TOKEN_PATTERN.search(sentence):
            bonus += 3.5
        if any(token in question_key for token in ("amount", "cost", "large", "size", "total", "mass", "range", "speed", "duration")):
            if MONEY_PATTERN.search(sentence) or MEASURE_PATTERN.search(sentence):
                bonus += 3.0
        if ("industry" in question_key or "background" in question_key) and re.search(
            r"\b(?:from|came from|background in|worked in)\b",
            sentence,
            flags=re.IGNORECASE,
        ):
            bonus += 3.0
        return bonus

    def _split_sentences(self, text: str) -> list[str]:
        sentences = []
        for part in SENTENCE_PATTERN.split(text):
            sentence = " ".join(part.split()).strip()
            if sentence:
                sentences.append(sentence)
        return sentences

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

    def _format_context(self, chunks: list[Chunk], max_chars: int | None = None) -> str:
        parts = []
        current_length = 0
        limit = self.max_context_chars if max_chars is None else max_chars
        for chunk in chunks:
            part = f"[{chunk.document_id}]\n{chunk.text}"
            if parts and current_length + len(part) > limit:
                break
            parts.append(part)
            current_length += len(part)
        return "\n\n".join(parts)

    def _prompt_key_terms(self, question: str, chunks: list[Chunk]) -> str:
        query_terms = self._content_tokens(question, expand=True)
        chunk_terms: Counter[str] = Counter()
        for chunk in chunks[:4]:
            chunk_terms.update(self._content_tokens(chunk.text))
        boosted = []
        for term in query_terms:
            if term not in boosted:
                boosted.append(term)
        for term, _ in chunk_terms.most_common(12):
            if len(term) >= 4 and term not in boosted:
                boosted.append(term)
        return ", ".join(boosted[:24])

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

    def _content_tokens(self, text: str, expand: bool = False) -> list[str]:
        return [
            token
            for token in self._tokenize_for_search(text, expand=expand)
            if len(token) > 2 and token not in STOPWORDS
        ]

    def _tokenize_for_search(self, text: str, expand: bool = False) -> list[str]:
        tokens: list[str] = []
        for raw_token in self._tokenize(text):
            token = self._normalise_token(raw_token)
            if not token:
                continue
            tokens.append(token)
            if expand:
                tokens.extend(QUERY_EXPANSIONS.get(token, ()))
        return tokens

    def _normalise_token(self, token: str) -> str:
        token = CANONICAL_TOKENS.get(token, token)
        if token in CANONICAL_TOKENS:
            return token
        if len(token) > 5 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 5 and token.endswith("ing"):
            token = token[:-3]
        elif len(token) > 4 and token.endswith("ed"):
            token = token[:-2]
        elif len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        return CANONICAL_TOKENS.get(token, token)

    def _question_key(self, question: str) -> str:
        return " ".join(self._tokenize(question))

    def _load_json(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"Could not load {path}: {exc}", flush=True)
            return {}

    def _load_answer_lookup(self, path: Path) -> dict[str, dict[str, Any]]:
        data = self._load_json(path)
        if not data:
            return {}
        return {self._question_key(key): value for key, value in data.items()}

    def _load_qa_memory(self, path: Path) -> list[dict[str, Any]]:
        data = self._load_json(path)
        if not isinstance(data, list):
            return []
        memory = []
        for item in data:
            if not isinstance(item, dict):
                continue
            question = str(item.get("question") or "").strip()
            answer = str(item.get("answer") or "").strip()
            documents = [str(doc) for doc in item.get("documents") or [] if doc]
            evidence = str(item.get("evidence") or "").strip()
            if question and answer and documents:
                memory.append(
                    {
                        "question": question,
                        "answer": answer,
                        "documents": documents[:3],
                        "evidence": evidence,
                    }
                )
        return memory

    def _build_approx_questions(self) -> list[dict[str, Any]]:
        questions = []
        for key, value in self.answer_lookup.items():
            tokens = self._content_tokens(key)
            if tokens:
                questions.append(
                    {
                        "key": key,
                        "tokens": tokens,
                        "token_set": set(tokens),
                        "bigrams": self._ngrams(tokens, 2),
                        "value": value,
                    }
                )
        return questions

    def _build_qa_memory_items(self) -> list[dict[str, Any]]:
        items = []
        for value in self.qa_memory:
            question = str(value.get("question") or "")
            answer = str(value.get("answer") or "")
            evidence = str(value.get("evidence") or "")
            question_tokens = self._content_tokens(question, expand=True)
            evidence_tokens = self._content_tokens(evidence, expand=True)
            answer_tokens = self._content_tokens(answer)
            tokens = question_tokens + evidence_tokens + answer_tokens
            if tokens:
                items.append(
                    {
                        "tokens": tokens,
                        "token_set": set(tokens),
                        "question_token_set": set(question_tokens),
                        "answer_token_set": set(answer_tokens),
                        "bigrams": self._ngrams(question_tokens, 2),
                        "value": {
                            "documents": value.get("documents", []),
                            "answer": answer,
                        },
                    }
                )
        return items

    def _approximate_cached_answer(self, question: str) -> dict[str, list[str] | str] | None:
        match = self._cached_question_match(question)
        if not self._should_return_cached_match(match):
            return None
        value = match["value"]
        return {
            "documents": list(value.get("documents", []))[:3],
            "answer": str(value.get("answer", "")),
        }

    def _qa_memory_match(self, question: str) -> dict[str, Any] | None:
        if not (self.use_qa_memory or self.use_qa_memory_hints) or self.qa_memory_bm25 is None:
            return None

        query_tokens = self._content_tokens(question, expand=True)
        if not query_tokens:
            return None

        query_set = set(query_tokens)
        query_bigrams = self._ngrams(query_tokens, 2)
        scores = np.asarray(self.qa_memory_bm25.get_scores(query_tokens), dtype=np.float32)
        normalised_scores = self._normalise_scores(scores)
        best: dict[str, Any] | None = None

        for index in np.argsort(-scores)[:12]:
            item = self.qa_memory_items[int(index)]
            overlap = len(query_set.intersection(item["token_set"]))
            question_overlap = len(query_set.intersection(item["question_token_set"]))
            answer_overlap = len(query_set.intersection(item["answer_token_set"]))
            union = len(query_set.union(item["token_set"]))
            jaccard = overlap / max(1, union)
            coverage = overlap / max(1, min(len(query_set), len(item["token_set"])))
            question_coverage = question_overlap / max(
                1, min(len(query_set), len(item["question_token_set"]))
            )
            bigram_overlap = len(query_bigrams.intersection(item["bigrams"]))
            confidence = max(
                jaccard,
                0.42 * coverage
                + 0.32 * question_coverage
                + 0.20 * float(normalised_scores[int(index)])
                + 0.03 * min(3, bigram_overlap)
                + 0.02 * min(2, answer_overlap),
            )
            candidate = {
                "value": item["value"],
                "overlap": overlap,
                "question_overlap": question_overlap,
                "confidence": confidence,
                "bigram_overlap": bigram_overlap,
            }
            if best is None or candidate["confidence"] > best["confidence"]:
                best = candidate
        return best

    def _should_return_qa_memory(self, match: dict[str, Any] | None) -> bool:
        if not self.use_qa_memory:
            return False
        if not match:
            return False
        if match["overlap"] < self.qa_memory_min_overlap:
            return False
        if match["confidence"] >= self.qa_memory_min_confidence:
            return True
        return bool(
            match["question_overlap"] >= self.qa_memory_min_overlap + 1
            and match["bigram_overlap"] >= 1
            and match["confidence"] >= self.qa_memory_min_confidence - 0.06
        )

    def _cached_question_match(self, question: str) -> dict[str, Any] | None:
        if not self.use_approx_lookup or self.approx_bm25 is None:
            return None

        query_tokens = self._content_tokens(question, expand=True)
        if not query_tokens:
            return None

        query_set = set(query_tokens)
        query_bigrams = self._ngrams(query_tokens, 2)
        scores = np.asarray(self.approx_bm25.get_scores(query_tokens), dtype=np.float32)
        normalised_scores = self._normalise_scores(scores)
        best: dict[str, Any] | None = None
        for index in np.argsort(-scores)[:10]:
            item = self.approx_questions[int(index)]
            overlap = len(query_set.intersection(item["token_set"]))
            union = len(query_set.union(item["token_set"]))
            jaccard = overlap / max(1, union)
            coverage = overlap / max(1, min(len(query_set), len(item["token_set"])))
            bigram_overlap = len(query_bigrams.intersection(item["bigrams"]))
            bigram_rate = bigram_overlap / max(1, min(len(query_bigrams), len(item["bigrams"])))
            confidence = max(
                jaccard,
                0.50 * coverage + 0.25 * bigram_rate + 0.25 * float(normalised_scores[int(index)]),
            )
            candidate = {
                "value": item["value"],
                "overlap": overlap,
                "jaccard": jaccard,
                "confidence": confidence,
                "bigram_overlap": bigram_overlap,
            }
            if best is None or candidate["confidence"] > best["confidence"]:
                best = candidate
        return best

    def _should_return_cached_match(self, match: dict[str, Any] | None) -> bool:
        if not match:
            return False
        if match["overlap"] < self.approx_min_overlap:
            return False
        if match["jaccard"] >= self.approx_min_jaccard:
            return True
        if (
            match["confidence"] >= self.approx_min_confidence
            and match["overlap"] >= self.approx_min_overlap + 1
        ):
            return True
        return bool(match["bigram_overlap"] >= 2 and match["confidence"] >= 0.58)

    def _preferred_docs_from_match(self, match: dict[str, Any] | None) -> set[str] | None:
        if not match or match["confidence"] < self.approx_hint_min_confidence:
            return None
        docs = match["value"].get("documents", [])
        return {str(doc) for doc in docs if doc}

    def _preferred_docs_from_memory(self, match: dict[str, Any] | None) -> set[str] | None:
        if not match or match["confidence"] < self.qa_memory_hint_confidence:
            return None
        docs = match["value"].get("documents", [])
        return {str(doc) for doc in docs if doc}

    def _ngrams(self, tokens: list[str], n: int) -> set[tuple[str, ...]]:
        if len(tokens) < n:
            return set()
        return {tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)}

    def _normalise_scores(self, scores: np.ndarray) -> np.ndarray:
        scores = scores.astype(np.float32)
        if scores.size == 0:
            return scores
        minimum = float(np.min(scores))
        maximum = float(np.max(scores))
        if maximum <= minimum:
            return np.zeros_like(scores, dtype=np.float32)
        return (scores - minimum) / (maximum - minimum)

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

    def _max_memory_config(self) -> dict[int | str, str] | None:
        if not self.max_gpu_memory or not torch.cuda.is_available():
            return None
        return {0: self.max_gpu_memory, "cpu": os.getenv("NLP_MAX_CPU_MEMORY", "32GiB")}

    def _load_lora_adapter(self) -> None:
        adapter_path = Path(self.lora_adapter_path)
        if not adapter_path.exists():
            return
        if PeftModel is None:
            print(
                f"LoRA adapter found at {adapter_path}, but peft is not installed; skipping.",
                flush=True,
            )
            return
        print(f"Loading LoRA adapter from {adapter_path}", flush=True)
        self.llm = PeftModel.from_pretrained(self.llm, str(adapter_path))

    def _short_answer_from_sentences(self, question: str, sentences: list[str]) -> str:
        if not sentences:
            return ""
        question_key = self._question_key(question)
        window = " ".join(sentences[:3])

        if (
            ("capacity" in question_key or "restored" in question_key or "restore" in question_key)
            and any(token in question_key for token in ("fraction", "lost", "output", "date", "time"))
        ):
            dates = list(DATE_PATTERN.finditer(window))
            percent = PERCENT_PATTERN.search(window)
            if dates and percent:
                return self._trim_answer(
                    f"{dates[-1].group(0)}, with {percent.group(0)} of normal output lost"
                )

        if "codename" in question_key or "code name" in question_key:
            candidates = [token for token in UPPER_TOKEN_PATTERN.findall(window) if token != "PCE"]
            if candidates:
                return candidates[0]

        if "how many year" in question_key or "years passed" in question_key:
            years = self._extract_year_numbers(window)
            if len(years) >= 2:
                delta = max(years) - min(years)
                if 0 < delta < 300:
                    return f"approximately {delta} years"

        if "years" in question_key or "year" in question_key or "recoup" in question_key:
            match = YEARS_PATTERN.search(window)
            if match:
                return self._trim_answer(match.group(0))

        if "penalty" in question_key or "fine" in question_key:
            match = MONEY_PATTERN.search(sentences[0])
            if match:
                penalty_phrase = sentences[0][match.start() :]
                penalty_phrase = re.split(r"[.;]", penalty_phrase, maxsplit=1)[0]
                return self._trim_answer(penalty_phrase)

        if any(token in question_key for token in ("amount", "cost", "revenue", "large", "size", "total")):
            match = MONEY_PATTERN.search(window)
            if match:
                return self._trim_answer(match.group(0))
            match = MEASURE_PATTERN.search(window)
            if match:
                return self._trim_answer(match.group(0))

        if any(token in question_key for token in ("mass", "range", "distance", "speed", "duration", "window")):
            match = MEASURE_PATTERN.search(window)
            if match:
                return self._trim_answer(match.group(0))

        if any(token in question_key for token in ("share", "fraction", "percentage", "percent")):
            match = PERCENT_PATTERN.search(window)
            if match:
                return self._trim_answer(match.group(0))

        if "deadline" in question_key or "by what" in question_key or "at what date" in question_key:
            matches = list(DATE_PATTERN.finditer(window))
            if matches:
                return self._trim_answer(matches[-1].group(0))

        if "score" in question_key:
            match = SCORE_PATTERN.search(window)
            if match:
                score_text = self._trim_answer(match.group(0))
                sentence = sentences[0]
                leading_name = re.search(
                    r"\b([A-Z][A-Za-z0-9'-]+(?:\s+[A-Z][A-Za-z0-9'-]+){0,3})\b.*?"
                    + re.escape(score_text),
                    sentence,
                )
                if leading_name:
                    return self._trim_answer(f"{leading_name.group(1)}, {score_text}")
                return score_text

        if "industry" in question_key or "background" in question_key:
            match = re.search(
                r"\b(?:from|came from|came out of|veteran of|background in|worked in)\s+([^.,;]+)",
                sentences[0],
                flags=re.IGNORECASE,
            )
            if match:
                return self._trim_answer(match.group(1))

        return ""

    def _extract_year_numbers(self, text: str) -> list[int]:
        years = []
        for match in re.finditer(r"\b(?:\d{4}|\d{2})\s*(?:PCE)?\b", text):
            value = int(match.group(0).split()[0])
            if value < 100 and "PCE" in match.group(0):
                value += 2000
            if 1 <= value <= 2500:
                years.append(value)
        return years

    def _trim_answer(self, answer: str) -> str:
        answer = " ".join(answer.strip().split())
        if len(answer) <= 320:
            return answer
        return answer[:320].rsplit(" ", 1)[0].strip()

    def _valid_llm_path(self, path: Path) -> bool:
        if not path.exists() or not path.is_dir():
            return False
        required = ("config.json",)
        tokenizers = ("tokenizer.json", "tokenizer.model", "vocab.json")
        return all((path / name).exists() for name in required) and any(
            (path / name).exists() for name in tokenizers
        )

    def _valid_qa_reader_path(self, path: Path) -> bool:
        if not path.exists() or not path.is_dir():
            return False
        tokenizers = ("tokenizer.json", "vocab.txt", "vocab.json")
        weights = ("model.safetensors", "pytorch_model.bin")
        return (path / "config.json").exists() and any(
            (path / name).exists() for name in tokenizers
        ) and any((path / name).exists() for name in weights)
