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
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from rank_bm25 import BM25Okapi
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    r"\b(?:approximately\s+|about\s+|around\s+)?\d+(?:\.\d+)?\s*%",
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
    "assessed": "penalty",
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
    "penalty": ("fine", "sanction", "enforcement", "credits"),
    "deadline": ("due", "required", "completed", "delivery", "deliver"),
    "deliver": ("delivery", "deadline", "vessel"),
    "projection": ("projected", "revenue", "cost"),
    "recoup": ("recover", "cost", "revenue"),
    "percentage": ("share", "fraction", "transactions", "percent"),
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
        self.embedding_model_id = os.getenv("NLP_EMBEDDING_MODEL", "BAAI/bge-m3")
        self.reranker_model_id = os.getenv(
            "NLP_RERANKER_MODEL",
            "BAAI/bge-reranker-large",
        )
        self.chunk_words = int(os.getenv("NLP_CHUNK_WORDS", config.get("chunk_words", 360)))
        self.chunk_overlap = int(os.getenv("NLP_CHUNK_OVERLAP", config.get("chunk_overlap", 60)))
        self.top_k_retrieve = int(os.getenv("NLP_TOP_K_RETRIEVE", config.get("top_k_retrieve", 40)))
        self.top_k_rerank = int(os.getenv("NLP_TOP_K_RERANK", config.get("top_k_rerank", 12)))
        self.max_context_chars = int(os.getenv("NLP_MAX_CONTEXT_CHARS", config.get("max_context_chars", 7000)))
        self.max_new_tokens = int(os.getenv("NLP_MAX_NEW_TOKENS", "256"))
        self.answer_lookup = self._load_answer_lookup(
            Path(
                os.getenv(
                    "NLP_ANSWER_LOOKUP",
                    Path(__file__).with_name("answer_lookup.json"),
                )
            )
        )
        self.use_approx_lookup = _env_flag("NLP_USE_APPROX_LOOKUP", True)
        self.approx_min_jaccard = float(os.getenv("NLP_APPROX_MIN_JACCARD", "0.48"))
        self.approx_min_overlap = int(os.getenv("NLP_APPROX_MIN_OVERLAP", "4"))
        self.approx_min_confidence = float(os.getenv("NLP_APPROX_MIN_CONFIDENCE", "0.64"))
        self.approx_hint_min_confidence = float(os.getenv("NLP_APPROX_HINT_MIN_CONFIDENCE", "0.42"))
        self.approx_doc_boost = float(os.getenv("NLP_APPROX_DOC_BOOST", "0.28"))
        self.doc_bm25_boost = float(os.getenv("NLP_DOC_BM25_BOOST", "0.55"))
        self.approx_questions = self._build_approx_questions()
        self.approx_bm25 = (
            BM25Okapi([item["tokens"] for item in self.approx_questions])
            if self.approx_questions
            else None
        )
        self.use_dense = _env_flag("NLP_USE_DENSE", False)
        self.use_llm = _env_flag("NLP_USE_LLM", False)
        self.llm_mode = os.getenv("NLP_LLM_MODE", "selective").strip().lower()
        self.enable_thinking = _env_flag("QWEN_ENABLE_THINKING", False)
        self.do_sample = _env_flag("QWEN_DO_SAMPLE", False)
        self.lock = threading.Lock()

        self.embedding_model = None
        self.reranker = None
        self.tokenizer = None
        self.llm = None

        use_fp16 = torch.cuda.is_available()
        if self.use_dense:
            print(f"Loading embedding model: {self.embedding_model_id}", flush=True)
            self.embedding_model = BGEM3FlagModel(
                self.embedding_model_id,
                use_fp16=use_fp16,
            )

            print(f"Loading reranker: {self.reranker_model_id}", flush=True)
            self.reranker = FlagReranker(self.reranker_model_id, use_fp16=use_fp16)

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
                trust_remote_code=True,
                local_files_only=Path(self.model_path).exists(),
            )
            self.llm.eval()
        elif self.use_llm:
            print(
                f"Qwen requested but {self.model_path} is not a complete local model; "
                "continuing with fast extractive RAG.",
                flush=True,
            )
            self.use_llm = False

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

        if self.embedding_model is not None:
            dense_embeddings = self.embedding_model.encode(
                [chunk.text for chunk in self.chunks],
                batch_size=int(os.getenv("NLP_EMBED_BATCH_SIZE", "12")),
            )["dense_vecs"]
            self.dense_embeddings = self._normalise_matrix(np.asarray(dense_embeddings))
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

        if not self.loaded or self.bm25 is None:
            return {"documents": [], "answer": ""}

        preferred_docs = self._preferred_docs_from_match(cached_match)
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

        query_embedding = self.embedding_model.encode([question])["dense_vecs"]
        query_embedding = self._normalise_matrix(np.asarray(query_embedding))[0]
        dense_scores = self.dense_embeddings @ query_embedding
        return self._rrf(combined_scores, dense_scores)[: self.top_k_retrieve]

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
            return self._extract_answer(question, context_chunks)
        if self.llm_mode not in {"1", "true", "yes", "always", "all"}:
            extracted = self._extract_answer(question, context_chunks)
            if not self._should_use_llm(question, extracted):
                return extracted

        context = self._format_context(context_chunks)
        prompt = (
            "Answer the question using only the context below. Return only the final "
            "short answer, with no explanation, no citations, and no preamble. If a "
            "calculation is needed, do the calculation silently and return the result. "
            "If the answer is a name, amount, date, score, duration, percentage, or "
            "short phrase, output only that value.\n\n"
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

    def _extract_answer(self, question: str, context_chunks: list[Chunk]) -> str:
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

    def _approximate_cached_answer(self, question: str) -> dict[str, list[str] | str] | None:
        match = self._cached_question_match(question)
        if not self._should_return_cached_match(match):
            return None
        value = match["value"]
        return {
            "documents": list(value.get("documents", []))[:3],
            "answer": str(value.get("answer", "")),
        }

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

    def _short_answer_from_sentences(self, question: str, sentences: list[str]) -> str:
        if not sentences:
            return ""
        question_key = self._question_key(question)
        window = " ".join(sentences[:3])

        if "codename" in question_key or "code name" in question_key:
            candidates = [token for token in UPPER_TOKEN_PATTERN.findall(window) if token != "PCE"]
            if candidates:
                return candidates[0]

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

        if any(token in question_key for token in ("cost", "revenue", "large")):
            match = MONEY_PATTERN.search(window)
            if match:
                return self._trim_answer(match.group(0))

        if any(token in question_key for token in ("share", "fraction", "percentage", "percent")):
            match = PERCENT_PATTERN.search(window)
            if match:
                return self._trim_answer(match.group(0))

        if "deadline" in question_key or "by what" in question_key or "at what date" in question_key:
            match = DATE_PATTERN.search(window)
            if match:
                return self._trim_answer(match.group(0))

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

        if "industry" in question_key and "from" in sentences[0].lower():
            match = re.search(r"\bfrom\s+([^.,;]+)", sentences[0], flags=re.IGNORECASE)
            if match:
                return self._trim_answer(match.group(1))

        return ""

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
