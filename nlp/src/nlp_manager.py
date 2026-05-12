"""Manages the NLP model — RAG-based Question Answering."""

import logging
import re
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class NLPManager:
    loaded = False

    def __init__(self):
        self.chunks: list[str] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix = None
        self.chunk_size = 400  # characters per chunk
        self.chunk_overlap = 150  # more overlap for better context
        self.top_k = 8  # retrieve more chunks for better coverage

    def _chunk_text(self, text: str) -> list[str]:
        """Splits text into overlapping chunks, respecting sentence boundaries."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_len = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if current_len + len(sentence) > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                # Keep some sentences for overlap
                overlap_text = " ".join(current_chunk)
                overlap_start = max(0, len(overlap_text) - self.chunk_overlap)
                overlap = overlap_text[overlap_start:]
                current_chunk = [overlap] if overlap.strip() else []
                current_len = len(overlap)

            current_chunk.append(sentence)
            current_len += len(sentence) + 1

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return [c for c in chunks if c.strip()]

    def load_corpus(self, documents: list[str]) -> None:
        """Loads and indexes the corpus of documents for RAG QA."""
        logger.info(f"Loading corpus of {len(documents)} documents...")

        # Chunk all documents
        self.chunks = []
        for doc in documents:
            self.chunks.extend(self._chunk_text(doc))

        logger.info(f"Created {len(self.chunks)} chunks from {len(documents)} documents.")

        # Build TF-IDF index for retrieval
        self.vectorizer = TfidfVectorizer(
            max_features=80000,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)

        logger.info("TF-IDF index built successfully.")
        self.loaded = True

    def _retrieve(self, question: str) -> list[str]:
        """Retrieves the most relevant chunks for a question."""
        if self.vectorizer is None or self.tfidf_matrix is None:
            return []

        query_vec = self.vectorizer.transform([question])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Get top-k most similar chunks
        top_indices = np.argsort(similarities)[-self.top_k:][::-1]
        return [self.chunks[i] for i in top_indices if similarities[i] > 0.01]

    def qa(self, question: str) -> str:
        """Performs question answering using retrieved context.

        Args:
            question: The question to answer.

        Returns:
            A string containing the answer to the question.
        """
        if not self.loaded:
            return ""

        # Retrieve relevant context
        context_chunks = self._retrieve(question)

        if not context_chunks:
            return ""

        # Build a combined context from retrieved chunks
        context = " ".join(context_chunks)

        # Split into sentences for ranking
        sentences = re.split(r'(?<=[.!?])\s+', context)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if not sentences:
            return context_chunks[0][:200] if context_chunks else ""

        # Use TF-IDF similarity to rank sentences against the question
        try:
            sent_vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
            )
            all_texts = [question] + sentences
            sent_matrix = sent_vectorizer.fit_transform(all_texts)
            q_vec = sent_matrix[0:1]
            s_matrix = sent_matrix[1:]
            sims = cosine_similarity(q_vec, s_matrix).flatten()

            # Get the top-3 most relevant sentences
            top_indices = np.argsort(sims)[-3:][::-1]
            best_sentences = [sentences[i] for i in top_indices if sims[i] > 0]

            if best_sentences:
                # Return the best matching sentence (or combine top 2 if short)
                answer = best_sentences[0]
                if len(answer) < 50 and len(best_sentences) > 1:
                    answer = " ".join(best_sentences[:2])
                return answer

        except Exception as e:
            logger.warning(f"Sentence ranking failed: {e}")

        # Fallback: return the first chunk
        return context_chunks[0][:200]
