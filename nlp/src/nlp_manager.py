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
        self.chunk_size = 500  # characters per chunk
        self.chunk_overlap = 100
        self.top_k = 5  # number of chunks to retrieve

    def _chunk_text(self, text: str) -> list[str]:
        """Splits text into overlapping chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk.strip())
            start += self.chunk_size - self.chunk_overlap
        return chunks

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
            max_features=50000,
            stop_words="english",
            ngram_range=(1, 2),
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
        return [self.chunks[i] for i in top_indices if similarities[i] > 0]

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

        # Simple extractive QA: find the most relevant sentence
        # from the retrieved chunks that best answers the question
        context = " ".join(context_chunks)
        sentences = re.split(r'[.!?]+', context)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return ""

        # Score each sentence against the question using word overlap
        question_words = set(question.lower().split())
        best_sentence = ""
        best_score = -1

        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            overlap = len(question_words & sentence_words)
            if overlap > best_score:
                best_score = overlap
                best_sentence = sentence

        return best_sentence
