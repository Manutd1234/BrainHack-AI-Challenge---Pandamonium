"""Manages the NLP model — v46 with Precision Zero-Score Fallback + Re-rank Fixes."""

from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "did", "do", "does", "doing", "down",
    "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
    "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if",
    "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so",
    "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "with", "you", "your", "yours", "yourself", "yourselves"
}


def _tokenize(text: str) -> list[str]:
    """
    Exact 0.981 Tokenizer:
    Baseline alphanumeric + split tokens + collapsed punctuation shingles.
    """
    lowercased = text.lower()

    # 1. Standard alphanumeric tokens (L1 Baseline)
    tokens_alpha = re.findall(r"[\w\u4e00-\u9fff]+", lowercased)

    # 2. Whitespace split preserving compound tags (e.g., tags, IDs, versions)
    tokens_split = [t.strip(".,;:!?()[]\"'") for t in lowercased.split()]
    tokens_split = [t for t in tokens_split if t]

    # 3. L2 Punctuation Collapse Shingling
    l2_shingles = []
    for t in tokens_split:
        collapsed = re.sub(r"[-_.:/\\\s]", "", t)
        if collapsed and collapsed != t and not collapsed.isdigit():
            l2_shingles.append(collapsed)

    combined_tokens = tokens_alpha + tokens_split + l2_shingles
    return [t for t in combined_tokens if t not in STOP_WORDS]


class NLPManager:
    loaded = False

    def __init__(self):
        self.texts: list[str] = []
        self.docids: list[str] = []
        self.bm25: Optional[BM25Okapi] = None
        self.top_k_retrieve = 20
        self.doc_sets: list[set[str]] = []

    def load_corpus(self, documents: list[dict[str, str]]) -> None:
        self.texts.clear()
        self.docids.clear()
        self.doc_sets.clear()
        self.bm25 = None
        self.loaded = False

        tokenized = []
        for doc in documents:
            doc_id = doc.get("id") or doc.get("doc_id") or doc.get("filename") or "unknown"
            text = doc.get("document") or doc.get("text") or doc.get("content") or ""
            text = str(text).strip()
            if not text:
                continue

            self.texts.append(text)
            self.docids.append(doc_id)

            body_tokens = _tokenize(text)
            id_tokens = _tokenize(str(doc_id).replace("_", " ").replace("-", " "))
            full_tokens = body_tokens + id_tokens

            tokenized.append(full_tokens)
            self.doc_sets.append(set(full_tokens))

        if tokenized:
            self.bm25 = BM25Okapi(tokenized, k1=1.5, b=0.55)
        else:
            self.bm25 = None

        self.loaded = True
        logger.info("Corpus loaded with %d documents", len(self.texts))

    def qa(self, question: str) -> dict[str, list[str] | str]:
        if not self.loaded or self.bm25 is None or not self.texts:
            return {"documents": [], "answer": ""}

        q = str(question).strip()
        if not q:
            return {"documents": [], "answer": ""}

        q_tokens_list = _tokenize(q)
        if not q_tokens_list:
            q_tokens_list = re.findall(r"[\w\u4e00-\u9fff]+", q.lower())

        scores = np.asarray(self.bm25.get_scores(q_tokens_list), dtype=np.float32)
        if scores.size == 0:
            return {"documents": [], "answer": ""}

        # FIX 1: Expand re-ranking window from 5 → 10
        top_candidates = np.argsort(scores)[::-1][:10]
        q_set = set(q_tokens_list)
        q_lower = q.lower()

        if len(q_set) >= 1:
            for idx in top_candidates:
                if scores[idx] <= 0.0:
                    continue

                # FIX 2: Stronger intersection boost (exponential, not linear)
                unique_matches = len(q_set.intersection(self.doc_sets[idx]))
                if unique_matches > 1:
                    scores[idx] *= (1.0 + (0.04 * unique_matches ** 1.3))

                # FIX 3: Exact trigram phrase signal
                words = q_lower.split()
                if len(words) >= 3:
                    text_lower = self.texts[idx].lower()
                    for start in range(len(words) - 2):
                        phrase = " ".join(words[start:start + 3])
                        if phrase in text_lower:
                            scores[idx] *= 1.15
                            break

        ranked_idx = np.argsort(scores)[::-1][: self.top_k_retrieve]
        if len(ranked_idx) == 0:
            return {"documents": [], "answer": ""}

        best_doc_idx = ranked_idx[0]

        # FIX 4: Smarter zero-score fallback
        if scores[best_doc_idx] <= 0.0:
            meaningful = sorted(
                [t for t in q_tokens_list if len(t) > 3],
                key=len, reverse=True
            )[:3]

            best_sub_idx = 0
            best_sub_count = -1
            for idx, text in enumerate(self.texts):
                text_lower = text.lower()
                count = sum(text_lower.count(tok) for tok in meaningful) if meaningful \
                        else text_lower.count(q_lower[:20])
                if count > best_sub_count:
                    best_sub_count = count
                    best_sub_idx = idx
            best_doc_idx = best_sub_idx

        best_answer = self.texts[best_doc_idx].strip()

        # Build deduplicated document list
        documents: list[str] = []
        seen = set()

        chosen_id = self.docids[best_doc_idx]
        seen.add(chosen_id)
        documents.append(chosen_id)

        for idx in ranked_idx:
            doc_id = self.docids[idx]
            if doc_id not in seen:
                seen.add(doc_id)
                documents.append(doc_id)
            if len(documents) >= 3:
                break

        # Fallback padding to guarantee a clean list size
        while len(documents) < min(3, len(self.docids)):
            for d_id in self.docids:
                if d_id not in seen:
                    seen.add(d_id)
                    documents.append(d_id)
                if len(documents) >= 3:
                    break

        return {
            "documents": documents,
            "answer": best_answer,
        }