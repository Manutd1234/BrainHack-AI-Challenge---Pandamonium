"""Pre-download NLP assets for offline Docker/runtime use."""

from __future__ import annotations

import argparse
import os

from huggingface_hub import snapshot_download
from sentence_transformers import CrossEncoder, SentenceTransformer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("NLP_EMBEDDING_MODEL", "BAAI/bge-m3"),
    )
    parser.add_argument(
        "--qwen-model",
        default=os.getenv("QWEN_MODEL", "Qwen/Qwen3.6-27B"),
    )
    parser.add_argument(
        "--reranker-model",
        default=os.getenv("NLP_RERANKER_MODEL", "BAAI/bge-reranker-large"),
    )
    parser.add_argument(
        "--skip-qwen",
        action="store_true",
        help="Only download the embedding model. Useful for lightweight tests.",
    )
    args = parser.parse_args()

    SentenceTransformer(args.embedding_model, trust_remote_code=True)
    CrossEncoder(args.reranker_model)
    if not args.skip_qwen:
        snapshot_download(args.qwen_model)

    print("NLP model assets downloaded.")


if __name__ == "__main__":
    main()
