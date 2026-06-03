# NLP: BM25-First RAG

This module serves the NLP challenge on `POST /nlp` at port `5004`.

The task is a corpus question-answering problem set in the fictional world of Clairos. The first request loads the corpus, later requests ask questions. The manager indexes the corpus in memory and answers with documents plus an answer string.

## Input and Output

Corpus load request:

```json
{
  "instances": [
    {
      "documents": [
        {"id": "DOC-0001", "document": "Document text"}
      ]
    }
  ]
}
```

Question request:

```json
{
  "instances": [
    {
      "question": "What happened?"
    }
  ]
}
```

Response:

```json
{
  "predictions": [
    {
      "documents": ["DOC-0001"],
      "answer": "Answer text"
    }
  ]
}
```

## Architecture

The production configuration is BM25-first because the Novice corpus questions are often lexical, entity-heavy, and benefit from exact document retrieval.

Main components:

- Tokenization with regex token extraction.
- Chunking by sentence/document boundaries.
- BM25 sparse retrieval using `rank_bm25`.
- Query expansion and canonical token mapping for domain terms.
- High-signal regex extractors for dates, money, percentages, years, measurements, scores, and uppercase identifiers.
- Optional dense retrieval with BGE-M3.
- Optional reranking with BGE-Reranker.
- Optional local quantized Qwen model for selective answer generation.
- Optional QA reader mode for span extraction.

The Dockerfile defaults to fast lexical mode:

```text
NLP_USE_DENSE=0
NLP_USE_LLM=0
NLP_USE_QA_READER=0
NLP_USE_APPROX_LOOKUP=1
```

This avoids GPU-heavy generation when direct retrieval is enough.

## BM25 Strategy

BM25 is used as the backbone because many questions reference:

- names,
- organizations,
- dates,
- credit amounts,
- code names,
- unique fictional-world terms,
- direct document facts.

The manager expands and canonicalizes query tokens so that related forms can still retrieve the correct chunk:

```text
amount -> credits, cost, program
deadline -> due, required, completed, delivery
penalty -> fine, sanction, enforcement
industry -> background, sector, logistics
```

## Optional LLM/Reranker Path

If the quantized model exists in `src/qwen-quantized`, it can be used selectively:

```bash
export NLP_USE_LLM=1
export QWEN_MODEL_PATH=/app/src/qwen-quantized
```

The code keeps LLM generation off by default because speed is part of the score and many questions can be answered more reliably by extraction.

## Training and Tuning

Useful scripts:

```text
build_raft_data.py      # create instruction/RAG fine-tuning data
train_raft_lora.py      # LoRA training
eval_raft_lora.py       # local evaluation
tune_rag.py             # retrieval and answer tuning
quantize.py             # quantize local Qwen checkpoint
analyze_int4.py         # inspect quantized model
```

BM25 tuning loop:

1. Load corpus and question-answer training JSONL.
2. Run local retrieval evaluation.
3. Inspect missed document IDs.
4. Add query expansions and canonical forms.
5. Increase/decrease BM25 document boost and approximate lookup thresholds.
6. Retest direct extraction before enabling neural generation.

## Build and Submit

```bash
export TIL_FOLDER=/home/jupyter/BrainHack_clean/BrainHack_V2
cd "$TIL_FOLDER/nlp"

til build nlp v1
til test nlp v1
til submit nlp v1
```

## Debug Checklist

- If the first `/nlp` request is slow, corpus indexing is running. Poll until loaded.
- If answers are empty too often, lower approximate confidence thresholds or enable QA reader.
- If latency is too high, keep dense retrieval and LLM generation disabled.
- If document IDs are wrong, inspect BM25 chunking before tuning the answer generator.
