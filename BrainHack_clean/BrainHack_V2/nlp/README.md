# NLP

Your NLP challenge is to answer questions using RAG.

## Input

The first `/nlp` request loads the corpus:

```JSON
{
  "instances": [
    {
      "documents": [
        {"id": "DOC-0001", "document": "Text of document one."},
        {"id": "DOC-0002", "document": "Text of document two."}
      ]
    }
  ]
}
```

Poll with `{"instances": [{"poll": true}]}` until the response status is
`loaded`.

Question requests use the TIL route on port `5004`:

```JSON
{
  "instances": [
    {"question": "QUESTION_TEXT"}
  ]
}
```

The response shape is:

```Python
{
    "predictions": [
        {"documents": ["DOC-0001"], "answer": "Answer text."}
    ]
}
```

## Implementation

This container uses:

- BM25 for sparse retrieval.
- BGE-M3 for dense retrieval.
- Reciprocal Rank Fusion to merge sparse and dense ranks.
- BGE-Reranker for the final context shortlist.
- A local quantized Qwen3 AWQ model in `src/qwen-quantized`.

Prepare the quantized model folder before building:

```bash
python quantize.py
```

Build and run:

```bash
docker build -t pandamonium-nlp:v3 .
docker run --gpus all -p 5004:5004 pandamonium-nlp:v3
```
