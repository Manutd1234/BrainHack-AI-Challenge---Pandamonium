# BrainHack V2: Team Pandamonium TIL-AI 2026 Finals Stack

This repository contains Team Pandamonium's model containers for the TIL-AI 2026 Novice Finals. It is based on the official `til-ai/til-26` template and keeps each challenge as a standalone Docker service that can be built, tested, and submitted with the `til` CLI.

The official challenge specifications define four scored model tasks: ASR, CV, NLP, and AE. Each task score is weighted 75% accuracy or reward and 25% inference speed. The overall score weights are ASR 20%, CV 20%, NLP 20%, and AE 40%. Noise is a finals gameplay component that adversarially perturbs opponents' CV inputs while respecting the image-similarity constraints.

## Repository Layout

```text
BrainHack_V2/
├── ae/       # Autonomous Exploration agent, MaskablePPO plus BFS/A* fallback logic
├── asr/      # Parakeet ASR, fine-tuned decoder checkpoints, correction pipeline
├── cv/       # YOLOv11/Ultralytics detector pipeline for aircraft, vehicles, and vessels
├── nlp/      # BM25-first RAG with optional dense/reranker/LLM components
└── noise/    # YOLO-surrogate adversarial image noise generator
```

## Runtime Endpoints

| Task | Route | Port | Output contract |
| --- | --- | --- | --- |
| ASR | `POST /asr` | `5001` | `{"predictions": ["transcript", ...]}` |
| CV | `POST /cv` | `5002` | `{"predictions": [[{"bbox": [l,t,w,h], "category_id": int}], ...]}` |
| Noise | `POST /noise` | `5003` | `{"predictions": ["BASE64_JPEG", ...]}` |
| NLP | `POST /nlp` | `5004` | Corpus load, poll, and question-answer responses |
| AE | `POST /ae`, `GET /reset` | `5005` | `{"predictions": [{"action": int}]}` |

## Build, Test, Submit

Set the repo path once per shell:

```bash
export TIL_FOLDER=/home/jupyter/BrainHack_clean/BrainHack_V2
```

Build and test an individual task:

```bash
cd "$TIL_FOLDER"
til build asr v43
til test asr v43
```

Submit to automatic evaluation:

```bash
til submit asr v43
```

Repeat the same pattern for `ae`, `cv`, `nlp`, and `noise`.

## Model Summary

### ASR

The ASR module uses NVIDIA Parakeet TDT as the acoustic model. The strongest local checkpoint is a decoder fine-tuned Parakeet v2 model with safe phrase corrections, TF32/FP16 inference, sorted batching, persistent temporary WAV reuse, and a carefully selected audio duration cap. See [asr/README.md](asr/README.md).

### CV

The CV module uses an Ultralytics YOLOv11-style detector workflow, with threshold tuning, optional model ensembling, optional high-resolution fallback, and LTWH output conversion for the challenge contract. See [cv/README.md](cv/README.md).

### Noise

The Noise module uses a YOLO surrogate to generate adversarial image perturbations under SSIM/RMSE-style visual constraints. When the surrogate checkpoint is missing, it falls back to deterministic bounded noise. See [noise/README.md](noise/README.md).

### NLP

The NLP module is BM25-first RAG. It indexes the provided corpus at runtime, retrieves candidate chunks/documents with sparse lexical scoring, optionally augments/reranks with neural components, and answers with extractive heuristics or a quantized LLM when enabled. See [nlp/README.md](nlp/README.md).

### AE

The AE module combines MaskablePPO training with feature engineering, action masking, reward shaping, and a rule-based BFS/A* fallback for robust navigation in the fixed Novice map. See [ae/README.md](ae/README.md).

## Files Not Committed

Large model artifacts are intentionally expected to be copied into each task's `model/` directory before build. The repository keeps `.gitkeep` placeholders where checkpoints should be placed. This avoids accidentally pushing multi-GB `.pt`, `.zip`, or `.nemo` files.

## Useful Official References

- Challenge specifications: https://github.com/til-ai/til-26/wiki/Challenge-specifications
- Template repository: https://github.com/til-ai/til-26
- Finals repository: https://github.com/til-ai/til-26-finals
- Finals submission flow: https://github.com/til-ai/til-26/wiki/Finals-competition-flow#how-to-submit
