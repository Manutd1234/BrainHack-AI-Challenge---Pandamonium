# BrainHack_V2

TIL-AI 2026 model containers based on the official `til-ai/til-26` template.

- `asr/`: MERaLiON-2-3B + DeepFilterNet3, serving `/asr` on port `5001`.
- `cv/`: YOLO26x + SAHI, serving `/cv` on port `5002`.
- `noise/`: PGD adversarial noising with a YOLO surrogate, serving `/noise` on port `5003`.
- `nlp/`: quantized Qwen3 + BGE RAG, serving `/nlp` on port `5004`.
- `ae/`: PPO-ready Bomberman agent with a BFS rule fallback, serving `/ae` on port `5005`.
