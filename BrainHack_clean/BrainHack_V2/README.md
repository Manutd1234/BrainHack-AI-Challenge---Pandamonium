# 🧠 BrainHack_V2 — Team Pandamonium

> **TIL-AI 2026 · Novice Track**
> Five containerized AI microservices for the TIL-AI 2026 hackathon.

---

## All 5 Tasks

All builds use the `BrainHack_V2` directory (the best implementations):

| Task | Model | Source Dir | Port |
|------|-------|------------|------|
| **ASR** | Parakeet-TDT 0.6B | `BrainHack_V2/asr` | `5001` |
| **CV** | YOLO26x + RF-DETR ensemble | `BrainHack_V2/cv` | `5002` |
| **Noise** | PGD w/ YOLO surrogate (speed-optimized) | `BrainHack_V2/noise` | `5003` |
| **NLP** | BGE-M3 RAG + Reranker | `BrainHack_V2/nlp` | `5004` |
| **AE** | SB3 PPO RL agent | `BrainHack_V2/ae` | `5005` |

---

## 🚀 Quick Start (GCP Workbench)

```bash
# 0. Auth & environment
cd ~
export TEAM_NAME=pandamonium
export TEAM_TRACK=novice
export TIL_FOLDER=/home/jupyter
gcloud auth configure-docker asia-southeast1-docker.pkg.dev

# 1. Pull latest code
cd ~/BrainHack_V2
git pull origin main

# 2. Prep: copy CV model as noise surrogate
cp ~/BrainHack_V2/cv/model/best.pt ~/BrainHack_V2/noise/model/best.pt

# 3. Build all 5 images
til build asr v20
til build cv v20
til build noise v20
til build nlp v20
til build ae v20

# 4. Test locally (optional)
til test asr v20
til test cv v20
til test noise v20
til test nlp v20
til test ae v20

# 5. Submit for evaluation
til submit asr v20
til submit cv v20
til submit noise v20
til submit nlp v20
til submit ae v20
```

---

## 📂 Project Structure

```
BrainHack_V2/
├── asr/                    # Automatic Speech Recognition
│   ├── Dockerfile
│   ├── src/
│   │   ├── asr_manager.py  # Parakeet-TDT 0.6B + optional Whisper rescue
│   │   └── asr_server.py   # FastAPI server on port 5001
│   ├── model/              # Pre-downloaded Parakeet .nemo checkpoint
│   └── requirements.txt
│
├── cv/                     # Computer Vision (Object Detection)
│   ├── Dockerfile
│   ├── src/
│   │   ├── cv_manager.py   # YOLO26x + RF-DETR ensemble + WBF fusion
│   │   └── cv_server.py    # FastAPI server on port 5002
│   ├── model/              # best.pt (YOLO) + rfdetr_best.pt
│   └── requirements.txt
│
├── noise/                  # Adversarial Noise Generation
│   ├── Dockerfile
│   ├── src/
│   │   ├── noise_manager.py # PGD attack with YOLO surrogate (speed-optimized)
│   │   └── noise_server.py  # FastAPI server on port 5003
│   ├── model/              # Copy of cv/model/best.pt as surrogate
│   └── requirements.txt
│
├── nlp/                    # Natural Language Processing (QA)
│   ├── Dockerfile
│   ├── src/
│   │   ├── nlp_manager.py  # BGE-M3 dense+sparse retrieval + BM25 + reranker
│   │   └── nlp_server.py   # FastAPI server on port 5004
│   └── requirements.txt
│
└── ae/                     # Agent Environment (Reinforcement Learning)
    ├── Dockerfile
    ├── src/
    │   ├── ae_manager.py   # PPO policy + BFS rule-based hybrid fallback
    │   └── ae_server.py    # FastAPI server on port 5005
    ├── model/              # policy.zip (SB3 PPO checkpoint)
    └── requirements.txt
```

---

## 🔧 Task Details

### ASR — Parakeet-TDT 0.6B
- **Model**: `nvidia/parakeet-tdt-0.6b-v2` (NeMo ASR)
- **Features**: Batch transcription, fp16 autocast, optional Whisper-large-v3-turbo rescue for blank outputs, domain term correction, audio memory caching
- **Input**: Base64 WAV audio → **Output**: Transcribed text

### CV — YOLO26x + RF-DETR Ensemble
- **Primary**: Fine-tuned YOLO26x on TIL-26 dataset (18 vehicle classes)
- **Ensemble**: RF-DETR Large as secondary detector
- **Post-processing**: Weighted Box Fusion (WBF), per-class confidence thresholds, high-resolution fallback at 1536px
- **Input**: Base64 JPEG image → **Output**: COCO-format detections `[{bbox, category_id}]`

### Noise — Speed-Optimized PGD Attack
- **Method**: Projected Gradient Descent using the fine-tuned YOLO model as a white-box surrogate
- **Speed optimizations** (quality=1.000, speed improved from 0.546):
  - PGD steps: 20 → **7**
  - SSIM bisection: 24 → **8** iterations
  - Attack resolution: 1280 → **640px**
  - GPU fp16 autocast enabled
  - SSIM window: 7 → **3**
- **Input**: Base64 JPEG image → **Output**: Base64 adversarial JPEG (SSIM ≥ 0.85)

### NLP — BGE-M3 RAG Pipeline
- **Retrieval**: BGE-M3 (dense + sparse) + BM25 hybrid scoring with sentence windowing
- **Reranking**: BGE-reranker-large for top-k re-scoring
- **Features**: Approximate answer lookup, QA memory cache, configurable LLM mode (Qwen3 optional)
- **Input**: Question + document corpus → **Output**: Answer string

### AE — PPO Reinforcement Learning Agent
- **Algorithm**: Stable-Baselines3 PPO with hybrid policy
- **Fallback**: BFS rule-based agent when no trained checkpoint exists
- **Input**: Game state observation → **Output**: Action integer

---

## 📋 Important Notes

- **Version tags are arbitrary** — use any string (`v20`, `v21`, etc.). Just ensure `til build` and `til submit` use the **same tag**.
- **Noise needs the CV model** — always copy `cv/model/best.pt` to `noise/model/best.pt` before building noise.
- **No internet at runtime** — all model weights must be baked into Docker images during build.
- **GPU required** — ASR, CV, and Noise containers need NVIDIA GPU access (`--gpus all`).

---

## 👥 Team Pandamonium

TIL-AI 2026 Hackathon — Novice Track
