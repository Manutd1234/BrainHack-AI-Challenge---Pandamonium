# TIL-AI 2026 — Full Pipeline Setup

## Repo structure

```
til-26/
├── build_all.sh          ← builds all 5 Docker images
├── asr/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── download_models.py   ← run ONCE before build
│   ├── models/              ← whisper weights land here (gitignored)
│   └── src/server.py
├── cv/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── download_models.py
│   ├── til26.yaml           ← YOLO dataset config
│   ├── models/              ← put best.pt here after fine-tuning
│   └── src/server.py
├── noise/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── src/server.py        ← no model weights needed (pure algorithm)
├── nlp/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── download_models.py
│   ├── models/              ← BGE + Mistral weights (gitignored)
│   └── src/server.py
└── ae/
    ├── Dockerfile
    ├── requirements.txt
    ├── train.py             ← run on Workbench to train PPO
    ├── models/              ← trained policy.zip lands here
    └── src/server.py
```

---

## Day-by-day setup plan

### Day 1 — Environment + dummy submission

```bash
cd /home/jupyter/
git clone https://github.com/YOUR_USERNAME/til-26.git
cd til-26
git submodule update --init
conda create --name env python=3.12
conda activate env
pip install -r requirements-dev.txt
```

Build a dummy image just to verify the submission pipeline works:
```bash
til build asr dummy
til test asr dummy
til submit asr dummy    # sends a dummy model — confirms pipeline OK
```

---

### Day 1–2 — Start AE training (RL needs time!)

```bash
cd /home/jupyter/til-26
conda activate env
pip install stable-baselines3 gymnasium
# Copy our train.py into the ae/ directory and run:
python ae/train.py
# This will run for hours — leave it going in a tmux session
```

Keep training running in background:
```bash
tmux new -s ae_train
python ae/train.py
# Ctrl+B then D to detach
```

---

### Day 2 — Download model weights

```bash
# ASR
cd /home/jupyter/til-26/asr
pip install faster-whisper
python download_models.py

# CV — download base weights
cd /home/jupyter/til-26/cv
pip install ultralytics
python download_models.py

# NLP — large download (~15GB), start early!
cd /home/jupyter/til-26/nlp
pip install sentence-transformers transformers bitsandbytes accelerate
python download_models.py
```

---

### Day 2–3 — Fine-tune CV

Edit `til26.yaml` to point to your training data path. Then:

```bash
cd /home/jupyter/til-26/cv
yolo train \
    model=models/cv/yolo11x.pt \
    data=til26.yaml \
    epochs=50 \
    imgsz=1280 \
    batch=8 \
    device=0 \
    project=runs/cv \
    name=til26_v1

# Copy best weights:
cp runs/cv/til26_v1/weights/best.pt models/cv/best.pt
```

---

### Day 3 — Fine-tune ASR Whisper (optional but recommended)

```bash
# Use HuggingFace Whisper fine-tuning on the TIL-AI ASR training data
# See: https://huggingface.co/blog/fine-tune-whisper
# After fine-tuning, convert to CTranslate2 format:
pip install ctranslate2
ct2-opus-mt-converter --model_name_or_path ./whisper-finetuned \
    --output_dir asr/models/whisper-large-v3-finetuned \
    --quantization float16
```

---

### Day 4–5 — Build & test all Docker images

```bash
cd /home/jupyter/til-26
chmod +x build_all.sh
./build_all.sh YOUR_TEAM_ID v1

# Test each
til test asr v1
til test cv v1
til test noise v1
til test nlp v1
til test ae v1
```

---

### Before May 24 — Submit

```bash
til submit asr v1
til submit cv v1
til submit noise v1
til submit nlp v1
til submit ae v1
```

---

## Ports reference

| Task  | Port | Route  |
|-------|------|--------|
| ASR   | 5001 | POST /asr   |
| CV    | 5002 | POST /cv    |
| Noise | 5003 | POST /noise |
| NLP   | 5004 | POST /nlp   |
| AE    | 5005 | POST /ae, GET /reset |

---

## Key tips

- **Weights go in `models/` (not `src/`)** — gitignored, but COPY'd into Docker
- Container has **no internet** at eval time — all weights must be bundled in
- Submit early with a basic working model; improve iteratively
- AE training takes the longest — start it first and leave it running overnight
- NLP model download is ~15GB — start early, use `nohup` or `tmux`
