#!/bin/bash
# ════════════════════════════════════════════════════════════════
# TIL-AI 2026 — COMPLETE BUILD + SUBMIT GUIDE (v3)
# ════════════════════════════════════════════════════════════════
# Run each section in order on your GCP Workbench terminal.
# ════════════════════════════════════════════════════════════════


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 0: VERIFY GCP + DOCKER AUTHENTICATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Check active GCP account
gcloud auth list

# Set your team's service account
gcloud config set account svc-pandamonium@til-ai-2026.iam.gserviceaccount.com

# Set project
gcloud config set project til-ai-2026

# Authenticate Docker with Artifact Registry
gcloud auth configure-docker asia-southeast1-docker.pkg.dev

# Verify Docker + GPU
docker info > /dev/null 2>&1 && echo "✓ Docker OK" || echo "✗ Docker NOT running"
nvidia-smi > /dev/null 2>&1 && echo "✓ GPU OK" || echo "⚠ No GPU"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 1: PULL LATEST CODE + INSTALL HOST TEST DEPENDENCIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cd ~/BrainHack-AI-Challenge---Pandamonium
git pull origin main
mkdir -p asr/models cv/models noise/models nlp/models ae/models

# Install ALL host-side test dependencies in one shot
pip install python-dotenv jiwer pycocotools scikit-image transformers

# Install til_environment (from the official TIL-26-AE repo)
pip install git+https://github.com/til-ai/til-26-ae.git


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 2: REBUILD ALL DOCKER IMAGES (--no-cache for fresh models)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CRITICAL: --no-cache forces Docker to download the NEW models
# (large-v3, yolo11l, Qwen2.5, MobileNetV2) instead of using
# the old cached layers (small, yolo11n).

cd ~/BrainHack-AI-Challenge---Pandamonium

docker build --no-cache -t pandamonium-asr:v2 ./asr
docker build --no-cache -t pandamonium-cv:v2 ./cv
docker build --no-cache -t pandamonium-noise:v2 ./noise
docker build --no-cache -t pandamonium-nlp:v2 ./nlp
docker build --no-cache -t pandamonium-ae:v2 ./ae

# Verify all images exist
docker image ls | grep pandamonium


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 3: TEST ALL IMAGES LOCALLY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

til test asr v2
til test cv v2
til test noise v2
til test nlp v2
til test ae v2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 4: SUBMIT ALL IMAGES FOR EVALUATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

til submit asr v2
til submit cv v2
til submit noise v2
til submit nlp v2
til submit ae v2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DONE! Check results:
#   → Discord team channel for evaluation notifications
#   → Leaderboard: https://tribegroup.notion.site/33a5263ef45a80c3bad7d6006752cba4
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


# ════════════════════════════════════════════════════════════════
# TROUBLESHOOTING
# ════════════════════════════════════════════════════════════════
#
# "til" command not found:
#   → You must be on the GCP Workbench instance
#
# Docker authentication fails:
#   gcloud auth print-access-token | docker login -u oauth2accesstoken --password-stdin https://asia-southeast1-docker.pkg.dev
#
# til_environment install fails:
#   git clone https://github.com/til-ai/til-26-ae.git /tmp/til-26-ae
#   cd /tmp/til-26-ae
#   pip install .
#
# Retry a single task:
#   docker build --no-cache -t pandamonium-TASK:v2 ./TASK
#   til test TASK v2
#   til submit TASK v2
