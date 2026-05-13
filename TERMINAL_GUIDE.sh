#!/bin/bash
# ════════════════════════════════════════════════════════════════
# TIL-AI 2026 — COMPLETE TERMINAL GUIDE
# ════════════════════════════════════════════════════════════════
# Paste each section into your GCP Workbench terminal in order.
# Do NOT run this as a single script — follow step by step.
# ════════════════════════════════════════════════════════════════


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 0: VERIFY GCP AUTHENTICATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 0a. Check which GCP account is active
gcloud auth list

# 0b. Set the correct service account (replace TEAM-NAME with yours, e.g. pandamonium)
gcloud config set account svc-pandamonium@til-ai-2026.iam.gserviceaccount.com

# 0c. Verify the correct project is set
gcloud config set project til-ai-2026

# 0d. Configure Docker to push to Artifact Registry
gcloud auth configure-docker asia-southeast1-docker.pkg.dev

# 0e. Verify Docker is running
docker info > /dev/null 2>&1 && echo "✓ Docker is running" || echo "✗ Docker NOT running"

# 0f. Verify GPU is available
nvidia-smi > /dev/null 2>&1 && echo "✓ GPU available" || echo "⚠ No GPU detected"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 1: PULL LATEST CODE FROM GITHUB
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cd /home/jupyter/til-26
git pull origin main


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 2: ENSURE MODELS DIRECTORIES EXIST
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

mkdir -p asr/models cv/models noise/models nlp/models ae/models

# Check if fine-tuned CV model exists
ls -la cv/models/best.pt 2>/dev/null && echo "✓ Fine-tuned CV model found" || echo "⚠ No best.pt — CV will score ~0 on TIL-26 classes"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 3: BUILD ALL DOCKER IMAGES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

til build asr v2
til build cv v2
til build noise v2
til build nlp v2
til build ae v2

# Verify all images were built
docker image ls | grep pandamonium


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 4: TEST ALL IMAGES LOCALLY (runs offline, scores models)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

til test asr v2
til test cv v2
til test noise v2
til test nlp v2
til test ae v2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 5: SUBMIT ALL IMAGES FOR EVALUATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

til submit asr v2
til submit cv v2
til submit noise v2
til submit nlp v2
til submit ae v2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DONE! Check your results:
#   → Discord team channel for evaluation notifications
#   → Leaderboard in the Strategist's Handbook
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


# ════════════════════════════════════════════════════════════════
# TROUBLESHOOTING
# ════════════════════════════════════════════════════════════════

# If "til" command not found:
#   → You must be on the GCP Workbench instance, not local machine

# If authentication fails:
#   gcloud auth activate-service-account svc-pandamonium@til-ai-2026.iam.gserviceaccount.com --key-file=/path/to/key.json
#   gcloud auth configure-docker asia-southeast1-docker.pkg.dev

# If Docker push fails (permission denied):
#   gcloud auth print-access-token | docker login -u oauth2accesstoken --password-stdin https://asia-southeast1-docker.pkg.dev

# If a single build/test/submit fails, retry individually:
#   til build TASK v2
#   til test TASK v2
#   til submit TASK v2

# To re-tag an existing image:
#   docker tag pandamonium-asr:v2 pandamonium-asr:latest
