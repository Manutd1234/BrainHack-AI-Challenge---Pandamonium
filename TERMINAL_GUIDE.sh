#!/bin/bash
# TIL-AI 2026 Pandamonium build/test/submit guide.
# Run commands from a GCP Workbench Bash terminal.

set -e

# 0. Authentication
gcloud config set account svc-pandamonium@til-ai-2026.iam.gserviceaccount.com
gcloud config set project til-ai-2026
gcloud auth configure-docker asia-southeast1-docker.pkg.dev

# 1. Pull latest pushed solution files
cd ~/BrainHack-AI-Challenge---Pandamonium
git pull --ff-only origin main
git submodule update --init --recursive

export TEAM_ID=pandamonium
export TAG="${TAG:-v4}"

# 2. Host dependencies for downloads and til test
pip install -U pip
pip install gdown faster-whisper ultralytics sahi sentence-transformers rank-bm25 openai huggingface-hub
pip install python-dotenv jiwer pycocotools scikit-image transformers stable-baselines3 gymnasium
pip install git+https://github.com/til-ai/til-26-ae.git

# 3. Dataset and model assets
python download_drive_data.py
python asr/download_models.py
python cv/download_models.py --model yolo26l.pt
python noise/download_models.py --model yolo26l.pt
python nlp/download_models.py

# Optional but recommended after training/fine-tuning:
#   cp runs/cv/.../weights/best.pt cv/models/best.pt
#   cp runs/noise/.../weights/best.pt noise/models/best.pt
#   python ae/train.py --mode advanced --envs 8 --total-steps 5000000

# 4. Build latest Docker images. Use a new tag so stale v3 images cannot be
# accidentally re-submitted.
docker build --no-cache -t "${TEAM_ID}-asr:${TAG}" ./asr
docker build --no-cache -t "${TEAM_ID}-cv:${TAG}" ./cv
docker build --no-cache -t "${TEAM_ID}-noise:${TAG}" ./noise
docker build --no-cache -t "${TEAM_ID}-ae:${TAG}" ./ae
docker build --no-cache --build-arg DOWNLOAD_QWEN=1 -t "${TEAM_ID}-nlp:${TAG}" ./nlp

docker image ls | grep "${TEAM_ID}"

# 5. Local evaluation scores
mkdir -p score_logs
for task in ae asr cv noise nlp; do
  til test "${task}" "${TAG}" 2>&1 | tee "score_logs/${task}_${TAG}.log"
done
grep -E "score:|1 - MER|mAP@|QA Accuracy|Noise Score|Image-level fairness|total rewards|equiv_rate" score_logs/*_"${TAG}".log || true

# 6. Submit
til submit ae "${TAG}"
til submit asr "${TAG}"
til submit cv "${TAG}"
til submit noise "${TAG}"
til submit nlp "${TAG}"
