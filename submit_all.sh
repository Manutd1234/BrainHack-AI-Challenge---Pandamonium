#!/bin/bash
# ============================================================
# TIL-AI 2026 — Complete Build, Test & Submit Pipeline
# ============================================================
# Run this on your GCP Workbench instance.
# Usage: bash submit_all.sh
#
# Prerequisites:
#   - til CLI available (Workbench only)
#   - Docker running with GPU support
#   - gcloud authenticated as your service account:
#     gcloud config set account svc-TEAM-NAME@til-ai-2026.iam.gserviceaccount.com
#   - Docker configured for Artifact Registry:
#     gcloud auth configure-docker asia-southeast1-docker.pkg.dev
# ============================================================

set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
TAG="${TAG:-v4}"  # Override with: TAG=v5 bash submit_all.sh

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  TIL-AI 2026 — Build, Test & Submit Pipeline            ║"
echo "║  Tag: ${TAG}                                             ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ─── STEP 0: Pre-flight checks ───────────────────────────────
echo "━━━ Step 0: Pre-flight checks ━━━"

# Ensure models/ directories exist (Docker COPY will fail otherwise)
for task in asr cv noise nlp ae; do
    mkdir -p "${REPO_ROOT}/${task}/models"
done

# Check for fine-tuned CV model
if [ -f "${REPO_ROOT}/cv/models/best.pt" ]; then
    echo "  ✓ Fine-tuned CV model (best.pt) found"
else
    echo "  ⚠ WARNING: No fine-tuned CV model found at cv/models/best.pt"
    echo "    CV will use pretrained YOLO26 fallback with COCO-to-TIL class mapping."
    echo "    To improve: train a YOLO26 checkpoint and copy best.pt to cv/models/best.pt"
    echo ""
fi

echo ""

# ─── STEP 1: Build all Docker images ─────────────────────────
echo "━━━ Step 1: Building all Docker images ━━━"
echo ""

TASKS=("asr" "cv" "noise" "nlp" "ae")
FAILED_BUILDS=()

for task in "${TASKS[@]}"; do
    echo "┌─────────────────────────────────────────────┐"
    echo "│  Building: ${task} (tag: ${TAG})              "
    echo "└─────────────────────────────────────────────┘"

    if til build "${task}" "${TAG}"; then
        echo "  ✓ ${task}:${TAG} built successfully"
    else
        echo "  ✗ ${task}:${TAG} build FAILED"
        FAILED_BUILDS+=("${task}")
    fi
    echo ""
done

if [ ${#FAILED_BUILDS[@]} -gt 0 ]; then
    echo "⚠ Build failures: ${FAILED_BUILDS[*]}"
    echo "  Fix these before proceeding."
    echo "  You can re-run individual builds with: til build TASK ${TAG}"
    exit 1
fi

echo "✓ All images built successfully"
echo ""

# ─── STEP 2: Test all images locally ─────────────────────────
echo "━━━ Step 2: Testing all images locally ━━━"
echo ""

FAILED_TESTS=()

for task in "${TASKS[@]}"; do
    echo "┌─────────────────────────────────────────────┐"
    echo "│  Testing: ${task} (tag: ${TAG})               "
    echo "└─────────────────────────────────────────────┘"

    if til test "${task}" "${TAG}"; then
        echo "  ✓ ${task}:${TAG} tests passed"
    else
        echo "  ✗ ${task}:${TAG} tests FAILED"
        FAILED_TESTS+=("${task}")
    fi
    echo ""
done

if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
    echo "⚠ Test failures: ${FAILED_TESTS[*]}"
    echo "  Review test output above."
    echo "  You can re-test with: til test TASK ${TAG}"
    echo ""
    read -p "  Continue with submission anyway? (y/N): " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        echo "  Aborting submission."
        exit 1
    fi
fi

echo "✓ All tests completed"
echo ""

# ─── STEP 3: Submit all images ───────────────────────────────
echo "━━━ Step 3: Submitting all images ━━━"
echo ""

FAILED_SUBMITS=()

for task in "${TASKS[@]}"; do
    echo "┌─────────────────────────────────────────────┐"
    echo "│  Submitting: ${task} (tag: ${TAG})            "
    echo "└─────────────────────────────────────────────┘"

    if til submit "${task}" "${TAG}"; then
        echo "  ✓ ${task}:${TAG} submitted successfully"
    else
        echo "  ✗ ${task}:${TAG} submission FAILED"
        FAILED_SUBMITS+=("${task}")
    fi
    echo ""
done

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                   SUBMISSION SUMMARY                     ║"
echo "╠══════════════════════════════════════════════════════════╣"

if [ ${#FAILED_SUBMITS[@]} -eq 0 ]; then
    echo "║  ✓ All 5 tasks submitted successfully!                  ║"
else
    echo "║  ⚠ Failed submissions: ${FAILED_SUBMITS[*]}"
    echo "║  Retry with: til submit TASK ${TAG}                     ║"
fi

echo "║                                                          ║"
echo "║  Check your scores:                                      ║"
echo "║  → Discord notifications (team channel)                  ║"
echo "║  → Leaderboard (Strategist's Handbook)                   ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
