#!/bin/bash
# build_all.sh
# Usage: ./build_all.sh <TEAM_ID> [TAG]
# Example: ./build_all.sh myteam v1

set -e

TEAM_ID=${1:?"Usage: $0 <TEAM_ID> [TAG]"}
TAG=${2:-latest}

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

TASKS=("asr" "cv" "noise" "nlp" "ae")

for TASK in "${TASKS[@]}"; do
    echo ""
    echo "========================================"
    echo "  Building ${TEAM_ID}-${TASK}:${TAG}"
    echo "========================================"
    cd "${REPO_ROOT}/${TASK}"
    docker build -t "${TEAM_ID}-${TASK}:${TAG}" .
    echo "✓ ${TEAM_ID}-${TASK}:${TAG} built successfully"
done

echo ""
echo "All images built:"
docker image ls | grep "${TEAM_ID}"
