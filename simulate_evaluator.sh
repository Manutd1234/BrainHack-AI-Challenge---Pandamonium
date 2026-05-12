#!/bin/bash
# simulate_evaluator.sh
# Usage: ./simulate_evaluator.sh <DOCKER_IMAGE_NAME> <PORT>
# Example: ./simulate_evaluator.sh myteam-asr:v1 5001

set -e

IMAGE=${1:?"Usage: $0 <IMAGE_NAME> <PORT>"}
PORT=${2:-5000}

echo "Simulating hackathon evaluation environment..."
echo "Mounting ./test_data to /data"
echo "Mapping port $PORT to $PORT"

mkdir -p ./test_data

docker run -it --rm \
    -v $(pwd)/test_data:/data \
    -p ${PORT}:${PORT} \
    ${IMAGE}
