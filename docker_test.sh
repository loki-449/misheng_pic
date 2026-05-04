#!/bin/bash
set -e

IMAGE_NAME="misheng-app-test"

echo "Building test image: ${IMAGE_NAME}"
docker build -t "${IMAGE_NAME}" .

echo "Running pytest in container"
docker run --rm "${IMAGE_NAME}" pytest -q

echo "Container tests passed"
