#!/usr/bin/env bash
# Build on the development machine. This script never pushes or uses a GPU.
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
repo="${1:?Usage: bash tools/build_cluster.sh DOCKERHUB_ID/robotwin-phys [TAG] [MODEL ...]}"
tag="${2:-v1}"
if [[ $# -ge 2 ]]; then shift 2; else shift; fi
models=("$@")
if [[ ${#models[@]} -eq 0 ]]; then models=(g05 pi05 fact motus fastwam); fi
for model in "${models[@]}"; do
  case "$model" in g05|pi05|fact|motus|fastwam) ;; *) echo "Unknown model: $model" >&2; exit 2;; esac
done
docker build --platform linux/amd64 -f docker/Dockerfile.blackwell \
  --build-arg 'TORCH_CUDA_ARCH_LIST=7.5;12.0' --build-arg MAX_JOBS=2 \
  -t "$repo:sim-cu128-$tag" .
for model in "${models[@]}"; do
  docker build --platform linux/amd64 -f docker/Dockerfile.policy --build-arg MODEL="$model" \
    -t "$repo:policy-$model-cu128-$tag" .
done
printf 'Built %s simulator and %s policy images. No images were pushed.\n' "$repo" "${#models[@]}"
