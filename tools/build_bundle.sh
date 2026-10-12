#!/usr/bin/env bash
# Builds only the combined image from existing simulator/policy images. Never pushes.
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
repo="${1:?Usage: bash tools/build_bundle.sh HUB_REPO [TAG=v2] [SOURCE_REPO SOURCE_TAG]}"
tag="${2:-v2}"
source_repo="${3:-robotwin-phys}"
source_tag="${4:-}"
suffix="${source_tag:+-$source_tag}"
args=(--build-arg "SIM_IMAGE=$source_repo:sim-cu128$suffix")
sources=("$source_repo:sim-cu128$suffix")
for model in pi05 fact motus fastwam g05; do
  source="$source_repo:policy-$model-cu128$suffix"
  args+=(--build-arg "${model^^}_IMAGE=$source")
  sources+=("$source")
done
for source in "${sources[@]}"; do
  docker image inspect "$source" >/dev/null || {
    echo "Missing build input: $source. Pull it or build the six input images first; see docs/cluster.md." >&2
    exit 2
  }
done
docker build --platform linux/amd64 -f docker/Dockerfile.bundle "${args[@]}" \
  -t "$repo:bundle-cu128-$tag" .
printf 'Built %s:bundle-cu128-%s. Push this one image for the OOD default template.\n' "$repo" "$tag"
