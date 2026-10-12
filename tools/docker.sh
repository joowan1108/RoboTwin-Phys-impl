#!/usr/bin/env bash
# Use system Docker only; do not activate or install anything in the host .env.
set -euo pipefail
ROBOTWIN_DOCKER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROBOTWIN_DOCKER_ROOT"
export ROBOTWIN_UID="${ROBOTWIN_UID:-$(id -u)}"
export ROBOTWIN_GID="${ROBOTWIN_GID:-$(id -g)}"
# Single-container default: use the second visible host GPU when available
# (GPU 0 often runs the desktop), otherwise the first. Never require GPU 1.
if [[ -z "${ROBOTWIN_GPU:-}" || "${ROBOTWIN_GPU:-}" == auto ]]; then
  mapfile -t detected_gpus < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || true)
  export ROBOTWIN_GPU="${detected_gpus[1]:-${detected_gpus[0]:-0}}"
fi
compose=(docker compose --env-file docker/runtime.env --project-directory "$ROBOTWIN_DOCKER_ROOT" -f compose.yaml)
select_gpu_config() {
  mode="${ROBOTWIN_GPU_MODE:-auto}"
  if [[ "$mode" == auto ]]; then
    runtimes="$(docker info --format '{{json .Runtimes}}' 2>/dev/null || true)"
    if [[ -z "$runtimes" || "$runtimes" == *'"nvidia"'* ]]; then mode=nvidia; else mode=bind; fi
  fi
  case "$mode" in
    nvidia) compose+=(-f docker/compose.gpu-nvidia.yaml) ;;
    bind)
      gpu_key="${ROBOTWIN_GPU:-0}"
      if [[ ! "$gpu_key" =~ ^([0-9]+|GPU-[a-fA-F0-9-]+)$ ]]; then echo 'Select one GPU index or UUID' >&2; exit 2; fi
      gpu_config=".container/compose.gpu-bind-${gpu_key}.json"
      python3 docker/host_gpu_bindings.py --gpu "$gpu_key" --output "$gpu_config" >&2
      compose+=(-f "$gpu_config")
      ;;
    *) echo "ROBOTWIN_GPU_MODE must be auto, nvidia or bind" >&2; exit 2 ;;
  esac
}
action="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
case "$action" in
  config) select_gpu_config; exec "${compose[@]}" config "$@" ;;
  help|-h|--help)
    cat <<'EOF'
Usage: bash tools/docker.sh COMMAND [ARGS...]
  build             Build simulator image (no host packages installed)
  run COMMAND...    Run a command in a disposable GPU container
  shell             Interactive shell in the image
  doctor            Check CUDA + Vulkan + dependencies, write docker-hardware.json
  validate          Check image, GPU, old manifest replay and tests
  config            Print resolved Compose configuration, no daemon required
  save [FILE]       Export the built image as a gzip-compressed Docker archive
Environment: ROBOTWIN_GPU=auto|INDEX|UUID, ROBOTWIN_CUDA_ARCH=auto, ROBOTWIN_BUILD_JOBS=4
GPU mode: ROBOTWIN_GPU_MODE=auto (NVIDIA Toolkit when available, otherwise driver binds)
EOF
    exit 0 ;;
esac
case "$action" in build|run|shell|doctor|validate|save) ;; *) echo "Unknown command: $action" >&2; exit 2 ;; esac
if ! docker info > /dev/null; then
  echo 'Docker daemon access is unavailable in this session. Run this command in a server terminal with authorized Docker access; no host permissions or drivers were changed.' >&2
  exit 1
fi
if [[ "$action" == build ]]; then
  if [[ -z "${ROBOTWIN_CUDA_ARCH:-}" || "${ROBOTWIN_CUDA_ARCH:-}" == auto ]]; then
    export ROBOTWIN_CUDA_ARCH
    ROBOTWIN_CUDA_ARCH="$(python3 docker/build_arch.py)"
  fi
  exec "${compose[@]}" build "$@" sim
fi
if [[ "$action" == save ]]; then
  archive="${1:-$ROBOTWIN_DOCKER_ROOT/.container/robotwin-phys-cuda12.1.tar.gz}"
  mkdir -p -- "$(dirname -- "$archive")"
  if [[ -e "$archive" ]]; then echo "Refusing to overwrite $archive" >&2; exit 1; fi
  temporary="$(mktemp "${archive}.XXXXXX")"
  trap 'rm -f -- "$temporary"' EXIT
  docker image save "${ROBOTWIN_IMAGE:-robotwin-phys:cuda12.1}" | gzip > "$temporary"
  mv -- "$temporary" "$archive"
  exit 0
fi
for asset in objects embodiments background_texture; do
  if [[ ! -d "vendor/RoboTwin/assets/$asset" ]]; then
    echo "Missing existing assets: vendor/RoboTwin/assets/$asset" >&2; exit 1
  fi
done
mkdir -p "${ROBOTWIN_CACHE_DIR:-.container/cache}" "${ROBOTWIN_REPORTS_DIR:-reports}" runs manifests checkpoints
select_gpu_config
run_options=(--rm --no-deps)
if [[ -n "${ROBOTWIN_CONTAINER_NAME:-}" ]]; then run_options+=(--name "$ROBOTWIN_CONTAINER_NAME"); fi
case "$action" in
  shell) exec "${compose[@]}" run "${run_options[@]}" sim bash "$@" ;;
  doctor|validate) exec "${compose[@]}" run "${run_options[@]}" -T sim "$action" "$@" ;;
  run) exec "${compose[@]}" run "${run_options[@]}" -T sim "$@" ;;
esac
