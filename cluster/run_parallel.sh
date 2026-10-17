#!/usr/bin/env bash
# Run INSIDE the bundle container. One model only; never installs host packages.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
# Bundle ENV paths point at the build user's home. The job supplies a writable HOME.
export XDG_CACHE_HOME="$HOME/.cache" XDG_CONFIG_HOME="$HOME/.config"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions"
options=(--model "${ROBOTWIN_MODEL:-fastwam}"
         --experiment "${ROBOTWIN_EXPERIMENT:-physical-random-100}"
         --phase "${ROBOTWIN_PHASE:-all}"
         --episodes "${ROBOTWIN_EPISODES:-100}"
         --video "${ROBOTWIN_VIDEO_SELECTION:-first}")
if [[ -n "${ROBOTWIN_POLICY_REPLICAS:-}" ]]; then options+=(--replicas "$ROBOTWIN_POLICY_REPLICAS"); fi
exec "$ROOT/.env/bin/python" tools/parallel_eval.py "${options[@]}" "$@"
