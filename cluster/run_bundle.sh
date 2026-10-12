#!/usr/bin/env bash
# Runs inside the bundle image. No Docker/Apptainer command is invoked here.
set -euo pipefail
ROOT="${ROBOTWIN_BUNDLE_SIM_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
SIM_PYTHON="$ROOT/.env/bin/python"
POLICY_ROOT="${ROBOTWIN_BUNDLE_POLICY_ROOT:-/opt/robotwin-policies}"
PHASE="${ROBOTWIN_PHASE:-all}"
GPU_LAYOUT="${ROBOTWIN_GPU_LAYOUT:-shared}"
MOTUS_T5_DEVICE="${ROBOTWIN_MOTUS_T5_DEVICE:-cuda:0}"
MODEL="${ROBOTWIN_MODEL:-pi05}"
EXPERIMENT="${ROBOTWIN_EXPERIMENT:-pro6000-bundle-v2}"
EPISODES="${ROBOTWIN_EPISODES:-100}"
TASKS="${ROBOTWIN_TASKS:-}"
PROTOCOLS="${ROBOTWIN_PROTOCOLS:-}"
VIDEO="${ROBOTWIN_VIDEO:-1}"
: "${CUDA_VISIBLE_DEVICES:?Allocated GPUs must be passed into the container}"
: "${SLURM_JOB_ID:?Run through a SLURM job}"
if [[ ! "$EXPERIMENT" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]; then echo 'Invalid EXPERIMENT name' >&2; exit 2; fi
for dir in checkpoints downloads manifests runs reports; do mkdir -p "$ROOT/$dir"; done
JOB_DIR="$ROOT/reports/jobs/$SLURM_JOB_ID"
mkdir -p "$JOB_DIR"
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export SAPIEN_RENDER_DEVICE=cuda:0
export XDG_CACHE_HOME="$HOME/.cache" XDG_CONFIG_HOME="$HOME/.config"
export MPLCONFIGDIR="$HOME/.cache/matplotlib" TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_extensions"
export HF_HOME="$HOME/.cache/huggingface"
sim() {
  # Existing commands use `sim python ...`; select the simulator interpreter explicitly.
  [[ "$1" = python ]] || return 2
  shift
  CUDA_VISIBLE_DEVICES="$SIM_GPU" "$SIM_PYTHON" "$@"
}
# --cleanenv must not replace SLURM's allocation with the image's GPU 0.
IFS=',' read -r -a ALLOCATED_GPUS <<< "$CUDA_VISIBLE_DEVICES"
SIM_GPU="${ALLOCATED_GPUS[0]}"
POLICY_GPUS="$SIM_GPU"
case "$GPU_LAYOUT" in
  shared)
    if [[ ${#ALLOCATED_GPUS[@]} -ne 1 ]]; then echo 'shared expects exactly one allocated GPU' >&2; exit 2; fi
    if [[ "$MOTUS_T5_DEVICE" != cuda:0 ]]; then echo 'T5 cuda:1 requires split layout' >&2; exit 2; fi
    ;;
  split)
    if [[ ${#ALLOCATED_GPUS[@]} -ne 2 || "${ALLOCATED_GPUS[0]}" = "${ALLOCATED_GPUS[1]}" ]]; then
      echo 'split expects exactly two distinct allocated GPUs' >&2; exit 2
    fi
    POLICY_GPUS="${ALLOCATED_GPUS[1]}"
    if [[ "$MODEL" = motus && "$MOTUS_T5_DEVICE" = cuda:1 ]]; then
      POLICY_GPUS="${ALLOCATED_GPUS[1]},${ALLOCATED_GPUS[0]}"
    fi
    ;;
  *) echo 'Unknown GPU_LAYOUT' >&2; exit 2;;
esac
EXP="$ROOT/runs/$EXPERIMENT"
SELECT=()
if [[ -n "$TASKS" ]]; then read -r -a task_list <<< "$TASKS"; SELECT+=(--tasks "${task_list[@]}"); fi
if [[ -n "$PROTOCOLS" ]]; then read -r -a protocol_list <<< "$PROTOCOLS"; SELECT+=(--protocols "${protocol_list[@]}"); fi
CAPTURE=()
if [[ "$VIDEO" = 1 ]]; then CAPTURE+=(--video); fi
POLICY_PID=""
cleanup() {
  if [[ -n "$POLICY_PID" ]]; then
    kill -TERM -- "-$POLICY_PID" 2>/dev/null || true
    wait "$POLICY_PID" 2>/dev/null || true
    POLICY_PID=""
  fi
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
printf 'Job %s on %s; phase=%s model=%s GPU allocation=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$PHASE" "$MODEL" "$CUDA_VISIBLE_DEVICES"
# These functions also support the optional individual phases.
prepare_inputs() {
  # Downloads/assets are shared by experiments. Serialize writers to the same DATA.
  exec 7> "$ROOT/downloads/.prepare.lock"
  flock 7
  sim python tools/prepare_cluster.py "$@"
  exec 7>&-
}
run_doctor() {
  sim python tools/cluster_preflight.py --output "reports/jobs/$SLURM_JOB_ID/preflight.json"
  if [[ "$GPU_LAYOUT" = split ]]; then
    CUDA_VISIBLE_DEVICES="${ALLOCATED_GPUS[1]}" "$SIM_PYTHON" tools/cluster_preflight.py --output "reports/jobs/$SLURM_JOB_ID/preflight-gpu-b.json"
  fi
}
evaluate_model() {
  MODEL="$1"
  POLICY_GPUS="$SIM_GPU"
  if [[ "$GPU_LAYOUT" = split ]]; then
    POLICY_GPUS="${ALLOCATED_GPUS[1]}"
    if [[ "$MODEL" = motus && "$MOTUS_T5_DEVICE" = cuda:1 ]]; then
      POLICY_GPUS="${ALLOCATED_GPUS[1]},${ALLOCATED_GPUS[0]}"
    fi
  fi
  # Stable per experiment/model, including resumed jobs.
  port_hash=$(printf '%s' "${ROBOTWIN_DATA_ID:-$(id -u)}/$EXPERIMENT/$MODEL" | cksum | cut -d' ' -f1)
  PORT=$((20000 + port_hash % 15000 * 2))
  NATIVE_PORT=$((PORT+1))
  mkdir -p "$ROOT/runs/.server-locks"
  exec 9> "$ROOT/runs/.server-locks/$EXPERIMENT-$MODEL.lock"
  flock -n 9 || { echo 'This experiment/model is already running' >&2; exit 2; }
  if [[ "$PHASE" != all ]]; then run_doctor; fi
  # Separate logs/config paths for all five servers in a single job.
  local record_dir="$ROOT/reports/jobs/$SLURM_JOB_ID/$MODEL"
  local runtime="$POLICY_ROOT/$MODEL"
  local workspace="$runtime/workspace"
  setsid env CUDA_VISIBLE_DEVICES="$POLICY_GPUS" ROBOTWIN_PHYS_ROOT="$workspace" \
    ROBOTWIN_BUILD_INFO="$runtime/build" VIRTUAL_ENV="$runtime/venv" \
    PYTHONPATH="$workspace/src" PATH="$runtime/venv/bin:/usr/local/cuda/bin:/usr/bin:/bin" \
    JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "$runtime/venv/bin/python" "$workspace/tools/serve_policy.py" --model "$MODEL" \
    --port "$PORT" --native-port "$NATIVE_PORT" --t5-device "$MOTUS_T5_DEVICE" --record-dir "$record_dir" \
    > "$JOB_DIR/$MODEL-policy.log" 2>&1 &
  POLICY_PID=$!
  sim python tools/wait_policy.py --model "$MODEL" --port "$PORT" --timeout 2400 --process-pid "$POLICY_PID" --config "$record_dir/policy.yaml" --config-root "$workspace"
  sim python tools/experiment.py evaluate --experiment "$EXP" --model "$MODEL" \
    --policy-port "$PORT" "${SELECT[@]}" "${CAPTURE[@]}"
  cleanup
  exec 9>&-
}
case "$PHASE" in
  all)
    mkdir -p "$ROOT/runs/.full-run-locks"
    exec 8> "$ROOT/runs/.full-run-locks/$EXPERIMENT.lock"
    flock -n 8 || { echo 'This full experiment is already running' >&2; exit 2; }
    prepare_inputs --assets
    run_doctor
    sim python tools/experiment.py init --experiment "$EXP" --episodes "$EPISODES" "${SELECT[@]}"
    sim python tools/experiment.py freeze --experiment "$EXP" "${SELECT[@]}" "${CAPTURE[@]}"
    # G05 is last because it additionally requires gated Hugging Face access.
    for MODEL in pi05 fact motus fastwam g05; do
      if sim python tools/experiment.py check-complete --experiment "$EXP" --model "$MODEL" "${SELECT[@]}"; then
        printf 'Already complete with verified conditions: %s; skipping model load.\n' "$MODEL"
        continue
      else
        completion=$?
        if [[ "$completion" -ne 1 ]]; then echo 'Invalid existing results; see tables/randomization_audit.json' >&2; exit "$completion"; fi
      fi
      prepare_inputs --models "$MODEL"
      evaluate_model "$MODEL"
    done
    sim python tools/experiment.py report --experiment "$EXP"
    ;;
  prepare) prepare_inputs --assets --models "$MODEL" ;;
  doctor) run_doctor ;;
  init) sim python tools/experiment.py init --experiment "$EXP" --episodes "$EPISODES" "${SELECT[@]}" ;;
  freeze) sim python tools/experiment.py freeze --experiment "$EXP" "${SELECT[@]}" "${CAPTURE[@]}" ;;
  evaluate) evaluate_model "$MODEL" ;;
  report) sim python tools/experiment.py report --experiment "$EXP" ;;
  *) echo "Unknown PHASE: $PHASE" >&2; exit 2 ;;
esac
printf 'Finished: %s\n' "$(date -Is)"
