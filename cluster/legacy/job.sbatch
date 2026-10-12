#!/usr/bin/env bash
#SBATCH --job-name=robotwin-phys
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --signal=B:TERM@90
set -euo pipefail

# Edit these in OOD Job Composer. Submit one phase/model at a time.
HUB_REPO="YOUR_DOCKERHUB_ID/robotwin-phys"
TAG="v1"
PHASE="all"                    # all | prepare | doctor | init | freeze | evaluate | report
GPU_LAYOUT="shared"             # shared: 1 GPU; split: simulator GPU A, policy GPU B
MOTUS_T5_DEVICE="cuda:0"        # split + Motus: cuda:1 puts T5 on the simulator GPU
MODEL="pi05"                   # g05 | pi05 | fact | motus | fastwam
EXPERIMENT="pro6000-v1"
EPISODES=100                   # full benchmark: 100 episodes per task/protocol
TASKS=""                       # empty = all 50; e.g. "stack_blocks_two grab_roller"
PROTOCOLS=""                   # empty = all three; e.g. "clean"
VIDEO=1                        # MP4 available through OOD Files; storage can be substantial
DATA="$HOME/robotwin-data"     # persistent NAS directory, independent of immutable image

# Private Docker Hub repository: uncomment both. Never put the token in an image.
# export APPTAINER_DOCKER_USERNAME="YOUR_DOCKERHUB_ID"
# export APPTAINER_DOCKER_PASSWORD="$(cat "$HOME/.dockerhub_token")"
# G05 download only, after obtaining access on Hugging Face:
# export APPTAINERENV_HF_TOKEN="$(cat "$HOME/.huggingface_token")"

: "${SLURM_JOB_ID:?Submit this script as a SLURM job through OOD}"
: "${CUDA_VISIBLE_DEVICES:?SLURM must set the allocated GPU visibility}"
command -v apptainer >/dev/null
export PYTHONUNBUFFERED=1
export APPTAINER_CACHEDIR="/tmp/apptainer-$(id -un)"
export APPTAINER_TMPDIR="${SLURM_TMPDIR:-/tmp}/robotwin-apptainer-$SLURM_JOB_ID"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
ROOT=/home/guest/robotwin-workspace
SIM="docker://$HUB_REPO:sim-cu128-$TAG"
POLICY="docker://$HUB_REPO:policy-$MODEL-cu128-$TAG"
for dir in assets checkpoints downloads manifests runs reports cache; do mkdir -p "$DATA/$dir"; done
JOB_DIR="$DATA/reports/jobs/$SLURM_JOB_ID"
mkdir -p "$JOB_DIR"

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
export APPTAINERENV_CUDA_VISIBLE_DEVICES="$SIM_GPU"
export APPTAINERENV_SAPIEN_RENDER_DEVICE=cuda:0
export APPTAINERENV_HOME=/tmp/robotwin-home
export APPTAINERENV_XDG_CACHE_HOME=/tmp/robotwin-home/.cache
export APPTAINERENV_XDG_CONFIG_HOME=/tmp/robotwin-home/.config
export APPTAINERENV_MPLCONFIGDIR=/tmp/robotwin-home/.cache/matplotlib
export APPTAINERENV_TORCH_EXTENSIONS_DIR=/tmp/robotwin-home/.cache/torch_extensions
export APPTAINERENV_HF_HOME=/tmp/robotwin-home/.cache/huggingface
export APPTAINERENV_PYTHONUNBUFFERED=1
export APPTAINERENV_PYTHONPATH="$ROOT/src"
export APPTAINERENV_PYTHONHASHSEED=0
export APPTAINERENV_CUBLAS_WORKSPACE_CONFIG=:4096:8
export APPTAINERENV_OMP_NUM_THREADS=4
export APPTAINERENV_OPENBLAS_NUM_THREADS=4
mkdir -p "$DATA/cache/home/.cache"
BIND=(--bind "$DATA/assets:$ROOT/vendor/RoboTwin/assets" \
      --bind "$DATA/checkpoints:$ROOT/checkpoints" --bind "$DATA/downloads:$ROOT/downloads" \
      --bind "$DATA/manifests:$ROOT/manifests" --bind "$DATA/runs:$ROOT/runs" \
      --bind "$DATA/reports:$ROOT/reports" --bind "$DATA/cache/home:/tmp/robotwin-home")
# --nv supplies the host driver libraries; the Vulkan loader also needs its ICD descriptor.
for icd in /usr/share/vulkan/icd.d /etc/vulkan/icd.d; do
  if [[ -d "$icd" ]]; then BIND+=(--bind "$icd:$icd:ro"); fi
done
APPTAINER=(apptainer exec --nv --cleanenv --no-home --no-mount cwd --pwd "$ROOT" "${BIND[@]}")
sim() { "${APPTAINER[@]}" "$SIM" "$@"; }
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
nvidia-smi
# These functions also support the optional individual phases.
prepare_inputs() {
  # Downloads/assets are shared by experiments. Serialize writers to the same DATA.
  exec 7> "$DATA/.prepare.lock"
  flock 7
  sim python tools/prepare_cluster.py "$@"
  exec 7>&-
}
run_doctor() {
  sim python tools/cluster_preflight.py --output "reports/jobs/$SLURM_JOB_ID/preflight.json"
  if [[ "$GPU_LAYOUT" = split ]]; then
    APPTAINERENV_CUDA_VISIBLE_DEVICES="${ALLOCATED_GPUS[1]}" "${APPTAINER[@]}" "$SIM" \
      python tools/cluster_preflight.py --output "reports/jobs/$SLURM_JOB_ID/preflight-gpu-b.json"
  fi
}
evaluate_model() {
  MODEL="$1"
  POLICY="docker://$HUB_REPO:policy-$MODEL-cu128-$TAG"
  POLICY_GPUS="$SIM_GPU"
  if [[ "$GPU_LAYOUT" = split ]]; then
    POLICY_GPUS="${ALLOCATED_GPUS[1]}"
    if [[ "$MODEL" = motus && "$MOTUS_T5_DEVICE" = cuda:1 ]]; then
      POLICY_GPUS="${ALLOCATED_GPUS[1]},${ALLOCATED_GPUS[0]}"
    fi
  fi
  # Stable per experiment/model, including resumed jobs.
  port_hash=$(printf '%s' "$DATA/$EXPERIMENT/$MODEL" | cksum | cut -d' ' -f1)
  PORT=$((20000 + port_hash % 15000 * 2))
  NATIVE_PORT=$((PORT+1))
  mkdir -p "$DATA/runs/.server-locks"
  exec 9> "$DATA/runs/.server-locks/$EXPERIMENT-$MODEL.lock"
  flock -n 9 || { echo 'This experiment/model is already running' >&2; exit 2; }
  if [[ "$PHASE" != all ]]; then run_doctor; fi
  # Separate logs/config paths for all five servers in a single job.
  local record_dir="$ROOT/reports/jobs/$SLURM_JOB_ID/$MODEL"
  APPTAINERENV_CUDA_VISIBLE_DEVICES="$POLICY_GPUS" setsid "${APPTAINER[@]}" "$POLICY" python tools/serve_policy.py --model "$MODEL" \
    --port "$PORT" --native-port "$NATIVE_PORT" --t5-device "$MOTUS_T5_DEVICE" --record-dir "$record_dir" \
    > "$JOB_DIR/$MODEL-policy.log" 2>&1 &
  POLICY_PID=$!
  sim python tools/wait_policy.py --model "$MODEL" --port "$PORT" --timeout 2400 --process-pid "$POLICY_PID" --config "$record_dir/policy.yaml"
  sim python tools/experiment.py evaluate --experiment "$EXP" --model "$MODEL" \
    --policy-port "$PORT" "${SELECT[@]}" "${CAPTURE[@]}"
  cleanup
  exec 9>&-
}
case "$PHASE" in
  all)
    mkdir -p "$DATA/runs/.full-run-locks"
    exec 8> "$DATA/runs/.full-run-locks/$EXPERIMENT.lock"
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
