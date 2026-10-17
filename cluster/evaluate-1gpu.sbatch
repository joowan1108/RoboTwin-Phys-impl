#!/bin/bash
#SBATCH --job-name=robotwin-phys
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-%j.out

# Settings: one image holds the simulator and the five policy environments.
set -euo pipefail
IMAGE="docker://YOUR_DOCKERHUB_ID/robotwin-phys:bundle-cu128"
# Set HOME in the payload: Apptainer rejects APPTAINERENV_HOME.
# CMD is assigned below after generating the asset-preparation payload.
DATA="$HOME/robotwin-data"
export APPTAINERENV_ROBOTWIN_EXPERIMENT="physical-random-100"
export APPTAINERENV_ROBOTWIN_MODEL="fastwam"
export APPTAINERENV_ROBOTWIN_PHASE="all"
export APPTAINERENV_ROBOTWIN_EPISODES=100
export APPTAINERENV_ROBOTWIN_VIDEO_SELECTION="first"
# One allocated PRO 6000 runs one simulator and one FastWAM policy process.
# The evaluator uses SLURM's assigned GPU ID; no host GPU number is assumed.
export APPTAINERENV_ROBOTWIN_POLICY_REPLICAS=1

# ---- Private image only: uncomment the next two lines ----
# export APPTAINER_DOCKER_USERNAME="YOUR_DOCKERHUB_ID"
# export APPTAINER_DOCKER_PASSWORD="$(cat ~/.dockerhub_token)"
# For the gated G0.5 checkpoint, save a Hugging Face token to ~/.huggingface_token and uncomment:
# export APPTAINERENV_HF_TOKEN="$(cat ~/.huggingface_token)"

# Runtime settings; the apptainer command at the bottom stays unchanged.
: "${SLURM_JOB_ID:?Submit with sbatch}"
: "${CUDA_VISIBLE_DEVICES:?SLURM must assign the GPUs}"
export APPTAINERENV_SLURM_JOB_ID="$SLURM_JOB_ID"
export APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
export APPTAINERENV_ROBOTWIN_DATA_ID="$DATA"
unset APPTAINERENV_HOME SINGULARITYENV_HOME
export APPTAINERENV_PYTHONUNBUFFERED=1
export PYTHONUNBUFFERED=1
export APPTAINER_CLEANENV=1
export APPTAINER_NO_HOME=1
export APPTAINER_NO_MOUNT="home,cwd"
export APPTAINER_TMPDIR="${SLURM_TMPDIR:-/tmp}/robotwin-apptainer-$SLURM_JOB_ID"
mkdir -p "$APPTAINER_TMPDIR" "$DATA/cache/home"
ROOT=/home/guest/robotwin-workspace
export APPTAINER_CWD="$ROOT"
# Use the image's glibc-compatible EGL/GLdispatch before host-injected copies.
# NVIDIA driver libraries remain available from /.singularity.d/libs.
export APPTAINERENV_LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/.singularity.d/libs"
export APPTAINER_BINDPATH="$DATA/cache/home:/tmp/robotwin-home"
for dir in assets checkpoints downloads manifests runs reports; do
  mkdir -p "$DATA/$dir"
  target="$ROOT/$dir"
  if [[ "$dir" = assets ]]; then target="$ROOT/vendor/RoboTwin/assets"; fi
  export APPTAINER_BINDPATH="$APPTAINER_BINDPATH,$DATA/$dir:$target"
done
for dir in /usr/share/vulkan/icd.d /etc/vulkan/icd.d /usr/share/glvnd/egl_vendor.d /etc/glvnd/egl_vendor.d; do
  if [[ -d "$dir" ]]; then export APPTAINER_BINDPATH="$APPTAINER_BINDPATH,$dir:$dir:ro"; fi
done
# Select NVIDIA only; unrelated host Mesa ICD files need not be loaded.
ROBOTWIN_NVIDIA_ICD=""
for icd in /etc/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json; do
  if [[ -r "$icd" ]]; then ROBOTWIN_NVIDIA_ICD="$icd"; break; fi
done
if [[ -z "$ROBOTWIN_NVIDIA_ICD" ]]; then
  echo "NVIDIA Vulkan ICD is missing on $(hostname)." >&2
  exit 1
fi
export APPTAINERENV_VK_ICD_FILENAMES="$ROBOTWIN_NVIDIA_ICD"

# Existing v5 images need the official template-generation step after unzip.
# Write the payload through the reports bind; no image rebuild is required.
ROBOTWIN_LAUNCH_HOST="$DATA/reports/launchers/$SLURM_JOB_ID"
ROBOTWIN_LAUNCH_CONTAINER="$ROOT/reports/launchers/$SLURM_JOB_ID"
mkdir -p "$ROBOTWIN_LAUNCH_HOST"
cat > "$ROBOTWIN_LAUNCH_HOST/run.sh" <<'ROBOTWIN_LAUNCH'
#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/guest/robotwin-workspace
export XDG_CACHE_HOME="$HOME/.cache" XDG_CONFIG_HOME="$HOME/.config"
export HF_HOME="$XDG_CACHE_HOME/huggingface" MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions"
mkdir -p "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$TORCH_EXTENSIONS_DIR"
# Use the same preparation lock as parallel_eval; assets are shared across jobs.
(
  flock -x 9
  cd "$ROOT"
  "$ROOT/.env/bin/python" tools/download_assets.py
  cd "$ROOT/vendor/RoboTwin"
  "$ROOT/.env/bin/python" script/update_embodiment_config_path.py < /dev/null
  for arm in left right; do
    target="assets/embodiments/aloha-agilex/curobo_${arm}.yml"
    if [[ ! -s "$target" ]]; then
      echo "Planner configuration was not generated: $target" >&2
      exit 1
    fi
  done
  echo "ASSET_PREPARATION_PASS: aloha-agilex left/right planner configs generated"
) 9> "$ROOT/downloads/.prepare.lock"
exec bash "$ROOT/cluster/run_parallel.sh"
ROBOTWIN_LAUNCH
CMD="env HOME=/tmp/robotwin-home bash $ROBOTWIN_LAUNCH_CONTAINER/run.sh"

# ---- Keep the Apptainer cache on node-local disk ----
export APPTAINER_CACHEDIR="/tmp/apptainer-$(whoami)"
mkdir -p "$APPTAINER_CACHEDIR"

# ============================================================
#  Nothing below needs to be changed.
# ============================================================

echo "=== Job start: $(date) on $(hostname) ==="
nvidia-smi                        # show the allocated GPUs

# Run the container (--nv passes the allocated GPUs through).
apptainer exec --nv "$IMAGE" $CMD

echo "=== Job end: $(date) ==="
