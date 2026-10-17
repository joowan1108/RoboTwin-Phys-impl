#!/bin/bash
#SBATCH --job-name=robotwin-phys
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx5090:2,gpu:pro6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-%j.out

# Settings: one image holds the simulator and the five policy environments.
set -euo pipefail
IMAGE="docker://YOUR_DOCKERHUB_ID/robotwin-phys:bundle-cu128"
CMD="bash /home/guest/robotwin-workspace/cluster/run_parallel.sh"
DATA="$HOME/robotwin-data"
export APPTAINERENV_ROBOTWIN_EXPERIMENT="physical-random-100"
export APPTAINERENV_ROBOTWIN_MODEL="fastwam"
export APPTAINERENV_ROBOTWIN_PHASE="all"
export APPTAINERENV_ROBOTWIN_EPISODES=100
export APPTAINERENV_ROBOTWIN_VIDEO_SELECTION="first"
# FastWAM defaults to two isolated policy replicas on the PRO 6000.
# Optional fallback if two replicas exceed memory: export APPTAINERENV_ROBOTWIN_POLICY_REPLICAS=1
# Confirm the site GRES type name rtx5090 before submitting; all three GPUs must be on one node.

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
export APPTAINERENV_HOME=/tmp/robotwin-home
export APPTAINERENV_PYTHONUNBUFFERED=1
export PYTHONUNBUFFERED=1
export APPTAINER_CLEANENV=1
export APPTAINER_NO_HOME=1
export APPTAINER_NO_MOUNT="home,cwd"
export APPTAINER_TMPDIR="${SLURM_TMPDIR:-/tmp}/robotwin-apptainer-$SLURM_JOB_ID"
mkdir -p "$APPTAINER_TMPDIR" "$DATA/cache/home"
ROOT=/home/guest/robotwin-workspace
export APPTAINER_BINDPATH="$DATA/cache/home:/tmp/robotwin-home"
for dir in assets checkpoints downloads manifests runs reports; do
  mkdir -p "$DATA/$dir"
  target="$ROOT/$dir"
  if [[ "$dir" = assets ]]; then target="$ROOT/vendor/RoboTwin/assets"; fi
  export APPTAINER_BINDPATH="$APPTAINER_BINDPATH,$DATA/$dir:$target"
done
for icd in /usr/share/vulkan/icd.d /etc/vulkan/icd.d; do
  if [[ -d "$icd" ]]; then export APPTAINER_BINDPATH="$APPTAINER_BINDPATH,$icd:$icd:ro"; fi
done

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
