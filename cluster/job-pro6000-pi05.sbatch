#!/bin/bash
#SBATCH --job-name=pi05-phys
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-%j.out

# PI 0.5: reuse the completed FastWAM episode pool. Submit after its job finishes.
set -euo pipefail
IMAGE="docker://joowan1108/robotwin-phys:bundle-cu128-v5"
# Set HOME in the payload: Apptainer rejects APPTAINERENV_HOME.
# CMD is assigned below after generating the asset-preparation payload.
DATA="$HOME/robotwin-data"
export APPTAINERENV_ROBOTWIN_EXPERIMENT="phys-pro6000-100-v1"
export APPTAINERENV_ROBOTWIN_MODEL="pi05"
export APPTAINERENV_ROBOTWIN_PHASE="evaluate"
export APPTAINERENV_ROBOTWIN_EPISODES=100
export APPTAINERENV_ROBOTWIN_VIDEO_SELECTION="first"
# One allocated PRO 6000 runs one simulator and one PI 0.5 policy process.
# The evaluator uses SLURM's assigned GPU ID; no host GPU number is assumed.
export APPTAINERENV_ROBOTWIN_POLICY_REPLICAS=1

# Evaluation uses the same sealed task/seed/condition pool as FastWAM.
ROBOTWIN_POOL="$DATA/manifests/$APPTAINERENV_ROBOTWIN_EXPERIMENT/pool-index.json"
if [[ ! -s "$ROBOTWIN_POOL" ]]; then
  echo "The shared pool is not complete yet: $ROBOTWIN_POOL" >&2
  echo "Submit this PI 0.5 job after the FastWAM job has finished." >&2
  exit 1
fi

# ---- (선택) private 이미지일 때만: 아래 두 줄 주석 해제 ----
# export APPTAINER_DOCKER_USERNAME="joowan1108"
# export APPTAINER_DOCKER_PASSWORD="$(cat ~/.dockerhub_token)"
# G05 접근 권한을 받은 HF token을 OOD Files에서 ~/.huggingface_token에 저장 후 주석 해제:
# export APPTAINERENV_HF_TOKEN="$(cat ~/.huggingface_token)"

# 사용자 영역의 실행 설정. 하단의 apptainer 명령은 그대로 사용합니다.
: "${SLURM_JOB_ID:?Submit through OOD Job Composer}"
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

# ---- 캐시를 노드 로컬로 (NAS 홈 부담 방지, 건드리지 마세요) ----
export APPTAINER_CACHEDIR="/tmp/apptainer-$(whoami)"
mkdir -p "$APPTAINER_CACHEDIR"

# ============================================================
#  아래는 수정할 필요 없습니다.
# ============================================================

echo "=== Job start: $(date) on $(hostname) ==="
nvidia-smi                        # 할당된 GPU 확인

# 컨테이너 실행 (--nv = GPU 전달). 할당된 GPU만 보입니다.
apptainer exec --nv "$IMAGE" $CMD

echo "=== Job end: $(date) ==="
