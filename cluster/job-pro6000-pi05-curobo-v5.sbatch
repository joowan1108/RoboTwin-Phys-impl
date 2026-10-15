#!/bin/bash
#SBATCH --job-name=pi05-fixed
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-%j.out

# 사용자 설정: 이미지 하나가 시뮬레이터와 다섯 policy 환경을 포함합니다.
set -euo pipefail
IMAGE="docker://joowan1108/robotwin-phys:bundle-cu128-v5"
# Set HOME in the payload: Apptainer rejects APPTAINERENV_HOME.
# CMD is assigned below after generating the asset-preparation payload.
DATA="$HOME/robotwin-data"
export APPTAINERENV_ROBOTWIN_EXPERIMENT="phys-pro6000-100-v4"
export APPTAINERENV_ROBOTWIN_MODEL="pi05"
export APPTAINERENV_ROBOTWIN_PHASE="evaluate"
export APPTAINERENV_ROBOTWIN_EPISODES=100
export APPTAINERENV_ROBOTWIN_VIDEO_SELECTION="first"
# One allocated PRO 6000 runs one simulator and one FastWAM policy process.
# The evaluator uses SLURM's assigned GPU ID; no host GPU number is assumed.
export APPTAINERENV_ROBOTWIN_POLICY_REPLICAS=1

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

# Apply the reviewed simulator snapshot to the existing v5 image.
ROBOTWIN_PATCH_DIR="$DATA/patches/robotwin-curobo-fix-v1"
ROBOTWIN_PATCH_ARCHIVE="$DATA/patches/robotwin-curobo-fix-v1.zip"
mkdir -p "$DATA/patches"
(
  flock -x 9
  if [[ ! -d "$ROBOTWIN_PATCH_DIR/src/robotwin_phys" ]]; then
    if [[ ! -r "$ROBOTWIN_PATCH_ARCHIVE" ]]; then
      echo "Upload robotwin-curobo-fix-v1.zip through OOD Files to $DATA/patches/" >&2
      exit 1
    fi
    printf '%s  %s\n' f7c7deb0045f0459ad9daa74d080c0a37f991fb841b0fdd436fddaaa51003454 "$ROBOTWIN_PATCH_ARCHIVE" | sha256sum -c -
    ROBOTWIN_PATCH_STAGE=$(mktemp -d "$DATA/patches/.expert-fix-XXXXXX")
    trap 'rm -rf -- "$ROBOTWIN_PATCH_STAGE"' EXIT
    if command -v unzip > /dev/null; then
      unzip -q "$ROBOTWIN_PATCH_ARCHIVE" -d "$ROBOTWIN_PATCH_STAGE"
    elif command -v python3 > /dev/null; then
      python3 -m zipfile -e "$ROBOTWIN_PATCH_ARCHIVE" "$ROBOTWIN_PATCH_STAGE"
    else
      echo "Extract the ZIP through OOD Files into $DATA/patches/ before submitting." >&2
      exit 1
    fi
    mv "$ROBOTWIN_PATCH_STAGE/robotwin-curobo-fix-v1" "$ROBOTWIN_PATCH_DIR"
  fi
) 9> "$DATA/patches/.expert-fix.lock"
export APPTAINER_BINDPATH="$APPTAINER_BINDPATH,$ROBOTWIN_PATCH_DIR/src:$ROOT/src:ro"
export APPTAINER_BINDPATH="$APPTAINER_BINDPATH,$ROBOTWIN_PATCH_DIR/tools:$ROOT/tools:ro"
export APPTAINER_BINDPATH="$APPTAINER_BINDPATH,$ROBOTWIN_PATCH_DIR/vendor/transforms.py:$ROOT/vendor/RoboTwin/envs/utils/transforms.py:ro"
export APPTAINER_BINDPATH="$APPTAINER_BINDPATH,$ROBOTWIN_PATCH_DIR/vendor/planner.py:$ROOT/vendor/RoboTwin/envs/robot/planner.py:ro"

# Existing v5 images need the official template-generation step after unzip.
# Write the payload through the reports bind; no image rebuild is required.
ROBOTWIN_LAUNCH_HOST="$DATA/reports/launchers/$SLURM_JOB_ID"
ROBOTWIN_LAUNCH_CONTAINER="$ROOT/reports/launchers/$SLURM_JOB_ID"
mkdir -p "$ROBOTWIN_LAUNCH_HOST"
cat > "$ROBOTWIN_LAUNCH_HOST/run.sh" <<'ROBOTWIN_LAUNCH'
#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/guest/robotwin-workspace

# Check the entire source snapshot before creating or reading any pool.
"$ROOT/.env/bin/python" - <<'PY_EXPERT_PATCH'
import hashlib, json
from pathlib import Path
root = Path('/home/guest/robotwin-workspace')
def hashes(folder, patterns):
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob('*')) if p.is_file() and p.suffix in patterns
            and '__pycache__' not in p.parts}
def checksum(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
if checksum(hashes(root / 'src', ('.py',))) != '6dfbef9791d525c4d9261a595f87efebf9955c4a0d976ec382024816300fe5e9':
    raise SystemExit('Expert patch source differs; restore uploaded snapshot')
if checksum(hashes(root / 'tools', ('.py', '.sh', '.json'))) != 'a0f6a038b0df311162c94fa5580adec8ccab31ec03e2b1ca7acb7ae757037dc8':
    raise SystemExit('Expert patch tools differ; restore uploaded snapshot')
if hashlib.sha256((root / 'vendor/RoboTwin/envs/utils/transforms.py').read_bytes()).hexdigest() != '942ed2b255094190f7380df113b78daed687803732928d23f2d9823d18862c53':
    raise SystemExit('Shared transform repair was not bound correctly')
if hashlib.sha256((root / 'vendor/RoboTwin/envs/robot/planner.py').read_bytes()).hexdigest() != '61e3d0d4fa53e58a2289a533a5bcf4d85a067bbd1939745a87bc19a7c6884f56':
    raise SystemExit('Planner exception repair was not bound correctly')
print('CUROBO_MASK_PATCH_SOURCE_PASS: simulator and reviewed helpers verified', flush=True)
PY_EXPERT_PATCH

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
