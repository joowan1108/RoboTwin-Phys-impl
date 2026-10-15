#!/bin/bash
#SBATCH --job-name=robotwin-diag
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out

# 진단 전용: 이미지 재빌드나 모델 다운로드 없이 CUDA/Vulkan/CuRobo를 검사합니다.
set -euo pipefail
IMAGE="docker://joowan1108/robotwin-phys:bundle-cu128-v5"
# Set HOME in the payload: Apptainer rejects APPTAINERENV_HOME.
# CMD is assigned below after creating the diagnostic payload.
DATA="$HOME/robotwin-data"
# Private 이미지일 때만 필요한 인증 설정입니다.
# export APPTAINER_DOCKER_USERNAME="joowan1108"
# export APPTAINER_DOCKER_PASSWORD="$(cat ~/.dockerhub_token)"

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

# The reports bind makes this generated script visible inside the existing image.
DIAG_HOST="$DATA/reports/diagnostics/$SLURM_JOB_ID"
DIAG_CONTAINER="$ROOT/reports/diagnostics/$SLURM_JOB_ID"
mkdir -p "$DIAG_HOST"
{
  date
  hostname
  apptainer --version
  printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES"
  getconf GNU_LIBC_VERSION
  cat "$ROBOTWIN_NVIDIA_ICD"
  ldconfig -p | grep -E 'libEGL|libGLdispatch|libGLX_nvidia|libnvidia-(glcore|eglcore|glvkspirv|gpucomp)' || true
} > "$DIAG_HOST/host.txt" 2>&1
cat "$DIAG_HOST/host.txt"
cat > "$DIAG_HOST/inside.sh" <<'ROBOTWIN_DIAGNOSTIC'
#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/guest/robotwin-workspace
cd "$ROOT"
DIAG="$ROOT/reports/diagnostics/$SLURM_JOB_ID"
exec > >(tee "$DIAG/diagnostic.log") 2>&1
export XDG_CACHE_HOME="$HOME/.cache" XDG_CONFIG_HOME="$HOME/.config"
export HF_HOME="$XDG_CACHE_HOME/huggingface" MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions"
export SAPIEN_RENDER_DEVICE=cuda:0
mkdir -p "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$TORCH_EXTENSIONS_DIR"
PYTHON="$ROOT/.env/bin/python"
printf 'HOME=%s\nLD_LIBRARY_PATH=%s\nVK_ICD_FILENAMES=%s\nCUDA_VISIBLE_DEVICES=%s\n' \
  "$HOME" "$LD_LIBRARY_PATH" "$VK_ICD_FILENAMES" "$CUDA_VISIBLE_DEVICES"
getconf GNU_LIBC_VERSION
cat "$VK_ICD_FILENAMES"
# Show actual loaded paths, since ldconfig alone cannot show runtime selection.
set +e
"$PYTHON" - <<'PY_LIBRARIES' > "$DIAG/libraries.log" 2>&1
import ctypes
from pathlib import Path
import sys
handles = []
errors = []
for name in ('libEGL.so.1', 'libGLdispatch.so.0', 'libEGL_nvidia.so.0', 'libGLX_nvidia.so.0'):
    try:
        handles.append(ctypes.CDLL(name))
        print(f'LOAD_OK {name}', flush=True)
    except OSError as exc:
        errors.append(name)
        print(f'LOAD_FAILED {name}: {exc}', flush=True)
paths = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                if any(name in line for name in ('libEGL', 'libGLdispatch', 'libGLX_nvidia', 'libnvidia', 'libc.so'))})
for path in paths:
    print(f'LOADED_PATH {path}', flush=True)
# The host's generic EGL/GLdispatch caused the observed GLIBC_2.38 failure.
for name in ('libEGL.so.', 'libGLdispatch.so.'):
    if any('/.singularity.d/libs/' in path and name in path for path in paths):
        errors.append(name)
        print(f'WRONG_LIBRARY_SOURCE {name}: host generic library is still selected', flush=True)
sys.exit(1 if errors else 0)
PY_LIBRARIES
libraries_rc=$?
vulkaninfo --summary > "$DIAG/vulkan.log" 2>&1
vulkan_rc=$?
set -e
cat "$DIAG/libraries.log"
cat "$DIAG/vulkan.log"
if (( libraries_rc != 0 || vulkan_rc != 0 )); then
  # Keep the verbose loader trace in a file instead of flooding the Slurm log.
  env LD_DEBUG=libs,files vulkaninfo --summary > "$DIAG/vulkan-loader.log" 2>&1 || true
  printf 'DIAGNOSIS_FAILED: libraries=%s vulkan=%s\n' "$libraries_rc" "$vulkan_rc" | tee "$DIAG/status.txt"
  printf 'Detailed log: %s/vulkan-loader.log\n' "$DIAG"
  exit 1
fi
# This performs real CUDA matmul, SAPIEN offscreen rendering, and CuRobo FK.
# No assets/checkpoints are downloaded and no evaluation episodes are started.
set +e
"$PYTHON" tools/cluster_preflight.py --output "$DIAG/preflight.json" > "$DIAG/preflight.log" 2>&1
preflight_rc=$?
set -e
cat "$DIAG/preflight.log"
if (( preflight_rc != 0 )); then
  printf 'DIAGNOSIS_FAILED: preflight=%s\n' "$preflight_rc" | tee "$DIAG/status.txt"
  exit "$preflight_rc"
fi
printf 'DIAGNOSIS_PASS: CUDA + Vulkan render + CuRobo forward kinematics\n' | tee "$DIAG/status.txt"
ROBOTWIN_DIAGNOSTIC
CMD="env HOME=/tmp/robotwin-home bash $DIAG_CONTAINER/inside.sh"
printf 'Diagnostic reports on host: %s\n' "$DIAG_HOST"

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
