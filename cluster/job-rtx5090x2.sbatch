#!/bin/bash
#SBATCH --job-name=robotwin-phys
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx5090:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-%j.out

# 사용자 설정: 이미지 하나가 시뮬레이터와 다섯 policy 환경을 포함합니다.
set -euo pipefail
IMAGE="docker://YOUR_DOCKERHUB_ID/robotwin-phys:bundle-cu128-v2"
CMD="bash /home/guest/robotwin-workspace/cluster/run_bundle.sh"
DATA="$HOME/robotwin-data"
export APPTAINERENV_ROBOTWIN_EXPERIMENT="rtx5090x2-bundle-v2"
export APPTAINERENV_ROBOTWIN_GPU_LAYOUT="split"
export APPTAINERENV_ROBOTWIN_MOTUS_T5_DEVICE="cuda:1"
export APPTAINERENV_ROBOTWIN_PHASE="all"
export APPTAINERENV_ROBOTWIN_EPISODES=100
export APPTAINERENV_ROBOTWIN_VIDEO=1

# ---- (선택) private 이미지일 때만: 아래 두 줄 주석 해제 ----
# export APPTAINER_DOCKER_USERNAME="YOUR_DOCKERHUB_ID"
# export APPTAINER_DOCKER_PASSWORD="$(cat ~/.dockerhub_token)"
# G05 접근 권한을 받은 HF token을 OOD Files에서 ~/.huggingface_token에 저장 후 주석 해제:
# export APPTAINERENV_HF_TOKEN="$(cat ~/.huggingface_token)"

# 사용자 영역의 실행 설정. 하단의 apptainer 명령은 그대로 사용합니다.
: "${SLURM_JOB_ID:?Submit through OOD Job Composer}"
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
