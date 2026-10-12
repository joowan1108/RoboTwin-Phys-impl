#!/bin/bash
#SBATCH --job-name=slurm-diagnose
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=slurm-diagnose-%j.out

# Submit as a separate OOD job. The known PRO 6000 request avoids guessing the 5090 type.
# Query the cluster's Slurm configuration on the host, then exit above the protected tail.
# No GPU computation, image pull, or evaluation is performed.
IMAGE="docker://joowan1108/robotwin-phys:bundle-cu128-v4"
CMD="/bin/true"

# Read-only Slurm queries. Run on a cluster shell or through the diagnostic job.
set -uo pipefail

for tool in sinfo scontrol; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Missing $tool. Run this on the CSI cluster, not the bs1 build host or inside the Docker image." >&2
    exit 127
  fi
done

diagnostic_status=0
query() {
  printf '\n>>> '
  printf '%q ' "$@"
  printf '\n'
  if "$@"; then
    :
  else
    local query_status=$?
    printf 'Query failed (exit %s); continuing with the remaining queries.\n' "$query_status" >&2
    diagnostic_status=1
  fi
}

printf 'Slurm resource report: %s on %s\n' "$(date -u +%FT%TZ)" "$(hostname)"
printf 'Evaluation request: one node; gpu:rtx5090:2,gpu:pro6000:1; 16 CPUs; 128G RAM; 3 days.\n'
printf 'The diagnostic job has its own smaller allocation; it does not validate the evaluation allocation.\n'

query sinfo --version
query sinfo -N -p gpu -o '%N | %G | %c | %m | %t'
query scontrol show partition gpu
query scontrol -o show nodes
query sinfo -R -p gpu
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  query scontrol show job "$SLURM_JOB_ID"
fi

printf '\nReport finished. Include query errors when sharing this output.\n'
# Always stop after the queries; no Docker/Apptainer or evaluation is launched.
exit "$diagnostic_status"

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
