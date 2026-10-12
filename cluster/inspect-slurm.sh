#!/usr/bin/env bash
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
