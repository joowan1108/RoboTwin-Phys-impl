#!/usr/bin/env python3
"""Embed the updated pool scheduler in an OOD script for the existing v5 image."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    scheduler = (ROOT / 'tools/parallel_eval.py').read_text()
    source_hash = hashlib.sha256(scheduler.encode()).hexdigest()
    host_payload = '''
# The simulator source stays identical to the uploaded expert patch, so v2
# progress can resume. Only the task scheduler is replaced for this job.
ROBOTWIN_SCHEDULER_HOST="$DATA/reports/launchers/$SLURM_JOB_ID/parallel_eval.py"
mkdir -p "$(dirname -- "$ROBOTWIN_SCHEDULER_HOST")"
cat > "$ROBOTWIN_SCHEDULER_HOST" <<'ROBOTWIN_POOL_SCHEDULER'
''' + scheduler + '''ROBOTWIN_POOL_SCHEDULER
printf '%s  %s\\n' SCHEDULER_HASH "$ROBOTWIN_SCHEDULER_HOST" | sha256sum -c -
export APPTAINER_BINDPATH="$APPTAINER_BINDPATH,$ROBOTWIN_SCHEDULER_HOST:$ROOT/tools/parallel_eval.py:ro"
'''.replace('SCHEDULER_HASH', source_hash)
    source = (ROOT / 'cluster/job-pro6000-expert-fix.sbatch').read_text()
    source = source.replace('#SBATCH --job-name=fastwam-phys', '#SBATCH --job-name=phys-pool-4workers')
    source = source.replace('#SBATCH --gres=gpu:pro6000:1', '#SBATCH --gres=gpu:pro6000:2')
    source = source.replace('#SBATCH --cpus-per-task=8', '#SBATCH --cpus-per-task=16')
    source = source.replace('#SBATCH --mem=96G', '#SBATCH --mem=128G')
    source = source.replace('export APPTAINERENV_ROBOTWIN_PHASE="all"', 'export APPTAINERENV_ROBOTWIN_PHASE="freeze"')
    source = source.replace('# One allocated PRO 6000 runs one simulator and one FastWAM policy process.',
                            '# Pool preparation only: two GPUs, two independent task processes per GPU.')
    source = source.replace('export APPTAINERENV_ROBOTWIN_POLICY_REPLICAS=1',
                            '# Reduce to 1 if two simultaneous planners per GPU exceed memory.\n'
                            'export APPTAINERENV_ROBOTWIN_FREEZE_WORKERS_PER_GPU=2')
    anchor = '# Existing v5 images need the official template-generation step after unzip.'
    source = source.replace(anchor, host_payload + '\n' + anchor)
    source = source.replace('exec bash "$ROOT/cluster/run_parallel.sh"',
                            'exec bash "$ROOT/cluster/run_parallel.sh" '
                            '--freeze-workers-per-gpu "$ROBOTWIN_FREEZE_WORKERS_PER_GPU"')
    for name in ('job-pro6000-pool-parallel-v5.sbatch', 'ood-pool-pro6000x2-v5.txt'):
        (ROOT / 'cluster' / name).write_text(source)
    print('Generated two-GPU OOD pool script; scheduler sha256=' + source_hash)


if __name__ == '__main__':
    main()
