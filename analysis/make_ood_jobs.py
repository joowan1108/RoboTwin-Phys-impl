#!/usr/bin/env python3
"""Generate Job Composer scripts that carry the analysis scripts inline (no shell access needed).

Writes cluster/ood-export-release-v6.txt and cluster/ood-analyze-episodes-v6.txt. Each embeds the
current analysis/*.py (and the paper table for the analyzer) as quoted heredocs, so uploading nothing
but the pasted job is enough. Re-run after editing the analysis scripts; a test checks they match.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = 'phys-pro6000-100-v6'
HEADER = '''#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output={name}-%j.out

# GPU request is intentionally absent: this job only reads records under $DATA and writes {writes}.
set -euo pipefail
DATA="$HOME/robotwin-data"
EXPERIMENT="{experiment}"
'''


def export_job():
    body = (ROOT / 'analysis/export_release.py').read_text()
    assert 'ROBOTWIN_EXPORT' not in body
    return (HEADER.format(name='phys-export-release', experiment=EXPERIMENT, writes='$DATA/release/<experiment>{,.zip}')
            + '''OUT="$DATA/release/$EXPERIMENT"
# Regenerated outputs only: remove the previous export so the script can write a fresh folder and ZIP.
rm -rf "$OUT" "$OUT.zip"
mkdir -p "$DATA/release"
python3 -u - --root "$DATA" --experiment "$EXPERIMENT" --out "$OUT" <<'ROBOTWIN_EXPORT'
''' + body + '''ROBOTWIN_EXPORT
echo "EXPORT_DONE: download $OUT.zip through OOD Files (robotwin-data/release/) and unzip it into data/ of the repository"
''')


def analyze_job():
    body = (ROOT / 'analysis/analyze_episodes.py').read_text()
    table = (ROOT / 'references/paper-table4-physical-random.json').read_text()
    assert 'ROBOTWIN_ANALYZE' not in body and 'ROBOTWIN_TABLE' not in table
    return (HEADER.format(name='phys-analyze-episodes', experiment=EXPERIMENT, writes='$DATA/reports/analysis/')
            + '''# Models to analyze (space separated); each gets its own report file.
MODELS="fastwam g05 pi05"
# Optional: an earlier run folder of the same experiment to compare episode by episode (fastwam only), or "".
COMPARE="$DATA/runs/$EXPERIMENT/fastwam-oidn201-discarded"
mkdir -p "$DATA/reports/analysis"
STAGE=$(mktemp -d)
cat > "$STAGE/paper-table4-physical-random.json" <<'ROBOTWIN_TABLE'
''' + table + '''ROBOTWIN_TABLE
cat > "$STAGE/analyze_episodes.py" <<'ROBOTWIN_ANALYZE'
''' + body + '''ROBOTWIN_ANALYZE
for model in $MODELS; do
  if [[ ! -d "$DATA/runs/$EXPERIMENT/$model/physical_random" ]]; then
    echo "ANALYZE_SKIP: no results yet for $model"
    continue
  fi
  extra=()
  if [[ "$model" == fastwam && -n "$COMPARE" && -d "$COMPARE" ]]; then extra=(--compare "$COMPARE"); fi
  report="$DATA/reports/analysis/$model-$SLURM_JOB_ID.txt"
  python3 -u "$STAGE/analyze_episodes.py" --root "$DATA" --experiment "$EXPERIMENT" --model "$model" \\
    --reference "$STAGE/paper-table4-physical-random.json" \\
    --step-limits "$DATA/vendor/RoboTwin/task_config/_eval_step_limit.yml" \\
    --errors --json "$DATA/reports/analysis/$model-$SLURM_JOB_ID.json" "${extra[@]}" | tee "$report"
  echo "ANALYZE_DONE: $report"
done
rm -rf "$STAGE"
''')


def main():
    for name, text in (('ood-export-release-v6.txt', export_job()), ('ood-analyze-episodes-v6.txt', analyze_job())):
        (ROOT / 'cluster' / name).write_text(text)
        print(ROOT / 'cluster' / name)


if __name__ == '__main__':
    main()
