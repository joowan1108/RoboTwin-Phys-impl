#!/usr/bin/env python3
"""Generate a controller fix and read-only OOD hash report using the existing ZIP."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    controller = (ROOT / 'tools/reverify_pool.py').read_text()
    controller_hash = hashlib.sha256(controller.encode()).hexdigest()
    with zipfile.ZipFile(ROOT / 'reports/robotwin-curobo-fix-v1.zip') as archive:
        manifest = json.loads(archive.read('robotwin-curobo-fix-v1/manifest.json'))
    tools = dict(manifest['tools'], **{'reverify_pool.py': controller_hash})
    tools_hash = hashlib.sha256(json.dumps(tools, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    overlay = '''
# Correct the reviewed-helper addition check without changing simulator source.
ROBOTWIN_REVERIFY_HOST="$DATA/reports/launchers/$SLURM_JOB_ID/reverify_pool.py"
mkdir -p "$(dirname -- "$ROBOTWIN_REVERIFY_HOST")"
cat > "$ROBOTWIN_REVERIFY_HOST" <<'ROBOTWIN_PROVENANCE_CONTROLLER'
''' + controller + '''ROBOTWIN_PROVENANCE_CONTROLLER
printf '%s  %s\\n' CONTROLLER_HASH "$ROBOTWIN_REVERIFY_HOST" | sha256sum -c -
export APPTAINER_BINDPATH="$APPTAINER_BINDPATH,$ROBOTWIN_REVERIFY_HOST:$ROOT/tools/reverify_pool.py:ro"
'''.replace('CONTROLLER_HASH', controller_hash)
    for template, script, copy in [
        ('job-pro6000-pool-curobo-v5.sbatch', 'job-pro6000-pool-provenance-v5.sbatch', 'ood-pool-pro6000x2-v5-provenance-fix.txt'),
        ('job-pro6000-fastwam-curobo-v5.sbatch', 'job-pro6000-fastwam-provenance-v5.sbatch', 'ood-fastwam-pro6000-v5-provenance-fix.txt'),
    ]:
        text = (ROOT / 'cluster' / template).read_text()
        text = text.replace('# Existing v5 images need the official template-generation step after unzip.',
                            overlay + '\n# Existing v5 images need the official template-generation step after unzip.')
        text = text.replace(manifest['tools_sha256'], tools_hash)
        (ROOT / 'cluster' / script).write_text(text)
        (ROOT / 'cluster' / copy).write_text(text)
    diagnostic = (ROOT / 'cluster/job-pro6000-pool-provenance-v5.sbatch').read_text()
    diagnostic = diagnostic.replace('#SBATCH --job-name=phys-pool-fixed', '#SBATCH --job-name=source-diag')
    diagnostic = diagnostic.replace('#SBATCH --gres=gpu:pro6000:2', '#SBATCH --gres=gpu:pro6000:1')
    diagnostic = diagnostic.replace('#SBATCH --cpus-per-task=32', '#SBATCH --cpus-per-task=2')
    diagnostic = diagnostic.replace('#SBATCH --mem=128G', '#SBATCH --mem=16G')
    diagnostic = diagnostic.replace('#SBATCH --time=3-00:00:00', '#SBATCH --time=00:10:00')
    helper = (ROOT / 'tools/diagnose_provenance.py').read_text()
    host_helper = '''
ROBOTWIN_DIAG_HOST="$DATA/reports/launchers/$SLURM_JOB_ID/diagnose_provenance.py"
cat > "$ROBOTWIN_DIAG_HOST" <<'ROBOTWIN_SOURCE_DIAG'
''' + helper + '''ROBOTWIN_SOURCE_DIAG
'''
    diagnostic = diagnostic.replace('cat > "$ROBOTWIN_LAUNCH_HOST/run.sh"', host_helper + '\ncat > "$ROBOTWIN_LAUNCH_HOST/run.sh"')
    start = diagnostic.index('# Use the same preparation lock as parallel_eval; assets are shared across jobs.')
    end = diagnostic.index('\nROBOTWIN_LAUNCH\n', start)
    diagnostic = diagnostic[:start] + '''cd "$ROOT"
exec "$ROOT/.env/bin/python" "$ROOT/reports/launchers/$SLURM_JOB_ID/diagnose_provenance.py" \\
  --root "$ROOT" --task adjust_bottle --job 1641 \\
  --output "$ROOT/reports/provenance-diagnostics/$SLURM_JOB_ID/summary.json"
''' + diagnostic[end:]
    (ROOT / 'cluster/job-pro6000-provenance-diagnose.sbatch').write_text(diagnostic)
    (ROOT / 'cluster/ood-provenance-diagnose-pro6000-v5.txt').write_text(diagnostic)
    print('Generated reviewed-helper addition fix and read-only hash diagnosis; no new ZIP')


if __name__ == '__main__':
    main()
