#!/usr/bin/env python3
"""Create a new immutable simulator snapshot and a pool-only OOD launcher."""
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
NAME = 'robotwin-fresh-pool-v1'
EXPERIMENT = 'phys-pro6000-100-v5'


def checksums(folder, extensions):
    return {str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(folder.rglob('*')) if path.is_file() and path.suffix in extensions
            and '__pycache__' not in path.parts}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def main():
    # Preserve the previously reviewed geometry/planner repairs byte for byte.
    with zipfile.ZipFile(ROOT / 'reports/robotwin-expert-fix-v2.zip') as old:
        prefix = 'robotwin-expert-fix-v2/'
        vendor = {name: old.read(prefix + 'vendor/' + name)
                  for name in ('transforms.py', 'planner.py', 'ROBOTWIN_LICENSE')}
    with zipfile.ZipFile(ROOT / 'reports/robotwin-curobo-fix-v1.zip') as old:
        prior = json.loads(old.read('robotwin-curobo-fix-v1/manifest.json'))
    source = checksums(ROOT / 'src', ('.py',))
    tools = checksums(ROOT / 'tools', ('.py', '.sh', '.json'))
    manifest = dict(name=NAME, files=source, tools=tools, source_sha256=digest(source),
                    tools_sha256=digest(tools), experiment=EXPERIMENT,
                    curobo_fix='consistent-batch-results-v2',
                    transforms_sha256=hashlib.sha256(vendor['transforms.py']).hexdigest(),
                    planner_sha256=hashlib.sha256(vendor['planner.py']).hexdigest(),
                    pool_mode='fresh experiment; resume only this experiment; no migration')
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        def write(relative, data):
            info = zipfile.ZipInfo(NAME + '/' + relative, date_time=(2026, 10, 12, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, data)
        for base, files in (('src', source), ('tools', tools)):
            for relative in files:
                write(base + '/' + relative, (ROOT / base / relative).read_bytes())
        for name, data in vendor.items():
            write('vendor/' + name, data)
        write('manifest.json', json.dumps(manifest, indent=2) + '\n')
    archive = ROOT / 'reports' / (NAME + '.zip')
    data = output.getvalue()
    if archive.exists() and archive.read_bytes() != data:
        raise FileExistsError('Do not overwrite a delivered snapshot; use a new package name')
    archive.write_bytes(data)
    archive_hash = hashlib.sha256(data).hexdigest()
    text = (ROOT / 'cluster/job-pro6000-pool-curobo-v5.sbatch').read_text()
    text = text.replace('robotwin-curobo-fix-v1', NAME)
    text = text.replace('phys-pro6000-100-v4', EXPERIMENT)
    text = text.replace('#SBATCH --job-name=phys-pool-fixed', '#SBATCH --job-name=phys-pool-fresh')
    text = text.replace('#SBATCH --gres=gpu:pro6000:2', '#SBATCH --gres=gpu:pro6000:1')
    text = text.replace('#SBATCH --cpus-per-task=32', '#SBATCH --cpus-per-task=16')
    text = text.replace('# One allocated PRO 6000 runs one simulator and one FastWAM policy process.',
                        '# Pool preparation only: four independent tasks per allocated GPU.')
    for name in ('source_sha256', 'tools_sha256'):
        text = text.replace(prior[name], manifest[name])
    text = re.sub(r"printf '%s  %s\\n' [0-9a-f]{64}",
                  lambda match: "printf '%s  %s\\n' " + archive_hash, text)
    text = text.replace(' --resume-from-experiment phys-pro6000-100-v3 --fallback-experiment phys-pro6000-100-v2', '')
    text = text.replace('CUROBO_MASK_PATCH_SOURCE_PASS: simulator and reviewed helpers verified',
                        'FRESH_POOL_PATCH_SOURCE_PASS: reviewed simulator snapshot verified')
    text = text.replace('exec bash "$ROOT/cluster/run_parallel.sh"',
                        'echo "FRESH_POOL: experiment=$ROBOTWIN_EXPERIMENT; all tasks; 100 episodes/task; '
                        'start seed=100000; no older pool import"\n'
                        'exec bash "$ROOT/cluster/run_parallel.sh"')
    text = text.replace('--freeze-workers-per-gpu "$ROBOTWIN_FREEZE_WORKERS_PER_GPU"',
                        '--seed 100000 --freeze-workers-per-gpu "$ROBOTWIN_FREEZE_WORKERS_PER_GPU"')
    for filename in ('job-pro6000-pool-fresh-v5.sbatch', 'ood-pool-pro6000-v5-fresh.txt'):
        (ROOT / 'cluster' / filename).write_text(text)
    print(f'Packaged {archive}: {EXPERIMENT}, all tasks, no older pool migration')


if __name__ == '__main__':
    main()
