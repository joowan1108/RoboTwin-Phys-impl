#!/usr/bin/env python3
"""Create a guarded v5 math repair and OOD launchers with v2 reverification."""
import hashlib
import json
from pathlib import Path
import re
import zipfile

from patch_robotwin import PIN, patch_transform_text, patch_planner_text

ROOT = Path(__file__).resolve().parents[1]
NAME = 'robotwin-expert-fix-v2'


def checksums(directory, patterns):
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob('*')) if p.is_file() and p.suffix in patterns
            and '__pycache__' not in p.parts}


def main():
    original = ROOT / 'vendor/RoboTwin/envs/utils/transforms.py'
    text = original.read_text()
    if '# Reviewed finite-rotation helpers' in text:
        raise RuntimeError('Package against the unmodified pinned transforms.py')
    fixed = patch_transform_text(text)
    planner = ROOT / 'vendor/RoboTwin/envs/robot/planner.py'
    planner_fixed = patch_planner_text(planner.read_text())
    with zipfile.ZipFile(ROOT / 'reports/robotwin-expert-fix-v1.zip') as old:
        previous = json.loads(old.read('robotwin-expert-fix-v1/manifest.json'))
    approved = {'src/' + p: checksum for p, checksum in previous['files'].items()}
    approved['vendor/RoboTwin/envs/utils/transforms.py'] = hashlib.sha256(text.encode()).hexdigest()
    approved['vendor/RoboTwin/envs/robot/planner.py'] = hashlib.sha256(planner.read_bytes()).hexdigest()
    (ROOT / 'tools/expert_fix_previous.json').write_text(json.dumps(approved, indent=2, sort_keys=True) + '\n')
    source_files = checksums(ROOT / 'src', ('.py',))
    tool_files = checksums(ROOT / 'tools', ('.py', '.sh', '.json'))
    checksum = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    manifest = dict(name=NAME, robotwin_revision=PIN, source_sha256=checksum(source_files),
                    tools_sha256=checksum(tool_files), files=source_files, tools=tool_files,
                    transforms_sha256=hashlib.sha256(fixed.encode()).hexdigest(),
                    planner_sha256=hashlib.sha256(planner_fixed.encode()).hexdigest())
    archive = ROOT / 'reports' / f'{NAME}.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        def write(name, data):
            info = zipfile.ZipInfo(f'{NAME}/{name}', date_time=(2026, 10, 12, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, data)
        for base, files in (('src', source_files), ('tools', tool_files)):
            for path in files:
                write(f'{base}/{path}', (ROOT / base / path).read_bytes())
        write('vendor/transforms.py', fixed)
        write('vendor/planner.py', planner_fixed)
        write('vendor/ROBOTWIN_LICENSE', (ROOT / 'vendor/RoboTwin/LICENSE').read_bytes())
        write('manifest.json', json.dumps(manifest, indent=2) + '\n')
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    validation = '''
import hashlib, json
from pathlib import Path
root = Path('/home/guest/robotwin-workspace')
def hashes(folder, patterns):
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob('*')) if p.is_file() and p.suffix in patterns
            and '__pycache__' not in p.parts}
def checksum(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
if checksum(hashes(root / 'src', ('.py',))) != 'SOURCE_HASH':
    raise SystemExit('Expert patch source differs; restore uploaded snapshot')
if checksum(hashes(root / 'tools', ('.py', '.sh', '.json'))) != 'TOOLS_HASH':
    raise SystemExit('Expert patch tools differ; restore uploaded snapshot')
if hashlib.sha256((root / 'vendor/RoboTwin/envs/utils/transforms.py').read_bytes()).hexdigest() != 'TRANSFORMS_HASH':
    raise SystemExit('Shared transform repair was not bound correctly')
if hashlib.sha256((root / 'vendor/RoboTwin/envs/robot/planner.py').read_bytes()).hexdigest() != 'PLANNER_HASH':
    raise SystemExit('Planner exception repair was not bound correctly')
print('EXPERT_GEOMETRY_PATCH_PASS: shared rotation helpers and task scheduler verified', flush=True)
'''.replace('SOURCE_HASH', manifest['source_sha256']).replace('TOOLS_HASH', manifest['tools_sha256']).replace(
        'TRANSFORMS_HASH', manifest['transforms_sha256']).replace('PLANNER_HASH', manifest['planner_sha256'])
    base = (ROOT / 'cluster/job-pro6000-expert-fix.sbatch').read_text()
    base = base.replace('robotwin-expert-fix-v1', NAME).replace('phys-pro6000-100-v2', 'phys-pro6000-100-v3')
    base = re.sub(r"printf '%s  %s\\n' [0-9a-f]{64}", lambda m: "printf '%s  %s\\n' " + archive_hash, base)
    base = base.replace('export APPTAINER_BINDPATH="$APPTAINER_BINDPATH,$ROBOTWIN_PATCH_DIR/src:$ROOT/src:ro"',
                        'export APPTAINER_BINDPATH="$APPTAINER_BINDPATH,$ROBOTWIN_PATCH_DIR/src:$ROOT/src:ro"\n'
                        'export APPTAINER_BINDPATH="$APPTAINER_BINDPATH,$ROBOTWIN_PATCH_DIR/tools:$ROOT/tools:ro"\n'
                        'export APPTAINER_BINDPATH="$APPTAINER_BINDPATH,$ROBOTWIN_PATCH_DIR/vendor/transforms.py:$ROOT/vendor/RoboTwin/envs/utils/transforms.py:ro"\n'
                        'export APPTAINER_BINDPATH="$APPTAINER_BINDPATH,$ROBOTWIN_PATCH_DIR/vendor/planner.py:$ROOT/vendor/RoboTwin/envs/robot/planner.py:ro"')
    start = base.index("<<'PY_EXPERT_PATCH'\n") + len("<<'PY_EXPERT_PATCH'\n")
    end = base.index('\nPY_EXPERT_PATCH', start)
    base = base[:start] + validation.strip() + base[end:]
    entries = [
        ('job-pro6000-pool-geometry-v5.sbatch', 'ood-pool-pro6000x2-v5-geometry-fix.txt', 'freeze'),
        ('job-pro6000-fastwam-geometry-v5.sbatch', 'ood-fastwam-pro6000-v5-geometry-fix.txt', 'all'),
        ('job-pro6000-pi05-geometry-v5.sbatch', 'ood-pi05-pro6000-v5-geometry-fix.txt', 'evaluate'),
        ('job-pro6000-geometry-diagnose.sbatch', 'ood-geometry-diagnose-pro6000-v5.txt', 'diagnosis'),
    ]
    for script, copy, phase in entries:
        content = base
        if phase == 'freeze':
            content = content.replace('#SBATCH --job-name=fastwam-phys', '#SBATCH --job-name=phys-pool-fixed')
            content = content.replace('#SBATCH --gres=gpu:pro6000:1', '#SBATCH --gres=gpu:pro6000:2')
            content = content.replace('#SBATCH --cpus-per-task=8', '#SBATCH --cpus-per-task=32')
            content = content.replace('#SBATCH --mem=96G', '#SBATCH --mem=128G')
            content = content.replace('ROBOTWIN_PHASE="all"', 'ROBOTWIN_PHASE="freeze"')
            content = content.replace('export APPTAINERENV_ROBOTWIN_POLICY_REPLICAS=1',
                                      'export APPTAINERENV_ROBOTWIN_FREEZE_WORKERS_PER_GPU=4')
            content = content.replace('exec bash "$ROOT/cluster/run_parallel.sh"',
                                      'exec bash "$ROOT/cluster/run_parallel.sh" '
                                      '--freeze-workers-per-gpu "$ROBOTWIN_FREEZE_WORKERS_PER_GPU" '
                                      '--resume-from-experiment phys-pro6000-100-v2')
        elif phase == 'evaluate':
            content = content.replace('#SBATCH --job-name=fastwam-phys', '#SBATCH --job-name=pi05-fixed')
            content = content.replace('ROBOTWIN_MODEL="fastwam"', 'ROBOTWIN_MODEL="pi05"')
            content = content.replace('ROBOTWIN_PHASE="all"', 'ROBOTWIN_PHASE="evaluate"')
        elif phase == 'all':
            content = content.replace('exec bash "$ROOT/cluster/run_parallel.sh"',
                                      'exec bash "$ROOT/cluster/run_parallel.sh" --resume-from-experiment phys-pro6000-100-v2')
        else:
            content = content.replace('#SBATCH --job-name=fastwam-phys', '#SBATCH --job-name=geometry-diag')
            content = content.replace('#SBATCH --time=3-00:00:00', '#SBATCH --time=01:30:00')
            content = content.replace('exec bash "$ROOT/cluster/run_parallel.sh"',
                                      'cd "$ROOT"\nexport SAPIEN_RENDER_DEVICE=cuda:0\n'
                                      'exec "$ROOT/.env/bin/python" -m robotwin_phys.expert_diagnosis '
                                      '--task beat_block_hammer --seed 100113 '
                                      '--output "$ROOT/reports/expert-diagnostics/$SLURM_JOB_ID"')
        (ROOT / 'cluster' / script).write_text(content)
        (ROOT / 'cluster' / copy).write_text(content)
    print(f'Packaged {archive}; new experiment phys-pro6000-100-v3')


if __name__ == '__main__':
    main()
