#!/usr/bin/env python3
"""Create the v6 fresh-pool snapshot: CoM offset on task objects only, clutter keeps its nominal CoM.

The simulator source changes (src/robotwin_phys/simulator.py), so the v5 pool and its saved
progress are not reusable: the runtime fingerprint differs and the saved initial states were
produced with the offset applied to clutter. This packages a new immutable snapshot and a
pool-only OOD launcher for a new experiment name. It includes the planner-failure (v3),
grasp-contact (v2) and empty-grasp (v1) expert repairs already in the tree.
"""
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
NAME = 'robotwin-fresh-pool-v2'
PREVIOUS = 'robotwin-fresh-pool-v1'
EXPERIMENT = 'phys-pro6000-100-v6'
PREVIOUS_EXPERIMENT = 'phys-pro6000-100-v5'
MARKER = 'FRESH_POOL_V2_PATCH_SOURCE_PASS'


def checksums(folder, extensions):
    return {str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(folder.rglob('*')) if path.is_file() and path.suffix in extensions
            and '__pycache__' not in path.parts}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def main():
    # Preserve the reviewed vendor repairs byte for byte from the delivered v5 snapshot.
    with zipfile.ZipFile(ROOT / f'reports/{PREVIOUS}.zip') as old:
        prefix = PREVIOUS + '/'
        vendor = {name: old.read(prefix + 'vendor/' + name)
                  for name in ('transforms.py', 'planner.py', 'ROBOTWIN_LICENSE')}
        prior = json.loads(old.read(prefix + 'manifest.json'))
    source = checksums(ROOT / 'src', ('.py',))
    tools = checksums(ROOT / 'tools', ('.py', '.sh', '.json'))
    simulator = (ROOT / 'src/robotwin_phys/simulator.py').read_text()
    hooks = (ROOT / 'src/robotwin_phys/hooks.py').read_text()
    if '_clutter_entity_ids(' not in simulator or '"clutter": is_clutter' not in simulator or 'clutter=True' not in simulator or '_clutter = ContextVar' not in hooks:
        raise SystemExit('Unreviewed simulator change: the CoM scope rule is missing')
    manifest = dict(name=NAME, files=source, tools=tools, source_sha256=digest(source),
                    tools_sha256=digest(tools), experiment=EXPERIMENT,
                    curobo_fix='consistent-batch-results-v2',
                    transforms_sha256=hashlib.sha256(vendor['transforms.py']).hexdigest(),
                    planner_sha256=hashlib.sha256(vendor['planner.py']).hexdigest(),
                    com_offset_scope='task objects only; official clutter keeps its nominal centre of mass',
                    clutter_scope='official clutter keeps nominal mass, CoM and size and receives no policy-stage force; contact material still applies',
                    includes_expert_repairs=['robotwin-empty-grasp-fix-v1', 'robotwin-grasp-contact-fix-v2',
                                             'robotwin-planner-failure-fix-v3'],
                    supersedes=f'{PREVIOUS_EXPERIMENT} pool built with the CoM offset applied to clutter',
                    pool_mode='fresh experiment; resume only this experiment; no migration')
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        def write(relative, data):
            info = zipfile.ZipInfo(NAME + '/' + relative, date_time=(2026, 10, 14, 0, 0, 0))
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
    text = (ROOT / 'cluster/job-pro6000-pool-fresh-v5.sbatch').read_text()
    text = text.replace(PREVIOUS, NAME)
    text = text.replace(PREVIOUS_EXPERIMENT, EXPERIMENT)
    text = text.replace('#SBATCH --job-name=phys-pool-fresh', '#SBATCH --job-name=phys-pool-fresh-v6')
    text = text.replace('#SBATCH --gres=gpu:pro6000:1', '#SBATCH --gres=gpu:pro6000:2')
    text = text.replace('#SBATCH --cpus-per-task=16', '#SBATCH --cpus-per-task=64')
    text = text.replace('export APPTAINERENV_ROBOTWIN_FREEZE_WORKERS_PER_GPU=4',
                        'export APPTAINERENV_ROBOTWIN_FREEZE_WORKERS_PER_GPU=8')
    text = text.replace('# Pool preparation only: four independent tasks per allocated GPU.',
                        '# Pool preparation only: eight independent tasks per allocated GPU (about 6 GB VRAM and 4 CPUs each).')
    for name in ('source_sha256', 'tools_sha256'):
        text = text.replace(prior[name], manifest[name])
    text = re.sub(r"printf '%s  %s\\n' [0-9a-f]{64}",
                  lambda match: "printf '%s  %s\\n' " + archive_hash, text)
    text = text.replace('FRESH_POOL_PATCH_SOURCE_PASS: reviewed simulator snapshot verified',
                        MARKER + ': reviewed simulator snapshot verified')
    text = text.replace('echo "FRESH_POOL: experiment=$ROBOTWIN_EXPERIMENT;',
                        'echo "FRESH_POOL_V2: CoM offset on task objects only; clutter keeps nominal CoM"\n'
                        'echo "FRESH_POOL: experiment=$ROBOTWIN_EXPERIMENT;')
    for check in (NAME, EXPERIMENT, archive_hash, manifest['source_sha256'], manifest['tools_sha256'], MARKER,
                  'FREEZE_WORKERS_PER_GPU=8', '--cpus-per-task=64', 'gpu:pro6000:2'):
        if check not in text:
            raise RuntimeError(f'Launcher substitution failed for {check}')
    if PREVIOUS_EXPERIMENT in text or PREVIOUS in text:
        raise RuntimeError('Launcher still references the superseded v5 pool')
    for filename in ('job-pro6000-pool-fresh-v6.sbatch', 'ood-pool-pro6000-v6-fresh.txt'):
        (ROOT / 'cluster' / filename).write_text(text)
    # Same snapshot, same pool: build it and then evaluate FastWAM in one job on both GPUs.
    full = text.replace('#SBATCH --job-name=phys-pool-fresh-v6', '#SBATCH --job-name=fastwam-phys-v6')
    full = full.replace('export APPTAINERENV_ROBOTWIN_PHASE="freeze"', 'export APPTAINERENV_ROBOTWIN_PHASE="all"')
    full = full.replace('export APPTAINERENV_ROBOTWIN_FREEZE_WORKERS_PER_GPU=8\n',
                        'export APPTAINERENV_ROBOTWIN_FREEZE_WORKERS_PER_GPU=8\n'
                        '# Evaluation on two GPUs: simulators on both, FastWAM replicas on the first.\n'
                        '# Each replica drives one simulator; 4 replicas = 4 concurrent episodes.\n'
                        'export APPTAINERENV_ROBOTWIN_POLICY_REPLICAS=4\n')
    full = full.replace('echo "FRESH_POOL: experiment=$ROBOTWIN_EXPERIMENT; all tasks; 100 episodes/task; start seed=100000; no older pool import"',
                        'echo "FRESH_POOL: experiment=$ROBOTWIN_EXPERIMENT; all tasks; 100 episodes/task; start seed=100000; no older pool import"\n'
                        'echo "POOL_THEN_FASTWAM: dual-GPU layout; policy replicas=$ROBOTWIN_POLICY_REPLICAS"')
    for check in ('ROBOTWIN_PHASE="all"', 'POLICY_REPLICAS=4', 'POOL_THEN_FASTWAM', 'fastwam-phys-v6'):
        if check not in full:
            raise RuntimeError(f'Full-run launcher substitution failed for {check}')
    for filename in ('job-pro6000x2-fastwam-v6-all.sbatch', 'ood-fastwam-pro6000x2-v6-all.txt'):
        (ROOT / 'cluster' / filename).write_text(full)
    print(f'Packaged {archive} ({archive_hash}): {EXPERIMENT}, all tasks, clutter keeps nominal CoM')


if __name__ == '__main__':
    main()
