#!/usr/bin/env python3
"""Package pinned CuRobo mask repair plus pool refill for existing v5 images."""
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
NAME = 'robotwin-curobo-fix-v1'


def checksums(folder, extensions):
    return {str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(folder.rglob('*')) if path.is_file() and path.suffix in extensions
            and '__pycache__' not in path.parts}


def main():
    approved = {}
    packages = {}
    for name in ('robotwin-expert-fix-v1', 'robotwin-expert-fix-v2'):
        with zipfile.ZipFile(ROOT / 'reports' / (name + '.zip')) as archive:
            manifest = json.loads(archive.read(name + '/manifest.json'))
            packages[name] = (manifest, archive.read(name + '/vendor/transforms.py') if name.endswith('v2') else None,
                              archive.read(name + '/vendor/planner.py') if name.endswith('v2') else None)
            for relative, checksum in manifest['files'].items():
                approved.setdefault('src/' + relative, set()).add(checksum)
    for relative, packaged in (('envs/utils/transforms.py', packages['robotwin-expert-fix-v2'][1]),
                              ('envs/robot/planner.py', packages['robotwin-expert-fix-v2'][2])):
        path = 'vendor/RoboTwin/' + relative
        approved[path] = {hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),
                          hashlib.sha256(packaged).hexdigest()}
    (ROOT / 'tools/expert_fix_previous.json').write_text(json.dumps(
        {path: sorted(values) for path, values in approved.items()}, sort_keys=True, indent=2) + '\n')
    source = checksums(ROOT / 'src', ('.py',))
    tools = checksums(ROOT / 'tools', ('.py', '.sh', '.json'))
    digest = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    old = packages['robotwin-expert-fix-v2'][0]
    manifest = dict(name=NAME, files=source, tools=tools, source_sha256=digest(source), tools_sha256=digest(tools),
                    transforms_sha256=old['transforms_sha256'], planner_sha256=old['planner_sha256'],
                    curobo_fix='independent-ik-result-masks-v1', experiment='phys-pro6000-100-v4')
    archive = ROOT / 'reports' / (NAME + '.zip')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        def write(relative, data):
            info = zipfile.ZipInfo(NAME + '/' + relative, date_time=(2026, 10, 12, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, data)
        for base, files in (('src', source), ('tools', tools)):
            for relative in files:
                write(base + '/' + relative, (ROOT / base / relative).read_bytes())
        write('vendor/transforms.py', packages['robotwin-expert-fix-v2'][1])
        write('vendor/planner.py', packages['robotwin-expert-fix-v2'][2])
        write('vendor/ROBOTWIN_LICENSE', (ROOT / 'vendor/RoboTwin/LICENSE').read_bytes())
        write('manifest.json', json.dumps(manifest, indent=2) + '\n')
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    entries = [
        ('job-pro6000-pool-geometry-v5.sbatch', 'job-pro6000-pool-curobo-v5.sbatch', 'ood-pool-pro6000x2-v5-curobo-fix.txt'),
        ('job-pro6000-fastwam-geometry-v5.sbatch', 'job-pro6000-fastwam-curobo-v5.sbatch', 'ood-fastwam-pro6000-v5-curobo-fix.txt'),
        ('job-pro6000-pi05-geometry-v5.sbatch', 'job-pro6000-pi05-curobo-v5.sbatch', 'ood-pi05-pro6000-v5-curobo-fix.txt'),
    ]
    for template, script, copy in entries:
        text = (ROOT / 'cluster' / template).read_text()
        text = text.replace('robotwin-expert-fix-v2', NAME).replace('phys-pro6000-100-v3', 'phys-pro6000-100-v4')
        text = text.replace(old['source_sha256'], manifest['source_sha256']).replace(old['tools_sha256'], manifest['tools_sha256'])
        text = re.sub(r"printf '%s  %s\\n' [0-9a-f]{64}", lambda match: "printf '%s  %s\\n' " + archive_hash, text)
        text = text.replace('--resume-from-experiment phys-pro6000-100-v2',
                            '--resume-from-experiment phys-pro6000-100-v3 --fallback-experiment phys-pro6000-100-v2')
        text = text.replace("print('EXPERT_GEOMETRY_PATCH_PASS: shared rotation helpers and task scheduler verified', flush=True)",
                            "print('CUROBO_MASK_PATCH_SOURCE_PASS: simulator and reviewed helpers verified', flush=True)")
        (ROOT / 'cluster' / script).write_text(text)
        (ROOT / 'cluster' / copy).write_text(text)
    diagnosis = (ROOT / 'cluster/job-pro6000-fastwam-curobo-v5.sbatch').read_text()
    diagnosis = diagnosis.replace('#SBATCH --job-name=fastwam-phys', '#SBATCH --job-name=curobo-diag')
    diagnosis = diagnosis.replace('#SBATCH --time=3-00:00:00', '#SBATCH --time=01:30:00')
    diagnosis = diagnosis.replace('exec bash "$ROOT/cluster/run_parallel.sh" --resume-from-experiment phys-pro6000-100-v3 --fallback-experiment phys-pro6000-100-v2',
        'cd "$ROOT"\nexport SAPIEN_RENDER_DEVICE=cuda:0\n'
        '"$ROOT/.env/bin/python" -m robotwin_phys.expert_diagnosis --task adjust_bottle --seed 100000 '
        '--output "$ROOT/reports/curobo-diagnostics/$SLURM_JOB_ID/adjust_bottle"\n'
        'exec "$ROOT/.env/bin/python" -m robotwin_phys.expert_diagnosis --task beat_block_hammer --seed 100115 '
        '--output "$ROOT/reports/curobo-diagnostics/$SLURM_JOB_ID/beat_block_hammer"')
    (ROOT / 'cluster/job-pro6000-curobo-diagnose.sbatch').write_text(diagnosis)
    (ROOT / 'cluster/ood-curobo-diagnose-pro6000-v5.txt').write_text(diagnosis)
    print('Packaged ' + str(archive) + '; preserve v3/v2 and reverify saved episodes into v4')


if __name__ == '__main__':
    main()
