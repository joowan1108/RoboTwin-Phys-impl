#!/usr/bin/env python3
"""Package the reviewed grasp-annotation repairs on the immutable v5 snapshot."""
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
NAME = 'robotwin-grasp-contact-fix-v2'
BASE = 'robotwin-fresh-pool-v1'
BASE_SHA = '4b2b19ccbc2581411019ac6c3c6e42c38434708d45140afcead3bd9116583ce6'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def main():
    # Delivered v2 is pinned to its own planning source; later tree changes
    # ship as new packages (see package_planner_failure_fix.py).
    from resume_contact_pool import BEFORE, EMPTY_ONLY, CONTACTS as AFTER, PLANNING
    RULE = 'grasp-annotation-availability-v2'
    original = ROOT / 'reports' / (BASE + '.zip')
    if hashlib.sha256(original.read_bytes()).hexdigest() != BASE_SHA:
        raise ValueError('Delivered baseline archive changed')
    with zipfile.ZipFile(original) as archive:
        payloads = {name.removeprefix(BASE + '/'): archive.read(name) for name in archive.namelist()}
    prior = json.loads(payloads.pop('manifest.json'))
    if hashlib.sha256(payloads[PLANNING]).hexdigest() != BEFORE:
        raise ValueError('Unexpected old planning source')
    payloads[PLANNING] = (ROOT / PLANNING).read_bytes()
    if hashlib.sha256(payloads[PLANNING]).hexdigest() != AFTER:
        raise ValueError('Unreviewed planning change; do not preserve evidence')
    payloads['tools/resume_contact_pool.py'] = (ROOT / 'tools/resume_contact_pool.py').read_bytes()
    def checksums(prefix):
        return {path.removeprefix(prefix): hashlib.sha256(data).hexdigest()
                for path, data in sorted(payloads.items()) if path.startswith(prefix)}
    source, tools = checksums('src/'), checksums('tools/')
    manifest = dict(prior, name=NAME, files=source, tools=tools,
                    source_sha256=digest(source), tools_sha256=digest(tools),
                    baseline_archive_sha256=BASE_SHA, preservation_rule=RULE,
                    planning_before=[BEFORE, EMPTY_ONLY], planning_after=AFTER,
                    pool_mode='same v5 pool; exact compatible change; preserve records without replay')
    payloads['manifest.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path, data in sorted(payloads.items()):
            info = zipfile.ZipInfo(NAME + '/' + path, date_time=(2026, 10, 13, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    data = output.getvalue()
    destination = ROOT / 'reports' / (NAME + '.zip')
    if destination.exists() and destination.read_bytes() != data:
        raise FileExistsError('Do not overwrite a delivered snapshot; use a new package name')
    destination.write_bytes(data)
    checksum = hashlib.sha256(data).hexdigest()
    script = (ROOT / 'cluster/job-pro6000-pool-fresh-v5.sbatch').read_text()
    script = script.replace(BASE, NAME).replace(BASE_SHA, checksum)
    script = script.replace('#SBATCH --job-name=phys-pool-fresh', '#SBATCH --job-name=phys-pool-contacts')
    script = script.replace('#SBATCH --gres=gpu:pro6000:1', '#SBATCH --gres=gpu:pro6000:2')
    script = script.replace('#SBATCH --cpus-per-task=16', '#SBATCH --cpus-per-task=32')
    script = script.replace('FRESH_POOL_PATCH_SOURCE_PASS', 'GRASP_CONTACT_PATCH_SOURCE_PASS')
    for key in ('source_sha256', 'tools_sha256'):
        script = script.replace(prior[key], manifest[key])
    script = re.sub(r'^echo "FRESH_POOL:.*$',
                    'echo "POOL_CONTINUE: same v5; all 50 tasks; preserve saved seeds; no expert replay"',
                    script, flags=re.MULTILINE)
    command = ('exec "$ROOT/.env/bin/python" "$ROOT/tools/resume_contact_pool.py" '
               '--model "$ROBOTWIN_MODEL" --experiment "$ROBOTWIN_EXPERIMENT" --phase freeze '
               '--episodes "$ROBOTWIN_EPISODES" --seed 100000 '
               '--freeze-workers-per-gpu "$ROBOTWIN_FREEZE_WORKERS_PER_GPU"')
    script, changed = re.subn(r'^exec bash "\$ROOT/cluster/run_parallel.sh".*$', command, script,
                              flags=re.MULTILINE)
    if changed != 1:
        raise ValueError('Unexpected baseline launcher')
    for name in ('job-pro6000-pool-contacts-v5.sbatch', 'ood-pool-pro6000-v5-contacts.txt'):
        (ROOT / 'cluster' / name).write_text(script)
    print(f'{destination}\nSHA256 {checksum}')


if __name__ == '__main__':
    main()
