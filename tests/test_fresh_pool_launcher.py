"""Exercise OOD ZIP extraction/source validation without an Apptainer server."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest

from robotwin_phys import curobo_compat

ROOT = Path(__file__).parents[1]
NAME = 'robotwin-fresh-pool-v1'
CURRENT = 'robotwin-fresh-pool-v2'
FRESH = {NAME: 'phys-pro6000-100-v5', CURRENT: 'phys-pro6000-100-v6'}
CONTEXT = 'robotwin-success-context-fix-v5'


@pytest.mark.parametrize('package,filename,marker,exec_prefix,gpus', [
    (NAME, 'ood-pool-pro6000-v5-fresh.txt', 'FRESH_POOL_PATCH_SOURCE_PASS', 'exec bash', 1),
    ('robotwin-empty-grasp-fix-v1', 'ood-pool-pro6000-v5-resume.txt',
     'EMPTY_GRASP_PATCH_SOURCE_PASS', 'exec "$ROOT/.env/bin/python"', 2),
    ('robotwin-grasp-contact-fix-v2', 'ood-pool-pro6000-v5-contacts.txt',
     'GRASP_CONTACT_PATCH_SOURCE_PASS', 'exec "$ROOT/.env/bin/python"', 2),
    ('robotwin-planner-failure-fix-v3', 'ood-pool-pro6000-v5-planfail.txt',
     'PLANNER_FAILURE_PATCH_SOURCE_PASS', 'exec "$ROOT/.env/bin/python"', 2),
    (CURRENT, 'ood-pool-pro6000-v6-fresh.txt', 'FRESH_POOL_V2_PATCH_SOURCE_PASS', 'exec bash', 2),
    (CURRENT, 'ood-fastwam-pro6000x2-v6-all.txt', 'FRESH_POOL_V2_PATCH_SOURCE_PASS', 'exec bash', 2),
    (CONTEXT, 'ood-fastwam-pro6000x2-v6-context.txt',
     'SUCCESS_CONTEXT_PATCH_SOURCE_PASS', 'exec "$ROOT/.env/bin/python"', 2),
])
def test_pool_launcher_extracts_snapshot_without_importing_older_experiments(
        tmp_path, package, filename, marker, exec_prefix, gpus):
    data = tmp_path / 'data'
    (data / 'patches').mkdir(parents=True)
    shutil.copyfile(ROOT / f'reports/{package}.zip', data / f'patches/{package}.zip')
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    record = tmp_path / 'record.json'
    for name, content in {'nvidia-smi': '#!/bin/sh\nexit 0\n', 'apptainer': '''#!/usr/bin/env python3
import os,json
from pathlib import Path
Path(os.environ['TEST_RECORD']).write_text(json.dumps({
 'experiment':os.environ['APPTAINERENV_ROBOTWIN_EXPERIMENT'],
 'phase':os.environ['APPTAINERENV_ROBOTWIN_PHASE'],
 'workers':os.environ['APPTAINERENV_ROBOTWIN_FREEZE_WORKERS_PER_GPU'],
 'binds':os.environ['APPTAINER_BINDPATH']}))
'''}.items():
        path = bindir / name
        path.write_text(content)
        path.chmod(0o755)
    # OIDN 2.3.3 is staged by the launcher from a pinned tarball; pre-stage it so the test never downloads.
    oidn = data / 'downloads/oidn-2.3.3/oidn_library'
    oidn.mkdir(parents=True)
    for name in ('libOpenImageDenoise.so.2.3.3', 'libOpenImageDenoise_core.so.2.3.3', 'libOpenImageDenoise_device_cuda.so.2.3.3'):
        (oidn / name).write_bytes(b'stub')
    icd = tmp_path / 'nvidia_icd.json'
    icd.write_text('{}')
    text = (ROOT / 'cluster' / filename).read_text()
    assert f'#SBATCH --gres=gpu:pro6000:{gpus}' in text
    assert '--resume-from-experiment' not in text and '--fallback-experiment' not in text
    text = text.replace('DATA="$HOME/robotwin-data"', f'DATA="{data}"')
    text = text.replace('/etc/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json', str(icd))
    script = tmp_path / 'job.sh'
    script.write_text(text)
    env = dict(os.environ, PATH=str(bindir) + ':' + os.environ['PATH'], SLURM_JOB_ID='fresh-test',
               CUDA_VISIBLE_DEVICES='GPU-a', TEST_RECORD=str(record))
    run = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True, timeout=15)
    assert run.returncode == 0, run.stdout + run.stderr
    metadata = json.loads(record.read_text())
    assert metadata['experiment'] == FRESH.get(package, 'phys-pro6000-100-v6' if package == CONTEXT else 'phys-pro6000-100-v5')
    assert metadata['phase'] == ('all' if ('all' in filename or package == CONTEXT) else 'freeze')
    assert metadata['workers'] == ('8' if package == CURRENT else '4')
    if 'all' in filename or package == CONTEXT:
        assert 'APPTAINERENV_ROBOTWIN_POLICY_REPLICAS=4' in text
    patch = data / 'patches' / package
    assert f'{patch}/src:/home/guest/robotwin-workspace/src:ro' in metadata['binds']
    assert f'{patch}/tools:/home/guest/robotwin-workspace/tools:ro' in metadata['binds']
    if package == CONTEXT:  # Fast-WAM profile (sigma_shift=5.0) reaches both the evaluator tree and the policy workspace
        for target in ('/home/guest/robotwin-workspace', '/opt/robotwin-policies/fastwam/workspace'):
            assert f'{patch}/configs/policies/fastwam.yaml:{target}/configs/policies/fastwam.yaml:ro' in metadata['binds']
        assert (patch / 'configs/policies/fastwam.yaml').read_bytes() == (ROOT / 'configs/policies/fastwam.yaml').read_bytes()
        sapien_pkg = '/home/guest/robotwin-workspace/.env/lib/python3.10/site-packages/sapien'
        assert f'{oidn}:{sapien_pkg}/oidn_library:ro' in metadata['binds']
        assert f'{patch}/oidn/_oidn_tricks.py:{sapien_pkg}/_oidn_tricks.py:ro' in metadata['binds']
        loader = (patch / 'oidn/_oidn_tricks.py').read_bytes()
        assert loader == (ROOT / 'docker/oidn/_oidn_tricks.py').read_bytes() and b'libOpenImageDenoise.so.2.3.3' in loader and b'.so.2.0.1' not in loader
        assert not (data / 'downloads/oidn-2.3.3/oidn.tar.gz').exists()  # pre-staged library: no download attempted
    launcher = (data / 'reports/launchers/fresh-test/run.sh').read_text()
    validation = launcher.split("<<'PY_EXPERT_PATCH'\n")[1].split('\nPY_EXPERT_PATCH')[0]
    validation = validation.replace("Path('/home/guest/robotwin-workspace')", f"Path({str(patch)!r})")
    validation = validation.replace("root / 'vendor/RoboTwin/envs/utils/transforms.py'", "root / 'vendor/transforms.py'")
    validation = validation.replace("root / 'vendor/RoboTwin/envs/robot/planner.py'", "root / 'vendor/planner.py'")
    validation = validation.replace("Path('/opt/robotwin-policies/fastwam/workspace/configs/policies/fastwam.yaml')",
                                    "root / 'configs/policies/fastwam.yaml'")
    validation = validation.replace("Path('/home/guest/robotwin-workspace/.env/lib/python3.10/site-packages/sapien')", "patch_sapien")
    validation = "from pathlib import Path\npatch_sapien = Path(" + repr(str(patch / 'sapien-stub')) + ")\n" + validation
    if package == CONTEXT:  # the in-container check expects the OIDN 2.3.3 files under the sapien package
        stub = patch / 'sapien-stub'
        (stub / 'oidn_library').mkdir(parents=True, exist_ok=True)
        for name in ('libOpenImageDenoise.so.2.3.3', 'libOpenImageDenoise_core.so.2.3.3', 'libOpenImageDenoise_device_cuda.so.2.3.3'):
            (stub / 'oidn_library' / name).write_bytes(b'stub')
        shutil.copyfile(patch / 'oidn/_oidn_tricks.py', stub / '_oidn_tricks.py')
    verified = subprocess.run(['python3', '-c', validation], capture_output=True, text=True)
    assert verified.returncode == 0 and marker in verified.stdout
    command = next(line for line in launcher.splitlines()
                   if line.startswith(exec_prefix) or (package == CONTEXT and 'continue_success_context.py' in line))
    assert '--seed 100000' in command and '--tasks' not in command
    assert 'reverify' not in command and '--resume-from-experiment' not in launcher
    assert '--fallback-experiment' not in launcher
    if package == CONTEXT:
        assert 'tools/continue_success_context.py' in command and '--replicas "$replicas"' in command
        assert 'ROBOTWIN_MODELS="fastwam g05 pi05"' in text and 'for model in $ROBOTWIN_MODELS' in launcher
        assert 'fact) replicas=1' in launcher and 'motus) replicas=2' in launcher
        assert 'tools/parallel_freeze_task.py' in launcher and 'PARALLEL_FREEZE_WORKERS=8' in text
        assert 'SUCCESS_CONTEXT_FIX' in launcher and 'FASTWAM_SIGMA_SHIFT' in verified.stdout
        assert '--timeout 3600' in launcher and 'PY_OIDN_PROBE' in launcher and 'OIDN_PASS' in launcher
        assert '#SBATCH --cpus-per-task=32' in text and '#SBATCH --mem=96G' in text and '--time=3-00:00:00' in text
    elif package not in FRESH:
        assert 'tools/resume_contact_pool.py' in command and '--phase freeze' in command
    if package == CURRENT:
        assert 'FRESH_POOL_V2: CoM offset on task objects only' in launcher
        assert ('POOL_THEN_FASTWAM' in launcher) == ('all' in filename)
        assert 'phys-pro6000-100-v5' not in text and NAME not in text
    # Validations must detect a stale/external extraction instead of silently using it.
    (patch / 'src/robotwin_phys/curobo_compat.py').write_text('stale snapshot')
    stale = subprocess.run(['python3', '-c', validation], capture_output=True, text=True)
    assert stale.returncode != 0 and 'source differs' in stale.stderr


def test_delivered_fresh_archive_remains_immutable_with_pinned_fix():
    # This delivered snapshot predates the empty-contact repair. Never silently
    # rebuild it to match current sources: running remote jobs mount this ZIP.
    assert hashlib.sha256((ROOT / f'reports/{NAME}.zip').read_bytes()).hexdigest() == (
        '4b2b19ccbc2581411019ac6c3c6e42c38434708d45140afcead3bd9116583ce6')
    with zipfile.ZipFile(ROOT / f'reports/{NAME}.zip') as archive:
        manifest = json.loads(archive.read(NAME + '/manifest.json'))
        assert manifest['curobo_fix'] == curobo_compat.FIX
        assert manifest['experiment'] == 'phys-pro6000-100-v5'
        for base, entries in (('src', manifest['files']), ('tools', manifest['tools'])):
            for relative, checksum in entries.items():
                payload = archive.read(f'{NAME}/{base}/{relative}')
                assert hashlib.sha256(payload).hexdigest() == checksum
        assert b'consistent-batch-results-v2' in archive.read(NAME + '/src/robotwin_phys/curobo_compat.py')


@pytest.mark.parametrize('sbatch,ood', [('job-pro6000-pool-fresh-v5.sbatch', 'ood-pool-pro6000-v5-fresh.txt'),
                                        ('job-pro6000-pool-resume-v5.sbatch', 'ood-pool-pro6000-v5-resume.txt'),
                                        ('job-pro6000-pool-contacts-v5.sbatch', 'ood-pool-pro6000-v5-contacts.txt'),
                                        ('job-pro6000-pool-planfail-v5.sbatch', 'ood-pool-pro6000-v5-planfail.txt'),
                                        ('job-pro6000-pool-fresh-v6.sbatch', 'ood-pool-pro6000-v6-fresh.txt'),
                                        ('job-pro6000x2-fastwam-v6-all.sbatch', 'ood-fastwam-pro6000x2-v6-all.txt'),
                                        ('job-pro6000x2-fastwam-v6-context.sbatch', 'ood-fastwam-pro6000x2-v6-context.txt')])
def test_pool_launcher_retains_protected_execution_and_has_valid_bash(sbatch, ood):
    path = ROOT / 'cluster' / sbatch
    text = path.read_text()
    assert text == (ROOT / 'cluster' / ood).read_text()
    assert text.endswith((ROOT / 'cluster/default-protected.txt').read_text())
    assert text.count('apptainer exec') == 1
    subprocess.run(['bash', '-n', str(path)], check=True)


DELIVERED = {'robotwin-empty-grasp-fix-v1': '1000849ae1dd224e72c38ae20b69040596c59d1c526eee4fc22bddccb4352bb8',
             'robotwin-grasp-contact-fix-v2': '1540f192866733ac05e8e1efa12a732747315abf16fa09c550353223711237cb',
             'robotwin-planner-failure-fix-v3': '4acb0d8aa351e1a91cc062bec742789faddca610be17f24c84c312eca2289791'}


@pytest.mark.parametrize('name', list(DELIVERED))
def test_resume_archive_changes_only_reviewed_branches_and_adds_audited_continuation_tool(name):
    with zipfile.ZipFile(ROOT / f'reports/{NAME}.zip') as before, \
            zipfile.ZipFile(ROOT / f'reports/{name}.zip') as after:
        old = {p.removeprefix(NAME + '/'): before.read(p) for p in before.namelist()}
        new = {p.removeprefix(name + '/'): after.read(p) for p in after.namelist()}
    assert set(new) - set(old) == {'tools/resume_contact_pool.py'}
    assert not set(old) - set(new)
    assert {p for p in old if old[p] != new[p]} == {'manifest.json', 'src/robotwin_phys/planning.py'}
    # Delivered snapshots are mounted by remote jobs; never rebuilt from the tree.
    assert hashlib.sha256((ROOT / 'reports' / (name + '.zip')).read_bytes()).hexdigest() == DELIVERED[name]
    if name == 'robotwin-planner-failure-fix-v3':
        # The v6 fresh snapshot carries the same expert repairs forward unchanged.
        for p in ('src/robotwin_phys/planning.py', 'tools/resume_contact_pool.py'):
            assert new[p] == (ROOT / p).read_bytes()
    manifest = json.loads(new['manifest.json'])
    assert manifest['planning_after'] == hashlib.sha256(new['src/robotwin_phys/planning.py']).hexdigest()
    for base, entries in (('src', manifest['files']), ('tools', manifest['tools'])):
        for path, checksum in entries.items():
            assert hashlib.sha256(new[base + '/' + path]).hexdigest() == checksum


DELIVERED_FRESH_V2 = '8e53a6f9f32591389c3026e7a33905d8c3452ddb555f038d6a6f9492be0ee594'
DELIVERED_CONTEXT_V4 = '45c83feaa4882884a83fbdac3ffe9ae9b9fc875f4cc607fa6c0a0f85153d0416'
DELIVERED_CONTEXT_V3 = '88c7734fbf4d4982721a9167d773a2d603b166dcdeb652c9d1960e77eac3d3ce'
DELIVERED_CONTEXT_V2 = 'e5a80ed1d5bd4d64b7f6b47f8ae5dc0fb33f7c7d52d38c92b9b626640b24a35d'
DELIVERED_CONTEXT_V1 = 'b88a9ba84d646c51ad5b5f91cbcea975d2b26e6b9ed145d501ada717f89203b9'


def test_delivered_context_archives_are_pinned_and_differ_only_in_tools():
    # Job 1682 mounted v1; job 1684 mounted v2 (adds the parallel seed scanner); v3 only spreads
    # policy replicas over both GPUs in tools/parallel_eval.py after the Fast-WAM OOM of job 1684.
    assert hashlib.sha256((ROOT / 'reports/robotwin-success-context-fix-v1.zip').read_bytes()).hexdigest() == DELIVERED_CONTEXT_V1
    assert hashlib.sha256((ROOT / 'reports/robotwin-success-context-fix-v2.zip').read_bytes()).hexdigest() == DELIVERED_CONTEXT_V2
    assert hashlib.sha256((ROOT / 'reports/robotwin-success-context-fix-v3.zip').read_bytes()).hexdigest() == DELIVERED_CONTEXT_V3
    assert hashlib.sha256((ROOT / 'reports/robotwin-success-context-fix-v4.zip').read_bytes()).hexdigest() == DELIVERED_CONTEXT_V4
    with zipfile.ZipFile(ROOT / 'reports/robotwin-success-context-fix-v1.zip') as v1, \
            zipfile.ZipFile(ROOT / 'reports/robotwin-success-context-fix-v2.zip') as v2, \
            zipfile.ZipFile(ROOT / 'reports/robotwin-success-context-fix-v3.zip') as v3, \
            zipfile.ZipFile(ROOT / 'reports/robotwin-success-context-fix-v4.zip') as v4, \
            zipfile.ZipFile(ROOT / f'reports/{CONTEXT}.zip') as v5:
        m1 = json.loads(v1.read('robotwin-success-context-fix-v1/manifest.json'))
        m2 = json.loads(v2.read('robotwin-success-context-fix-v2/manifest.json'))
        m3 = json.loads(v3.read('robotwin-success-context-fix-v3/manifest.json'))
        m4 = json.loads(v4.read('robotwin-success-context-fix-v4/manifest.json'))
        m5 = json.loads(v5.read(CONTEXT + '/manifest.json'))
        assert m1['files'] == m2['files'] == m3['files'] == m4['files'] == m5['files']
        # v5: only the packager changed in tools; configs identical to v4; the OIDN 2.3.3 loader is shipped under oidn/.
        assert {t for t in m4['tools'] if m4['tools'][t] != m5['tools'][t]} == {'package_success_context_fix.py'}
        assert m5['configs'] == m4['configs'] and 'oidn' not in m4
        assert m5['oidn']['version'] == '2.3.3' and m5['oidn']['tarball_sha256'] == '3c385230d9e6f63527ba72f2229594dbac5051674219d72e0044b5d0b841796f'
        assert hashlib.sha256(v5.read(CONTEXT + '/oidn/_oidn_tricks.py')).hexdigest() == m5['oidn']['loader_sha256'] \
            == hashlib.sha256((ROOT / 'docker/oidn/_oidn_tricks.py').read_bytes()).hexdigest()
        assert m5['episode_timeout_seconds'] == 3600
        assert set(m2['tools']) - set(m1['tools']) == {'parallel_freeze_task.py'}
        assert {t for t in m1['tools'] if m1['tools'][t] != m2['tools'].get(t)} == {'package_success_context_fix.py'}
        assert set(m3['tools']) == set(m2['tools'])
        assert {t for t in m2['tools'] if m2['tools'][t] != m3['tools'][t]} == {'parallel_eval.py', 'package_success_context_fix.py'}
        # v4: only the packager changed in tools; the Fast-WAM profile (sigma_shift=5.0) is shipped under configs/.
        assert {t for t in m3['tools'] if m3['tools'][t] != m4['tools'][t]} == {'package_success_context_fix.py'}
        assert 'configs' not in m3 and list(m4['configs']) == ['policies/fastwam.yaml']
        assert hashlib.sha256(v4.read('robotwin-success-context-fix-v4/configs/policies/fastwam.yaml')).hexdigest() == m4['configs']['policies/fastwam.yaml'] \
            == hashlib.sha256((ROOT / 'configs/policies/fastwam.yaml').read_bytes()).hexdigest()
        assert b'sigma_shift: 5.0' in v5.read(CONTEXT + '/configs/policies/fastwam.yaml')


def test_delivered_v6_archive_is_pinned_and_declares_clutter_scope():
    # Job 1681 mounts this snapshot; it is never rebuilt from the moved tree.
    assert hashlib.sha256((ROOT / f'reports/{CURRENT}.zip').read_bytes()).hexdigest() == DELIVERED_FRESH_V2
    with zipfile.ZipFile(ROOT / f'reports/{CURRENT}.zip') as archive:
        manifest = json.loads(archive.read(CURRENT + '/manifest.json'))
        assert manifest['experiment'] == 'phys-pro6000-100-v6'
        assert manifest['curobo_fix'] == curobo_compat.FIX
        assert 'task objects only' in manifest['com_offset_scope'] and 'no policy-stage force' in manifest['clutter_scope']
        assert manifest['includes_expert_repairs'][-1] == 'robotwin-planner-failure-fix-v3'
        for base, entries in (('src', manifest['files']), ('tools', manifest['tools'])):
            for relative, checksum in entries.items():
                assert hashlib.sha256(archive.read(f'{CURRENT}/{base}/{relative}')).hexdigest() == checksum
        simulator = archive.read(CURRENT + '/src/robotwin_phys/simulator.py').decode()
        assert '_clutter_entity_ids(' in simulator and 'clutter=True' in simulator
        # Only the reviewed evaluation-path files moved since this delivery.
        tree = {p: hashlib.sha256((ROOT / 'src' / p).read_bytes()).hexdigest() for p in manifest['files']}
        assert {p for p in manifest['files'] if tree[p] != manifest['files'][p]} == {'robotwin_phys/worker.py'}
        with zipfile.ZipFile(ROOT / f'reports/{NAME}.zip') as previous:
            for name in ('transforms.py', 'planner.py', 'ROBOTWIN_LICENSE'):
                assert archive.read(f'{CURRENT}/vendor/{name}') == previous.read(f'{NAME}/vendor/{name}')


def test_package_generation_is_reproducible_without_changing_old_deliveries():
    archives = list((ROOT / 'reports').glob('robotwin-*.zip'))
    checksums = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in archives}
    for packager in ('tools/package_success_context_fix.py', 'tools/package_planner_failure_fix.py'):
        # The current packager rebuilds its archive byte for byte; v3 grafts only its reviewed
        # planning branch onto the pinned v1 baseline, so it stays reproducible as well.
        subprocess.run([sys.executable, str(ROOT / packager)], check=True, capture_output=True)
        assert checksums == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in archives}
    for packager, message in (('tools/package_grasp_contact_fix.py', 'Unreviewed planning change'),
                              ('tools/package_fresh_pool_v2.py', 'Do not overwrite a delivered snapshot')):
        stale = subprocess.run([sys.executable, str(ROOT / packager)], capture_output=True, text=True)
        assert stale.returncode != 0 and message in stale.stderr
    assert checksums == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in archives}
