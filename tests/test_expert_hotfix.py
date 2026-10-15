"""Run the OOD host script to verify extraction, binds and snapshot validation."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).parents[1]
ARCHIVE = ROOT / "reports/robotwin-expert-fix-v1.zip"


@pytest.mark.parametrize("name", ["job-pro6000-expert-fix.sbatch", "job-pro6000-pi05-expert-fix.sbatch",
                                  "job-pro6000-expert-diagnose.sbatch", "job-pro6000-pool-parallel-v5.sbatch"])
def test_ood_patch_is_extracted_and_bound_without_overwriting_v1(tmp_path, name):
    if not ARCHIVE.exists():
        pytest.skip("Run tools/package_expert_fix.py first")
    data = tmp_path / "data"
    (data / "patches").mkdir(parents=True)
    shutil.copyfile(ARCHIVE, data / "patches" / ARCHIVE.name)
    (data / "manifests/phys-pro6000-100-v2").mkdir(parents=True)
    (data / "manifests/phys-pro6000-100-v2/pool-index.json").write_text('{}')
    old = data / "manifests/phys-pro6000-100-v1/adjust_bottle.progress.json"
    old.parent.mkdir()
    old.write_text('original progress')
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for tool in ("nvidia-smi",):
        path = bindir / tool
        path.write_text('#!/bin/sh\nexit 0\n')
        path.chmod(0o755)
    apptainer = bindir / "apptainer"
    record = tmp_path / "apptainer.json"
    apptainer.write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
Path(os.environ['TEST_RECORD']).write_text(json.dumps({
    'args': sys.argv[1:], 'binds': os.environ['APPTAINER_BINDPATH'],
    'experiment': os.environ['APPTAINERENV_ROBOTWIN_EXPERIMENT']}))
''')
    apptainer.chmod(0o755)
    source = (ROOT / "cluster" / name).read_text().replace('DATA="$HOME/robotwin-data"',
                                                             f'DATA="{data}"')
    # No NVIDIA ICD on the test machine; use a test manifest to reach Apptainer.
    icd = tmp_path / "nvidia_icd.json"
    icd.write_text('{}')
    source = source.replace('/etc/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json', str(icd))
    script = tmp_path / "job.sh"
    script.write_text(source)
    env = dict(os.environ, PATH=str(bindir) + ':' + os.environ['PATH'], SLURM_JOB_ID='hotfix-test',
               CUDA_VISIBLE_DEVICES='GPU-test', TEST_RECORD=str(record))
    for _ in range(2):
        result = subprocess.run(['bash', str(script)], env=env, text=True, capture_output=True, timeout=15)
        assert result.returncode == 0, result.stdout + result.stderr
    command = json.loads(record.read_text())
    assert command['experiment'] == 'phys-pro6000-100-v2'
    assert '/src:/home/guest/robotwin-workspace/src:ro' in command['binds']
    assert command['args'][:3] == ['exec', '--nv', 'docker://joowan1108/robotwin-phys:bundle-cu128-v5']
    assert old.read_text() == 'original progress'
    if name == 'job-pro6000-pool-parallel-v5.sbatch':
        scheduler = data / 'reports/launchers/hotfix-test/parallel_eval.py'
        assert scheduler.read_text() == (ROOT / 'tools/parallel_eval.py').read_text()
        assert f'{scheduler}:/home/guest/robotwin-workspace/tools/parallel_eval.py:ro' in command['binds']
        assert '--freeze-workers-per-gpu "$ROBOTWIN_FREEZE_WORKERS_PER_GPU"' in (
            data / 'reports/launchers/hotfix-test/run.sh').read_text()
    patch = data / 'patches/robotwin-expert-fix-v1'
    manifest = json.loads((patch / 'manifest.json').read_text())
    for relative, expected in manifest['files'].items():
        assert hashlib.sha256((patch / 'src' / relative).read_bytes()).hexdigest() == expected
    # Execute the generated source verification against the extracted snapshot.
    launcher = (data / 'reports/launchers/hotfix-test/run.sh').read_text()
    validation = launcher.split("<<'PY_EXPERT_PATCH'\n")[1].split('\nPY_EXPERT_PATCH')[0]
    validation = validation.replace("Path('/home/guest/robotwin-workspace/src')", f"Path({str(patch / 'src')!r})")
    verified = subprocess.run(['python3', '-c', validation], text=True, capture_output=True)
    assert verified.returncode == 0 and 'EXPERT_PLANNING_PATCH_PASS' in verified.stdout
    target = patch / 'src/robotwin_phys/planning.py'
    target.write_text(target.read_text() + '\n# changed\n')
    assert subprocess.run(['python3', '-c', validation], capture_output=True).returncode != 0


def test_ood_copies_and_protected_tail_are_preserved():
    pairs = [('job-pro6000-expert-fix.sbatch', 'ood-fastwam-pro6000-v5-expert-fix.txt'),
             ('job-pro6000-pi05-expert-fix.sbatch', 'ood-pi05-pro6000-v5-expert-fix.txt'),
             ('job-pro6000-expert-diagnose.sbatch', 'ood-expert-diagnose-pro6000-v5.txt'),
             ('job-pro6000-pool-parallel-v5.sbatch', 'ood-pool-pro6000x2-v5.txt')]
    marker = '# ---- 캐시를 노드 로컬로'
    original = (ROOT / 'cluster/job-pro6000-single.sbatch').read_text().split(marker)[1]
    for script, copy in pairs:
        text = (ROOT / 'cluster' / script).read_text()
        assert text == (ROOT / 'cluster' / copy).read_text()
        assert text.split(marker)[1] == original
