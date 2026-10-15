"""Validate the uploaded CuRobo repair snapshot and OOD source binds."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).parents[1]
NAME = 'robotwin-curobo-fix-v1'


@pytest.mark.parametrize('filename', ['ood-pool-pro6000x2-v5-curobo-fix.txt',
                                     'ood-fastwam-pro6000-v5-curobo-fix.txt',
                                     'ood-pi05-pro6000-v5-curobo-fix.txt',
                                     'ood-curobo-diagnose-pro6000-v5.txt'])
def test_curobo_launcher_extracts_and_validates_snapshot(tmp_path, filename):
    data = tmp_path / 'data'
    (data / 'patches').mkdir(parents=True)
    shutil.copyfile(ROOT / f'reports/{NAME}.zip', data / f'patches/{NAME}.zip')
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    record = tmp_path / 'record.json'
    for name, content in {'nvidia-smi': '#!/bin/sh\nexit 0\n', 'apptainer': '''#!/usr/bin/env python3
import os,json
from pathlib import Path
Path(os.environ['TEST_RECORD']).write_text(json.dumps({
 'experiment':os.environ['APPTAINERENV_ROBOTWIN_EXPERIMENT'], 'binds':os.environ['APPTAINER_BINDPATH']}))
'''}.items():
        path = bindir / name
        path.write_text(content)
        path.chmod(0o755)
    icd = tmp_path / 'nvidia_icd.json'
    icd.write_text('{}')
    text = (ROOT / 'cluster' / filename).read_text().replace('DATA="$HOME/robotwin-data"', f'DATA="{data}"')
    text = text.replace('/etc/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json', str(icd))
    script = tmp_path / 'job.sh'
    script.write_text(text)
    env = dict(os.environ, PATH=str(bindir) + ':' + os.environ['PATH'], SLURM_JOB_ID='curobo-test',
               CUDA_VISIBLE_DEVICES='GPU-a,GPU-b', TEST_RECORD=str(record))
    run = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True, timeout=15)
    assert run.returncode == 0, run.stdout + run.stderr
    metadata = json.loads(record.read_text())
    assert metadata['experiment'] == 'phys-pro6000-100-v4'
    patch = data / 'patches' / NAME
    launcher = (data / 'reports/launchers/curobo-test/run.sh').read_text()
    validation = launcher.split("<<'PY_EXPERT_PATCH'\n")[1].split('\nPY_EXPERT_PATCH')[0]
    validation = validation.replace("Path('/home/guest/robotwin-workspace')", f"Path({str(patch)!r})")
    validation = validation.replace("root / 'vendor/RoboTwin/envs/utils/transforms.py'", "root / 'vendor/transforms.py'")
    validation = validation.replace("root / 'vendor/RoboTwin/envs/robot/planner.py'", "root / 'vendor/planner.py'")
    verified = subprocess.run(['python3', '-c', validation], capture_output=True, text=True)
    assert verified.returncode == 0 and 'CUROBO_MASK_PATCH_SOURCE_PASS' in verified.stdout
    assert (patch / 'src/robotwin_phys/curobo_compat.py').exists()
    if 'diagnose' not in filename and 'pi05' not in filename:
        assert '--resume-from-experiment phys-pro6000-100-v3 --fallback-experiment phys-pro6000-100-v2' in launcher


def test_curobo_launchers_preserve_protected_execution():
    protected = (ROOT / 'cluster/default-protected.txt').read_text()
    for script, copy in [('job-pro6000-pool-curobo-v5.sbatch', 'ood-pool-pro6000x2-v5-curobo-fix.txt'),
                         ('job-pro6000-fastwam-curobo-v5.sbatch', 'ood-fastwam-pro6000-v5-curobo-fix.txt'),
                         ('job-pro6000-pi05-curobo-v5.sbatch', 'ood-pi05-pro6000-v5-curobo-fix.txt'),
                         ('job-pro6000-curobo-diagnose.sbatch', 'ood-curobo-diagnose-pro6000-v5.txt')]:
        text = (ROOT / 'cluster' / script).read_text()
        assert text == (ROOT / 'cluster' / copy).read_text() and text.endswith(protected)
        assert text.count('apptainer exec') == 1
        subprocess.run(['bash', '-n', str(ROOT / 'cluster' / script)], check=True)
