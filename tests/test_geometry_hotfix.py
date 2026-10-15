"""Exercise the OOD archive, transforms bind and snapshot verification."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).parents[1]
NAME = 'robotwin-expert-fix-v2'


@pytest.mark.parametrize('filename', ['ood-pool-pro6000x2-v5-geometry-fix.txt',
                                     'ood-fastwam-pro6000-v5-geometry-fix.txt',
                                     'ood-pi05-pro6000-v5-geometry-fix.txt',
                                     'ood-geometry-diagnose-pro6000-v5.txt'])
def test_ood_geometry_archive_is_bound_and_validated(tmp_path, filename):
    data = tmp_path / 'data'
    (data / 'patches').mkdir(parents=True)
    shutil.copyfile(ROOT / f'reports/{NAME}.zip', data / f'patches/{NAME}.zip')
    bin_path = tmp_path / 'bin'
    bin_path.mkdir()
    record = tmp_path / 'apptainer.json'
    for name, content in {
        'nvidia-smi': '#!/bin/sh\nexit 0\n',
        'apptainer': '''#!/usr/bin/env python3
import os, json
from pathlib import Path
Path(os.environ['TEST_RECORD']).write_text(json.dumps({
    'binds': os.environ['APPTAINER_BINDPATH'],
    'experiment': os.environ['APPTAINERENV_ROBOTWIN_EXPERIMENT']}))
''',
    }.items():
        path = bin_path / name
        path.write_text(content)
        path.chmod(0o755)
    icd = tmp_path / 'nvidia_icd.json'
    icd.write_text('{}')
    text = (ROOT / 'cluster' / filename).read_text().replace('DATA="$HOME/robotwin-data"', f'DATA="{data}"')
    text = text.replace('/etc/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json', str(icd))
    script = tmp_path / 'job.sh'
    script.write_text(text)
    env = dict(os.environ, PATH=str(bin_path) + ':' + os.environ['PATH'], SLURM_JOB_ID='math-test',
               CUDA_VISIBLE_DEVICES='GPU-a,GPU-b', TEST_RECORD=str(record))
    for _ in range(2):
        result = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True, timeout=15)
        assert result.returncode == 0, result.stdout + result.stderr
    command = json.loads(record.read_text())
    assert command['experiment'] == 'phys-pro6000-100-v3'
    assert '/tools:/home/guest/robotwin-workspace/tools:ro' in command['binds']
    assert '/vendor/transforms.py:/home/guest/robotwin-workspace/vendor/RoboTwin/envs/utils/transforms.py:ro' in command['binds']
    assert '/vendor/planner.py:/home/guest/robotwin-workspace/vendor/RoboTwin/envs/robot/planner.py:ro' in command['binds']
    patch = data / 'patches' / NAME
    launcher = (data / 'reports/launchers/math-test/run.sh').read_text()
    validation = launcher.split("<<'PY_EXPERT_PATCH'\n")[1].split('\nPY_EXPERT_PATCH')[0]
    validation = validation.replace("Path('/home/guest/robotwin-workspace')", f"Path({str(patch)!r})")
    validation = validation.replace("root / 'vendor/RoboTwin/envs/utils/transforms.py'", "root / 'vendor/transforms.py'")
    validation = validation.replace("root / 'vendor/RoboTwin/envs/robot/planner.py'", "root / 'vendor/planner.py'")
    check = subprocess.run(['python3', '-c', validation], capture_output=True, text=True)
    assert check.returncode == 0 and 'EXPERT_GEOMETRY_PATCH_PASS' in check.stdout
    target = patch / 'vendor/transforms.py'
    target.write_text(target.read_text() + '\n# modified\n')
    assert subprocess.run(['python3', '-c', validation], capture_output=True).returncode != 0


def test_geometry_script_copies_preserve_protected_section_and_old_patch():
    pairs = [('job-pro6000-pool-geometry-v5.sbatch', 'ood-pool-pro6000x2-v5-geometry-fix.txt'),
             ('job-pro6000-fastwam-geometry-v5.sbatch', 'ood-fastwam-pro6000-v5-geometry-fix.txt'),
             ('job-pro6000-pi05-geometry-v5.sbatch', 'ood-pi05-pro6000-v5-geometry-fix.txt'),
             ('job-pro6000-geometry-diagnose.sbatch', 'ood-geometry-diagnose-pro6000-v5.txt')]
    protected = (ROOT / 'cluster/default-protected.txt').read_text()
    for script, copy in pairs:
        text = (ROOT / 'cluster' / script).read_text()
        assert text == (ROOT / 'cluster' / copy).read_text() and text.endswith(protected)
        assert text.count('apptainer exec') == 1
        subprocess.run(['bash', '-n', str(ROOT / 'cluster' / script)], check=True)
    # Prior package remains available for provenance comparison and rollback.
    assert (ROOT / 'reports/robotwin-expert-fix-v1.zip').exists()
    assert (ROOT / 'tools/expert_fix_previous.json').exists()
