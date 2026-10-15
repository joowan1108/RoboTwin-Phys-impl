"""New OOD scripts override only the pool controller in the existing math ZIP."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

import pytest

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize('filename', ['ood-pool-pro6000x2-v5-refill.txt',
                                     'ood-fastwam-pro6000-v5-refill.txt'])
def test_refill_launcher_reuses_existing_math_patch(tmp_path, filename):
    data = tmp_path / 'data'
    (data / 'patches').mkdir(parents=True)
    archive_path = ROOT / 'reports/robotwin-expert-fix-v2.zip'
    shutil.copyfile(archive_path, data / 'patches' / archive_path.name)
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    record = tmp_path / 'record.json'
    for name, content in {
        'nvidia-smi': '#!/bin/sh\nexit 0\n',
        'apptainer': '''#!/usr/bin/env python3
import os, json
from pathlib import Path
Path(os.environ['TEST_RECORD']).write_text(json.dumps({'binds': os.environ['APPTAINER_BINDPATH']}))
''',
    }.items():
        path = bindir / name
        path.write_text(content)
        path.chmod(0o755)
    icd = tmp_path / 'nvidia_icd.json'
    icd.write_text('{}')
    source = (ROOT / 'cluster' / filename).read_text().replace('DATA="$HOME/robotwin-data"', f'DATA="{data}"')
    source = source.replace('/etc/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json', str(icd))
    script = tmp_path / 'job.sh'
    script.write_text(source)
    env = dict(os.environ, PATH=str(bindir) + ':' + os.environ['PATH'], SLURM_JOB_ID='refill-test',
               CUDA_VISIBLE_DEVICES='GPU-a,GPU-b', TEST_RECORD=str(record))
    result = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    override = data / 'reports/launchers/refill-test/reverify_pool.py'
    assert override.read_text() == (ROOT / 'tools/reverify_pool.py').read_text()
    binds = json.loads(record.read_text())['binds']
    assert f'{override}:/home/guest/robotwin-workspace/tools/reverify_pool.py:ro' in binds
    assert binds.index('/tools:/home/guest/robotwin-workspace/tools:ro') < binds.index(str(override))
    patch = data / 'patches/robotwin-expert-fix-v2'
    # Model the nested read-only file bind to validate the generated controller.
    shutil.copyfile(override, patch / 'tools/reverify_pool.py')
    launcher = (data / 'reports/launchers/refill-test/run.sh').read_text()
    validation = launcher.split("<<'PY_EXPERT_PATCH'\n")[1].split('\nPY_EXPERT_PATCH')[0]
    validation = validation.replace("Path('/home/guest/robotwin-workspace')", f"Path({str(patch)!r})")
    validation = validation.replace("root / 'vendor/RoboTwin/envs/utils/transforms.py'", "root / 'vendor/transforms.py'")
    validation = validation.replace("root / 'vendor/RoboTwin/envs/robot/planner.py'", "root / 'vendor/planner.py'")
    checked = subprocess.run(['python3', '-c', validation], capture_output=True, text=True)
    assert checked.returncode == 0, checked.stdout + checked.stderr
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read('robotwin-expert-fix-v2/manifest.json'))
        for relative, checksum in manifest['files'].items():
            assert hashlib.sha256((patch / 'src' / relative).read_bytes()).hexdigest() == checksum


def test_refill_copies_and_protected_execution_are_preserved():
    protected = (ROOT / 'cluster/default-protected.txt').read_text()
    for script, copy in [('job-pro6000-pool-refill-v5.sbatch', 'ood-pool-pro6000x2-v5-refill.txt'),
                         ('job-pro6000-fastwam-refill-v5.sbatch', 'ood-fastwam-pro6000-v5-refill.txt')]:
        text = (ROOT / 'cluster' / script).read_text()
        assert text == (ROOT / 'cluster' / copy).read_text() and text.endswith(protected)
        assert text.count('apptainer exec') == 1
        subprocess.run(['bash', '-n', str(ROOT / 'cluster' / script)], check=True)
