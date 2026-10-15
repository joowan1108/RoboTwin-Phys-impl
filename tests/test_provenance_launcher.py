"""Check the controller file override and read-only diagnosis launcher."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize('filename', ['ood-pool-pro6000x2-v5-provenance-fix.txt',
                                     'ood-provenance-diagnose-pro6000-v5.txt'])
def test_provenance_launcher_reuses_curobo_zip_and_overrides_existing_controller(tmp_path, filename):
    data = tmp_path / 'data'
    (data / 'patches').mkdir(parents=True)
    shutil.copyfile(ROOT / 'reports/robotwin-curobo-fix-v1.zip', data / 'patches/robotwin-curobo-fix-v1.zip')
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    record = tmp_path / 'record.json'
    for name, text in {'nvidia-smi': '#!/bin/sh\nexit 0\n', 'apptainer': '''#!/usr/bin/env python3
import json,os
from pathlib import Path
Path(os.environ['TEST_RECORD']).write_text(json.dumps({'binds':os.environ['APPTAINER_BINDPATH']}))
'''}.items():
        path = bindir / name
        path.write_text(text)
        path.chmod(0o755)
    icd = tmp_path / 'nvidia_icd.json'
    icd.write_text('{}')
    text = (ROOT / 'cluster' / filename).read_text().replace('DATA="$HOME/robotwin-data"', f'DATA="{data}"')
    text = text.replace('/etc/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json', str(icd))
    script = tmp_path / 'job.sh'
    script.write_text(text)
    env = dict(os.environ, PATH=str(bindir) + ':' + os.environ['PATH'], SLURM_JOB_ID='source-test',
               CUDA_VISIBLE_DEVICES='GPU-a', TEST_RECORD=str(record))
    run = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True, timeout=15)
    assert run.returncode == 0, run.stdout + run.stderr
    patch = data / 'patches/robotwin-curobo-fix-v1'
    controller = data / 'reports/launchers/source-test/reverify_pool.py'
    assert controller.read_text() == (ROOT / 'tools/reverify_pool.py').read_text()
    assert f'{controller}:/home/guest/robotwin-workspace/tools/reverify_pool.py:ro' in json.loads(record.read_text())['binds']
    shutil.copyfile(controller, patch / 'tools/reverify_pool.py')
    launcher = (data / 'reports/launchers/source-test/run.sh').read_text()
    validation = launcher.split("<<'PY_EXPERT_PATCH'\n")[1].split('\nPY_EXPERT_PATCH')[0]
    validation = validation.replace("Path('/home/guest/robotwin-workspace')", f"Path({str(patch)!r})")
    validation = validation.replace("root / 'vendor/RoboTwin/envs/utils/transforms.py'", "root / 'vendor/transforms.py'")
    validation = validation.replace("root / 'vendor/RoboTwin/envs/robot/planner.py'", "root / 'vendor/planner.py'")
    verified = subprocess.run(['python3', '-c', validation], capture_output=True, text=True)
    assert verified.returncode == 0, verified.stdout + verified.stderr
    if 'diagnose' in filename:
        helper = data / 'reports/launchers/source-test/diagnose_provenance.py'
        assert helper.read_text() == (ROOT / 'tools/diagnose_provenance.py').read_text()
        assert 'download_assets.py' not in launcher and 'run_parallel.sh' not in launcher
        output = data / 'reports/provenance-diagnostics/source-test/summary.json'
        diagnosed = subprocess.run(['python3', str(helper), '--root', str(patch), '--output', str(output)],
                                   capture_output=True, text=True)
        assert diagnosed.returncode == 0 and output.exists()


def test_provenance_script_copies_and_protected_tail():
    protected = (ROOT / 'cluster/default-protected.txt').read_text()
    for script, copy in [('job-pro6000-pool-provenance-v5.sbatch', 'ood-pool-pro6000x2-v5-provenance-fix.txt'),
                         ('job-pro6000-fastwam-provenance-v5.sbatch', 'ood-fastwam-pro6000-v5-provenance-fix.txt'),
                         ('job-pro6000-provenance-diagnose.sbatch', 'ood-provenance-diagnose-pro6000-v5.txt')]:
        text = (ROOT / 'cluster' / script).read_text()
        assert text == (ROOT / 'cluster' / copy).read_text() and text.endswith(protected)
        assert text.count('apptainer exec') == 1
        subprocess.run(['bash', '-n', str(ROOT / 'cluster' / script)], check=True)
