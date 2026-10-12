"""Exercise real batch-shell orchestration with a fake Apptainer command, no GPUs."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import pytest

ROOT = Path(__file__).parents[1]
FAKE = '''#!/usr/bin/env python3
import json, os, sys, time
args = sys.argv[1:]
tool = next((a for a in args if a.startswith('tools/')), '')
rest = args[args.index(tool)+1:] if tool else []
record = dict(tool=tool, args=rest, gpu=os.environ.get('APPTAINERENV_CUDA_VISIBLE_DEVICES'))
with open(os.environ['TEST_LOG'], 'a') as f: f.write(json.dumps(record)+'\\n')
def value(name): return rest[rest.index(name)+1]
if tool.endswith('serve_policy.py'):
    while True: time.sleep(1)
if tool.endswith('wait_policy.py'):
    for _ in range(100):
        records = [json.loads(l) for l in open(os.environ['TEST_LOG'])]
        if any(r['tool'].endswith('serve_policy.py') and value('--model') in r['args'] for r in records): break
        time.sleep(.01)
    else: sys.exit(90)
if tool.endswith('experiment.py'):
    if rest[0] == 'check-complete': sys.exit(0 if value('--model') in os.environ.get('COMPLETED', '').split(',') else 1)
    if rest[0] == 'freeze' and os.environ.get('FAIL_FREEZE') == '1': sys.exit(23)
'''


def run_batch(tmp_path, template, **extra):
    bindir = tmp_path/'bin'; bindir.mkdir()
    apptainer = bindir/'apptainer'; apptainer.write_text(FAKE); apptainer.chmod(0o755)
    smi = bindir/'nvidia-smi'; smi.write_text('#!/bin/sh\nexit 0\n'); smi.chmod(0o755)
    source = (ROOT/'cluster/legacy'/template).read_text().replace('DATA="$HOME/robotwin-data"', 'DATA='+shlex.quote(str(tmp_path/'data')))
    script = tmp_path/'job.sh'; script.write_text(source)
    log = tmp_path/'commands.jsonl'
    env = dict(os.environ, PATH=str(bindir)+':'+os.environ['PATH'], TEST_LOG=str(log), SLURM_JOB_ID='test-job',
               CUDA_VISIBLE_DEVICES='GPU-A,GPU-B' if '5090' in template else 'GPU-A', **extra)
    run = subprocess.run(['bash', str(script)], env=env, text=True, capture_output=True, timeout=20)
    records = [json.loads(l) for l in log.read_text().splitlines()]
    return run, records


@pytest.mark.parametrize('template', ['job.sbatch', 'job-rtx5090x2.sbatch'])
def test_full_job_automatically_freezes_then_runs_all_models(tmp_path, template):
    run, records = run_batch(tmp_path, template)
    assert run.returncode == 0, run.stdout+run.stderr
    experiments = [r['args'][0] for r in records if r['tool'].endswith('experiment.py')]
    assert experiments[:2] == ['init', 'freeze']
    assert experiments.count('evaluate') == 5 and experiments[-1] == 'report'
    assert next(r for r in records if r['tool'].endswith('experiment.py') and r['args'][0] == 'init')['args'][-2:] == ['--episodes', '100']
    servers = [r for r in records if r['tool'].endswith('serve_policy.py')]
    assert [r['args'][1] for r in servers] == ['pi05', 'fact', 'motus', 'fastwam', 'g05']
    for server in servers:
        model = server['args'][1]
        expected = 'GPU-A' if template == 'job.sbatch' else ('GPU-B,GPU-A' if model == 'motus' else 'GPU-B')
        assert server['gpu'] == expected


def test_completed_models_are_not_downloaded_or_loaded_again(tmp_path):
    run, records = run_batch(tmp_path, 'job-rtx5090x2.sbatch', COMPLETED='pi05,motus')
    assert run.returncode == 0, run.stderr
    assert [r['args'][1] for r in records if r['tool'].endswith('serve_policy.py')] == ['fact', 'fastwam', 'g05']
    assert all(not ('--models' in r['args'] and r['args'][-1] in ['pi05', 'motus']) for r in records if r['tool'].endswith('prepare_cluster.py'))


def test_no_policy_is_run_if_shared_pool_freeze_fails(tmp_path):
    run, records = run_batch(tmp_path, 'job.sbatch', FAIL_FREEZE='1')
    assert run.returncode == 23
    assert not any(r['tool'].endswith('serve_policy.py') for r in records)
