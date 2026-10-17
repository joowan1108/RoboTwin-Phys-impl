"""Run the unchanged template tail through a mock runtime and the real bundle controller."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import pytest

ROOT = Path(__file__).parents[1]
FAKE_PYTHON = '''#!/usr/bin/python3
import json, os, sys, time
from pathlib import Path
tool=Path(sys.argv[1]).name
args=sys.argv[2:]
def value(key): return args[args.index(key)+1]
record=dict(tool=tool,args=args,gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),pid=os.getpid(),runtime=os.environ.get('ROBOTWIN_PHYS_ROOT'),venv=os.environ.get('VIRTUAL_ENV'))
with open(os.environ['TEST_LOG'],'a') as f: f.write(json.dumps(record)+'\\n')
if tool=='serve_policy.py':
    while True: time.sleep(1)
if tool=='wait_policy.py':
    for _ in range(200):
        records=[json.loads(l) for l in open(os.environ['TEST_LOG'])]
        if any(r.get('tool')=='serve_policy.py' and value('--model') in r['args'] for r in records): break
        time.sleep(.01)
    else: sys.exit(90)
if tool=='experiment.py':
    if args[0]=='check-complete': sys.exit(0 if value('--model') in os.environ.get('COMPLETED','').split(',') else 1)
    if args[0]=='freeze' and os.environ.get('FAIL_FREEZE')=='1': sys.exit(23)
'''
FAKE_APPTAINER = '''#!/usr/bin/python3
import json,os,sys
args=sys.argv[1:]
assert args[:2]==['exec','--nv']
assert ':bundle-cu128' in args[2]
with open(os.environ['TEST_LOG'],'a') as f: f.write(json.dumps(dict(tool='apptainer',args=args))+'\\n')
env=os.environ.copy()
for key,value in os.environ.items():
    if key.startswith('APPTAINERENV_'): env[key[len('APPTAINERENV_'):]]=value
os.execvpe(args[3],args[3:],env)
'''


def executable(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text); path.chmod(0o755)


def run_bundle(tmp_path, template, **extra):
    simroot=tmp_path/'sim'; policies=tmp_path/'policies'; bindir=tmp_path/'bin'
    executable(simroot/'.env/bin/python',FAKE_PYTHON)
    for model in ['pi05','fact','motus','fastwam','g05']:
        executable(policies/model/'venv/bin/python',FAKE_PYTHON)
    executable(bindir/'apptainer',FAKE_APPTAINER)
    executable(bindir/'nvidia-smi','#!/bin/sh\nexit 0\n')
    text=(ROOT/'cluster'/template).read_text()
    text=text.replace('DATA="$HOME/robotwin-data"','DATA='+shlex.quote(str(tmp_path/'data')))
    text=text.replace('bash /home/guest/robotwin-workspace/cluster/run_bundle.sh','bash '+str(ROOT/'cluster/run_bundle.sh'))
    script=tmp_path/'job.sh';script.write_text(text)
    log=tmp_path/'calls.jsonl'
    env=dict(os.environ,PATH=str(bindir)+':'+os.environ['PATH'],TEST_LOG=str(log),
             SLURM_JOB_ID='test-bundle',CUDA_VISIBLE_DEVICES='GPU-A,GPU-B' if '2gpu' in template else 'GPU-A',
             APPTAINERENV_ROBOTWIN_BUNDLE_SIM_ROOT=str(simroot),APPTAINERENV_ROBOTWIN_BUNDLE_POLICY_ROOT=str(policies),**extra)
    result=subprocess.run(['bash',str(script)],env=env,text=True,capture_output=True,timeout=20)
    records=[json.loads(l) for l in log.read_text().splitlines()]
    return result,records


@pytest.mark.parametrize('template',['bundle-1gpu.sbatch','bundle-2gpu.sbatch'])
def test_protected_default_tail_and_cache_are_unchanged(template):
    protected=(ROOT/'cluster/default-protected.txt').read_text()
    assert (ROOT/'cluster'/template).read_text().endswith(protected)
    assert 'export APPTAINER_CACHEDIR="/tmp/apptainer-$(whoami)"\nmkdir -p "$APPTAINER_CACHEDIR"' in protected
    assert 'apptainer exec --nv "$IMAGE" $CMD' in protected


@pytest.mark.parametrize('template',['bundle-1gpu.sbatch','bundle-2gpu.sbatch'])
def test_one_apptainer_call_runs_all_models_in_isolated_environments(tmp_path,template):
    result,records=run_bundle(tmp_path,template)
    assert result.returncode==0,result.stdout+result.stderr
    assert sum(r['tool']=='apptainer' for r in records)==1
    phases=[r['args'][0] for r in records if r['tool']=='experiment.py']
    assert phases[:2]==['init','freeze'] and phases.count('evaluate')==5 and phases[-1]=='report'
    servers=[r for r in records if r['tool']=='serve_policy.py']
    assert [r['args'][1] for r in servers]==['pi05','fact','motus','fastwam','g05']
    for record in servers:
        model=record['args'][1]
        assert record['runtime']==str(tmp_path/'policies'/model/'workspace')
        assert record['venv']==str(tmp_path/'policies'/model/'venv')
        assert record['gpu']==('GPU-A' if template=='bundle-1gpu.sbatch' else ('GPU-B,GPU-A' if model=='motus' else 'GPU-B'))
        with pytest.raises(ProcessLookupError): os.kill(record['pid'],0)
    assert all(r['gpu']=='GPU-A' for r in records if r['tool']=='experiment.py')


def test_bundle_skips_completed_models(tmp_path):
    result,records=run_bundle(tmp_path,'bundle-1gpu.sbatch',COMPLETED='pi05,motus')
    assert result.returncode==0,result.stderr
    assert [r['args'][1] for r in records if r['tool']=='serve_policy.py']==['fact','fastwam','g05']


def test_pipeline_failure_propagates_through_unmodified_default_tail(tmp_path):
    result,records=run_bundle(tmp_path,'bundle-1gpu.sbatch',FAIL_FREEZE='1')
    assert result.returncode==23
    assert 'Job end:' not in result.stdout
    assert not any(r['tool']=='serve_policy.py' for r in records)
