"""Fairness, resume and concurrency checks without loading learned checkpoints."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import socket
import sys
import threading

import pytest

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location('parallel_eval', ROOT / 'tools/parallel_eval.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
from robotwin_phys.conditions import digest, sample
from robotwin_phys.manifest import atomic_json, seal


@pytest.fixture
def pool(tmp_path, monkeypatch):
    monkeypatch.setattr(p, 'ROOT', tmp_path)
    monkeypatch.setattr(p.experiment, 'ROOT', tmp_path)
    tasks = ['grab_roller', 'stack_blocks_two']
    suite = seal(dict(schema='robotwin-experiment-v1', name='shared', tasks=tasks,
        protocols=['physical_random'], models=list(p.experiment.MODELS), episodes=3,
        seed=100000, policy_seed=0, instruction_type='unseen'))
    folder = tmp_path / 'runs/shared'
    atomic_json(folder / 'suite.json', suite)
    for task in tasks:
        episodes = []
        for i in range(3):
            condition = sample(task, 100000 + i, 'physical_random')
            initial = dict(physical={'mass': condition['mass_scale']}, pose=[i, 0, 0])
            episodes.append(dict(episode_index=i, seed=100000+i, condition=condition,
                condition_sha256=digest(condition), expert_success=True, initial_state=initial,
                initial_state_sha256=digest(initial), instruction=f'Perform {task}'))
        manifest = seal(dict(schema='robotwin-phys-local-v1', status='verified', task=task,
            protocol='physical_random', start_seed=100000, requested_episodes=3,
            instruction_type='unseen', only=[], without=[], provenance={'test': True}, episodes=episodes))
        atomic_json(p.experiment.manifest_path(suite, 'physical_random', task), manifest)
    index = tmp_path / 'manifests/shared/pool-index.json'
    manifests = p.seal_pool(suite, index)
    return folder, suite, index, manifests


@pytest.mark.parametrize('sim_model', ['5060', '5090'])
def test_device_assignment_uses_allocation_not_host_order(sim_model):
    devices = {'GPU-pro': {'name': 'NVIDIA RTX PRO 6000 Blackwell'},
               'GPU-sim2': {'name': f'NVIDIA GeForce RTX {sim_model}'},
               'GPU-sim1': {'name': f'NVIDIA GeForce RTX {sim_model}'}}
    assert p.select_gpus(devices) == (['GPU-sim2', 'GPU-sim1'], 'GPU-pro')
    with pytest.raises(ValueError, match='outside'):
        p.select_gpus(devices, ['GPU-sim1', 'host-0'], 'GPU-pro')
    with pytest.raises(ValueError, match='distinct'):
        p.select_gpus(devices, ['GPU-sim1', 'GPU-sim1'], 'GPU-pro')
    # Two allocated GPUs: simulators on both, replicas on the first listed.
    two = {key: value for key, value in devices.items() if key != 'GPU-pro'}
    assert p.select_gpus(two) == (['GPU-sim2', 'GPU-sim1'], 'GPU-sim2')
    with pytest.raises(ValueError, match='one, two or three'):
        p.select_gpus({})


def test_two_pro6000_dual_layout_and_replica_defaults():
    devices = {'GPU-a': {'name': 'NVIDIA RTX PRO 6000 Blackwell'}, 'GPU-b': {'name': 'NVIDIA RTX PRO 6000 Blackwell'}}
    assert p.select_gpus(devices) == (['GPU-a', 'GPU-b'], 'GPU-a')
    assert p.select_gpus(devices, ['GPU-b', 'GPU-a'], 'GPU-b') == (['GPU-b', 'GPU-a'], 'GPU-b')
    with pytest.raises(ValueError, match='simulators on both'):
        p.select_gpus(devices, ['GPU-a'], 'GPU-a')
    with pytest.raises(ValueError, match='outside'):
        p.select_gpus(devices, ['GPU-a', 'GPU-b'], 'GPU-c')
    assert p.gpu_layout(['GPU-a', 'GPU-b'], 'GPU-a') == 'dual'
    assert p.gpu_layout(['GPU-a'], 'GPU-a') == 'shared'
    assert p.gpu_layout(['GPU-a', 'GPU-b'], 'GPU-c') == 'split'
    # Dual spreads replicas over both GPUs starting at the policy GPU; shared/split keep them on one.
    assert p.replica_gpus(['GPU-a', 'GPU-b'], 'GPU-a', 4) == ['GPU-a', 'GPU-b', 'GPU-a', 'GPU-b']
    assert p.replica_gpus(['GPU-a', 'GPU-b'], 'GPU-b', 3) == ['GPU-b', 'GPU-a', 'GPU-b']
    assert p.replica_gpus(['GPU-a', 'GPU-b'], 'GPU-c', 2) == ['GPU-c', 'GPU-c']
    assert p.replica_gpus(['GPU-a'], 'GPU-a', 1) == ['GPU-a']
    ns = lambda model, replicas: argparse.Namespace(model=model, replicas=replicas)
    assert p.policy_replica_count(ns('fastwam', None), ['GPU-a', 'GPU-b'], 'GPU-a') == 4
    assert p.policy_replica_count(ns('fastwam', 6), ['GPU-a', 'GPU-b'], 'GPU-a') == 6
    assert p.policy_replica_count(ns('fact', None), ['GPU-a', 'GPU-b'], 'GPU-a') == 1
    assert p.policy_replica_count(ns('fastwam', None), ['GPU-a', 'GPU-b'], 'GPU-c') == 2
    assert p.policy_replica_count(ns('fastwam', None), ['GPU-a'], 'GPU-a') == 1
    with pytest.raises(ValueError, match='between 1 and 8'):
        p.policy_replica_count(ns('fastwam', 9), ['GPU-a', 'GPU-b'], 'GPU-a')
    with pytest.raises(ValueError, match='at most two'):
        p.policy_replica_count(ns('fastwam', 3), ['GPU-a', 'GPU-b'], 'GPU-c')
    with pytest.raises(ValueError, match='FACT'):
        p.policy_replica_count(ns('fact', 2), ['GPU-a', 'GPU-b'], 'GPU-a')


@pytest.mark.parametrize('gpu', ['3', 'GPU-allocated-pro'])
def test_single_gpu_discovery_and_selection_preserve_slurm_allocation(monkeypatch, gpu):
    observed = []
    monkeypatch.setenv('SLURM_JOB_ID', 'test-shared-gpu')

    def probe(argv, env, text):
        observed.append(env['CUDA_VISIBLE_DEVICES'])
        return json.dumps(dict(name='NVIDIA RTX PRO 6000 Blackwell', memory_gib=96))

    monkeypatch.setattr(p.subprocess, 'check_output', probe)
    devices = p.discover_gpus({'CUDA_VISIBLE_DEVICES': gpu})
    assert observed == [gpu]
    assert p.select_gpus(devices) == ([gpu], gpu)
    assert p.select_gpus(devices, [gpu], gpu) == ([gpu], gpu)
    with pytest.raises(ValueError, match='outside'):
        p.select_gpus(devices, ['not-allocated'], gpu)
    with pytest.raises(ValueError, match='single GPU'):
        p.select_gpus(devices, [gpu, gpu], gpu)


def test_pool_reused_byte_for_byte_across_models(pool):
    folder, suite, index, manifests = pool
    original = index.read_bytes()
    assert p.verify_pool(suite, index) == manifests
    assert p.seal_pool(suite, index) == manifests
    assert index.read_bytes() == original
    assert len((index.parent / 'conditions.csv').read_text().splitlines()) == 7


def test_resealed_manifest_cannot_replace_an_indexed_pool(pool):
    folder, suite, index, manifests = pool
    task = suite['tasks'][0]
    path = p.experiment.manifest_path(suite, 'physical_random', task)
    changed = json.loads(path.read_text())
    ep = changed['episodes'][0]
    ep['condition']['mass_scale'] *= 2
    ep['condition_sha256'] = digest(ep['condition'])
    atomic_json(path, seal(changed))
    with pytest.raises(ValueError, match='index differs'):
        p.verify_pool(suite, index)


def test_corrupt_initial_state_is_rejected_even_when_manifest_resealed(pool):
    folder, suite, index, manifests = pool
    path = p.experiment.manifest_path(suite, 'physical_random', suite['tasks'][0])
    changed = json.loads(path.read_text())
    changed['episodes'][0]['initial_state']['pose'][0] = 999
    atomic_json(path, seal(changed))
    with pytest.raises(ValueError, match='Initial state hash'):
        p.verify_pool(suite, index)


class FakeProcesses:
    def __init__(self, workers=2):
        self.cancelled = threading.Event()
        self.requests, self.starts, self.stops = [], [], []
        self.guard = threading.Lock()
        self.barrier = threading.Barrier(workers)

    def run(self, argv, env, logfile, timeout=None):
        req, res = Path(argv[-2]), Path(argv[-1])
        request = json.loads(req.read_text())
        # Every concurrent slot must dispatch before any result completes.
        self.barrier.wait(timeout=5)
        with self.guard:
            self.requests.append((env['CUDA_VISIBLE_DEVICES'], request))
        ep = request['episode']
        result = {key: ep[key] for key in ('episode_index', 'seed', 'condition_sha256', 'initial_state_sha256')}
        atomic_json(res, dict(result, status='completed', success=ep['episode_index'] != 1,
            policy=request['policy_identity'], benchmark_eligible=True, seconds=1.0))

    def stop(self, proc):
        self.stops.append(proc)

    def close(self):
        self.cancelled.set()


@pytest.mark.parametrize('sim_gpus,policy_gpu,workers', [(['GPU-sim2', 'GPU-sim1'], 'GPU-pro', 2), (['GPU-pro'], 'GPU-pro', 1),
                                                          (['GPU-pro', 'GPU-sim1'], 'GPU-pro', 3)])  # dual: 3 replicas, 6 episodes = 2 full rounds
def test_same_conditions_next_model_and_resume_for_gpu_layouts(pool, monkeypatch, sim_gpus, policy_gpu, workers):
    folder, suite, index, manifests = pool
    original = index.read_bytes()
    processes = FakeProcesses(workers)

    def start(processes, base, args, gpu, slot, job_dir):
        processes.starts.append((args.model, gpu, slot))
        return dict(identity={'name': f'{args.model}_robotwin2', 'config_sha256': 'fixed'},
                    port=23000 + slot*2, process=f'{args.model}-{slot}')

    monkeypatch.setattr(p, 'start_policy', start)
    # Dual layout defaults to 4 replicas; the barrier harness needs the worker count to divide the 6 episodes.
    args = argparse.Namespace(model='fastwam', replicas=workers if workers == 3 else None, video='first', timeout=20)
    base = {'CUDA_VISIBLE_DEVICES': ','.join(dict.fromkeys([policy_gpu, *sim_gpus]))}
    job_dir = folder / 'test-job'
    p.evaluate_pool(processes, args, suite, manifests, base, sim_gpus, policy_gpu, folder, job_dir)
    assert processes.starts == [('fastwam', gpu, i) for i, gpu in enumerate(p.replica_gpus(sim_gpus, policy_gpu, workers))]
    assert len(processes.requests) == 6
    assert {gpu for gpu, req in processes.requests} == set(sim_gpus)
    if p.gpu_layout(sim_gpus, policy_gpu) == 'dual':  # each simulator runs on its replica's GPU
        placement = {23000 + slot*2: gpu for slot, gpu in enumerate(p.replica_gpus(sim_gpus, policy_gpu, workers))}
        assert all(placement[req['policy_port']] == gpu for gpu, req in processes.requests)
    assert len([req for gpu, req in processes.requests if 'video' in req]) == 2
    assert all(req['episode']['episode_index'] == 0 for gpu, req in processes.requests if 'video' in req)
    assert json.loads((job_dir/'timing.json').read_text())['newly_completed'] == 6
    assert index.read_bytes() == original
    # A completed run must not load models or rerun episodes.
    p.evaluate_pool(processes, args, suite, manifests, base, sim_gpus, policy_gpu, folder, job_dir)
    assert len(processes.starts) == workers and len(processes.requests) == 6
    # Another model uses the exact same pool on reversed worker/GPU ordering.
    args.model = 'pi05'
    p.evaluate_pool(processes, args, suite, manifests, base, list(reversed(sim_gpus)), policy_gpu, folder, job_dir)
    first = {(r['episode']['condition']['task'], r['episode']['seed']): r['episode'] for g, r in processes.requests[:6]}
    second = {(r['episode']['condition']['task'], r['episode']['seed']): r['episode'] for g, r in processes.requests[6:]}
    assert first == second
    assert index.read_bytes() == original
    result = p.experiment.report(folder)
    assert result['randomization_audit']['verified_episodes'] == 12
    assert result['randomization_audit']['partial'] == 6  # Three other models have not run.
    for row in result['aggregate']:
        assert row['success_rate'] == (pytest.approx(2/3) if row['model'] in ('pi05', 'fastwam') else None)


def test_single_gpu_refuses_two_policy_replicas_before_model_start(pool):
    folder, suite, index, manifests = pool
    processes = FakeProcesses(workers=1)
    args = argparse.Namespace(model='fastwam', replicas=2, video='first', timeout=20)
    with pytest.raises(ValueError, match='exactly one policy replica'):
        p.evaluate_pool(processes, args, suite, manifests, {}, ['GPU-pro'], 'GPU-pro', folder, folder/'logs')
    assert not processes.starts and not processes.requests


def test_policy_replica_identity_mismatch_stops_before_any_episode(pool, monkeypatch):
    folder, suite, index, manifests = pool
    processes = FakeProcesses()
    monkeypatch.setattr(p, 'start_policy', lambda *a: dict(identity={'name': 'fastwam_robotwin2', 'revision': a[4]},
                                                        port=23000+a[4]*2, process=a[4]))
    args = argparse.Namespace(model='fastwam', replicas=2, video='first', timeout=20)
    with pytest.raises(ValueError, match='different identities'):
        p.evaluate_pool(processes, args, suite, manifests, {}, ['a', 'b'], 'c', folder, folder/'logs')
    assert not processes.requests and len(processes.stops) == 2


def test_resume_refuses_modified_worker_conditions(pool):
    folder, suite, index, manifests = pool
    identity = {'name': 'fastwam_robotwin2'}
    jobs = p.pending_episodes(folder, suite, 'fastwam', manifests, identity)
    task, ep = jobs[0]
    dest = folder/'fastwam/physical_random'/task
    request = p.make_request(ep, suite, identity, 23000, dest, 'first')
    request = json.loads(json.dumps(request))
    request['episode']['condition']['mass_scale'] *= 2
    atomic_json(dest/'episode-0000.request.json', request)
    with pytest.raises(ValueError, match='Invalid existing'):
        p.pending_episodes(folder, suite, 'fastwam', manifests, identity)


def test_failure_cancels_queue_without_substituting_seeds():
    processes = FakeProcesses()
    observed = []
    def fail(slot, item):
        observed.append(item)
        raise RuntimeError('GPU error')
    with pytest.raises(RuntimeError, match='GPU error'):
        p.dispatch(list(range(100)), [0, 1], fail, processes)
    assert processes.cancelled.is_set() and len(observed) <= 2


def test_owned_processes_are_cleaned_up_and_finished_pids_are_forgotten(tmp_path):
    processes = p.Processes(tmp_path)
    processes.run([sys.executable, '-c', 'pass'], os.environ.copy(), tmp_path/'completed.log')
    assert processes.children == []
    child = processes.start([sys.executable, '-c', 'import time; time.sleep(60)'], os.environ.copy(), tmp_path/'child.log')
    processes.close()
    assert child.poll() is not None and processes.children == []


@pytest.mark.parametrize('template', ['job-5060x2-pro6000.sbatch', 'job-rtx5090x2-pro6000.sbatch',
                                    'job-pro6000-single.sbatch'])
def test_new_cluster_job_keeps_the_protected_template(template):
    protected = (ROOT/'cluster/default-protected.txt').read_text()
    script = (ROOT/'cluster'/template).read_text()
    assert script.endswith(protected)
    assert script.count('apptainer exec') == 1
    assert 'run_parallel.sh' in script
    assert 'ROBOTWIN_MODEL="fastwam"' in script
    assert 'ROBOTWIN_EPISODES=100' in script
    assert '#SBATCH --nodes=1' in script
    subprocess.run(['bash', '-n', ROOT/'cluster/run_parallel.sh', ROOT/'cluster'/template], check=True)


@pytest.mark.parametrize('shared', [False, True])
def test_all_builds_once_then_second_model_only_reuses_pool(pool, monkeypatch, shared):
    folder, suite, index, manifests = pool
    index.unlink()
    frozen, evaluated, calls = [], [], []

    class ControllerProcesses:
        def __init__(self):
            self.cancelled = threading.Event()
        def run(self, argv, env, logfile, timeout=None):
            calls.append([str(x) for x in argv])
        def close(self):
            self.cancelled.set()

    monkeypatch.setattr(p, 'Processes', ControllerProcesses)
    monkeypatch.setattr(p, 'task_names', lambda: suite['tasks'])
    devices = {'GPU-p': dict(name='RTX PRO 6000 Blackwell')}
    if not shared:
        devices.update({'GPU-a': dict(name='RTX 5060'), 'GPU-b': dict(name='RTX 5060')})
    monkeypatch.setattr(p, 'discover_gpus', lambda base: devices)
    monkeypatch.setattr(p, 'provenance_on_gpu', lambda *a: {'test': True})
    monkeypatch.setattr(p, 'freeze_pool', lambda *a: frozen.append(a[4]))
    monkeypatch.setattr(p, 'evaluate_pool', lambda *a: evaluated.append((a[1].model, a[3])))
    # signal handlers belong to the test process, not to the mocked controller.
    monkeypatch.setattr(p.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(sys, 'argv', ['parallel_eval.py', '--experiment', 'shared', '--episodes', '3'])
    p.main()
    assert frozen == [['GPU-p'] if shared else ['GPU-a', 'GPU-b']]
    assert evaluated[0] == ('fastwam', manifests)
    assert sum(any('cluster_preflight.py' in item for item in call) for call in calls) == len(devices)
    original = index.read_bytes()
    assert any(call[-2:] == ['--models', 'fastwam'] for call in calls)
    assert not any('--models' in c and 'pi05' in c for c in calls)

    calls.clear()
    monkeypatch.setattr(sys, 'argv', ['parallel_eval.py', '--experiment', 'shared', '--episodes', '3',
                                     '--phase', 'evaluate', '--model', 'pi05'])
    p.main()
    assert len(frozen) == 1 and evaluated[-1] == ('pi05', manifests)
    assert index.read_bytes() == original
    assert any(call[-2:] == ['--models', 'pi05'] for call in calls)
    # Evaluation must never rebuild a missing shared pool.
    index.unlink()
    with pytest.raises(FileNotFoundError, match='No sealed pool'):
        p.main()
    assert len(frozen) == 1 and len(evaluated) == 2


def test_wrapper_uses_writable_home_and_passes_model_override(tmp_path):
    import shutil
    root = tmp_path/'container'
    (root/'cluster').mkdir(parents=True)
    shutil.copyfile(ROOT/'cluster/run_parallel.sh', root/'cluster/run_parallel.sh')
    (root/'.env/bin').mkdir(parents=True)
    python = root/'.env/bin/python'
    python.write_text('#!/usr/bin/python3\nimport json,os,sys\n'
                      'print(json.dumps(dict(args=sys.argv[1:],cache=os.environ["XDG_CACHE_HOME"],'
                      'extensions=os.environ["TORCH_EXTENSIONS_DIR"])))\n')
    python.chmod(0o755)
    home = str(tmp_path/'writable-home')
    env = dict(os.environ, HOME=home, XDG_CACHE_HOME='/read-only-image/.cache',
               TORCH_EXTENSIONS_DIR='/read-only-image/.cache/torch_extensions')
    output = subprocess.check_output(['bash', root/'cluster/run_parallel.sh', '--model', 'pi05',
                                     '--phase', 'evaluate'], env=env, text=True)
    record = json.loads(output)
    assert record['cache'] == home+'/.cache'
    assert record['extensions'] == home+'/.cache/torch_extensions'
    assert record['args'][-4:] == ['--model', 'pi05', '--phase', 'evaluate']


def test_start_policy_handshake_resolves_the_isolated_workspace(tmp_path):
    runtime = tmp_path/'policies/fastwam'
    workspace = runtime/'workspace'
    (workspace/'tools').mkdir(parents=True)
    (runtime/'venv/bin').mkdir(parents=True)
    (runtime/'venv/bin/python').symlink_to(sys.executable)
    # Real TCP handshake, no GPU or learned model. The policy resolves ${ROOT}
    # differently from the simulator, just as the relocated bundle does.
    (workspace/'tools/serve_policy.py').write_text('''
import argparse, hashlib, json, os, socket, struct
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--model');p.add_argument('--port',type=int);p.add_argument('--native-port')
p.add_argument('--t5-device');p.add_argument('--record-dir',type=Path)
a=p.parse_args()
config={'name':a.model+'_robotwin2','args':{'checkpoint':'${ROOT}/checkpoints/weights'}}
a.record_dir.mkdir(parents=True)
(a.record_dir/'policy.yaml').write_text(json.dumps(config))
config['args']['checkpoint']=os.environ['ROBOTWIN_PHYS_ROOT']+'/checkpoints/weights'
identity={'type':'hello','name':config['name'],'config_sha256':hashlib.sha256(
    json.dumps(config,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()}
packet=json.dumps(identity).encode()
with socket.socket() as server:
    server.bind(('127.0.0.1',a.port));server.listen(1)
    while True:
        connection,_=server.accept()
        with connection: connection.sendall(struct.pack('!I',len(packet))+packet)
''')
    with socket.socket() as port_probe:
        port_probe.bind(('127.0.0.1', 0))
        port = port_probe.getsockname()[1]
    args = argparse.Namespace(policy_root=str(tmp_path/'policies'), model='fastwam',
                              base_port=port, startup_timeout=10)
    processes = p.Processes(tmp_path)
    try:
        server = p.start_policy(processes, os.environ.copy(), args, 'GPU-p', 0, tmp_path/'logs')
        assert server['identity']['name'] == 'fastwam_robotwin2'
        # The legacy bundle readiness command must use the same isolated root.
        subprocess.run([sys.executable, ROOT/'tools/wait_policy.py', '--model', 'fastwam',
                        '--port', str(port), '--config', tmp_path/'logs/policy-0/policy.yaml',
                        '--config-root', workspace, '--timeout', '5'], check=True, timeout=10)
    finally:
        processes.close()
