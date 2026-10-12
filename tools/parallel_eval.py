#!/usr/bin/env python3
"""One model on a shared GPU or three GPUs, with immutable episode pools.

Run with the simulator Python inside the bundle image. No host installation,
inference batching, precision change, or seed resampling during evaluation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import csv
import fcntl
import json
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment as experiment
from policy_runtime import read_policy_config
from robotwin_phys.conditions import ROOT, digest, task_names
from robotwin_phys.manifest import atomic_json, seal
from robotwin_phys.rpc import PolicyClient
from randomization_audit import episode_record

PROTOCOL = 'physical_random'


@contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def select_gpus(devices, sim_gpus=None, policy_gpu=None):
    """IDs refer only to the job/container allocation, never host GPU order."""
    if len(devices) not in (1, 3):
        raise ValueError('This evaluator requires one or three allocated/visible GPUs')
    if (sim_gpus is None) != (policy_gpu is None):
        raise ValueError('Specify both --sim-gpus and --policy-gpu, or neither')
    if sim_gpus is None:
        if len(devices) == 1:
            policy_gpu = next(iter(devices))
            sim_gpus = [policy_gpu]
        else:
            sim_gpus = [key for key, value in devices.items() if re.search(r'RTX (?:5060|5090)\b', value['name'])]
            candidates = [key for key, value in devices.items() if 'PRO 6000' in value['name']]
            if len(sim_gpus) != 2 or len(candidates) != 1:
                raise ValueError('Expected two RTX 5060 or RTX 5090 GPUs and one RTX PRO 6000; use explicit GPU IDs for another layout')
            policy_gpu = candidates[0]
    if not set([*sim_gpus, policy_gpu]) <= set(devices):
        raise ValueError('GPU selection is outside the allocation')
    if len(devices) == 1:
        if list(sim_gpus) != [policy_gpu]:
            raise ValueError('A single GPU must be shared by one simulator and one policy replica')
    elif len(sim_gpus) != 2 or len(set([*sim_gpus, policy_gpu])) != 3:
        raise ValueError('Two distinct simulation GPUs and a separate inference GPU are required')
    return list(sim_gpus), policy_gpu


def policy_replica_count(args, sim_gpus, policy_gpu):
    shared = sim_gpus == [policy_gpu]
    replicas = args.replicas if args.replicas is not None else (1 if shared or args.model == 'fact' else 2)
    if shared and replicas != 1:
        raise ValueError('A shared GPU requires exactly one policy replica')
    if args.model == 'fact' and replicas != 1:
        raise ValueError('FACT currently uses one replica to keep its native-port policy identity fixed')
    return replicas


class Processes:
    """Own and clean up complete child process groups on failure or interruption."""
    def __init__(self, root=ROOT):
        self.root = root
        self.children = []
        self.guard = threading.Lock()
        self.cancelled = threading.Event()

    def start(self, argv, env, logfile):
        logfile.parent.mkdir(parents=True, exist_ok=True)
        with self.guard, logfile.open('w') as log:
            if self.cancelled.is_set():
                raise RuntimeError('Run cancelled')
            proc = subprocess.Popen([str(x) for x in argv], cwd=self.root, env=env,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            self.children.append(proc)
        return proc

    def run(self, argv, env, logfile, timeout=None):
        proc = self.start(argv, env, logfile)
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.stop(proc)
            raise TimeoutError(f'Process timed out; see {logfile}')
        if code:
            self.stop(proc)
            raise RuntimeError(f'Process exited {code}; see {logfile}. No episode is replaced or scored as a failure.')
        # Never retain a completed PID across a multi-day run (PID reuse).
        with self.guard:
            if proc in self.children:
                self.children.remove(proc)

    def stop(self, proc):
        with self.guard:
            if proc not in self.children:
                return
            self.children.remove(proc)
        # A wrapper may already have exited while its descendants are still alive.
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()

    def close(self):
        with self.guard:
            self.cancelled.set()
            children = list(self.children)
        for proc in reversed(children):
            self.stop(proc)


def dispatch(items, slots, action, processes):
    """Each slot owns one policy replica. Queue order does not determine seeds."""
    work = queue.Queue()
    for item in items:
        work.put(item)

    def consume(slot):
        while not processes.cancelled.is_set():
            try:
                item = work.get_nowait()
            except queue.Empty:
                return
            action(slot, item)

    with ThreadPoolExecutor(max_workers=len(slots)) as executor:
        futures = [executor.submit(consume, slot) for slot in slots]
        try:
            for future in as_completed(futures):
                future.result()
        except BaseException:
            processes.close()
            raise


def read_pool(suite):
    manifests = {}
    for task in suite['tasks']:
        manifest = experiment.check_manifest(suite, PROTOCOL, task)
        for episode in manifest['episodes']:
            if digest(episode['initial_state']) != episode['initial_state_sha256']:
                raise ValueError(f'Initial state hash mismatch: {task}/{episode["episode_index"]}')
        manifests[task] = manifest
    return manifests


def pool_index(suite, manifests):
    return seal(dict(schema='robotwin-shared-pool-v1', suite_sha256=suite['sha256'],
                     protocol=PROTOCOL, episodes_per_task=suite['episodes'],
                     manifests={task: manifest['sha256'] for task, manifest in manifests.items()}))


def verify_pool(suite, path):
    saved = json.loads(path.read_text())
    manifests = read_pool(suite)
    if saved != pool_index(suite, manifests):
        raise ValueError('Frozen pool index differs from the task manifests; do not regenerate or mix this pool')
    return manifests


def seal_pool(suite, path):
    manifests = read_pool(suite)
    index = pool_index(suite, manifests)
    if path.exists():
        return verify_pool(suite, path)
    # Human-readable export is derived from the immutable manifests, never sampled again.
    export = path.parent / 'conditions.csv'
    temporary = export.with_suffix('.csv.tmp')
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['task', 'episode_index', 'seed', 'condition_sha256',
                                                   'initial_state_sha256', 'condition_json'])
        writer.writeheader()
        for task, manifest in manifests.items():
            for ep in manifest['episodes']:
                writer.writerow(dict(task=task, **{key: ep[key] for key in
                    ('episode_index', 'seed', 'condition_sha256', 'initial_state_sha256')},
                    condition_json=json.dumps(ep['condition'], sort_keys=True)))
    temporary.replace(export)
    atomic_json(path, index)
    return manifests


def pending_episodes(folder, suite, model, manifests, identity):
    jobs = []
    for task, manifest in manifests.items():
        dest = folder / model / PROTOCOL / task
        expected = dict(manifest_sha256=manifest['sha256'], policy=identity,
                        policy_seed=suite['policy_seed'], smoke_max_actions=None, mode='evaluate')
        meta = dest / 'run.json'
        if meta.exists() and json.loads(meta.read_text()) != expected:
            raise ValueError(f'Policy or manifest changed for existing results: {dest}')
        atomic_json(meta, expected)
        allowed = {f"episode-{e['episode_index']:04d}.result.json" for e in manifest['episodes']}
        if any(p.name not in allowed for p in dest.glob('episode-*.result.json')):
            raise ValueError(f'Unexpected episode results: {dest}')
        for ep in manifest['episodes']:
            audit = episode_record(folder, suite, model, PROTOCOL, task, manifest, ep)
            if audit and audit['errors']:
                raise ValueError(f'Invalid existing episode evidence: {task}/{ep["episode_index"]}')
            if audit and audit['verified']:
                continue
            jobs.append((task, ep))
    # Interleave tasks to keep early timing less dependent on one long task.
    return sorted(jobs, key=lambda job: (job[1]['episode_index'], job[0]))


def make_request(ep, suite, identity, port, destination, video):
    request = dict(mode='evaluate', episode=ep, host='127.0.0.1', policy_port=port,
                   policy_seed=suite['policy_seed'], policy_identity=identity,
                   live=False, port=8765, viewer=False, smoke_max_actions=None)
    if video == 'all' or (video == 'first' and ep['episode_index'] == 0):
        request['video'] = str(destination / f"episode-{ep['episode_index']:04d}.mp4")
    return request


def base_environment():
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', SAPIEN_RENDER_DEVICE='cuda:0',
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONPATH=str(ROOT / 'src'))
    return env


def simulator_environment(base, gpu):
    return dict(base, CUDA_VISIBLE_DEVICES=gpu)


def discover_gpus(base):
    if os.environ.get('SLURM_JOB_ID') and not base.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('SLURM must supply CUDA_VISIBLE_DEVICES')
    allocated = base.get('CUDA_VISIBLE_DEVICES')
    if allocated is None:
        code = 'import torch; print(torch.cuda.device_count())'
        count = int(subprocess.check_output([sys.executable, '-c', code], env=base, text=True))
        allocated = ','.join(str(i) for i in range(count))
    ids = allocated.split(',')
    if len(ids) not in (1, 3) or len(set(ids)) != len(ids) or any(not x for x in ids):
        raise ValueError('Pass one or three distinct GPUs through the job/container allocation')
    code = ('import json,torch; p=torch.cuda.get_device_properties(0); '
            'print(json.dumps(dict(name=p.name,memory_gib=p.total_memory/2**30)))')
    return {gpu: json.loads(subprocess.check_output([sys.executable, '-c', code],
                env=simulator_environment(base, gpu), text=True)) for gpu in ids}


def provenance_on_gpu(processes, base, gpu, path):
    code = ('from robotwin_phys.provenance import fingerprint; '
            'from robotwin_phys.manifest import atomic_json; import sys; '
            'atomic_json(sys.argv[1], fingerprint())')
    processes.run([sys.executable, '-c', code, path], simulator_environment(base, gpu), path.with_suffix('.log'))
    return json.loads(path.read_text())


def start_policy(processes, base, args, policy_gpu, slot, job_dir):
    runtime = Path(args.policy_root) / args.model
    workspace, venv = runtime / 'workspace', runtime / 'venv'
    if not (venv / 'bin/python').exists():
        raise FileNotFoundError('Use the bundle image; isolated policy runtime is missing')
    record = job_dir / f'policy-{slot}'
    port = args.base_port + slot * 2
    env = dict(base, CUDA_VISIBLE_DEVICES=policy_gpu, ROBOTWIN_PHYS_ROOT=str(workspace),
               ROBOTWIN_BUILD_INFO=str(runtime / 'build'), VIRTUAL_ENV=str(venv),
               PYTHONPATH=str(workspace / 'src'), PATH=f'{venv}/bin:/usr/local/cuda/bin:/usr/bin:/bin',
               JAX_PLATFORMS='cpu', XLA_PYTHON_CLIENT_PREALLOCATE='false')
    proc = processes.start([venv / 'bin/python', workspace / 'tools/serve_policy.py', '--model', args.model,
        '--port', port, '--native-port', port + 1, '--t5-device', 'cuda:0', '--record-dir', record],
        env, job_dir / f'policy-{slot}.log')
    deadline = time.monotonic() + args.startup_timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f'Policy replica {slot} exited; inspect {job_dir}/policy-{slot}.log')
        if (record / 'policy.yaml').exists():
            try:
                client = PolicyClient('127.0.0.1', port, timeout=2)
                identity = client.identity
                client.close()
                cfg = read_policy_config(record / 'policy.yaml', workspace)
                if identity['name'] != f'{args.model}_robotwin2' or identity['config_sha256'] != digest(cfg):
                    raise ValueError('Policy endpoint/configuration mismatch')
                return dict(port=port, identity=identity, process=proc)
            except (OSError, ConnectionError):
                pass
        if processes.cancelled.wait(1):
            raise RuntimeError('Run cancelled during policy startup')
    raise TimeoutError(f'Policy replica {slot} startup timed out; inspect {job_dir}')


def freeze_pool(processes, args, suite, base, sim_gpus, fingerprints, job_dir):
    remaining = []
    for task in suite['tasks']:
        path = experiment.manifest_path(suite, PROTOCOL, task)
        if path.exists():
            manifest = experiment.check_manifest(suite, PROTOCOL, task)
            if manifest['provenance'] != fingerprints[0]:
                raise ValueError(f'Existing manifest runtime differs: {task}')
        else:
            remaining.append(task)

    def work(gpu, task):
        path = experiment.manifest_path(suite, PROTOCOL, task)
        with lock(ROOT / 'runs' / suite['name'] / '.locks' / f'freeze-None-{PROTOCOL}-{task}.lock'):
            # Candidates remain sequential within each task, so timing cannot select seeds.
            argv = [sys.executable, '-m', 'robotwin_phys.cli', 'freeze', '--task', task,
                    '--protocol', PROTOCOL, '--episodes', suite['episodes'], '--seed', suite['seed'],
                    '--instruction-type', suite['instruction_type'], '--max-candidates', args.max_candidates,
                    '--timeout', args.timeout, '--resume', '--output', path]
            processes.run(argv, simulator_environment(base, gpu), job_dir / f'freeze-{task}.log')
            print(f'Frozen {task}: {suite["episodes"]} episodes', flush=True)

    dispatch(remaining, sim_gpus, work, processes)


def evaluate_pool(processes, args, suite, manifests, base, sim_gpus, policy_gpu, folder, job_dir):
    # Check existing evidence before model startup; invalid results never get silently replaced.
    result = experiment.report(folder)
    selected = [cell for cell in result['tasks'] if cell['model'] == args.model]
    if any(cell['status'] == 'invalid' for cell in selected):
        raise ValueError('Invalid existing results; inspect tables/randomization_audit.json')
    if all(cell['status'] == 'complete' for cell in selected):
        print('All episodes already complete and verified; no policy load needed.', flush=True)
        return
    replicas = policy_replica_count(args, sim_gpus, policy_gpu)
    servers = []
    try:
        # Sequential startup avoids concurrent checkpoint reads/initialization memory peaks.
        for slot in range(replicas):
            servers.append(start_policy(processes, base, args, policy_gpu, slot, job_dir))
        if any(server['identity'] != servers[0]['identity'] for server in servers):
            raise ValueError('Policy replicas have different identities; evaluation refused')
        identity = servers[0]['identity']
        jobs = pending_episodes(folder, suite, args.model, manifests, identity)
        slots = [(sim_gpus[i % len(sim_gpus)], server) for i, server in enumerate(servers)]
        begin, completed = time.monotonic(), []
        progress_lock = threading.Lock()

        def work(slot, job):
            gpu, server = slot
            task, ep = job
            dest = folder / args.model / PROTOCOL / task
            label = f"episode-{ep['episode_index']:04d}"
            request = make_request(ep, suite, identity, server['port'], dest, args.video)
            req, res = dest / (label + '.request.json'), dest / (label + '.result.json')
            with lock(folder / '.episode-locks' / args.model / task / (label + '.lock')):
                atomic_json(req, request)
                processes.run([sys.executable, '-m', 'robotwin_phys.worker', req, res],
                              simulator_environment(base, gpu), dest / (label + '.log'), args.timeout)
                audit = episode_record(folder, suite, args.model, PROTOCOL, task, manifests[task], ep)
                if not audit or not audit['verified']:
                    raise ValueError(f'Episode failed validation: {task}/{label}; no seed replacement')
                with progress_lock:
                    completed.append((task, ep['episode_index']))
                    elapsed = time.monotonic() - begin
                    record = dict(model=args.model, pool_sha256=json.loads((experiment.manifest_path(suite,
                        PROTOCOL, suite['tasks'][0]).parent.parent / 'pool-index.json').read_text())['sha256'],
                        newly_completed=len(completed), scheduled_this_run=len(jobs), elapsed_seconds=elapsed,
                        episodes_per_hour=3600 * len(completed) / elapsed,
                        estimated_seconds_remaining=elapsed / len(completed) * (len(jobs) - len(completed)),
                        estimate_basis='Observed wall-clock throughput for this model and current run; task mix can change ETA')
                    atomic_json(job_dir / 'timing.json', record)
                    print(f'{args.model} {task}/{label}: success={audit["success"]}; '
                          f'{len(completed)}/{len(jobs)}; {record["episodes_per_hour"]:.1f} episodes/hour', flush=True)

        dispatch(jobs, slots, work, processes)
    finally:
        for server in servers:
            processes.stop(server['process'])
        experiment.report(folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=experiment.MODELS, default='fastwam')
    parser.add_argument('--experiment', default='phys-5060-100-v1')
    parser.add_argument('--phase', choices=['all', 'freeze', 'evaluate', 'report'], default='all')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=100000)
    parser.add_argument('--policy-seed', type=int, default=0)
    parser.add_argument('--tasks', nargs='+')
    parser.add_argument('--sim-gpus', nargs='+', help='One shared GPU or two simulator GPUs; defaults to allocation-based selection')
    parser.add_argument('--policy-gpu')
    parser.add_argument('--replicas', type=int, choices=[1, 2])
    parser.add_argument('--video', choices=['first', 'none', 'all'], default='first')
    parser.add_argument('--max-candidates', type=int, default=10000)
    parser.add_argument('--timeout', type=int, default=1800)
    parser.add_argument('--startup-timeout', type=int, default=2400)
    parser.add_argument('--base-port', type=int, default=23000)
    parser.add_argument('--policy-root', default='/opt/robotwin-policies')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9._-]*', args.experiment):
        parser.error('Experiment must be a simple directory name')
    if not 1024 <= args.base_port <= 65531 or args.timeout <= 0 or args.startup_timeout <= 0:
        parser.error('Invalid port or timeout')
    folder = ROOT / 'runs' / args.experiment
    folder.mkdir(parents=True, exist_ok=True)
    if args.phase == 'report':
        experiment.report(folder)
        return
    tasks = args.tasks or task_names()
    if not tasks or len(set(tasks)) != len(tasks) or not set(tasks) <= set(task_names()):
        parser.error('Invalid task selection')
    if not 0 <= args.seed < 2**32 or args.episodes < 1 or args.max_candidates < args.episodes:
        parser.error('Invalid seed/episode/candidate count')
    suite = seal(dict(schema='robotwin-experiment-v1', name=args.experiment, tasks=tasks,
        protocols=[PROTOCOL], models=list(experiment.MODELS), episodes=args.episodes,
        seed=args.seed, policy_seed=args.policy_seed, instruction_type='unseen'))
    processes = Processes()
    job_id = os.environ.get('SLURM_JOB_ID', f'local-{int(time.time())}-{os.getpid()}')
    job_dir = ROOT / 'reports' / 'parallel' / args.experiment / f'{args.model}-{job_id}'
    job_dir.mkdir(parents=True, exist_ok=True)

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    with lock(ROOT / 'runs/.full-run-locks' / f'{args.experiment}.lock'), \
            lock(ROOT / 'runs/.server-locks' / f'{args.experiment}-{args.model}.lock'):
        try:
            if (folder / 'suite.json').exists():
                if experiment.read_suite(folder) != suite:
                    raise ValueError('Existing suite settings differ; keep the same pool or choose a new experiment')
            elif args.phase == 'evaluate':
                raise FileNotFoundError('Evaluation requires an existing frozen suite')
            else:
                atomic_json(folder / 'suite.json', suite)
            base = base_environment()
            devices = discover_gpus(base)
            sim_gpus, policy_gpu = select_gpus(devices, args.sim_gpus, args.policy_gpu)
            args.replicas = policy_replica_count(args, sim_gpus, policy_gpu)
            atomic_json(job_dir / 'allocation.json', dict(devices=devices, sim_gpus=sim_gpus,
                        policy_gpu=policy_gpu, model=args.model, video=args.video, suite_sha256=suite['sha256'],
                        gpu_layout='shared' if sim_gpus == [policy_gpu] else 'split', policy_replicas=args.replicas))
            print(f'Model={args.model}; simulator GPUs={sim_gpus}; policy GPU={policy_gpu}; replicas={args.replicas}', flush=True)
            with lock(ROOT / 'downloads/.prepare.lock'):
                argv = [sys.executable, ROOT / 'tools/prepare_cluster.py', '--assets']
                processes.run(argv, base, job_dir / 'prepare-assets.log')
            for i, gpu in enumerate(dict.fromkeys([*sim_gpus, policy_gpu])):
                processes.run([sys.executable, ROOT / 'tools/cluster_preflight.py', '--output',
                               job_dir / f'gpu-{i}.json'], simulator_environment(base, gpu), job_dir / f'gpu-{i}.log')
            fingerprints = [provenance_on_gpu(processes, base, gpu, job_dir / f'provenance-{i}.json')
                            for i, gpu in enumerate(sim_gpus)]
            if any(fingerprint != fingerprints[0] for fingerprint in fingerprints[1:]):
                raise ValueError('Simulator GPU runtimes differ; cannot share an episode pool')
            index = ROOT / 'manifests' / args.experiment / 'pool-index.json'
            if index.exists():
                manifests = verify_pool(suite, index)
                print(f'Reusing frozen pool: {index}', flush=True)
            elif args.phase == 'evaluate':
                raise FileNotFoundError('No sealed pool. Run --phase freeze or all first')
            else:
                if any((folder / model).exists() for model in suite['models']):
                    raise ValueError('Evaluation files exist but pool index is missing; restore the original index')
                freeze_pool(processes, args, suite, base, sim_gpus, fingerprints, job_dir)
                index.parent.mkdir(parents=True, exist_ok=True)
                manifests = seal_pool(suite, index)
            if any(m['provenance'] != fingerprints[0] for m in manifests.values()):
                raise ValueError('Frozen pool runtime differs; do not regenerate it inside this experiment')
            if args.phase != 'freeze':
                with lock(ROOT / 'downloads/.prepare.lock'):
                    processes.run([sys.executable, ROOT / 'tools/prepare_cluster.py', '--models', args.model],
                                  base, job_dir / 'prepare-model.log')
                evaluate_pool(processes, args, suite, manifests, base, sim_gpus, policy_gpu, folder, job_dir)
            else:
                experiment.report(folder)
        finally:
            processes.close()


if __name__ == '__main__':
    main()
