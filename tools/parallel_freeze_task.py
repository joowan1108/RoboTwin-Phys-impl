#!/usr/bin/env python3
"""Finish one task's expert pool with several workers scanning seeds concurrently.

The sequential freeze (robotwin_phys.cli freeze) evaluates candidates one seed at
a time. Candidates are independent (each runs in a fresh worker process on its
own seed), so this scanner evaluates the next N seeds at once and commits the
results strictly in seed order. The saved records are the same as the sequential
loop would write: the rejected list, the accepted episodes with consecutive
episode_index values, the contiguous seed prefix, the sealed progress file and
the final verified manifest. Results for seeds beyond the one that completed the
pool are discarded, exactly as the sequential loop never evaluates them.

Lives in tools/ (outside the runtime fingerprint); the worker and the expert are
the untouched src/vendor code of the bound snapshot. Runs the success-context
provenance continuation first when the saved records still carry the previous
source, so the manifest it finishes has the current provenance.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))
from robotwin_phys.cli import episode as make_episode
from robotwin_phys.conditions import task_names
from robotwin_phys.manifest import atomic_json, seal

PROTOCOL = 'physical_random'
WORKER_COMMAND = [sys.executable, '-m', 'robotwin_phys.worker']


def base_env(gpu):
    import parallel_eval
    return parallel_eval.simulator_environment(parallel_eval.base_environment(), gpu)


class Candidate:
    def __init__(self, seed, index, folder, request, env, timeout):
        self.seed, self.index, self.folder = seed, index, folder
        self.label = f'seed-{seed}'
        self.req, self.res = folder / (self.label + '.request.json'), folder / (self.label + '.result.json')
        self.logfile = folder / (self.label + '.log')
        atomic_json(self.req, request)
        self.log = self.logfile.open('w')
        self.started = time.monotonic()
        self.timeout = timeout
        self.proc = subprocess.Popen([*WORKER_COMMAND, str(self.req), str(self.res)],
                                     stdout=self.log, stderr=subprocess.STDOUT, env=env)

    def done(self):
        if self.proc.poll() is None:
            if time.monotonic() - self.started > self.timeout:
                self.proc.kill()
                self.proc.wait()
                self.log.close()
                raise RuntimeError(f'Episode worker timed out after {self.timeout}s; see {self.logfile}')
            return False
        self.log.close()
        return True

    def result(self):
        if self.proc.returncode:
            detail = '\n'.join(self.logfile.read_text(errors='replace').splitlines()[-24:])
            raise RuntimeError(f'Episode worker failed (exit {self.proc.returncode}); see {self.logfile}. '
                               f'This error is not excluded as infeasible.\nWorker log tail:\n{detail}')
        return json.loads(self.res.read_text())

    def kill(self):
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()
        self.log.close()


def finish_task(task, out, config, gpus, workers, timeout, max_candidates, args):
    """Advance config (a loaded progress record) to its requested episode count."""
    folder = out.parent / (out.stem + '-verification')
    folder.mkdir(parents=True, exist_ok=True)
    used = [e['seed'] for e in config['episodes'] + config['rejected']]
    next_seed = (max(used) + 1) if used else config['start_seed']
    limit_seed = config['start_seed'] + max_candidates
    running, results = {}, {}
    commit_seed = next_seed
    slots = [gpus[i % len(gpus)] for i in range(workers)]  # slot -> GPU, round robin
    free = list(range(len(slots)))
    try:
        while len(config['episodes']) < config['requested_episodes']:
            while free and next_seed < limit_seed:
                slot = free.pop(0)
                index = len(config['episodes'])  # provisional, patched at commit
                ep = make_episode(task, next_seed, index, args)
                request = {'mode': 'expert', 'episode': ep, 'instruction_type': config['instruction_type'],
                           'live': False, 'port': 8765, 'viewer': False}
                candidate = Candidate(next_seed, index, folder, request, base_env(slots[slot]), timeout)
                candidate.slot = slot
                running[next_seed] = candidate
                next_seed += 1
            if not running:
                raise RuntimeError('Candidate budget exhausted. Partial pool is not eligible for evaluation; see progress.json')
            for seed, candidate in list(running.items()):
                if candidate.done():
                    results[seed] = candidate.result()
                    free.append(candidate.slot)
                    del running[seed]
            while commit_seed in results and len(config['episodes']) < config['requested_episodes']:
                result = results.pop(commit_seed)
                candidate_folder = folder
                if result['status'] == 'rejected':
                    config['rejected'].append({'seed': commit_seed, **result})
                    atomic_json(out.with_suffix('.progress.json'), seal(config))
                    print(f'Rejected seed {commit_seed}: {result["reason"]}', flush=True)
                elif result['status'] != 'verified':
                    raise RuntimeError(result)
                else:
                    ep = result['episode']
                    ep['episode_index'] = len(config['episodes'])
                    # Keep the per-seed evidence consistent with the committed index.
                    for name in ('.request.json', '.result.json'):
                        path = candidate_folder / (f'seed-{commit_seed}' + name)
                        record = json.loads(path.read_text())
                        record['episode']['episode_index'] = ep['episode_index']
                        atomic_json(path, record)
                    config['episodes'].append(ep)
                    print(f'{len(config["episodes"])}/{config["requested_episodes"]}: seed {commit_seed}', flush=True)
                    atomic_json(out.with_suffix('.progress.json'), seal(config))
                commit_seed += 1
            if len(config['episodes']) < config['requested_episodes']:
                time.sleep(0.2)
    finally:
        for candidate in running.values():
            candidate.kill()
    # Seeds evaluated (or started) beyond the completing one are not part of the
    # contiguous prefix; drop their evidence so the folder matches the sequential loop.
    for seed in list(results) + list(running):
        for name in ('.request.json', '.result.json', '.log', '.mp4'):
            path = folder / (f'seed-{seed}' + name)
            if path.exists():
                path.unlink()
    config['status'] = 'verified'
    atomic_json(out, seal(config))
    print(f'Manifest: {out}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--tasks', nargs='*', help='Unfinished tasks to finish; default: every task with progress but no manifest')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--timeout', type=int, default=1800)
    parser.add_argument('--max-candidates', type=int, default=10000)
    args = parser.parse_args()
    import parallel_eval
    import continue_success_context
    from robotwin_phys.provenance import fingerprint
    folder = ROOT / 'runs' / args.experiment
    suite = parallel_eval.experiment.read_suite(folder)
    allocated = os.environ.get('CUDA_VISIBLE_DEVICES')
    if not allocated:
        raise ValueError('SLURM must supply CUDA_VISIBLE_DEVICES')
    gpus = allocated.split(',')
    with parallel_eval.lock(ROOT / 'runs/.full-run-locks' / f'{args.experiment}.lock'):
        current = fingerprint()
        continue_success_context.preserve(ROOT, suite, current)
        manifests = ROOT / 'manifests' / args.experiment / PROTOCOL
        tasks = args.tasks or [t for t in suite['tasks'] if (manifests / (t + '.progress.json')).exists()
                               and not (manifests / (t + '.json')).exists()]
        if not tasks:
            print('PARALLEL_FREEZE: no unfinished task with saved progress', flush=True)
            return
        for task in tasks:
            if task not in task_names():
                raise ValueError(f'Unknown task: {task}')
            out = manifests / (task + '.json')
            if out.exists():
                print(f'PARALLEL_FREEZE: {task} already frozen', flush=True)
                continue
            config = json.loads(out.with_suffix('.progress.json').read_text())
            if seal({k: v for k, v in config.items() if k != 'sha256'})['sha256'] != config['sha256']:
                raise ValueError(f'Partial manifest was modified: {task}')
            if config['provenance'] != current:
                raise ValueError(f'Saved progress provenance differs from the current runtime: {task}')
            if (config['task'], config['protocol'], config['start_seed'], config['requested_episodes'], config['instruction_type']) != (
                    task, PROTOCOL, suite['seed'], suite['episodes'], suite['instruction_type']):
                raise ValueError(f'Saved progress does not match the suite: {task}')
            episode_args = argparse.Namespace(protocol=PROTOCOL, only=config['only'], without=config['without'])
            print(f'PARALLEL_FREEZE: {task} accepted={len(config["episodes"])} rejected={len(config["rejected"])} '
                  f'workers={args.workers} gpus={gpus}', flush=True)
            config.pop('sha256', None)
            finish_task(task, out, config, gpus, args.workers, args.timeout, args.max_candidates, episode_args)


if __name__ == '__main__':
    main()
