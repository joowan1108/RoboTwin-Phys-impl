"""Parallel task scheduling keeps each task's candidate stream and saved pools."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import threading

import pytest

from robotwin_phys.conditions import digest, sample
from robotwin_phys.manifest import atomic_json, seal

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location('pool_scheduler', ROOT / 'tools/parallel_eval.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


@pytest.mark.parametrize('gpu_ids', [['GPU-b', 'GPU-a'], ['5'], ['GPU-a', 'GPU-b', 'GPU-c']])
def test_freeze_uses_allocated_gpus_without_reserving_policy_device(gpu_ids):
    devices = {gpu: {'name': 'NVIDIA RTX PRO 6000'} for gpu in gpu_ids}
    assert p.select_freeze_gpus(devices) == (gpu_ids, None)
    assert p.freeze_slots(gpu_ids, 2) == [gpu for gpu in gpu_ids for _ in range(2)]
    with pytest.raises(ValueError, match='outside'):
        p.select_freeze_gpus(devices, ['host-gpu'])
    with pytest.raises(ValueError, match='distinct'):
        p.select_freeze_gpus(devices, [gpu_ids[0], gpu_ids[0]])
    with pytest.raises(ValueError, match='positive'):
        p.freeze_slots(gpu_ids, 0)


def test_two_gpu_discovery_probes_each_allocated_gpu_individually(monkeypatch):
    seen = []

    def probe(argv, env, text):
        seen.append(env['CUDA_VISIBLE_DEVICES'])
        return json.dumps({'name': 'NVIDIA RTX PRO 6000', 'memory_gib': 95})

    monkeypatch.setattr(p.subprocess, 'check_output', probe)
    assert list(p.discover_gpus({'CUDA_VISIBLE_DEVICES': 'GPU-b,GPU-a'})) == ['GPU-b', 'GPU-a']
    assert seen == ['GPU-b', 'GPU-a']


def test_fresh_experiment_generates_all_fifty_tasks_without_reading_old_pools(tmp_path, monkeypatch):
    from robotwin_phys.conditions import task_names
    tasks = task_names()
    assert len(tasks) == 50
    suite = fixture_suite(tmp_path, monkeypatch)
    suite['name'], suite['tasks'] = 'phys-pro6000-100-v5', tasks
    for version in (2, 3, 4):
        for task in tasks:
            atomic_json(tmp_path / f'manifests/phys-pro6000-100-v{version}/physical_random/{task}.json',
                        {'do_not_read_or_reverify': True})
    read_text = Path.read_text

    def guarded_read(path, *args, **kwargs):
        if any(f'phys-pro6000-100-v{v}' in path.parts for v in (2, 3, 4)):
            pytest.fail(f'Fresh pool accessed previous experiment: {path}')
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', guarded_read)
    attempted = []

    class Processes:
        cancelled = threading.Event()

        def run(self, argv, env, logfile):
            assert argv[1:4] == ['-m', 'robotwin_phys.cli', 'freeze']
            assert '--source' not in argv and env['CUDA_VISIBLE_DEVICES'] == 'GPU-a'
            assert argv[argv.index('--seed') + 1] == 100000
            task = argv[argv.index('--task') + 1]
            attempted.append(task)
            write_manifest(suite, task)

        def close(self):
            self.cancelled.set()

    args = argparse.Namespace(max_candidates=10000, timeout=1800, freeze_workers_per_gpu=4)
    p.freeze_pool(Processes(), args, suite, {}, ['GPU-a'], [{'test': True}], tmp_path / 'logs')
    assert sorted(attempted) == tasks
    manifests = p.seal_pool(suite, tmp_path / 'manifests/phys-pro6000-100-v5/pool-index.json')
    assert sorted(manifests) == tasks


def fixture_suite(tmp_path, monkeypatch):
    monkeypatch.setattr(p, 'ROOT', tmp_path)
    monkeypatch.setattr(p.experiment, 'ROOT', tmp_path)
    tasks = ['adjust_bottle', 'grab_roller', 'stack_blocks_two', 'beat_block_hammer', 'open_microwave']
    return seal(dict(schema='robotwin-experiment-v1', name='shared', tasks=tasks,
                     protocols=['physical_random'], models=list(p.experiment.MODELS), episodes=2,
                     seed=100000, policy_seed=0, instruction_type='unseen'))


def write_manifest(suite, task):
    episodes = []
    for i, seed in enumerate((100000, 100002)):
        condition = sample(task, seed)
        initial = {'entities': [{'name': task}]}
        episodes.append(dict(episode_index=i, seed=seed, condition=condition,
                             condition_sha256=digest(condition), expert_success=True,
                             initial_state=initial, initial_state_sha256=digest(initial), instruction=task))
    path = p.experiment.manifest_path(suite, 'physical_random', task)
    atomic_json(path, seal(dict(schema='robotwin-phys-local-v1', status='verified', task=task,
                               protocol='physical_random', start_seed=100000, requested_episodes=2,
                               instruction_type='unseen', only=[], without=[], provenance={'test': True},
                               episodes=episodes)))
    return path


def test_four_simultaneous_tasks_resume_and_share_one_sealed_pool(tmp_path, monkeypatch):
    suite = fixture_suite(tmp_path, monkeypatch)
    complete = write_manifest(suite, suite['tasks'][0])
    saved = complete.read_bytes()
    started = []
    guard = threading.Lock()
    barrier = threading.Barrier(4)

    class Processes:
        cancelled = threading.Event()

        def run(self, argv, env, logfile):
            task = argv[argv.index('--task') + 1]
            assert '--resume' in argv and argv[argv.index('--seed') + 1] == 100000
            with guard:
                started.append((env['CUDA_VISIBLE_DEVICES'], task))
            barrier.wait(timeout=5)  # All four tasks must be active concurrently.
            write_manifest(suite, task)

        def close(self):
            self.cancelled.set()

    args = argparse.Namespace(max_candidates=10000, timeout=1800, freeze_workers_per_gpu=2)
    p.freeze_pool(Processes(), args, suite, {}, ['GPU-b', 'GPU-a'], [{'test': True}], tmp_path / 'logs')
    assert len(started) == len({task for gpu, task in started}) == 4
    assert [gpu for gpu, task in started].count('GPU-a') == 2
    assert [gpu for gpu, task in started].count('GPU-b') == 2
    assert complete.read_bytes() == saved
    index = tmp_path / 'manifests/shared/pool-index.json'
    manifests = p.seal_pool(suite, index)
    assert set(manifests) == set(suite['tasks'])
    assert all([e['seed'] for e in m['episodes']] == [100000, 100002] for m in manifests.values())
    assert p.verify_pool(suite, index) == manifests


def test_freeze_only_controller_never_starts_or_downloads_policy(tmp_path, monkeypatch):
    suite = fixture_suite(tmp_path, monkeypatch)
    for task in suite['tasks']:
        write_manifest(suite, task)
    atomic_json(tmp_path / 'runs/shared/suite.json', suite)
    calls = []

    class Processes:
        def __init__(self):
            self.cancelled = threading.Event()

        def run(self, argv, *a, **kw):
            calls.append([str(x) for x in argv])

        def close(self):
            self.cancelled.set()

    monkeypatch.setattr(p, 'Processes', Processes)
    monkeypatch.setattr(p, 'task_names', lambda: suite['tasks'])
    monkeypatch.setattr(p, 'discover_gpus', lambda base: {
        'GPU-a': {'name': 'RTX PRO 6000'}, 'GPU-b': {'name': 'RTX PRO 6000'}})
    monkeypatch.setattr(p, 'provenance_on_gpu', lambda *a: {'test': True})
    monkeypatch.setattr(p.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(p, 'evaluate_pool', lambda *a: pytest.fail('Freeze must not evaluate a policy'))
    monkeypatch.setattr(sys, 'argv', ['parallel_eval.py', '--experiment', 'shared', '--episodes', '2',
                                     '--phase', 'freeze', '--freeze-workers-per-gpu', '2'])
    p.main()
    assert not any('--models' in call for call in calls)
    assert sum(any('cluster_preflight.py' in arg for arg in call) for call in calls) == 2
    assert (tmp_path / 'manifests/shared/pool-index.json').exists()


def test_partial_manifest_resumes_after_interrupt_without_replacing_accepted_seeds(tmp_path, monkeypatch):
    from robotwin_phys import cli, provenance
    monkeypatch.setattr(cli, 'task_names', lambda: ['adjust_bottle'])
    monkeypatch.setattr(provenance, 'fingerprint', lambda: {'test': True})
    requested = []

    def worker(request, *args):
        episode = request['episode']
        seed = episode['seed']
        requested.append(seed)
        if seed == 100001:
            return {'status': 'rejected', 'reason': 'unstable_initialization'}
        if seed == 100002 and len(requested) == 3:
            raise RuntimeError('Interrupted current candidate')
        initial = {'entities': []}
        return {'status': 'verified', 'episode': dict(episode, expert_success=True, initial_state=initial,
                initial_state_sha256=digest(initial), instruction='Adjust bottle')}

    monkeypatch.setattr(cli, 'worker', worker)
    output = tmp_path / 'adjust_bottle.json'
    args = argparse.Namespace(output=str(output), task='adjust_bottle', protocol='physical_random',
                              episodes=2, max_candidates=4, seed=100000, instruction_type='unseen',
                              only=None, without=None, resume=False, live=False, port=8765, viewer=False,
                              video=False, timeout=1800)
    with pytest.raises(RuntimeError, match='Interrupted'):
        cli.freeze(args)
    progress = json.loads(output.with_suffix('.progress.json').read_text())
    accepted = progress['episodes'][0]
    args.resume = True
    cli.freeze(args)
    manifest = json.loads(output.read_text())
    assert requested == [100000, 100001, 100002, 100002]
    assert manifest['episodes'][0] == accepted
    assert [e['seed'] for e in manifest['episodes']] == [100000, 100002]
    assert [e['seed'] for e in manifest['rejected']] == [100001]


def test_one_task_error_does_not_cancel_other_tasks_or_seal_incomplete_pool(tmp_path, monkeypatch):
    suite = fixture_suite(tmp_path, monkeypatch)
    attempted = []

    class Processes:
        cancelled = threading.Event()

        def run(self, argv, env, logfile):
            task = argv[argv.index('--task') + 1]
            attempted.append(task)
            if task == 'beat_block_hammer':
                logfile.parent.mkdir(parents=True, exist_ok=True)
                logfile.write_text('RuntimeError: actual worker error\n')
                raise RuntimeError('Worker error')
            write_manifest(suite, task)

        def close(self):
            self.cancelled.set()

    processes = Processes()
    args = argparse.Namespace(max_candidates=10000, timeout=1800, freeze_workers_per_gpu=1)
    logs = tmp_path / 'logs'
    with pytest.raises(RuntimeError, match=r'1 task\(s\)'):
        p.freeze_pool(processes, args, suite, {}, ['GPU-a'], [{'test': True}], logs)
    assert attempted == suite['tasks']
    assert not processes.cancelled.is_set()
    errors = json.loads((logs / 'freeze-errors.json').read_text())
    assert errors[0]['task'] == 'beat_block_hammer' and 'actual worker error' in errors[0]['log_tail']
    for task in suite['tasks']:
        assert p.experiment.manifest_path(suite, 'physical_random', task).exists() == (task != 'beat_block_hammer')
    assert not (tmp_path / 'manifests/shared/pool-index.json').exists()


def test_reverification_prefers_latest_saved_pool_and_falls_back_per_task(tmp_path, monkeypatch):
    suite = fixture_suite(tmp_path, monkeypatch)
    suite['name'] = 'v4'
    suite['tasks'] = ['adjust_bottle', 'beat_block_hammer']
    first = tmp_path / 'manifests/v3/physical_random/adjust_bottle.progress.json'
    second = tmp_path / 'manifests/v2/physical_random/beat_block_hammer.json'
    atomic_json(first, {})
    atomic_json(second, {})
    chosen = []

    class Processes:
        cancelled = threading.Event()

        def run(self, argv, env, logfile):
            assert '--source' in argv
            source = Path(argv[argv.index('--source') + 1])
            target = Path(argv[argv.index('--target') + 1])
            chosen.append(source)
            write_manifest(suite, target.stem)

        def close(self):
            self.cancelled.set()

    args = argparse.Namespace(max_candidates=10000, timeout=1800, freeze_workers_per_gpu=1,
                              resume_from_experiment='v3', fallback_experiment=['v2'])
    p.freeze_pool(Processes(), args, suite, {}, ['GPU-a'], [{'test': True}], tmp_path / 'logs')
    assert chosen == [first.parent / 'adjust_bottle.json', second]
