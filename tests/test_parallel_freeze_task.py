"""Concurrent seed scanning writes the same records as the sequential freeze loop."""
import argparse
import importlib.util
import json
from pathlib import Path
import os
import sys

import pytest

from robotwin_phys import cli
from robotwin_phys.conditions import digest, sample
from robotwin_phys.manifest import atomic_json, seal

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location('parallel_freeze_task', ROOT / 'tools/parallel_freeze_task.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)

TASK = 'adjust_bottle'
# Deterministic outcome per seed: accept every third seed, make one seed unstable, reject the rest.
def outcome(seed):
    if seed % 7 == 3:
        return {'status': 'rejected', 'reason': 'unstable_initialization', 'details': f'seed({seed})'}
    if seed % 3 == 0:
        return 'verified'
    return {'status': 'rejected', 'reason': 'expert_failed', 'plan_success': False}


def fake_worker_script(tmp_path):
    script = tmp_path / 'fake_worker.py'
    script.write_text('''
import json, sys, time
req = json.load(open(sys.argv[1])); ep = req["episode"]; seed = ep["seed"]
time.sleep(0.01 * (seed % 5))  # finish out of order
if seed % 7 == 3:
    out = {"status": "rejected", "reason": "unstable_initialization", "details": f"seed({seed})"}
elif seed % 3 == 0:
    out = {"status": "verified", "episode": {**ep, "expert_success": True, "initial_state": {"entities": [seed]},
           "initial_state_sha256": "h%d" % seed, "instruction": "do it", "expert_info": {"info": {}}, "expert_seconds": 1.0}}
else:
    out = {"status": "rejected", "reason": "expert_failed", "plan_success": False}
json.dump(out, open(sys.argv[2], "w"))
''')
    return script


@pytest.fixture
def fake_worker(tmp_path, monkeypatch):
    script = fake_worker_script(tmp_path)
    monkeypatch.setattr(p, 'WORKER_COMMAND', [sys.executable, str(script)])
    monkeypatch.setattr(p, 'base_env', lambda gpu: dict(FAKE_GPU=gpu, PATH=os.environ.get('PATH', '')))
    return script


def sequential_reference(tmp_path, episodes, start_seed, prior=None):
    """What robotwin_phys.cli freeze would write, using the same deterministic outcomes."""
    out = tmp_path / 'reference' / f'{TASK}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(task=TASK, protocol='physical_random', episodes=episodes, seed=start_seed, max_candidates=10000,
                              only=None, without=None, instruction_type='unseen', output=str(out), resume=prior is not None,
                              live=False, port=8765, viewer=False, video=False, timeout=5)
    if prior is not None:
        atomic_json(out.with_suffix('.progress.json'), seal(prior))

    def worker(request, folder, label, timeout):
        ep = request['episode']; seed = ep['seed']
        o = outcome(seed)
        if o == 'verified':
            return {'status': 'verified', 'episode': {**ep, 'expert_success': True, 'initial_state': {'entities': [seed]},
                    'initial_state_sha256': 'h%d' % seed, 'instruction': 'do it', 'expert_info': {'info': {}}, 'expert_seconds': 1.0}}
        return o

    original = cli.worker
    cli.worker = worker
    try:
        import robotwin_phys.provenance as prov
        saved = prov.fingerprint
        prov.fingerprint = lambda: {'test': True}
        try:
            cli.freeze(args, verify=True)
        finally:
            prov.fingerprint = saved
    finally:
        cli.worker = original
    return json.loads(out.read_text())


def progress_record(start_seed, episodes, accepted_seeds=(), rejected_seeds=()):
    rec = dict(schema='robotwin-phys-local-v1', implementation='independent-paper-reproduction', task=TASK,
               protocol='physical_random', start_seed=start_seed, requested_episodes=episodes, instruction_type='unseen',
               only=[], without=[], status='building', provenance={'test': True}, episodes=[], rejected=[])
    for i, seed in enumerate(accepted_seeds):
        cond = sample(TASK, seed)
        rec['episodes'].append(dict(episode_index=i, seed=seed, condition=cond, condition_sha256=digest(cond), expert_success=True,
                                    initial_state={'entities': [seed]}, initial_state_sha256='h%d' % seed, instruction='do it',
                                    expert_info={'info': {}}, expert_seconds=1.0))
    for seed in rejected_seeds:
        rec['rejected'].append({'seed': seed, **outcome(seed)})
    return rec


@pytest.mark.parametrize('workers', [1, 3, 8])
def test_parallel_scan_matches_sequential_records_exactly(tmp_path, fake_worker, workers):
    start, episodes = 100000, 6
    reference = sequential_reference(tmp_path, episodes, start)
    out = tmp_path / 'manifests' / f'{TASK}.json'
    out.parent.mkdir(parents=True)
    config = progress_record(start, episodes)
    atomic_json(out.with_suffix('.progress.json'), seal(config))
    p.finish_task(TASK, out, config, ['GPU-a', 'GPU-b'], workers, 5, 10000,
                  argparse.Namespace(protocol='physical_random', only=[], without=[]))
    produced = json.loads(out.read_text())
    assert produced == reference
    seeds = sorted(e['seed'] for e in produced['episodes']) + sorted(r['seed'] for r in produced['rejected'])
    assert sorted(seeds) == list(range(start, start + len(seeds)))
    assert [e['episode_index'] for e in produced['episodes']] == list(range(episodes))
    folder = out.parent / f'{TASK}-verification'
    evidence = sorted(int(x.name[len('seed-'):-len('.result.json')]) for x in folder.glob('seed-*.result.json'))
    assert evidence == list(range(start, start + len(seeds)))  # nothing beyond the completing seed is kept
    for ep in produced['episodes']:
        assert json.loads((folder / f"seed-{ep['seed']}.result.json").read_text())['episode']['episode_index'] == ep['episode_index']
        assert json.loads((folder / f"seed-{ep['seed']}.request.json").read_text())['episode']['episode_index'] == ep['episode_index']


def test_resumes_from_saved_prefix_and_appends_consecutive_indices(tmp_path, fake_worker):
    start, episodes = 100000, 5
    # Consistent with outcome(): 100002 accepted, the others rejected; the scan resumes at 100005.
    prior = progress_record(start, episodes, accepted_seeds=[100002], rejected_seeds=[100000, 100001, 100003, 100004])
    reference = sequential_reference(tmp_path, episodes, start, prior=prior)
    out = tmp_path / 'manifests' / f'{TASK}.json'
    out.parent.mkdir(parents=True)
    atomic_json(out.with_suffix('.progress.json'), seal(prior))
    p.finish_task(TASK, out, dict(prior), ['GPU-a'], 4, 5, 10000, argparse.Namespace(protocol='physical_random', only=[], without=[]))
    assert json.loads(out.read_text()) == reference


def test_worker_error_aborts_without_touching_saved_progress(tmp_path, fake_worker, monkeypatch):
    start, episodes = 100000, 3
    out = tmp_path / 'manifests' / f'{TASK}.json'
    out.parent.mkdir(parents=True)
    config = progress_record(start, episodes)
    atomic_json(out.with_suffix('.progress.json'), seal(config))
    before = out.with_suffix('.progress.json').read_bytes()
    crash = tmp_path / 'crash.py'
    crash.write_text('import sys; print("boom"); sys.exit(3)')
    monkeypatch.setattr(p, 'WORKER_COMMAND', [sys.executable, str(crash)])
    with pytest.raises(RuntimeError, match='not excluded as infeasible'):
        p.finish_task(TASK, out, config, ['GPU-a'], 2, 5, 10000, argparse.Namespace(protocol='physical_random', only=[], without=[]))
    assert out.with_suffix('.progress.json').read_bytes() == before and not out.exists()


def test_budget_exhaustion_is_an_error_not_a_partial_manifest(tmp_path, fake_worker):
    out = tmp_path / 'manifests' / f'{TASK}.json'
    out.parent.mkdir(parents=True)
    config = progress_record(100000, 50)
    atomic_json(out.with_suffix('.progress.json'), seal(config))
    with pytest.raises(RuntimeError, match='Candidate budget exhausted'):
        p.finish_task(TASK, out, config, ['GPU-a'], 4, 5, 10, argparse.Namespace(protocol='physical_random', only=[], without=[]))
    assert not out.exists()
