"""export_release publishes the pool and results; verify_release reproduces every condition from its seed."""
import csv
import importlib.util
import json
from pathlib import Path

import pytest

from robotwin_phys.conditions import digest, sample
from robotwin_phys.manifest import atomic_json, seal

ROOT = Path(__file__).parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f'analysis/{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


e, v = load('export_release'), load('verify_release')
EXP = 'exp-test'


def make_data(tmp_path, tasks=('adjust_bottle', 'open_laptop'), episodes=3):
    manifests = tmp_path / 'manifests' / EXP / 'physical_random'
    runs = tmp_path / 'runs' / EXP
    sealed = {}
    for task in tasks:
        accepted, rejected, seed = [], [], 100000
        while len(accepted) < episodes:
            if seed % 3 == 1:
                rejected.append({'seed': seed, 'status': 'rejected', 'reason': 'expert_failed', 'plan_success': False})
            else:
                cond = sample(task, seed)
                state = {'entities': [{'name': 'bottle', 'pose': [seed, 0, 0]}]}
                accepted.append(dict(episode_index=len(accepted), seed=seed, condition=cond, condition_sha256=digest(cond),
                                     expert_success=True, initial_state=state, initial_state_sha256=digest(state),
                                     instruction='Adjust it', expert_info={'info': {}}, expert_seconds=3.5))
            seed += 1
        record = seal(dict(schema='robotwin-phys-local-v1', implementation='independent-paper-reproduction', task=task,
                           protocol='physical_random', start_seed=100000, requested_episodes=episodes, instruction_type='unseen',
                           only=[], without=[], status='verified', provenance={'source_sha256': 'abc', 'gpu': {'name': 'PRO 6000'}},
                           episodes=accepted, rejected=rejected))
        atomic_json(manifests / f'{task}.json', record)
        sealed[task] = record
    atomic_json(manifests.parent / 'pool-index.json', seal(dict(schema='robotwin-shared-pool-v1', suite_sha256='s',
                protocol='physical_random', episodes_per_task=episodes, manifests={t: m['sha256'] for t, m in sealed.items()})))
    atomic_json(runs / 'suite.json', seal(dict(schema='robotwin-experiment-v1', name=EXP, tasks=list(tasks), models=['fastwam'],
                                               episodes=episodes, seed=100000, protocols=['physical_random'])))
    identity = {'name': 'fastwam_robotwin2', 'config_sha256': 'c', 'adapter_sha256': 'a'}
    for task in tasks:
        folder = runs / 'fastwam/physical_random' / task
        atomic_json(folder / 'run.json', dict(mode='evaluate', manifest_sha256=sealed[task]['sha256'], policy=identity))
        for ep in sealed[task]['episodes']:
            i = ep['episode_index']
            atomic_json(folder / f'episode-{i:04d}.result.json', dict(status='completed', success=i % 2 == 0, actions=400 + i,
                        seconds=55.5, condition_sha256=ep['condition_sha256'], policy=identity, benchmark_eligible=True))
    (runs / 'tables').mkdir()
    (runs / 'tables/summary.csv').write_text('model,success_rate\nfastwam,0.5\n')
    return sealed


def test_export_then_verify_roundtrip(tmp_path, capsys):
    sealed = make_data(tmp_path)
    out = e.export(tmp_path, EXP, tmp_path / 'release' / EXP, 'physical_random')
    assert (tmp_path / 'release' / f'{EXP}.zip').exists()
    seeds = json.loads((out / 'pool/seeds.json').read_text())['tasks']
    assert seeds['adjust_bottle']['accepted'] == [100001, 100002, 100004] and seeds['adjust_bottle']['rejected'] == {'100000': 'expert_failed', '100003': 'expert_failed'}
    with (out / 'pool/conditions.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 6 and list(rows[0]) == e.COLUMNS and rows[0]['instruction'] == 'Adjust it'
    with (out / 'results/fastwam/episodes.csv').open(newline='') as f:
        res = list(csv.DictReader(f))
    assert [r['success'] for r in res] == ['1', '0', '1'] * 2 and res[0]['seed'] == '100001'
    identity = json.loads((out / 'results/fastwam/identity.json').read_text())
    assert identity['manifests']['open_laptop'] == sealed['open_laptop']['sha256'] and identity['policy']['name'] == 'fastwam_robotwin2'
    prov = json.loads((out / 'provenance.json').read_text())
    assert prov['runtime_fingerprint_is_uniform'] and prov['pool_counts']['adjust_bottle'] == dict(accepted=3, rejected=2, candidates=5)
    assert (out / 'tables/summary.csv').exists() and (out / 'full/pool-index.json').exists() and (out / 'full/suite.json').exists()
    summary = (out / 'summary.md').read_text()
    assert '| adjust_bottle | 67% (2/3) |' in summary and 'mean of task rates' in summary
    checks = v.verify(out)
    assert checks == {'conditions_reproduced': 6, 'tasks': 2, 'manifests_sealed': 2, 'pool_index_sealed': 1, 'results_fastwam': 6}
    assert v.main([str(out)]) == 0 and 'OK' in capsys.readouterr().out


@pytest.mark.parametrize('tamper', ['condition_column', 'seed', 'manifest', 'result_episode'])
def test_verify_detects_tampering(tmp_path, tamper, capsys):
    make_data(tmp_path)
    out = e.export(tmp_path, EXP, tmp_path / 'release' / EXP, 'physical_random')
    if tamper in ('condition_column', 'seed'):
        path = out / 'pool/conditions.csv'
        with path.open(newline='') as f:
            rows = list(csv.DictReader(f))
        if tamper == 'condition_column':
            rows[2]['mass_scale'] = str(float(rows[2]['mass_scale']) + 0.01)
        else:
            rows[2]['seed'] = '100009'
        with path.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=e.COLUMNS)
            w.writeheader()
            w.writerows(rows)
    elif tamper == 'manifest':
        path = out / 'full/manifests/physical_random/open_laptop.json'
        m = json.loads(path.read_text())
        m['episodes'][0]['seed'] = 999
        path.write_text(json.dumps(m))
    else:
        path = out / 'results/fastwam/episodes.csv'
        path.write_text(path.read_text().replace('adjust_bottle,2,', 'adjust_bottle,7,'))
    assert v.main([str(out)]) == 1 and 'FAIL' in capsys.readouterr().out


def test_export_refuses_result_whose_condition_differs_from_manifest(tmp_path):
    make_data(tmp_path)
    path = tmp_path / 'runs' / EXP / 'fastwam/physical_random/adjust_bottle/episode-0001.result.json'
    r = json.loads(path.read_text())
    r['condition_sha256'] = 'other'
    path.write_text(json.dumps(r))
    with pytest.raises(ValueError, match='condition hash differs'):
        e.export(tmp_path, EXP, tmp_path / 'release' / EXP, 'physical_random')


def test_discarded_run_folders_are_skipped_and_refresh_rewrites_summary(tmp_path):
    sealed = make_data(tmp_path)
    folder = tmp_path / 'runs' / EXP / 'fastwam-discarded/physical_random/adjust_bottle'
    e.shutil.copytree(tmp_path / 'runs' / EXP / 'fastwam/physical_random/adjust_bottle', folder)
    out = e.export(tmp_path, EXP, tmp_path / 'release' / EXP, 'physical_random')
    assert sorted(p.name for p in (out / 'results').iterdir()) == ['fastwam']
    assert list(json.loads((out / 'provenance.json').read_text())['policy_identities']) == ['fastwam']
    summary = (out / 'summary.md').read_text()
    assert 'discarded' not in summary and 'paper fastwam' in summary and '| adjust_bottle | 67% (2/3) | 64% |' in summary
    (out / 'summary.md').write_text('stale')
    e.main(['--refresh', str(out)])
    assert (out / 'summary.md').read_text() == summary
    # Without the large manifests (GitHub layout) the remaining checks still pass.
    e.shutil.rmtree(out / 'full/manifests')
    assert v.verify(out)['manifests_sealed'] == 0 and v.verify(out)['pool_index_sealed'] == 1
