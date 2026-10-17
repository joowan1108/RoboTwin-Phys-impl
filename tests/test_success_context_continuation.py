"""The v6 pool is advanced to the success-context source without replay, with evidence."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

from robotwin_phys.conditions import digest, sample
from robotwin_phys.manifest import atomic_json, seal

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location('continue_success_context', ROOT / 'tools/continue_success_context.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def fingerprint(worker, context=None):
    files = {c.WORKER: worker, 'src/robotwin_phys/simulator.py': 'unchanged',
             'vendor/RoboTwin/envs/open_laptop.py': 'official', 'configs/physical_domains.json': 'same'}
    if context is not None:
        files[c.CONTEXT] = context
    return dict(files=files, source_sha256=digest(files), gpu={'name': 'PRO 6000'},
                curobo={'compatibility_fix': 'consistent-batch-results-v2'}, assets={'sha256': 'same'})


BEFORE = fingerprint(c.BEFORE_WORKER)
AFTER = fingerprint(c.AFTER_WORKER, c.CONTEXT_SHA256)


def setup(tmp_path, tasks=('open_laptop', 'adjust_bottle'), frozen=True, sealed=True, runs=()):
    suite = dict(name=c.EXPERIMENT, tasks=list(tasks), models=['fastwam', 'pi05'], episodes=1,
                 seed=100000, instruction_type='unseen', protocols=['physical_random'])
    folder = tmp_path / 'manifests' / c.EXPERIMENT / 'physical_random'
    manifests = {}
    for task in tasks:
        condition = sample(task, 100000)
        episode = dict(episode_index=0, seed=100000, condition=condition, condition_sha256=digest(condition),
                       initial_state={'entities': []}, initial_state_sha256=digest({'entities': []}),
                       instruction='Open it', expert_success=True, expert_info={'info': {'{a}': 'left'}})
        record = seal(dict(schema='robotwin-phys-local-v1', implementation='independent-paper-reproduction',
                           status='verified' if frozen else 'building', task=task, protocol='physical_random',
                           start_seed=100000, requested_episodes=1, instruction_type='unseen', only=[], without=[],
                           provenance=BEFORE, episodes=[episode], rejected=[]))
        atomic_json(folder / (task + ('.json' if frozen else '.progress.json')), record)
        manifests[task] = record
    if sealed and frozen:
        atomic_json(folder.parent / 'pool-index.json', seal(dict(schema='robotwin-shared-pool-v1', suite_sha256='s',
                    protocol='physical_random', episodes_per_task=1,
                    manifests={t: m['sha256'] for t, m in manifests.items()})))
    for model, task in runs:
        atomic_json(tmp_path / 'runs' / c.EXPERIMENT / model / 'physical_random' / task / 'run.json',
                    dict(mode='evaluate', manifest_sha256=manifests[task]['sha256'], policy={'name': f'{model}_robotwin2'}))
    return suite, folder, manifests


def test_sealed_pool_and_started_runs_are_advanced_with_backups_and_receipts(tmp_path):
    suite, folder, manifests = setup(tmp_path, runs=[('fastwam', 'open_laptop')])
    originals = {p: p.read_bytes() for p in folder.glob('*.json')}
    c.preserve(tmp_path, suite, AFTER)
    index = json.loads((folder.parent / 'pool-index.json').read_text())
    for task, old in manifests.items():
        new = json.loads((folder / (task + '.json')).read_text())
        assert new['provenance'] == AFTER and new['episodes'] == old['episodes'] and new['sha256'] != old['sha256']
        assert index['manifests'][task] == new['sha256']
    assert index['sha256'] == seal({k: v for k, v in index.items() if k != 'sha256'})['sha256']
    run = json.loads((tmp_path / 'runs' / c.EXPERIMENT / 'fastwam/physical_random/open_laptop/run.json').read_text())
    assert run['manifest_sha256'] == index['manifests']['open_laptop']
    audit = tmp_path / 'reports/pool-preservation' / c.EXPERIMENT / c.RULE
    backups = {p.read_bytes() for p in (audit / 'originals').iterdir()}
    assert set(originals.values()) <= backups
    receipts = [json.loads(p.read_text()) for p in audit.glob('*.receipt.json')]
    assert len(receipts) == 4 and all(r['expert_reexecuted'] is False and r['rule'] == c.RULE for r in receipts)
    # Rerunning is a no-op.
    snapshot = {p: p.read_bytes() for p in folder.rglob('*.json')}
    c.preserve(tmp_path, suite, AFTER)
    assert snapshot == {p: p.read_bytes() for p in folder.rglob('*.json')}


def test_unsealed_progress_is_advanced_too(tmp_path):
    suite, folder, manifests = setup(tmp_path, frozen=False, sealed=False)
    c.preserve(tmp_path, suite, AFTER)
    for task in suite['tasks']:
        assert json.loads((folder / (task + '.progress.json')).read_text())['provenance'] == AFTER
    assert not (folder.parent / 'pool-index.json').exists()


@pytest.mark.parametrize('change', ['gpu', 'simulator', 'worker', 'extra'])
def test_unrelated_changes_are_rejected_before_any_write(tmp_path, change):
    suite, folder, manifests = setup(tmp_path)
    current = copy.deepcopy(AFTER)
    if change == 'gpu':
        current['gpu'] = {'name': 'RTX 5090'}
    elif change == 'simulator':
        current['files']['src/robotwin_phys/simulator.py'] = 'edited'
    elif change == 'worker':
        current['files'][c.WORKER] = 'other'
    else:
        current['files']['src/robotwin_phys/new_module.py'] = 'added'
    current['source_sha256'] = digest(current['files'])
    before = {p: p.read_bytes() for p in folder.rglob('*.json')}
    with pytest.raises(ValueError):
        c.preserve(tmp_path, suite, current)
    assert before == {p: p.read_bytes() for p in folder.rglob('*.json')}


def test_index_or_run_pinned_elsewhere_is_refused(tmp_path):
    suite, folder, manifests = setup(tmp_path, runs=[('fastwam', 'adjust_bottle')])
    meta = tmp_path / 'runs' / c.EXPERIMENT / 'fastwam/physical_random/adjust_bottle/run.json'
    run = json.loads(meta.read_text())
    run['manifest_sha256'] = 'different'
    atomic_json(meta, run)
    with pytest.raises(ValueError, match='unknown manifest'):
        c.preserve(tmp_path, suite, AFTER)
    index = folder.parent / 'pool-index.json'
    saved = json.loads(index.read_text())
    saved['manifests']['adjust_bottle'] = 'tampered'
    atomic_json(index, saved)
    with pytest.raises(ValueError, match='does not match'):
        c.preserve(tmp_path, suite, AFTER)


def test_packaged_archive_pins_the_tree_and_only_the_reviewed_source_changes():
    name = 'robotwin-success-context-fix-v7'
    with zipfile.ZipFile(ROOT / f'reports/{name}.zip') as archive, \
            zipfile.ZipFile(ROOT / 'reports/robotwin-fresh-pool-v2.zip') as base:
        manifest = json.loads(archive.read(name + '/manifest.json'))
        assert manifest['source_changes'] == ['robotwin_phys/success_context.py', 'robotwin_phys/worker.py']
        assert manifest['files'][c.WORKER.removeprefix('src/')] == c.AFTER_WORKER
        assert manifest['files'][c.CONTEXT.removeprefix('src/')] == c.CONTEXT_SHA256
        prior = json.loads(base.read('robotwin-fresh-pool-v2/manifest.json'))
        assert prior['files']['robotwin_phys/worker.py'] == c.BEFORE_WORKER
        for base_dir, entries in (('src', manifest['files']), ('tools', manifest['tools'])):
            for relative, checksum in entries.items():
                payload = archive.read(f'{name}/{base_dir}/{relative}')
                assert hashlib.sha256(payload).hexdigest() == checksum == hashlib.sha256((ROOT / base_dir / relative).read_bytes()).hexdigest()
        for vendor in ('transforms.py', 'planner.py', 'ROBOTWIN_LICENSE'):
            assert archive.read(f'{name}/vendor/{vendor}') == base.read(f'robotwin-fresh-pool-v2/vendor/{vendor}')
