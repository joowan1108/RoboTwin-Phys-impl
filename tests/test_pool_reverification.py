"""Upgrade saved pools only after fresh expert validation and matching states."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

from robotwin_phys.conditions import digest, sample
from robotwin_phys.manifest import atomic_json, seal

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location('reverify_pool', ROOT / 'tools/reverify_pool.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def fixture(tmp_path, complete=False):
    files = {'src/robotwin_phys/planning.py': 'old-planning',
             'vendor/RoboTwin/envs/utils/transforms.py': 'old-transforms',
             'configs/physical_domains.json': 'unchanged'}
    previous = dict(files=files, source_sha256=digest(files), gpu={'name': 'PRO 6000'},
                    assets={'sha256': 'unchanged'}, versions={'torch': 'same'})
    current = copy.deepcopy(previous)
    current['files'].update({'src/robotwin_phys/planning.py': 'new-planning',
                            'vendor/RoboTwin/envs/utils/transforms.py': 'new-transforms',
                            'src/robotwin_phys/numerics.py': 'new-numerics'})
    current['source_sha256'] = digest(current['files'])
    episodes = []
    for index, seed in enumerate((100000, 100002)):
        condition = sample('beat_block_hammer', seed)
        initial = {'entities': [{'pose': [0., .1, .7]}]}
        episodes.append(dict(seed=seed, episode_index=index, condition=condition,
                             condition_sha256=digest(condition), expert_success=True,
                             initial_state=initial, initial_state_sha256=digest(initial), instruction='Beat block'))
    prior = seal(dict(schema='robotwin-phys-local-v1', implementation='independent-paper-reproduction',
                      task='beat_block_hammer', protocol='physical_random', start_seed=100000,
                      requested_episodes=2 if complete else 3, instruction_type='unseen', only=[], without=[],
                      episodes=episodes, rejected=[{'seed': 100001, 'status': 'rejected', 'reason': 'expert_failed'}],
                      status='verified' if complete else 'building', provenance=previous))
    source = tmp_path / 'v2/beat_block_hammer.json'
    atomic_json(source if complete else source.with_suffix('.progress.json'), prior)
    target = tmp_path / 'v3/beat_block_hammer.json'
    return source, target, prior, current, files


@pytest.mark.parametrize('complete', [False, True])
def test_saved_seed_conditions_are_reverified_before_provenance_upgrade(tmp_path, monkeypatch, complete):
    source, target, prior, current, approved = fixture(tmp_path, complete)
    old_path = source if complete else source.with_suffix('.progress.json')
    old_bytes = old_path.read_bytes()
    requests = []

    def worker(request, *args):
        requests.append(request)
        return {'status': 'verified', 'episode': request['episode']}

    monkeypatch.setattr(r, 'worker', worker)
    r.reverify(source, target, current, approved)
    migrated = json.loads(target.with_suffix('.progress.json').read_text())
    assert [req['episode']['seed'] for req in requests] == [100000, 100002]
    assert migrated['episodes'] == prior['episodes']
    assert migrated['rejected'] == prior['rejected']
    assert migrated['provenance'] == current
    assert target.exists() == complete
    assert old_path.read_bytes() == old_bytes
    assert json.loads(target.with_suffix('.reverification.json').read_text())['status'] == 'reverified'


def test_interrupt_resume_does_not_repeat_already_reverified_seed(tmp_path, monkeypatch):
    source, target, prior, current, approved = fixture(tmp_path)
    seen = []

    def worker(request, *args):
        seed = request['episode']['seed']
        seen.append(seed)
        if len(seen) == 2:
            raise RuntimeError('Interrupted')
        return {'status': 'verified', 'episode': request['episode']}

    monkeypatch.setattr(r, 'worker', worker)
    with pytest.raises(RuntimeError, match='Interrupted'):
        r.reverify(source, target, current, approved)
    assert not target.with_suffix('.progress.json').exists()
    r.reverify(source, target, current, approved)
    assert seen == [100000, 100002, 100002]
    assert len(json.loads(target.with_suffix('.progress.json').read_text())['episodes']) == 2


@pytest.mark.parametrize('failure', ['state', 'instruction'])
def test_changed_expert_outcome_or_scene_never_silently_relabels_old_pool(tmp_path, monkeypatch, failure):
    source, target, prior, current, approved = fixture(tmp_path)

    def worker(request, *args):
        episode = copy.deepcopy(request['episode'])
        if failure == 'state':
            episode['initial_state']['entities'][0]['pose'][0] = .5
        if failure == 'instruction':
            episode['instruction'] = 'Different instruction'
        return {'status': 'verified', 'episode': episode}

    monkeypatch.setattr(r, 'worker', worker)
    with pytest.raises((ValueError, RuntimeError)):
        r.reverify(source, target, current, approved)
    assert not target.exists() and not target.with_suffix('.progress.json').exists()


@pytest.mark.parametrize('reason', ['expert_failed', 'unstable_initialization'])
def test_infeasible_reverified_seed_is_removed_and_refilled_without_resetting_pool(tmp_path, monkeypatch, reason):
    from robotwin_phys import cli, provenance
    from robotwin_phys.manifest import load
    source, target, prior, current, approved = fixture(tmp_path, complete=True)
    old_bytes = source.read_bytes()
    reverified, filled = [], []

    def worker(request, *args):
        episode = request['episode']
        reverified.append(episode['seed'])
        if episode['seed'] == 100000:
            return {'status': 'rejected', 'reason': reason, 'details': 'New expert cannot execute this condition'}
        return {'status': 'verified', 'episode': episode}

    monkeypatch.setattr(r, 'worker', worker)
    r.reverify(source, target, current, approved)
    assert not target.exists()
    progress = json.loads(target.with_suffix('.progress.json').read_text())
    assert progress['status'] == 'building'
    assert [e['seed'] for e in progress['episodes']] == [100002]
    assert progress['episodes'][0]['episode_index'] == 0
    assert {e['seed'] for e in progress['rejected']} == {100000, 100001}
    assert progress['rejected'][-1]['reason'] == reason

    def fill_worker(request, *args):
        episode = request['episode']
        filled.append(episode['seed'])
        initial = {'entities': []}
        return {'status': 'verified', 'episode': dict(episode, expert_success=True,
                initial_state=initial, initial_state_sha256=digest(initial), instruction='Beat block')}

    monkeypatch.setattr(cli, 'worker', fill_worker)
    monkeypatch.setattr(cli, 'task_names', lambda: ['beat_block_hammer'])
    monkeypatch.setattr(provenance, 'fingerprint', lambda: current)
    args = __import__('argparse').Namespace(output=str(target), task='beat_block_hammer',
            protocol='physical_random', episodes=2, max_candidates=10000, seed=100000,
            instruction_type='unseen', only=None, without=None, resume=True, live=False,
            port=8765, viewer=False, video=False, timeout=30)
    cli.freeze(args)
    completed = load(target)
    assert reverified == [100000, 100002] and filled == [100003]
    assert [e['seed'] for e in completed['episodes']] == [100002, 100003]
    assert [e['episode_index'] for e in completed['episodes']] == [0, 1]
    assert source.read_bytes() == old_bytes


def test_resume_after_rejection_and_interrupt_does_not_repeat_processed_seed(tmp_path, monkeypatch):
    source, target, prior, current, approved = fixture(tmp_path)
    seen = []

    def worker(request, *args):
        seed = request['episode']['seed']
        seen.append(seed)
        if seed == 100000:
            return {'status': 'rejected', 'reason': 'expert_failed'}
        if len(seen) == 2:
            raise RuntimeError('CUDA error')
        return {'status': 'verified', 'episode': request['episode']}

    monkeypatch.setattr(r, 'worker', worker)
    with pytest.raises(RuntimeError, match='CUDA error'):
        r.reverify(source, target, current, approved)
    r.reverify(source, target, current, approved)
    assert seen == [100000, 100002, 100002]
    migrated = json.loads(target.with_suffix('.progress.json').read_text())
    assert len(migrated['episodes']) == 1 and migrated['episodes'][0]['episode_index'] == 0
    assert len([e for e in migrated['rejected'] if e['seed'] == 100000]) == 1


def test_legacy_reverification_progress_is_resumed(tmp_path, monkeypatch):
    source, target, prior, current, approved = fixture(tmp_path)
    atomic_json(target.with_suffix('.reverify.progress.json'), seal(dict(source=str(source),
                source_sha256=prior['sha256'], current_provenance_sha256=digest(current),
                episodes=prior['episodes'][:1], status='reverifying')))
    seen = []

    def worker(request, *args):
        seen.append(request['episode']['seed'])
        return {'status': 'verified', 'episode': request['episode']}

    monkeypatch.setattr(r, 'worker', worker)
    r.reverify(source, target, current, approved)
    assert seen == [100002]


@pytest.mark.parametrize('result', [{'status': 'error', 'error': 'CUDA failure'},
                                  {'status': 'rejected', 'reason': 'missing_asset'}])
def test_runtime_errors_are_never_replaced_with_other_seeds(tmp_path, monkeypatch, result):
    source, target, prior, current, approved = fixture(tmp_path)
    monkeypatch.setattr(r, 'worker', lambda *args: result)
    with pytest.raises(RuntimeError, match='runtime errors'):
        r.reverify(source, target, current, approved)
    assert not target.with_suffix('.progress.json').exists()


@pytest.mark.parametrize('key', ['gpu', 'assets', 'versions'])
def test_upgrade_requires_same_runtime_and_assets(tmp_path, key):
    source, target, prior, current, approved = fixture(tmp_path)
    current[key] = {'changed': True}
    with pytest.raises(ValueError, match='unchanged'):
        r.reverify(source, target, current, approved)


def test_unreviewed_physics_change_cannot_bypass_fingerprint(tmp_path):
    source, target, prior, current, approved = fixture(tmp_path)
    current['files']['configs/physical_domains.json'] = 'different bounds'
    current['source_sha256'] = digest(current['files'])
    with pytest.raises(ValueError, match='Unreviewed benchmark'):
        r.reverify(source, target, current, approved)


def test_reviewed_curobo_metadata_can_be_added_to_older_pool(tmp_path):
    from robotwin_phys.curobo_compat import PIN, SOURCE_SHA256, FIX
    source, target, prior, current, approved = fixture(tmp_path)
    current['curobo'] = dict(revision=PIN, motion_gen_sha256=SOURCE_SHA256, compatibility_fix=FIX)
    r.approved_upgrade(prior['provenance'], current, approved)
    current['curobo']['revision'] = 'different'
    with pytest.raises(ValueError, match='unchanged'):
        r.approved_upgrade(prior['provenance'], current, approved)


def test_reviewed_source_checksums_accept_both_prior_patch_versions(tmp_path):
    source, target, prior, current, approved = fixture(tmp_path)
    approved = {path: [checksum, 'another reviewed version'] for path, checksum in approved.items()}
    r.approved_upgrade(prior['provenance'], current, approved)
    previous = copy.deepcopy(prior['provenance'])
    previous['files']['src/robotwin_phys/planning.py'] = 'unreviewed'
    previous['source_sha256'] = digest(previous['files'])
    with pytest.raises(ValueError, match='Unreviewed'):
        r.approved_upgrade(previous, current, approved)


def test_known_helper_absent_in_v2_is_allowed_even_if_allowlist_has_v3_hash(tmp_path):
    source, target, prior, current, approved = fixture(tmp_path)
    # Real packaged approval combines v2 (no numerics.py) and v3 (has numerics.py).
    approved = copy.deepcopy(approved)
    approved['src/robotwin_phys/numerics.py'] = ['new-numerics']
    r.approved_upgrade(prior['provenance'], current, approved)


def test_existing_unknown_numerics_hash_is_not_treated_as_an_added_file(tmp_path):
    source, target, prior, current, approved = fixture(tmp_path)
    previous = copy.deepcopy(prior['provenance'])
    previous['files']['src/robotwin_phys/numerics.py'] = 'unknown version'
    previous['source_sha256'] = digest(previous['files'])
    approved['src/robotwin_phys/numerics.py'] = ['new-numerics']
    with pytest.raises(ValueError, match='previous_sha256=unknown version'):
        r.approved_upgrade(previous, current, approved)


def test_unknown_new_numerics_hash_is_not_approved_as_known_addition(tmp_path):
    source, target, prior, current, approved = fixture(tmp_path)
    approved = copy.deepcopy(approved)
    approved['src/robotwin_phys/numerics.py'] = ['another reviewed hash']
    with pytest.raises(ValueError, match='Unreviewed previous simulator source'):
        r.approved_upgrade(prior['provenance'], current, approved)
