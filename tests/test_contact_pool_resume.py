"""Preserve saved v5 evidence only for the exact non-success-path change."""
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
spec = importlib.util.spec_from_file_location('resume_contacts', ROOT / 'tools/resume_contact_pool.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def fingerprint(planning):
    files = {r.PLANNING: planning, 'src/robotwin_phys/simulator.py': 'unchanged',
             'vendor/RoboTwin/envs/place_a2b_left.py': 'official', 'configs/physical_domains.json': 'same'}
    return dict(files=files, source_sha256=digest(files), gpu={'driver': '580.82.07'},
                curobo={'compatibility_fix': 'consistent-batch-results-v2'}, assets={'sha256': 'same'})


def setup(tmp_path, tasks=('adjust_bottle', 'place_a2b_right'), previous=r.BEFORE):
    suite = dict(name=r.EXPERIMENT, tasks=list(tasks), models=['fastwam', 'pi05'], episodes=2,
                 seed=100000, instruction_type='unseen')
    paths = []
    for task in tasks:
        condition = sample(task, 100000)
        episode = dict(episode_index=0, seed=100000, condition=condition, condition_sha256=digest(condition),
                       initial_state={'entities': []}, initial_state_sha256=digest({'entities': []}),
                       instruction='Move object', expert_success=True)
        record = dict(schema='robotwin-phys-local-v1', implementation='independent-paper-reproduction',
                      status='building', task=task,
                      protocol='physical_random', start_seed=100000, requested_episodes=2,
                      instruction_type='unseen', only=[], without=[], provenance=fingerprint(previous),
                      episodes=[episode], rejected=[dict(seed=100001, status='rejected', reason='expert_failed')])
        path = tmp_path / 'manifests' / r.EXPERIMENT / 'physical_random' / (task + '.progress.json')
        atomic_json(path, seal(record))
        paths.append(path)
    return suite, paths


@pytest.mark.parametrize('previous', [r.BEFORE, r.EMPTY_ONLY, r.CONTACTS])
def test_completed_and_partial_progress_are_preserved_with_original_evidence(tmp_path, previous):
    suite, paths = setup(tmp_path, previous=previous)
    full = json.loads(paths[0].read_text())
    full['episodes'].append(copy.deepcopy(full['episodes'][0]))
    last = full['episodes'][-1]
    last['episode_index'], last['seed'] = 1, 100002
    last['condition'] = sample(full['task'], 100002)
    last['condition_sha256'] = digest(last['condition'])
    full['status'] = 'verified'
    atomic_json(paths[0], seal(full))
    frozen = paths[0].with_name(full['task'] + '.json')
    frozen.write_bytes(paths[0].read_bytes())
    originals = {p: p.read_bytes() for p in [*paths, frozen]}
    r.preserve_progress(tmp_path, suite, fingerprint(r.AFTER))
    for path, raw in originals.items():
        prior, new = json.loads(raw), json.loads(path.read_text())
        assert prior['episodes'] == new['episodes'] and prior['rejected'] == new['rejected']
        assert new['provenance'] == fingerprint(r.AFTER)
        assert new['sha256'] == seal(new)['sha256']
        receipt_path = tmp_path / 'reports/pool-preservation' / r.EXPERIMENT / r.RULE / (path.name + '.receipt.json')
        receipt = json.loads(receipt_path.read_text())
        assert receipt['expert_reexecuted'] is False
        assert Path(receipt['original_backup']).read_bytes() == raw
        assert receipt['previous_provenance'] == prior['provenance']
    saved = {p: p.read_bytes() for p in originals}
    r.preserve_progress(tmp_path, suite, fingerprint(r.AFTER))
    assert saved == {p: p.read_bytes() for p in originals}


@pytest.mark.parametrize('change', ['assets', 'gpu', 'curobo', 'another_source', 'old_planning', 'checksum'])
def test_unrelated_changes_are_rejected_before_any_manifest_write(tmp_path, change):
    suite, paths = setup(tmp_path)
    bad = json.loads(paths[1].read_text())
    prov = bad['provenance']
    if change == 'another_source':
        prov['files']['src/robotwin_phys/simulator.py'] = 'unknown'
        prov['source_sha256'] = digest(prov['files'])
    elif change == 'old_planning':
        prov['files'][r.PLANNING] = 'unknown'
        prov['source_sha256'] = digest(prov['files'])
    elif change == 'checksum':
        prov['source_sha256'] = 'bad'
    else:
        prov[change] = {'changed': True}
    atomic_json(paths[1], seal(bad))
    originals = {p: p.read_bytes() for p in paths}
    with pytest.raises(ValueError):
        r.preserve_progress(tmp_path, suite, fingerprint(r.AFTER))
    assert originals == {p: p.read_bytes() for p in paths}
    assert not (tmp_path / 'reports/pool-preservation').exists()


@pytest.mark.parametrize('which', ['pool-index', 'evaluation'])
def test_existing_evaluation_or_sealed_pool_is_never_rewritten(tmp_path, which):
    suite, paths = setup(tmp_path)
    if which == 'pool-index':
        atomic_json(paths[0].parent.parent / 'pool-index.json', {})
    else:
        (tmp_path / 'runs' / r.EXPERIMENT / 'fastwam').mkdir(parents=True)
    original = paths[0].read_bytes()
    with pytest.raises(ValueError):
        r.preserve_progress(tmp_path, suite, fingerprint(r.AFTER))
    assert paths[0].read_bytes() == original


def packaged_planning(package):
    with zipfile.ZipFile(ROOT / 'reports' / (package + '.zip')) as bundle:
        return bundle.read(package + '/' + r.PLANNING)


@pytest.mark.parametrize('package,checksum', [('robotwin-fresh-pool-v1', r.BEFORE),
                                            ('robotwin-empty-grasp-fix-v1', r.EMPTY_ONLY)])
def test_v2_planning_diff_preserves_success_paths_from_both_previous_versions(package, checksum):
    import ast
    old = packaged_planning(package)
    new = packaged_planning('robotwin-grasp-contact-fix-v2')
    assert hashlib.sha256(old).hexdigest() == checksum
    assert hashlib.sha256(new).hexdigest() == r.CONTACTS
    before, after = ast.parse(old), ast.parse(new)
    def branch(tree):
        return next(n for n in ast.walk(tree) if isinstance(n, ast.If) and
                    isinstance(n.test, ast.UnaryOp) and isinstance(n.test.op, ast.Not) and
                    isinstance(n.test.operand, ast.Name) and n.test.operand.id == 'contacts')
    original_branch = branch(before)
    new_branch = branch(after)
    new_branch.body = original_branch.body
    # The second change is confined to the previously-fatal None result.
    # Passing the arm name only adds diagnostic context on that failure path.
    def validator(tree):
        return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and
                    n.name == '_validate_grasp_contact')
    original_validator, new_validator = validator(before), validator(after)
    assert [a.arg for a in original_validator.args.args] == ['actor', 'index']
    assert [a.arg for a in new_validator.args.args] == ['actor', 'index', 'arm_tag']
    assert ast.dump(new_validator.args.defaults[0]) == ast.dump(ast.Constant(None))
    new_validator.args = original_validator.args
    old_none = next(n for n in original_validator.body if isinstance(n, ast.If))
    new_none = next(n for n in new_validator.body if isinstance(n, ast.If))
    assert ast.dump(old_none.test) == ast.dump(new_none.test)
    assert len(old_none.body) == 1 and isinstance(old_none.body[0], ast.Raise)
    new_none.body = old_none.body
    count = 0
    for node in ast.walk(after):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and
                node.func.attr == '_validate_grasp_contact'):
            assert len(node.args) == 3 and ast.dump(node.args[-1]) == ast.dump(ast.Name('arm_tag', ast.Load()))
            node.args.pop()
            count += 1
    assert count == 2
    assert ast.dump(before) == ast.dump(after)


def test_v3_planning_diff_adds_only_post_failure_classification():
    import ast
    old = packaged_planning('robotwin-grasp-contact-fix-v2')
    new = (ROOT / r.PLANNING).read_bytes()
    assert hashlib.sha256(old).hexdigest() == r.CONTACTS
    assert hashlib.sha256(new).hexdigest() == r.AFTER
    before, after = ast.parse(old), ast.parse(new)

    def member(tree, cls, name):
        klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
        return klass, [n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == name]
    # 1. `import traceback` is the only new module-level statement.
    added = [n for n in after.body if isinstance(n, ast.Import) and n.names[0].name == 'traceback']
    assert len(added) == 1 and not any(isinstance(n, ast.Import) and n.names[0].name == 'traceback'
                                       for n in before.body)
    after.body.remove(added[0])
    # 2. PlanningMixin only gains play_once, which re-raises unless the official
    #    planner already recorded plan_success = False.
    mixin, play = member(after, 'PlanningMixin', 'play_once')
    assert len(play) == 1 and member(before, 'PlanningMixin', 'play_once')[1] == []
    source = ast.get_source_segment(new.decode(), play[0])
    assert 'if self.plan_success is not False:\n                raise' in source
    assert 'except ExpertPlanningFailure:\n            raise' in source
    assert 'kind="exception_after_planning_failure"' in source
    mixin.body.remove(play[0])
    # 3. ExpertPlanningFailure.__init__ accepts a missing actor with the same
    #    signature; equal diagnostics for real actors are checked separately.
    failure, new_init = member(after, 'ExpertPlanningFailure', '__init__')
    _, old_init = member(before, 'ExpertPlanningFailure', '__init__')
    assert ast.dump(old_init[0].args) == ast.dump(new_init[0].args)
    failure.body[failure.body.index(new_init[0])] = old_init[0]
    assert ast.dump(before) == ast.dump(after)


def test_v3_failure_diagnostics_are_identical_for_annotated_actors():
    import types
    from types import SimpleNamespace
    from robotwin_phys import planning as current
    old = types.ModuleType('planning_v2')
    old.__package__ = 'robotwin_phys'
    exec(compile(packaged_planning('robotwin-grasp-contact-fix-v2'), 'planning_v2.py', 'exec'), old.__dict__)
    actor = SimpleNamespace(get_name=lambda: '021_cup',
                            get_pose=lambda: SimpleNamespace(p=[0.1, -0.2, 0.75], q=[1., 0., 0., 0.]))
    for args in ((actor, 'left', [2]), (actor, 'right', [0, 1], 'unavailable_grasp_annotation', 'x'),
                 (actor, 'left', [], 'no_grasp_annotations')):
        assert old.ExpertPlanningFailure(*args).diagnostics == current.ExpertPlanningFailure(*args).diagnostics
        assert str(old.ExpertPlanningFailure(*args)) == str(current.ExpertPlanningFailure(*args))
    with pytest.raises(AttributeError):
        old.ExpertPlanningFailure(None, None, [])
    generic = current.ExpertPlanningFailure(None, None, [], kind='exception_after_planning_failure', details='x')
    assert generic.diagnostics == {'kind': 'exception_after_planning_failure', 'actor': None, 'arm': None,
                                   'contact_point_ids': []}


def test_metadata_audit_distinguishes_explicit_empty_from_corrupt_extraction(tmp_path):
    archive = tmp_path / 'downloads/objects.zip'
    archive.parent.mkdir()
    relative = 'objects/086_woodenblock/model_data0.json'
    raw = (ROOT / 'tests/fixtures/woodenblock/model_data0.json').read_bytes()
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr(relative, raw)
    asset = tmp_path / 'vendor/RoboTwin/assets' / relative
    asset.parent.mkdir(parents=True)
    asset.write_bytes(raw)
    output = tmp_path / 'reports/grasp-assets.json'
    report = r.audit_assets(tmp_path, output)
    assert report['empty_contact_files'] == [relative] and report['issues'] == []
    asset.write_bytes(raw.replace(b'1.0', b'2.0', 1))  # Same byte count, changed content.
    with pytest.raises(ValueError, match='integrity'):
        r.audit_assets(tmp_path, output)
    assert 'differs' in json.loads(output.read_text())['issues'][0]['error']


def test_asset_audit_identifies_official_task_contact_mismatches_without_failing(tmp_path):
    archive = tmp_path / 'downloads/objects.zip'
    archive.parent.mkdir()
    with zipfile.ZipFile(archive, 'w') as z:
        for path in (ROOT / 'tests/fixtures/grasp_contacts').glob('*/model_data*.json'):
            relative = Path('objects') / path.relative_to(ROOT / 'tests/fixtures/grasp_contacts')
            z.writestr(str(relative), path.read_bytes())
            asset = tmp_path / 'vendor/RoboTwin/assets' / relative
            asset.parent.mkdir(parents=True, exist_ok=True)
            asset.write_bytes(path.read_bytes())
    report = r.audit_assets(tmp_path, tmp_path / 'report.json')
    assert report['issues'] == []
    assert {(x['path'], x['arm'], x['requested_contact']) for x in report['unavailable_contact_selections']} == {
        ('objects/021_cup/model_data6.json', 'left', 2),
        ('objects/021_cup/model_data7.json', 'left', 2)}


@pytest.mark.parametrize('previous,kind', [(r.BEFORE, 'no_grasp_annotations'),
                                         (r.BEFORE, 'unavailable_grasp_annotation'),
                                         (r.EMPTY_ONLY, 'unavailable_grasp_annotation'),
                                         (r.BEFORE, 'exception_after_planning_failure'),
                                         (r.CONTACTS, 'exception_after_planning_failure')])
def test_resume_retries_only_failed_candidate_then_continues(tmp_path, monkeypatch, previous, kind):
    from robotwin_phys import cli, provenance
    from types import SimpleNamespace
    suite, paths = setup(tmp_path, tasks=('place_a2b_right',), previous=previous)
    current = fingerprint(r.AFTER)
    r.preserve_progress(tmp_path, suite, current)
    monkeypatch.setattr(provenance, 'fingerprint', lambda: current)
    monkeypatch.setattr(cli, 'task_names', lambda: suite['tasks'])
    seen = []
    def run(request, *a):
        episode = request['episode']
        seen.append(episode['seed'])
        if episode['seed'] == 100002:
            return dict(status='rejected', reason='expert_failed',
                        planning_failure={'kind': kind})
        initial = {'entities': []}
        return dict(status='verified', episode=dict(episode, initial_state=initial,
                    initial_state_sha256=digest(initial), expert_success=True, instruction='Move object'))
    monkeypatch.setattr(cli, 'worker', run)
    output = paths[0].with_name('place_a2b_right.json')
    args = SimpleNamespace(output=str(output), task='place_a2b_right', protocol='physical_random',
        episodes=2, max_candidates=100, seed=100000, instruction_type='unseen', only=None, without=None,
        resume=True, live=False, port=8765, viewer=False, video=False, timeout=30)
    cli.freeze(args)
    result = json.loads(output.read_text())
    assert seen == [100002, 100003]
    assert [ep['seed'] for ep in result['episodes']] == [100000, 100003]
    assert [ep['seed'] for ep in result['rejected']] == [100001, 100002]
