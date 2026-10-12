"""Real filesystem fixtures exercise score integrity, missing data, and immutable identities."""
import importlib.util
import json
from pathlib import Path
import pytest
from robotwin_phys.conditions import sample, digest
from robotwin_phys.manifest import seal, atomic_json

spec=importlib.util.spec_from_file_location('experiment',Path(__file__).parents[1]/'tools/experiment.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)

@pytest.fixture
def suite(tmp_path,monkeypatch):
    monkeypatch.setattr(e,'ROOT',tmp_path)
    folder=tmp_path/'runs/demo';folder.mkdir(parents=True)
    s=seal(dict(name='demo',tasks=['stack_blocks_two','grab_roller'],protocols=e.PROTOCOLS,
                models=['pi05'],episodes=2,seed=100000,policy_seed=0,instruction_type='unseen'))
    atomic_json(folder/'suite.json',s)
    for protocol in s['protocols']:
        for task in s['tasks']:
            eps=[]
            for i in range(2):
                c=sample(task,100000+i,protocol)
                eps.append(dict(episode_index=i,seed=100000+i,condition=c,condition_sha256=digest(c),
                    expert_success=True,initial_state={'example':True},initial_state_sha256=digest({'example':True}),instruction='test'))
            m=seal(dict(schema='robotwin-phys-local-v1',status='verified',task=task,protocol=protocol,
                   start_seed=s['seed'],requested_episodes=2,instruction_type='unseen',only=[],without=[],episodes=eps))
            atomic_json(e.manifest_path(s,protocol,task),m)
            policy=dict(name='pi05_robotwin2',checkpoint={'revision':'test-only'})
            dest=folder/'pi05'/protocol/task
            atomic_json(dest/'run.json',dict(mode='evaluate',policy=policy,manifest_sha256=m['sha256'],policy_seed=0,smoke_max_actions=None))
            for ep in eps:
                result={k:ep[k] for k in ('episode_index','seed','condition_sha256','initial_state_sha256')}
                atomic_json(dest/f"episode-{ep['episode_index']:04d}.result.json",dict(result,status='completed',success=ep['episode_index']==0,policy=policy,benchmark_eligible=True))
                atomic_json(dest/f"episode-{ep['episode_index']:04d}.request.json",dict(mode='evaluate',episode=ep,policy_identity=policy,policy_seed=0,smoke_max_actions=None))
    return folder,s


def test_macro_table_and_formats(suite):
    folder,s=suite
    result=e.report(folder)
    assert not result['full_benchmark_scope']
    assert all(r['success_rate']==.5 for r in result['aggregate'])
    assert all((folder/'tables'/n).exists() for n in ['results.md','results.tex','results.json','summary.csv','tasks.csv'])


def test_missing_does_not_become_zero_or_partial_average(suite):
    folder,s=suite
    (folder/'pi05/clean/grab_roller/episode-0001.result.json').unlink()
    r=e.report(folder)['aggregate'][0]
    assert r['success_rate'] is None and r['completed_episodes']==3


@pytest.mark.parametrize('mutation',[{'seed':42},{'success':1},{'benchmark_eligible':False},{'condition_sha256':'bad'}])
def test_reject_invalid_episode(suite,mutation):
    folder,s=suite;p=folder/'pi05/clean/grab_roller/episode-0000.result.json'
    r=json.loads(p.read_text());r.update(mutation);atomic_json(p,r)
    result=e.report(folder)
    assert result['aggregate'][0]['success_rate'] is None
    assert any(r['status']=='invalid' for r in result['tasks'])


def test_refuse_mixed_model_settings(suite):
    folder,s=suite;p=folder/'pi05/clean/grab_roller/run.json'
    run=json.loads(p.read_text());run['policy']['checkpoint']['revision']='other';atomic_json(p,run)
    result=e.report(folder)
    assert all(r['success_rate'] is None for r in result['aggregate'])


def test_modified_manifest_never_scored(suite):
    folder,s=suite;p=e.manifest_path(s,'clean','grab_roller');m=json.loads(p.read_text());m['start_seed']=0;atomic_json(p,m)
    assert e.report(folder)['aggregate'][0]['success_rate'] is None


def add_second_model(folder, s):
    import shutil
    s = dict(s, models=['pi05', 'fact'])
    atomic_json(folder/'suite.json', seal(s))
    shutil.copytree(folder/'pi05', folder/'fact')
    for path in (folder/'fact').rglob('*.json'):
        text = path.read_text().replace('pi05_robotwin2', 'fact_robotwin2')
        path.write_text(text)


def test_each_models_actual_request_is_exported_and_compared(suite):
    import csv
    folder, s = suite
    add_second_model(folder, s)
    result = e.report(folder)
    assert result['randomization_audit']['all_models_match'] == 12
    assert result['randomization_audit']['verified_episodes'] == 24
    records = [json.loads(line) for line in (folder/'tables/randomization.jsonl').read_text().splitlines()]
    assert len(records) == 24
    for protocol in s['protocols']:
        peers = [r for r in records if r['protocol'] == protocol and r['task'] == 'grab_roller' and r['episode_index'] == 1]
        assert {r['model'] for r in peers} == {'pi05', 'fact'}
        assert peers[0]['condition'] == peers[1]['condition']
        assert peers[0]['observed_initial_state_sha256'] == peers[1]['observed_initial_state_sha256']
        assert all((folder/r['audit_file']).exists() for r in peers)
    with (folder/'tables/randomization.csv').open() as f:
        assert len(list(csv.DictReader(f))) == 24


def test_request_tampering_cannot_hide_behind_copied_result_hash(suite):
    folder, s = suite
    add_second_model(folder, s)
    path = folder/'fact/physical_random/grab_roller/episode-0000.request.json'
    request = json.loads(path.read_text())
    request['episode']['condition']['mass_scale'] *= 2
    atomic_json(path, request)  # Keep the old header/result hashes to test actual condition hashing.
    result = e.report(folder)
    audit = result['randomization_audit']
    assert audit['mismatch'] == 1
    assert audit['all_models_match'] == 11
    assert ['fact', 'physical_random', 'grab_roller'] in audit['invalid_cells']
    assert next(r for r in result['aggregate'] if r['model'] == 'fact' and r['protocol'] == 'physical_random')['success_rate'] is None
    assert e.completion_status(folder, 'fact', s['tasks'], s['protocols']) == 2


def test_missing_model_is_partial_not_all_models_match(suite):
    folder, s = suite
    add_second_model(folder, s)
    for suffix in ['request', 'result']:
        (folder/f'fact/clean/grab_roller/episode-0001.{suffix}.json').unlink()
    result = e.report(folder)
    assert result['randomization_audit']['partial'] == 1
    assert result['randomization_audit']['all_models_match'] == 11
    assert e.completion_status(folder, 'fact', s['tasks'], s['protocols']) == 1
    assert e.completion_status(folder, 'pi05', s['tasks'], s['protocols']) == 0


def test_completed_result_without_request_is_not_accepted(suite):
    folder, s = suite
    (folder/'pi05/clean/grab_roller/episode-0000.request.json').unlink()
    result = e.report(folder)
    assert result['randomization_audit']['invalid_episodes'] == 1
    assert result['aggregate'][0]['success_rate'] is None
