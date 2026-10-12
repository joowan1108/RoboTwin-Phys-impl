"""Export the actual worker requests and verified results, never invent used conditions."""
import csv
import json

from robotwin_phys.conditions import digest
from robotwin_phys.manifest import atomic_json

FACTORS = ['mass_scale', 'friction', 'restitution', 'com_offset_m', 'geometry_scale',
           'object_joint_damping', 'arm_damping_scale', 'gripper_damping_scale',
           'table_height_offset_m', 'table_tilt_deg', 'table_tilt_azimuth_deg',
           'force_mg', 'force_direction', 'camera', 'active_factors']


def episode_record(folder, suite, model, protocol, task, manifest, episode):
    directory = folder / model / protocol / task
    stem = f"episode-{episode['episode_index']:04d}"
    request_path, result_path = directory / (stem+'.request.json'), directory / (stem+'.result.json')
    if not request_path.exists() and not result_path.exists():
        return None
    record = dict(model=model, task=task, protocol=protocol, episode_index=episode['episode_index'],
                  seed=episode['seed'], manifest_sha256=manifest['sha256'],
                  expected_condition_sha256=episode['condition_sha256'],
                  expected_initial_state_sha256=episode['initial_state_sha256'],
                  status='requested', verified=False, errors=[], condition=None,
                  units=dict(body_mass='kg', inertia='kg*m^2', com_offset='m', table_height='m',
                             angles='degrees', friction='dimensionless', force_mg='fraction of mass*9.81'),
                  audit_file=str((directory/(stem+'.randomization.json')).relative_to(folder)))
    try:
        request = json.loads(request_path.read_text())
        used = request['episode']
        record['condition'] = used['condition']
        record['requested_condition_sha256'] = digest(used['condition'])
        record['request_sha256'] = digest(request)
        run = json.loads((directory/'run.json').read_text())
        record['policy'] = run['policy']
        record['policy_sha256'] = digest(run['policy'])
        if (request['mode'] != 'evaluate' or run['mode'] != 'evaluate'
                or request.get('smoke_max_actions') is not None or run.get('smoke_max_actions') is not None
                or run['policy']['name'] != f'{model}_robotwin2'
                or run['manifest_sha256'] != manifest['sha256']
                or request['policy_identity'] != run['policy']
                or request['policy_seed'] != suite['policy_seed'] or run['policy_seed'] != suite['policy_seed']):
            raise ValueError('Policy/run/request identity mismatch or debug evaluation')
        for key in ('episode_index', 'seed', 'condition_sha256', 'initial_state_sha256', 'instruction'):
            if used[key] != episode[key]:
                raise ValueError(f'Request {key} differs from shared manifest')
        if digest(used['condition']) != episode['condition_sha256']:
            raise ValueError('Actual requested randomization differs from shared manifest')
        if digest(used['initial_state']) != episode['initial_state_sha256']:
            raise ValueError('Request initial state differs from shared manifest')
        if result_path.exists():
            result = json.loads(result_path.read_text())
            record.update(status=result['status'], result_sha256=digest(result), success=result.get('success'))
            if result['status'] == 'completed':
                for key in ('episode_index', 'seed', 'condition_sha256', 'initial_state_sha256'):
                    if result[key] != episode[key]:
                        raise ValueError(f'Result {key} differs from shared manifest')
                if result['policy'] != run['policy'] or not result.get('benchmark_eligible') or type(result['success']) is not bool:
                    raise ValueError('Result policy/eligibility mismatch')
                record['verified'] = True
                record['observed_initial_state_sha256'] = result['initial_state_sha256']
                # The worker checked this state before its first policy action and returned its hash.
                # This is initial physics, not the later trajectory (which depends on the policy).
                state = used['initial_state']
                record['verified_initial_physical'] = state.get('physical', {})
                record['verified_initial_scene'] = {k: state[k] for k in ('table_height_m', 'cameras', 'lighting', 'textures') if k in state}
                record['external_force'] = dict(phase='policy_only', magnitude_rule='force_mg * body_mass_kg * 9.81 N',
                    force_mg=used['condition'].get('force_mg'), direction=used['condition'].get('force_direction'))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        record['status'] = 'invalid'
        record['verified'] = False
        record['errors'].append(str(exc))
    target = folder / record['audit_file']
    try:
        previous = json.loads(target.read_text())
    except (OSError, ValueError):
        previous = None
    if previous != record:
        atomic_json(target, record)
    return record


def export_audit(folder, suite, get_manifest):
    out = folder/'tables'
    out.mkdir(exist_ok=True)
    invalid = set()
    counts = dict(recorded_episodes=0, verified_episodes=0, invalid_episodes=0,
                  shared_episodes=0, all_models_match=0, partial=0, pending=0, mismatch=0,
                  missing_or_invalid_manifests=0)
    fields = ['model', 'protocol', 'task', 'episode_index', 'seed', 'status', 'verified',
              'expected_condition_sha256', 'requested_condition_sha256', 'expected_initial_state_sha256',
              'observed_initial_state_sha256', 'manifest_sha256', 'policy_sha256', *FACTORS,
              'condition_json', 'errors', 'audit_file']
    compare_fields = ['protocol', 'task', 'episode_index', 'seed', 'condition_sha256',
                      'initial_state_sha256', 'expected_models', 'recorded_models', 'verified_models',
                      'mismatched_models', 'status']
    names = ['randomization.jsonl', 'randomization.csv', 'condition_consistency.csv']
    with (out/('.'+names[0]+'.tmp')).open('w') as stream, (out/('.'+names[1]+'.tmp')).open('w', newline='') as table, (out/('.'+names[2]+'.tmp')).open('w', newline='') as compare:
        writer = csv.DictWriter(table, fieldnames=fields)
        writer.writeheader()
        comparison = csv.DictWriter(compare, fieldnames=compare_fields)
        comparison.writeheader()
        for protocol in suite['protocols']:
            for task in suite['tasks']:
                try:
                    manifest = get_manifest(protocol, task)
                except (OSError, ValueError, KeyError, TypeError):
                    counts['missing_or_invalid_manifests'] += 1
                    continue
                for episode in manifest['episodes']:
                    records = []
                    for model in suite['models']:
                        record = episode_record(folder, suite, model, protocol, task, manifest, episode)
                        if record is None:
                            continue
                        records.append(record)
                        counts['recorded_episodes'] += 1
                        counts['verified_episodes'] += record['verified']
                        if record['errors']:
                            counts['invalid_episodes'] += 1
                            invalid.add((model, protocol, task))
                        stream.write(json.dumps(record, sort_keys=True)+'\n')
                        values = {k: record.get(k) for k in fields}
                        values.update({k: (record['condition'] or {}).get(k) for k in FACTORS})
                        values['condition_json'] = json.dumps(record['condition'], sort_keys=True)
                        writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in values.items()})
                    verified = [r['model'] for r in records if r['verified']]
                    mismatched = [r['model'] for r in records if r['errors']]
                    status = 'mismatch' if mismatched else ('all_models_match' if len(verified) == len(suite['models']) else ('partial' if verified else 'pending'))
                    counts['shared_episodes'] += 1
                    counts[status] += 1
                    comparison.writerow(dict(protocol=protocol, task=task, episode_index=episode['episode_index'], seed=episode['seed'],
                        condition_sha256=episode['condition_sha256'], initial_state_sha256=episode['initial_state_sha256'],
                        expected_models=json.dumps(suite['models']), recorded_models=json.dumps([r['model'] for r in records]),
                        verified_models=json.dumps(verified), mismatched_models=json.dumps(mismatched), status=status))
    for name in names:
        (out/('.'+name+'.tmp')).replace(out/name)
    summary = dict(schema='randomization-audit-v1', suite_sha256=suite['sha256'], **counts,
                   expected_shared_episodes=len(suite['tasks'])*len(suite['protocols'])*suite['episodes'],
                   invalid_cells=[list(x) for x in sorted(invalid)],
                   evidence='Worker request conditions + completed result hashes; no completion is inferred from a planned manifest alone.')
    atomic_json(out/'randomization_audit.json', summary)
    return summary
