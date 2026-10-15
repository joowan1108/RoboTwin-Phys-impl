#!/usr/bin/env python3
"""Reverify v2 expert episodes before continuing a pool with the math repair."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from robotwin_phys.cli import worker
from robotwin_phys.conditions import digest
from robotwin_phys.manifest import atomic_json, seal
from robotwin_phys.simulator import assert_same_state

REVIEWED_SOURCE_CHANGES = {'src/robotwin_phys/numerics.py', 'src/robotwin_phys/planning.py',
                           'src/robotwin_phys/cli.py', 'src/robotwin_phys/curobo_compat.py',
                           'src/robotwin_phys/simulator.py', 'src/robotwin_phys/provenance.py'}
REVIEWED_SOURCE_ADDITIONS = {'src/robotwin_phys/numerics.py', 'src/robotwin_phys/curobo_compat.py'}


def approved_upgrade(previous, current, approved):
    if previous.get('source_sha256') != digest(previous['files']):
        raise ValueError('Previous source provenance hash is invalid')
    for key in previous.keys() | current.keys():
        if key not in ('source_sha256', 'files') and previous.get(key) != current.get(key):
            if key == 'curobo' and previous.get(key) is None:
                from robotwin_phys.curobo_compat import PIN, SOURCE_SHA256, FIX
                if current.get(key) == dict(revision=PIN, motion_gen_sha256=SOURCE_SHA256, compatibility_fix=FIX):
                    continue
            raise ValueError(f'Reverification requires unchanged runtime/config/assets: {key}')
    def reviewed(path, old):
        value = approved.get(path)
        if path not in previous['files'] and path in REVIEWED_SOURCE_ADDITIONS:
            # Known v3 helper hashes also describe a reviewed addition to v2.
            return value is None or current['files'].get(path) in (value if isinstance(value, list) else [value])
        return old in value if isinstance(value, list) else old == value
    for path in previous['files'].keys() | current['files'].keys():
        old, new = previous['files'].get(path), current['files'].get(path)
        if old == new:
            continue
        if path.startswith('src/robotwin_phys/'):
            if path not in REVIEWED_SOURCE_CHANGES or not reviewed(path, old):
                raise ValueError(f'Unreviewed previous simulator source: {path}; '
                                 f'previous_sha256={old}; current_sha256={new}; approved={approved.get(path)}')
        elif path in ('vendor/RoboTwin/envs/utils/transforms.py', 'vendor/RoboTwin/envs/robot/planner.py'):
            if not reviewed(path, old):
                raise ValueError('Unreviewed previous transform implementation')
        else:
            raise ValueError(f'Unreviewed benchmark change: {path}')


def reverify(source, target, current, approved, timeout=1800):
    source, target = Path(source), Path(target)
    if target.exists() or target.with_suffix('.progress.json').exists():
        return  # Already migrated; normal freeze validates and resumes this pool.
    source = source if source.exists() else source.with_suffix('.progress.json')
    prior = json.loads(source.read_text())
    if prior.get('sha256') != seal(prior)['sha256'] or prior.get('schema') != 'robotwin-phys-local-v1':
        raise ValueError('Source manifest/progress failed integrity validation')
    if prior.get('status') not in ('verified', 'building'):
        raise ValueError('Only verified or expert-building source pools can be reverified')
    approved_upgrade(prior['provenance'], current, approved)
    source_hash = prior['sha256']
    folder = target.parent / (target.stem + '-reverification')
    progress = target.with_suffix('.reverify.progress.json')
    audit = dict(source=str(source), source_sha256=source_hash, current_provenance_sha256=digest(current),
                 episodes=[], decisions=[], rejected=[], status='reverifying')
    if progress.exists():
        saved = json.loads(progress.read_text())
        if saved.get('sha256') != seal(saved)['sha256']:
            raise ValueError('Reverification progress was modified')
        if saved['source_sha256'] != source_hash or saved['current_provenance_sha256'] != digest(current):
            raise ValueError('Reverification source or runtime changed')
        audit = {k: v for k, v in saved.items() if k != 'sha256'}
    old_episodes = prior['episodes']
    if len(audit['episodes']) > len(old_episodes):
        raise ValueError('Invalid reverification episode count')
    # The previous tool stopped on the first rejected candidate. Its saved
    # episodes are therefore a verified prefix of the source episode list.
    if 'decisions' not in audit:
        audit['decisions'] = [dict(source_episode_index=index, seed=episode['seed'],
                                  condition_sha256=episode['condition_sha256'], status='verified')
                              for index, episode in enumerate(audit['episodes'])]
        audit['rejected'] = []
    decisions = audit['decisions']
    if len(decisions) > len(old_episodes):
        raise ValueError('Invalid reverification decision count')
    retained = iter(audit['episodes'])
    for index, decision in enumerate(decisions):
        original = old_episodes[index]
        if (decision['source_episode_index'] != index or decision['seed'] != original['seed'] or
                decision['condition_sha256'] != original['condition_sha256']):
            raise ValueError('Reverification decision does not match saved seed conditions')
        if decision['status'] == 'verified':
            episode = next(retained, None)
            if (episode is None or episode['seed'] != original['seed'] or
                    episode['condition_sha256'] != original['condition_sha256']):
                raise ValueError('Reverification progress does not match saved seed conditions')
        elif decision['status'] != 'rejected':
            raise ValueError('Invalid reverification decision status')
    if next(retained, None) is not None:
        raise ValueError('Reverification episodes have no matching decisions')
    for index, episode in enumerate(audit['episodes']):
        if (episode['episode_index'] != index or digest(episode['condition']) != episode['condition_sha256'] or
                digest(episode['initial_state']) != episode['initial_state_sha256'] or
                episode.get('expert_success') is not True):
            raise ValueError('Invalid retained reverification episode')
    rejected_decisions = [decision for decision in decisions if decision['status'] == 'rejected']
    if len(rejected_decisions) != len(audit['rejected']):
        raise ValueError('Reverification rejection history has no matching decisions')
    for decision, rejected in zip(rejected_decisions, audit['rejected']):
        if (rejected['seed'] != decision['seed'] or rejected['reason'] != decision['reason'] or
                rejected.get('reason') not in ('expert_failed', 'unstable_initialization')):
            raise ValueError('Invalid reverification rejection history')
    for index, episode in enumerate(old_episodes):
        if (episode['episode_index'] != index or episode['seed'] != episode['condition']['episode_seed'] or
                digest(episode['condition']) != episode['condition_sha256'] or
                digest(episode['initial_state']) != episode['initial_state_sha256'] or
                episode.get('expert_success') is not True):
            raise ValueError('Invalid saved expert episode')
        if index < len(decisions):
            continue
        result = worker(dict(mode='expert', episode=episode, instruction_type=prior['instruction_type']),
                        folder, f'seed-{episode["seed"]}', timeout)
        decision = dict(source_episode_index=index, seed=episode['seed'],
                        condition_sha256=episode['condition_sha256'])
        if result.get('status') == 'rejected' and result.get('reason') in ('expert_failed', 'unstable_initialization'):
            audit['rejected'].append(dict(result, seed=episode['seed'],
                                          condition_sha256=episode['condition_sha256'],
                                          reverification=True, prior_expert_success=True))
            decisions.append(dict(decision, status='rejected', reason=result['reason']))
            atomic_json(progress, seal(audit))
            print(f'Reverification rejected seed {episode["seed"]}: {result["reason"]}; '
                  'the missing episode will be filled with a new expert-verified seed', flush=True)
            continue
        if result.get('status') != 'verified':
            raise RuntimeError(f'Saved expert seed {episode["seed"]} no longer verifies: {result}; '
                               'runtime errors are not excluded as infeasible')
        verified = result['episode']
        assert_same_state(episode['initial_state'], verified['initial_state'])
        if verified['instruction'] != episode['instruction']:
            raise ValueError('Reverification changed the saved instruction')
        verified = dict(verified, episode_index=len(audit['episodes']))
        audit['episodes'].append(verified)
        decisions.append(dict(decision, status='verified'))
        atomic_json(progress, seal(audit))
        print(f'Reverified {prior["task"]} {index+1}/{len(old_episodes)}: seed {episode["seed"]}', flush=True)
    migrated = {k: v for k, v in prior.items() if k != 'sha256'}
    migrated.update(provenance=current, episodes=audit['episodes'], rejected=prior['rejected'] + audit['rejected'])
    migrated['status'] = 'verified' if len(migrated['episodes']) == migrated['requested_episodes'] else 'building'
    atomic_json(target.with_suffix('.progress.json'), seal(migrated))
    if migrated['status'] == 'verified':
        atomic_json(target, seal(migrated))
    audit['status'] = 'reverified'
    audit['selection'] = 'Retain reverified prior seeds; reject infeasible seeds and refill from the next unused candidate'
    atomic_json(target.with_suffix('.reverification.json'), seal(audit))
    print(f'REVERIFICATION_PASS {prior["task"]}: {len(audit["episodes"])}/{len(old_episodes)} saved episodes retained; '
          f'{len(audit["rejected"])} rejected; {migrated["requested_episodes"] - len(audit["episodes"])} '
          'episodes still needed', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--timeout', type=int, default=1800)
    args = parser.parse_args()
    from robotwin_phys.provenance import fingerprint
    approved = json.loads((ROOT / 'tools/expert_fix_previous.json').read_text())
    reverify(args.source, args.target, fingerprint(), approved, args.timeout)


if __name__ == '__main__':
    main()
