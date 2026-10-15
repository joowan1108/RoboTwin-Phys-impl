#!/usr/bin/env python3
"""Continue v5 after reviewed expert-failure classification fixes, without replay.

Only an exact planning.py change is compatible. Existing accepted/rejected
records are retained byte-equivalent as JSON values; original manifests are
backed up before their source provenance advances. This hook runs inside the
parallel controller's exclusive experiment lock, after GPU/runtime preflight.

Accepted previous planning sources: the original v5 pool (BEFORE), the empty
annotation patch v1 (EMPTY_ONLY) and the grasp-contact patch v2 (CONTACTS).
Every one of them aborted the task on the branches that now reject a candidate:
unavailable grasp annotations, and task exceptions raised after the official
planner had already recorded plan_success = False.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys
import zlib
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))
from robotwin_phys.conditions import digest
from robotwin_phys.manifest import atomic_json, seal

PLANNING = 'src/robotwin_phys/planning.py'
BEFORE = '4c97b9ec3cd252837ffc6461f2b3a766eee373f46a1a4c51fc01b242220ab864'
EMPTY_ONLY = '2714bd70d21392e6732e5a939d8fd7880a9abf1f9091ef84bc14aa8ea45447fd'
CONTACTS = '1dafe49674a85370d678111b6a82139de79ede0ab4adc4ad603a21f0c36f700f'
AFTER = '1cf08d6682077dc5952755684d8709603bfc4eead9adf70caa1006305513a4f6'
PREVIOUS = (BEFORE, EMPTY_ONLY, CONTACTS)
EXPERIMENT = 'phys-pro6000-100-v5'
RULE = 'expert-exception-after-planning-failure-v3'


def compatible(previous, current):
    for value in (previous, current):
        if value.get('source_sha256') != digest(value['files']):
            raise ValueError('Source provenance checksum is invalid')
    if current['files'].get(PLANNING) != AFTER:
        raise ValueError('The current planning source is not the reviewed expert-failure repair')
    if previous == current:
        return
    if previous['files'].get(PLANNING) not in PREVIOUS:
        raise ValueError('Only the reviewed v5 planning sources can be preserved without expert replay')
    expected = copy.deepcopy(previous)
    expected['files'][PLANNING] = AFTER
    expected['source_sha256'] = digest(expected['files'])
    if expected != current:
        raise ValueError('Runtime/assets/config/source changed beyond the reviewed expert-failure repair')


def audit_assets(root, output):
    """Audit all object metadata against the already SHA-verified objects ZIP."""
    archive = root / 'downloads/objects.zip'
    asset_root = root / 'vendor/RoboTwin/assets'
    records, issues, empty = [], [], []
    with zipfile.ZipFile(archive) as bundle:
        for entry in bundle.infolist():
            relative = Path(entry.filename)
            if (not relative.parts or relative.parts[0] != 'objects' or
                    not relative.name.startswith('model_data') or relative.suffix != '.json'):
                continue
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe object metadata archive path')
            path = asset_root / relative
            if not path.is_file():
                issues.append(dict(path=str(relative), error='missing extracted metadata'))
                continue
            raw = path.read_bytes()
            if len(raw) != entry.file_size or zlib.crc32(raw) != entry.CRC:
                issues.append(dict(path=str(relative), error='extracted metadata differs from verified ZIP'))
                continue
            try:
                config = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                issues.append(dict(path=str(relative), error='invalid JSON in verified ZIP'))
                continue
            if not isinstance(config, dict):
                issues.append(dict(path=str(relative), error='metadata is not a JSON object'))
                continue
            key = 'contact_points_pose' if 'contact_points_pose' in config else 'contact_points'
            points = config.get(key)
            state = ('empty' if isinstance(points, list) and not points else
                     'annotated' if isinstance(points, list) else 'missing_or_nonlist')
            record = dict(path=str(relative), contacts=len(points) if isinstance(points, list) else None,
                          state=state, sha256=hashlib.sha256(raw).hexdigest())
            records.append(record)
            if state == 'empty':
                empty.append(str(relative))
    # Pinned official place_container_plate selects contact 2 for the left arm
    # and 0 for the right arm across these variants. Report unsupported choices
    # before starting workers; do not alter that task's sampling or contacts.
    counts = {record['path']: record['contacts'] for record in records}
    unavailable = []
    for name, variants in (('002_bowl', (1, 2, 3, 5)), ('021_cup', (1, 2, 3, 4, 5, 6, 7))):
        for variant in variants:
            path = f'objects/{name}/model_data{variant}.json'
            count = counts.get(path)
            for arm, index in (('left', 2), ('right', 0)):
                if count is not None and index >= count:
                    unavailable.append(dict(task='place_container_plate', path=path, arm=arm,
                                            requested_contact=index, available_contacts=count))
    report = dict(rule=RULE, metadata_files=len(records), empty_contact_files=empty,
                  unavailable_contact_selections=unavailable,
                  note='Empty annotations limit the scripted expert; missing metadata stays an error when used.',
                  issues=issues, files=records)
    atomic_json(output, report)
    if issues:
        raise ValueError(f'Object metadata integrity audit failed; see {output}')
    if not records:
        raise ValueError('No object metadata found in objects.zip')
    print(f'GRASP_ASSET_AUDIT: {len(records)} metadata files; {len(empty)} explicit empty contact lists; '
          f'{len(unavailable)} unavailable fixed contact selections; '
          f'report={output}', flush=True)
    return report


def validate_record(value, suite, task):
    if value.get('sha256') != seal(value)['sha256'] or value.get('schema') != 'robotwin-phys-local-v1':
        raise ValueError(f'Manifest integrity failure: {task}')
    expected = dict(implementation='independent-paper-reproduction',
                    task=task, protocol='physical_random', start_seed=suite['seed'],
                    requested_episodes=suite['episodes'], instruction_type=suite['instruction_type'],
                    only=[], without=[])
    if any(value.get(key) != v for key, v in expected.items()):
        raise ValueError(f'Manifest settings differ: {task}')
    if value.get('status') not in ('verified', 'building'):
        raise ValueError(f'Unsupported progress status: {task}')
    episodes, rejected = value['episodes'], value['rejected']
    if len(episodes) > suite['episodes'] or (value['status'] == 'verified' and len(episodes) != suite['episodes']):
        raise ValueError(f'Invalid episode count: {task}')
    used = []
    for index, episode in enumerate(episodes):
        condition = episode['condition']
        if (episode['episode_index'] != index or episode['seed'] != condition['episode_seed'] or
                condition['task'] != task or condition['protocol'] != 'physical_random' or
                digest(condition) != episode['condition_sha256'] or
                digest(episode['initial_state']) != episode['initial_state_sha256'] or
                episode.get('expert_success') is not True or not episode.get('instruction')):
            raise ValueError(f'Invalid saved expert evidence: {task}/{index}')
        used.append(episode['seed'])
    for entry in rejected:
        if entry.get('status') != 'rejected' or entry.get('reason') not in ('expert_failed', 'unstable_initialization'):
            raise ValueError(f'Unexpected rejection history: {task}')
        used.append(entry['seed'])
    if len(set(used)) != len(used) or sorted(used) != list(range(suite['seed'], suite['seed'] + len(used))):
        raise ValueError(f'Candidate history is not an uninterrupted saved prefix: {task}')


def preserve_progress(root, suite, current):
    if suite['name'] != EXPERIMENT:
        raise ValueError(f'This reviewed continuation applies only to {EXPERIMENT}')
    compatible(current, current)
    folder = root / 'manifests' / EXPERIMENT / 'physical_random'
    if (folder.parent / 'pool-index.json').exists():
        raise ValueError('The pool is already sealed; do not rewrite it')
    if any((root / 'runs' / EXPERIMENT / model).exists() for model in suite['models']):
        raise ValueError('Evaluation has already started; do not rewrite its pool provenance')
    pending = []
    # Validate every file before making any change. No accepted episode is rerun.
    for task in suite['tasks']:
        versions = []
        for suffix in ('.json', '.progress.json'):
            path = folder / (task + suffix)
            if not path.exists():
                continue
            raw = path.read_bytes()
            prior = json.loads(raw)
            validate_record(prior, suite, task)
            compatible(prior['provenance'], current)
            versions.append(prior)
            if prior['provenance'] == current:
                continue
            updated = seal(dict(prior, provenance=current))
            pending.append((path, raw, prior, updated))
        if len(versions) == 2 and (versions[0]['episodes'] != versions[1]['episodes'] or
                                    versions[0]['rejected'] != versions[1]['rejected']):
            raise ValueError(f'Frozen and progress records disagree: {task}')
    audit = root / 'reports/pool-preservation' / EXPERIMENT / RULE
    for path, raw, prior, updated in pending:
        backup = audit / 'originals' / (path.name + '.' + hashlib.sha256(raw).hexdigest())
        backup.parent.mkdir(parents=True, exist_ok=True)
        if backup.exists():
            if backup.read_bytes() != raw:
                raise ValueError('Original evidence backup differs')
        else:
            with backup.open('xb') as target:
                target.write(raw)
        receipt = dict(rule=RULE, manifest=str(path), original_backup=str(backup),
                       previous_sha256=prior['sha256'], current_sha256=updated['sha256'],
                       previous_provenance=prior['provenance'], current_provenance=current,
                       accepted_unchanged=len(prior['episodes']), rejected_unchanged=len(prior['rejected']),
                       episodes_sha256=digest(prior['episodes']), rejected_sha256=digest(prior['rejected']),
                       expert_reexecuted=False,
                       rationale='Only branches that formerly aborted the task (unavailable grasp annotations; task exceptions '
                                 'raised after the official planner had recorded plan_success = False) now reject the candidate; '
                                 'all previously returned accepted/rejected results follow identical code paths.')
        # Write the receipt before replacing the manifest; reruns are idempotent.
        atomic_json(audit / (path.name + '.receipt.json'), receipt)
        atomic_json(path, updated)
        print(f'POOL_PROGRESS_PRESERVED {path.name}: accepted={len(prior["episodes"])}; '
              f'rejected={len(prior["rejected"])}; expert replay=0; backup={backup}', flush=True)
    print(f'POOL_CONTINUATION_READY: {len(pending)} saved files advanced by reviewed rule; '
          'same v5 pool, no older experiment import', flush=True)


def main():
    import parallel_eval as controller
    original = controller.freeze_pool

    def resume(processes, args, suite, base, sim_gpus, fingerprints, job_dir):
        # main() already holds the exclusive experiment lock and has checked all
        # allocated GPUs, assets and absence of policy evaluation directories.
        audit_assets(ROOT, job_dir / 'grasp-assets.json')
        preserve_progress(ROOT, suite, fingerprints[0])
        return original(processes, args, suite, base, sim_gpus, fingerprints, job_dir)

    controller.freeze_pool = resume
    controller.main()


if __name__ == '__main__':
    main()
