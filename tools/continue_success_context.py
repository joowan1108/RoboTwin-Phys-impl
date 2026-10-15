#!/usr/bin/env python3
"""Continue the v6 pool/evaluation after the reviewed success-context fix, without replay.

The fix touches only the evaluation path of src/robotwin_phys/worker.py (three
tasks' check_success attributes are derived from the initial scene) and adds
src/robotwin_phys/success_context.py. Expert verification, conditions, initial
states and the sampled physics are unchanged, so every saved expert record stays
valid. The runtime fingerprint hashes every source file, so the saved records'
provenance is advanced by this exact reviewed transition, with original backups
and receipts. A sealed pool index is re-sealed over the advanced manifests, and
run.json files of already evaluated tasks are re-pinned to the advanced
manifest hashes. Nothing else may differ.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))
from robotwin_phys.conditions import digest
from robotwin_phys.manifest import atomic_json, seal

WORKER = 'src/robotwin_phys/worker.py'
CONTEXT = 'src/robotwin_phys/success_context.py'
BEFORE_WORKER = '2f3b7af34a1801812ae6effb2612e0ec6f88043107e00e834313e5d40abb4b62'
AFTER_WORKER = '158efc15d48b0aba357f9d14e892c0e4a69dc003e4a4211ec08d074f7ab08970'
CONTEXT_SHA256 = '2cb9de14ead38c1d47ad429691011893f8fca925035b414ee9625ed45a18ff23'
EXPERIMENT = 'phys-pro6000-100-v6'
RULE = 'success-context-from-initial-scene-v1'
PROTOCOL = 'physical_random'


def compatible(previous, current):
    for value in (previous, current):
        if value.get('source_sha256') != digest(value['files']):
            raise ValueError('Source provenance checksum is invalid')
    if current['files'].get(WORKER) != AFTER_WORKER or current['files'].get(CONTEXT) != CONTEXT_SHA256:
        raise ValueError('The current source is not the reviewed success-context fix')
    if previous == current:
        return
    if previous['files'].get(WORKER) != BEFORE_WORKER or CONTEXT in previous['files']:
        raise ValueError('Only the delivered v6 pool source can be advanced by this rule')
    expected = copy.deepcopy(previous)
    expected['files'][WORKER] = AFTER_WORKER
    expected['files'][CONTEXT] = CONTEXT_SHA256
    expected['source_sha256'] = digest(expected['files'])
    if expected != current:
        raise ValueError('Runtime/assets/config/source changed beyond the reviewed success-context fix')


def backup_and_receipt(audit, path, raw, extra):
    backup = audit / 'originals' / (path.name + '.' + hashlib.sha256(raw).hexdigest())
    backup.parent.mkdir(parents=True, exist_ok=True)
    if backup.exists():
        if backup.read_bytes() != raw:
            raise ValueError('Original evidence backup differs')
    else:
        with backup.open('xb') as target:
            target.write(raw)
    receipt = dict(rule=RULE, file=str(path), original_backup=str(backup), expert_reexecuted=False,
                   rationale='Evaluation-only change: three tasks\' check_success attributes are derived from the '
                             'verified initial scene with the official play_once expressions. Expert records, '
                             'conditions and initial states are untouched.', **extra)
    atomic_json(audit / (path.name + '.' + hashlib.sha256(str(path).encode()).hexdigest()[:12] + '.receipt.json'), receipt)
    return backup


def preserve(root, suite, current):
    from resume_contact_pool import validate_record
    import parallel_eval as controller
    if suite['name'] != EXPERIMENT:
        raise ValueError(f'This reviewed continuation applies only to {EXPERIMENT}')
    compatible(current, current)
    folder = root / 'manifests' / EXPERIMENT / PROTOCOL
    audit = root / 'reports/pool-preservation' / EXPERIMENT / RULE
    pending, old_sha, new_sha = [], {}, {}
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
            if suffix == '.json':
                old_sha[task] = prior['sha256']
            if prior['provenance'] == current:
                if suffix == '.json':
                    new_sha[task] = prior['sha256']
                continue
            updated = seal(dict(prior, provenance=current))
            if suffix == '.json':
                new_sha[task] = updated['sha256']
            pending.append((path, raw, prior, updated))
        if len(versions) == 2 and (versions[0]['episodes'] != versions[1]['episodes'] or
                                    versions[0]['rejected'] != versions[1]['rejected']):
            raise ValueError(f'Frozen and progress records disagree: {task}')
    index_path = folder.parent / 'pool-index.json'
    index_update = None
    if index_path.exists():
        saved = json.loads(index_path.read_bytes())
        if saved['manifests'] != old_sha or set(old_sha) != set(suite['tasks']):
            raise ValueError('Sealed pool index does not match the frozen manifests; refusing to rewrite it')
        index_update = seal(dict(saved, manifests=new_sha))
    run_updates = []
    for model in suite['models']:
        for task in suite['tasks']:
            meta = root / 'runs' / EXPERIMENT / model / PROTOCOL / task / 'run.json'
            if not meta.exists():
                continue
            run = json.loads(meta.read_bytes())
            if task not in old_sha or run.get('manifest_sha256') != old_sha[task]:
                raise ValueError(f'Evaluation run {model}/{task} is pinned to an unknown manifest; refusing')
            if old_sha[task] != new_sha[task]:
                run_updates.append((meta, meta.read_bytes(), run, dict(run, manifest_sha256=new_sha[task])))
    # Validate everything first; then write receipts before each replacement. Reruns are idempotent.
    for path, raw, prior, updated in pending:
        backup = backup_and_receipt(audit, path, raw, dict(
            previous_sha256=prior['sha256'], current_sha256=updated['sha256'],
            previous_provenance=prior['provenance'], current_provenance=current,
            accepted_unchanged=len(prior['episodes']), rejected_unchanged=len(prior['rejected']),
            episodes_sha256=digest(prior['episodes']), rejected_sha256=digest(prior['rejected'])))
        atomic_json(path, updated)
        print(f'POOL_PROGRESS_PRESERVED {path.name}: accepted={len(prior["episodes"])}; '
              f'rejected={len(prior["rejected"])}; expert replay=0; backup={backup}', flush=True)
    if index_update is not None and index_update != json.loads(index_path.read_bytes()):
        raw = index_path.read_bytes()
        backup_and_receipt(audit, index_path, raw, dict(previous_sha256=json.loads(raw)['sha256'],
                                                        current_sha256=index_update['sha256']))
        atomic_json(index_path, index_update)
        print(f'POOL_INDEX_RESEALED {index_path}: {len(new_sha)} manifests', flush=True)
    for meta, raw, run, updated in run_updates:
        backup_and_receipt(audit, meta, raw, dict(previous_manifest_sha256=run['manifest_sha256'],
                                                  current_manifest_sha256=updated['manifest_sha256']))
        atomic_json(meta, updated)
        print(f'EVAL_RUN_REPINNED {meta}', flush=True)
    print(f'POOL_CONTINUATION_READY: {len(pending)} saved files advanced by reviewed rule {RULE}; '
          f'index resealed={index_update is not None}; runs repinned={len(run_updates)}', flush=True)


def main():
    import parallel_eval as controller
    captured = {}
    original_provenance, original_verify, original_freeze = (
        controller.provenance_on_gpu, controller.verify_pool, controller.freeze_pool)

    def provenance_on_gpu(processes, base, gpu, path):
        fingerprint = original_provenance(processes, base, gpu, path)
        captured.setdefault('fingerprint', fingerprint)
        return fingerprint

    def verify_pool(suite, path):
        preserve(ROOT, suite, captured['fingerprint'])
        return original_verify(suite, path)

    def freeze_pool(processes, args, suite, base, sim_gpus, fingerprints, job_dir):
        preserve(ROOT, suite, fingerprints[0])
        return original_freeze(processes, args, suite, base, sim_gpus, fingerprints, job_dir)

    controller.provenance_on_gpu = provenance_on_gpu
    controller.verify_pool = verify_pool
    controller.freeze_pool = freeze_pool
    controller.main()


if __name__ == '__main__':
    main()
