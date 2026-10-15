#!/usr/bin/env python3
"""Read-only report of the source hashes involved in job 1641 pool migration."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def report(root=ROOT, task='adjust_bottle', job='1641'):
    paths = ['src/robotwin_phys/numerics.py', 'src/robotwin_phys/planning.py',
             'src/robotwin_phys/simulator.py', 'src/robotwin_phys/provenance.py',
             'src/robotwin_phys/curobo_compat.py', 'src/robotwin_phys/cli.py']
    selected = None
    sources = []
    for experiment in ('phys-pro6000-100-v3', 'phys-pro6000-100-v2'):
        folder = root / 'manifests' / experiment / 'physical_random'
        chosen = next((p for p in (folder / f'{task}.json', folder / f'{task}.progress.json') if p.exists()), None)
        record = {'experiment': experiment, 'exists': chosen is not None}
        if chosen is not None:
            data = json.loads(chosen.read_text())
            provenance = data.get('provenance', {})
            record.update(path=str(chosen), status=data.get('status'), accepted=len(data.get('episodes', [])),
                          rejected=len(data.get('rejected', [])), source_sha256=provenance.get('source_sha256'),
                          files={p: {'present': p in provenance.get('files', {}),
                                     'sha256': provenance.get('files', {}).get(p)} for p in paths})
            if selected is None:
                selected = record
        sources.append(record)
    directory = root / 'reports/parallel/phys-pro6000-100-v4' / f'fastwam-{job}'
    current_path = directory / 'provenance-0.json'
    current = json.loads(current_path.read_text()) if current_path.exists() else {}
    approved_path = root / 'tools/expert_fix_previous.json'
    approved = json.loads(approved_path.read_text()) if approved_path.exists() else {}
    key = paths[0]
    previous = selected['files'][key] if selected else None
    known = approved.get(key)
    diagnosis = 'No saved source pool was found'
    if previous is not None:
        if not previous['present']:
            diagnosis = 'Reviewed helper was absent in the selected older source; the old migration guard rejected this addition'
        elif previous['sha256'] in (known if isinstance(known, list) else [known]):
            diagnosis = 'Previous numerics.py hash is approved; compare the actually loaded approval file/tool version'
        else:
            diagnosis = 'Previous numerics.py exists with an unknown hash; its old source contents need review'
    result = dict(task=task, job=job, selected_source=selected['path'] if selected else None,
                  diagnosis=diagnosis, sources=sources, current_provenance_path=str(current_path),
                  current_recorded_hashes={p: current.get('files', {}).get(p) for p in paths},
                  current_loaded_hashes={p: hashlib.sha256((root / p).read_bytes()).hexdigest()
                                         if (root / p).exists() else None for p in paths},
                  approved_hashes={p: approved.get(p) for p in paths},
                  loaded_reverify_tool_sha256=hashlib.sha256((root / 'tools/reverify_pool.py').read_bytes()).hexdigest()
                  if (root / 'tools/reverify_pool.py').exists() else None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', default='adjust_bottle')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--job', default='1641')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = report(root=args.root, task=args.task, job=args.job)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
