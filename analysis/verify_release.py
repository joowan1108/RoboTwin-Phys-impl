#!/usr/bin/env python3
"""Verify an exported dataset against this repository's condition sampler and seals.

Run inside the repository (needs src/ and configs/physical_domains.json):
  python3 analysis/verify_release.py data/phys-pro6000-100-v6

Checks, and prints a count for each:
  - every row of pool/conditions.csv: sample(task, seed, protocol) reproduces condition_sha256 and the
    flattened attribute columns;
  - pool/seeds.json agrees with conditions.csv (accepted seeds, order, counts);
  - full/manifests/*.json (if present): seal intact, seeds and hashes match the CSV, status verified;
  - full/pool-index.json (if present): seal intact and manifest hashes match;
  - results/<model>/episodes.csv: every (task, episode_index) exists in the pool and identity.json
    pins the manifest hashes of the exported manifests.
Exit status 1 on the first failed check.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'analysis'))
from robotwin_phys.conditions import digest, sample  # noqa: E402
from robotwin_phys.manifest import seal  # noqa: E402
from export_release import COLUMNS, flatten  # noqa: E402


def same(a, b):
    if a in ('', None) or b in ('', None):
        return (a in ('', None)) and (b in ('', None))
    try:
        return math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return str(a) == str(b)


def verify(folder):
    folder = Path(folder)
    checks = {}
    with (folder / 'pool/conditions.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    if [c for c in rows[0]] != COLUMNS:
        raise AssertionError('conditions.csv columns differ from export_release.COLUMNS')
    for r in rows:
        cond = sample(r['task'], int(r['seed']), r['protocol'], tuple(x for x in r['active_factors'].split('|') if x) or (), ())
        if digest(cond) != r['condition_sha256']:
            raise AssertionError(f'{r["task"]} seed {r["seed"]}: condition hash does not reproduce')
        expected = flatten(r['task'], dict(episode_index=int(r['episode_index']), seed=int(r['seed']), condition=cond,
                                           condition_sha256=r['condition_sha256']))
        for k in COLUMNS:
            if k in ('initial_state_sha256', 'expert_seconds', 'instruction'):
                continue
            if not same(expected[k], r[k]):
                raise AssertionError(f'{r["task"]} seed {r["seed"]}: column {k} = {r[k]!r}, sampler gives {expected[k]!r}')
    checks['conditions_reproduced'] = len(rows)
    seeds = json.loads((folder / 'pool/seeds.json').read_text())['tasks']
    by_task = {}
    for r in rows:
        by_task.setdefault(r['task'], []).append(r)
    for task, rec in seeds.items():
        got = [int(r['seed']) for r in sorted(by_task.get(task, []), key=lambda r: int(r['episode_index']))]
        if got != rec['accepted']:
            raise AssertionError(f'{task}: seeds.json accepted list differs from conditions.csv')
        if set(map(int, rec['rejected'])) & set(rec['accepted']):
            raise AssertionError(f'{task}: a seed is both accepted and rejected')
    if set(seeds) != set(by_task):
        raise AssertionError('seeds.json and conditions.csv cover different tasks')
    checks['tasks'] = len(seeds)
    manifests = sorted((folder / 'full/manifests').rglob('*.json')) if (folder / 'full/manifests').exists() else []
    hashes = {}
    for path in manifests:
        m = json.loads(path.read_text())
        if seal({k: v for k, v in m.items() if k != 'sha256'})['sha256'] != m['sha256']:
            raise AssertionError(f'{path.name}: seal broken')
        if m['status'] != 'verified' or m['sha256'] != seeds[m['task']]['manifest_sha256']:
            raise AssertionError(f'{path.name}: status or hash differs from seeds.json')
        csv_rows = sorted(by_task[m['task']], key=lambda r: int(r['episode_index']))
        if [(e['seed'], e['condition_sha256'], e['initial_state_sha256']) for e in m['episodes']] != \
                [(int(r['seed']), r['condition_sha256'], r['initial_state_sha256']) for r in csv_rows]:
            raise AssertionError(f'{path.name}: episodes differ from conditions.csv')
        hashes[m['task']] = m['sha256']
    checks['manifests_sealed'] = len(manifests)
    index_path = folder / 'full/pool-index.json'
    if index_path.exists():
        index = json.loads(index_path.read_text())
        if seal({k: v for k, v in index.items() if k != 'sha256'})['sha256'] != index['sha256']:
            raise AssertionError('pool-index.json: seal broken')
        if hashes and any(index['manifests'].get(t) != h for t, h in hashes.items()):
            raise AssertionError('pool-index.json: manifest hashes differ from the exported manifests')
        checks['pool_index_sealed'] = 1
    keys = {(r['task'], r['episode_index']) for r in rows}
    for model_dir in sorted((folder / 'results').iterdir()) if (folder / 'results').exists() else []:
        with (model_dir / 'episodes.csv').open(newline='') as f:
            res = list(csv.DictReader(f))
        for r in res:
            if (r['task'], r['episode_index']) not in keys:
                raise AssertionError(f'{model_dir.name}: result for unknown episode {r["task"]}/{r["episode_index"]}')
        identity = json.loads((model_dir / 'identity.json').read_text())
        for task, h in identity['manifests'].items():
            if task in hashes and h != hashes[task]:
                raise AssertionError(f'{model_dir.name}: pinned manifest hash for {task} differs from the exported manifest')
        checks[f'results_{model_dir.name}'] = len(res)
    return checks


def main(argv=None):
    folder = (argv or sys.argv[1:])[0]
    try:
        checks = verify(folder)
    except AssertionError as exc:
        print(f'FAIL: {exc}')
        return 1
    for k, v in checks.items():
        print(f'{k}: {v}')
    print('OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
