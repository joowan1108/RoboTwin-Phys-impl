#!/usr/bin/env python3
"""Frozen multi-policy suite; partial results never become paper-table scores."""
import argparse
import csv
import fcntl
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from robotwin_phys.conditions import ROOT, digest, task_names
from robotwin_phys.manifest import atomic_json, load, seal
from randomization_audit import export_audit, episode_record

MODELS = {'g05': 'G0.5', 'pi05': 'π0.5', 'fact': 'FACT', 'motus': 'Motus', 'fastwam': 'Fast-WAM'}
PROTOCOLS = ['clean', 'official_random', 'physical_random']
NAMES = ['Clean', 'RoboTwin 2.0 randomization', 'RoboTwin-Phys randomization']


def read_suite(folder):
    suite = json.loads((folder / 'suite.json').read_text())
    if seal(suite)['sha256'] != suite['sha256']:
        raise ValueError('Suite was modified; create a new experiment')
    return suite


def manifest_path(suite, protocol, task):
    return ROOT / 'manifests' / suite['name'] / protocol / f'{task}.json'


def check_manifest(suite, protocol, task):
    m = load(manifest_path(suite, protocol, task))
    expected = {'task': task, 'protocol': protocol, 'start_seed': suite['seed'],
                'requested_episodes': suite['episodes'], 'instruction_type': suite['instruction_type'],
                'only': [], 'without': []}
    for key, value in expected.items():
        if m[key] != value:
            raise ValueError(f'Manifest {task}/{protocol}: {key} mismatch')
    if len(m['episodes']) != suite['episodes']:
        raise ValueError('Manifest episode count mismatch')
    return m


def inspect_cell(folder, suite, model, protocol, task):
    row = dict(model=model, protocol=protocol, task=task, expected=suite['episodes'],
               completed=0, successes=0, success_rate=None, status='missing', manifest_sha256=None)
    try:
        path = manifest_path(suite, protocol, task)
        if not path.exists():
            row['status'] = 'manifest_missing'
            return row
        m = check_manifest(suite, protocol, task)
        row['manifest_sha256'] = m['sha256']
        run_dir = folder / model / protocol / task
        if not (run_dir / 'run.json').exists():
            return row
        run = json.loads((run_dir / 'run.json').read_text())
        policy = run['policy']
        if (run['mode'] != 'evaluate' or run.get('smoke_max_actions') is not None
                or policy['name'] != f'{model}_robotwin2' or run['policy_seed'] != suite['policy_seed']
                or run['manifest_sha256'] != m['sha256']):
            raise ValueError('Run identity/debug/manifest mismatch')
        row['policy_sha256'] = digest(policy)
        expected_files = {f"episode-{e['episode_index']:04d}.result.json" for e in m['episodes']}
        if any(p.name not in expected_files for p in run_dir.glob('episode-*.result.json')):
            raise ValueError('Unexpected extra episode results')
        for ep in m['episodes']:
            file = run_dir / f"episode-{ep['episode_index']:04d}.result.json"
            if not file.exists():
                continue
            result = json.loads(file.read_text())
            if result['status'] != 'completed':
                continue
            for key in ('episode_index', 'seed', 'condition_sha256', 'initial_state_sha256'):
                if result[key] != ep[key]:
                    raise ValueError(f'Episode {ep["episode_index"]}: {key} mismatch')
            if result['policy'] != policy or not result.get('benchmark_eligible'):
                raise ValueError('Policy mismatch or debug/noop result')
            if type(result['success']) is not bool:
                raise ValueError('Success must be a boolean')
            row['completed'] += 1
            row['successes'] += result['success']
        row['status'] = 'complete' if row['completed'] == row['expected'] else 'partial'
        if row['status'] == 'complete':
            row['success_rate'] = row['successes'] / row['expected']
    except (ValueError, KeyError, TypeError, OSError) as e:
        row['status'], row['error'] = 'invalid', str(e)
        row['success_rate'] = None
    return row


def report(folder):
    with (folder / '.report.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _report(folder)


def _report(folder):
    suite = read_suite(folder)
    rows = [inspect_cell(folder, suite, model, protocol, task)
            for model in suite['models'] for protocol in suite['protocols'] for task in suite['tasks']]
    audit = export_audit(folder, suite, lambda protocol, task: check_manifest(suite, protocol, task))
    for row in rows:
        if [row['model'], row['protocol'], row['task']] in audit['invalid_cells']:
            row.update(status='invalid', success_rate=None, error='Randomization evidence mismatch; see randomization audit')
    # One immutable model configuration across all tasks/conditions.
    for model in suite['models']:
        identities = {r['policy_sha256'] for r in rows if r['model'] == model and r.get('policy_sha256')}
        if len(identities) > 1:
            for r in rows:
                if r['model'] == model:
                    r.update(status='invalid', success_rate=None, error='Policy identity changed within suite')
    aggregate = []
    for model in suite['models']:
        for protocol in suite['protocols']:
            group = [r for r in rows if r['model'] == model and r['protocol'] == protocol]
            complete = all(r['status'] == 'complete' for r in group)
            aggregate.append(dict(model=model, protocol=protocol, complete_tasks=sum(r['status'] == 'complete' for r in group),
                expected_tasks=len(group), completed_episodes=sum(r['completed'] for r in group),
                expected_episodes=len(group)*suite['episodes'],
                success_rate=sum(r['success_rate'] for r in group)/len(group) if complete else None))
    full_tasks = set(suite['tasks']) == set(task_names()) and suite['episodes'] == 100
    full = full_tasks and suite['protocols'] == PROTOCOLS
    result = dict(implementation='independent-paper-reproduction', suite_sha256=suite['sha256'],
                  full_benchmark_scope=full, full_task_episode_scope=full_tasks,
                  selected_protocols=suite['protocols'], aggregate=aggregate, tasks=rows, randomization_audit=audit)
    out = folder / 'tables'
    out.mkdir(exist_ok=True)
    atomic_json(out / 'results.json', result)
    for name, data in [('summary', aggregate), ('tasks', rows)]:
        keys = sorted(set().union(*(r.keys() for r in data)))
        tmp = out / f'.{name}.csv.tmp'
        with tmp.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader(); writer.writerows(data)
        tmp.replace(out / f'{name}.csv')
    scope = f"{len(suite['tasks'])} tasks × {suite['episodes']} episodes per model/protocol."
    scope += ' All 50 tasks with 100 episodes.' if full_tasks else ' Reduced task/episode scope.'
    lines = ['# Policy success rate (%)', '', 'Independent paper reproduction; ' + scope, '',
             '| Model | ' + ' | '.join(NAMES[PROTOCOLS.index(p)] for p in suite['protocols']) + ' |',
             '|---|' + '---:|'*len(suite['protocols'])]
    latex = [r'\begin{tabular}{l' + 'r'*len(suite['protocols']) + '}', r'\hline',
             'Model & ' + ' & '.join(['Clean' if p == 'clean' else ('Official' if p == 'official_random' else 'Phys') for p in suite['protocols']]) + r' \\', r'\hline']
    for model in suite['models']:
        cells, tex = [], []
        for protocol in suite['protocols']:
            r = next(r for r in aggregate if r['model'] == model and r['protocol'] == protocol)
            score = r['success_rate']
            cells.append(f'{100*score:.2f}' if score is not None else f'— ({r["completed_episodes"]}/{r["expected_episodes"]} episodes)')
            tex.append(f'{100*score:.2f}' if score is not None else '--')
        lines.append('| ' + MODELS[model] + ' | ' + ' | '.join(cells) + ' |')
        label = r'$\pi_{0.5}$' if model == 'pi05' else MODELS[model]
        latex.append(label + ' & ' + ' & '.join(tex) + r' \\')
    lines += ['', 'Each score is the arithmetic mean of task success rates. Missing, interrupted, invalid, or debug results do not count as failures and suppress the aggregate score. Completed policy failures remain in the denominator.', '',
              'The same verified manifest is shared by every policy for each task/protocol. Different protocols can accept different seeds after expert filtering; this is not a paired comparison across protocols.', '',
              'Detailed task results and completion counts: `tasks.csv`, `summary.csv`, `results.json`.']
    latex += [r'\hline', r'\end{tabular}', '% Independent reproduction. Missing/incomplete cells are --. See results.json for scope and coverage.']
    for name, content in [('results.md', lines), ('results.tex', latex)]:
        tmp = out / ('.' + name + '.tmp'); tmp.write_text('\n'.join(content)+'\n'); tmp.replace(out / name)
    print('\n'.join(lines), flush=True)
    return result


def completion_status(folder, model, tasks, protocols):
    result = report(folder)
    selected = [r for r in result['tasks'] if r['model'] == model and r['task'] in tasks and r['protocol'] in protocols]
    if any(r['status'] == 'invalid' for r in result['tasks'] if r['model'] == model):
        return 2
    return 0 if selected and all(r['status'] == 'complete' for r in selected) else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['init', 'freeze', 'evaluate', 'report', 'check-complete'])
    p.add_argument('--experiment', required=True, type=Path)
    p.add_argument('--tasks', nargs='+', help='Init: defaults to all 50; freeze/evaluate: optional subset')
    p.add_argument('--protocols', nargs='+', choices=PROTOCOLS)
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--seed', type=int, default=100000)
    p.add_argument('--policy-seed', type=int, default=0)
    p.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    p.add_argument('--model', choices=MODELS)
    p.add_argument('--policy-port', type=int, default=9000)
    p.add_argument('--max-candidates', type=int, default=10000)
    p.add_argument('--timeout', type=int, default=1800)
    p.add_argument('--video', action='store_true')
    p.add_argument('--live', action='store_true')
    p.add_argument('--port', type=int, default=8765)
    args = p.parse_args()
    folder = args.experiment.resolve(); folder.mkdir(parents=True, exist_ok=True)
    if args.command == 'init':
        tasks = args.tasks or task_names()
        if len(set(args.models)) != len(args.models) or (args.protocols and len(set(args.protocols)) != len(args.protocols)):
            p.error('Duplicate models/protocols')
        if not tasks or len(set(tasks)) != len(tasks) or not set(tasks) <= set(task_names()) or args.episodes < 1:
            p.error('Invalid tasks/episode count')
        suite = seal(dict(schema='robotwin-experiment-v1', name=folder.name, tasks=tasks,
            protocols=args.protocols or PROTOCOLS, models=args.models, episodes=args.episodes, seed=args.seed,
            policy_seed=args.policy_seed, instruction_type='unseen'))
        if (folder / 'suite.json').exists() and read_suite(folder) != suite:
            raise ValueError('Existing experiment has different settings; choose another directory')
        atomic_json(folder / 'suite.json', suite)
        report(folder); return
    suite = read_suite(folder)
    if args.command == 'report':
        report(folder); return
    tasks, protocols = args.tasks or suite['tasks'], args.protocols or suite['protocols']
    if not set(tasks) <= set(suite['tasks']) or not set(protocols) <= set(suite['protocols']):
        p.error('Selection is outside frozen suite')
    if args.command in ('evaluate', 'check-complete') and args.model not in suite['models']:
        p.error('Select --model from the suite')
    if args.command == 'check-complete':
        raise SystemExit(completion_status(folder, args.model, tasks, protocols))
    try:
        for protocol in protocols:
            for task in tasks:
                lock = folder / '.locks' / f'{args.command}-{args.model}-{protocol}-{task}.lock'
                lock.parent.mkdir(exist_ok=True)
                with lock.open('w') as f:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    path = manifest_path(suite, protocol, task)
                    cmd = [sys.executable, '-m', 'robotwin_phys.cli', args.command]
                    if args.command == 'freeze':
                        if path.exists():
                            m = check_manifest(suite, protocol, task)
                            from robotwin_phys.provenance import fingerprint
                            if m['provenance'] != fingerprint():
                                raise ValueError('Existing manifest runtime differs; use a new experiment')
                            continue
                        cmd += ['--task', task, '--protocol', protocol, '--episodes', str(suite['episodes']),
                            '--seed', str(suite['seed']), '--max-candidates', str(args.max_candidates), '--output', str(path)]
                    else:
                        check_manifest(suite, protocol, task)
                        from robotwin_phys.rpc import PolicyClient
                        c = PolicyClient('127.0.0.1', args.policy_port)
                        identity = c.identity; c.close()
                        if identity['name'] != f'{args.model}_robotwin2':
                            raise ValueError('Wrong policy server for selected model')
                        cmd += ['--manifest', str(path), '--output', str(folder/args.model/protocol/task),
                                '--policy-port', str(args.policy_port), '--policy-seed', str(suite['policy_seed'])]
                    cmd += ['--resume', '--timeout', str(args.timeout), '--port', str(args.port)]
                    if args.video: cmd.append('--video')
                    if args.live: cmd.append('--live')
                    subprocess.run(cmd, check=True)
                    if args.command == 'evaluate':
                        manifest = check_manifest(suite, protocol, task)
                        for ep in manifest['episodes']:
                            audit = episode_record(folder, suite, args.model, protocol, task, manifest, ep)
                            if audit is None or not audit['verified']:
                                raise ValueError(f'Randomization audit failed: {args.model}/{protocol}/{task}/{ep["episode_index"]}')
    finally:
        report(folder)


if __name__ == '__main__':
    main()
