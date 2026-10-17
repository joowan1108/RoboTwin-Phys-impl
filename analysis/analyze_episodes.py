#!/usr/bin/env python3
"""Analyze every evaluated episode of one model at once.

Reads runs/<experiment>/<model>/<protocol>/<task>/episode-NNNN.{result,request}.json
and episode-NNNN.log and prints:

  1. overall and per-task success with a 95% interval, next to the paper's
     per-task rate (references/paper-table4-physical-random.json) and a flag
     when the observed count is unlikely under the paper rate;
  2. log triage: which known failure signatures (OIDN, CUDA, timeout, state
     mismatch, policy errors) appear in how many episodes, plus the most common
     unclassified error-like lines;
  3. which sampled physical attributes separate failures from successes
     (median split of each scalar factor, camera mode), across all tasks;
  4. record anomalies: result status other than completed, episodes that
     stopped before the step limit without success, more than one policy identity;
  5. with --compare, episode-by-episode agreement with another run folder
     (same task and episode index), e.g. a discarded earlier job.

Standard library only, so it also runs on the cluster login shell:

  python3 analysis/analyze_episodes.py --root <data-root> --experiment <experiment> --model fastwam
  python3 analysis/analyze_episodes.py ... --compare <other-run-folder>
  python3 analysis/analyze_episodes.py ... --errors          # log tails of every failed or errored episode
  python3 analysis/analyze_episodes.py ... --json report.json
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = ROOT / 'references/paper-table4-physical-random.json'

# Known signatures. Order matters only for display.
SIGNATURES = {
    'oidn_error': re.compile(r'OIDN Error'),
    'render_error': re.compile(r'\[svulkan2\] \[error\]'),
    'cuda_error': re.compile(r'CUDA (error|out of memory)|OutOfMemoryError|CUBLAS_STATUS|cudaError'),
    'timeout': re.compile(r'TimeoutError|timed out'),
    'policy_no_action': re.compile(r'Policy returned without taking an action'),
    'policy_identity': re.compile(r'Policy server identity changed'),
    'state_mismatch': re.compile(r'assert_same_state|Initial state .*differ|does not match the verified'),
    'rpc_error': re.compile(r'ConnectionRefusedError|ConnectionResetError|BrokenPipeError|RPC'),
    'traceback': re.compile(r'^Traceback \(most recent call last\)'),
    'planner': re.compile(r'(?i)plan(ning)? (failed|failure)|curobo.*(error|fail)'),
}
HARMLESS = re.compile(r'pytorch3d|DeprecationWarning|UserWarning|FutureWarning|Warp |warp\.|TF_CPP|XLA|'
                      r'tokenizers|HF_HUB|torch\.load|weights_only|triton|flash_attn|is deprecated')
ERRORISH = re.compile(r'(?i)\b(error|exception|failed|fatal|assert(ion)?|refused|killed|abort)\b')
NORMALIZE = re.compile(r'0x[0-9a-fA-F]+|\d+(\.\d+)?')

SCALARS = [
    ('mass_scale', lambda c: c.get('mass_scale')),
    ('com_offset_norm_m', lambda c: math.sqrt(sum(x * x for x in c.get('com_offset_m', [0, 0, 0])))),
    ('geometry_scale', lambda c: c.get('geometry_scale')),
    ('table_tilt_deg', lambda c: c.get('table_tilt_deg')),
    ('table_height_offset_m', lambda c: c.get('table_height_offset_m')),
    ('force_mg', lambda c: c.get('force_mg')),
    ('camera_orbit_m', lambda c: c.get('camera', {}).get('orbit_m')),
    ('camera_translation_m', lambda c: c.get('camera', {}).get('translation_m')),
    ('camera_rotation_deg', lambda c: c.get('camera', {}).get('rotation_deg')),
    ('friction', lambda c: c.get('friction')),
    ('restitution', lambda c: c.get('restitution')),
    ('object_joint_damping', lambda c: c.get('object_joint_damping')),
    ('arm_damping_scale', lambda c: c.get('arm_damping_scale')),
]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def binom_two_sided(k, n, p):
    """Exact two-sided probability of a count at least as extreme as k under Binomial(n, p)."""
    if n == 0:
        return 1.0
    pmf = [math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    return min(1.0, sum(x for x in pmf if x <= pmf[k] + 1e-12))


def load_step_limits(path):
    limits = {}
    if path and Path(path).exists():
        for line in Path(path).read_text().splitlines():
            m = re.match(r'^\s*([A-Za-z0-9_]+)\s*:\s*(\d+)\s*$', line)
            if m:
                limits[m.group(1)] = int(m.group(2))
    return limits


def scan_log(path):
    """Return (set of signatures, Counter of normalized unclassified error-like lines)."""
    hits, other = set(), Counter()
    if not path.exists():
        return hits, other
    for line in path.read_text(errors='replace').splitlines():
        matched = False
        for name, rx in SIGNATURES.items():
            if rx.search(line):
                hits.add(name)
                matched = True
        if not matched and ERRORISH.search(line) and not HARMLESS.search(line):
            other[NORMALIZE.sub('#', line.strip())[:160]] += 1
    return hits, other


def load_episodes(folder, tasks=None):
    """folder = runs/<experiment>/<model>/<protocol>. Returns list of episode dicts."""
    episodes = []
    for task_dir in sorted(p for p in folder.iterdir() if p.is_dir() and not p.name.startswith('.')):
        if tasks and task_dir.name not in tasks:
            continue
        for req in sorted(task_dir.glob('episode-*.request.json')):
            stem = req.name[:-len('.request.json')]
            res = task_dir / (stem + '.result.json')
            log = task_dir / (stem + '.log')
            try:
                request = json.loads(req.read_text())
            except Exception as exc:
                request = {'_unreadable': str(exc)}
            result = None
            if res.exists():
                try:
                    result = json.loads(res.read_text())
                except Exception as exc:
                    result = {'status': 'unreadable', 'error': str(exc)}
            sigs, other = scan_log(log)
            episodes.append(dict(task=task_dir.name, index=int(stem.split('-')[1]), stem=stem, request=request,
                                 result=result, log=log, signatures=sigs, other_errors=other,
                                 condition=(request.get('episode') or {}).get('condition') or {}))
    return episodes


def running_episodes(runs_model_folder, model):
    locks = runs_model_folder.parent / '.episode-locks' / model
    if not locks.exists():
        return []
    return sorted(str(p.relative_to(locks)) for p in locks.rglob('*') if p.is_file())


def fmt_pct(k, n):
    return f'{100 * k / n:5.1f}%' if n else '    - '


def analyze(args):
    runs = Path(args.root).expanduser() / 'runs' / args.experiment / args.model / args.protocol
    if not runs.exists():
        raise SystemExit(f'No run folder: {runs}')
    reference = {}
    if args.reference and Path(args.reference).exists():
        ref = json.loads(Path(args.reference).read_text())
        reference = {t: v.get(args.model) for t, v in ref['tasks'].items()}
        overall_ref = ref.get('overall', {}).get(args.model)
    else:
        overall_ref = None
    limits = load_step_limits(args.step_limits)
    episodes = load_episodes(runs, set(args.tasks) if args.tasks else None)
    completed = [e for e in episodes if e['result'] and e['result'].get('status') == 'completed']
    errored = [e for e in episodes if e['result'] and e['result'].get('status') != 'completed']
    pending = [e for e in episodes if e['result'] is None]
    succ = sum(bool(e['result']['success']) for e in completed)
    report = {'experiment': args.experiment, 'model': args.model, 'protocol': args.protocol,
              'episodes': len(episodes), 'completed': len(completed), 'errored': len(errored), 'pending': len(pending),
              'successes': succ, 'tasks': {}, 'signatures': {}, 'factors': {}, 'anomalies': []}
    out = []
    lo, hi = wilson(succ, len(completed))
    out.append(f'== {args.experiment} / {args.model} / {args.protocol}')
    out.append(f'episodes with request: {len(episodes)}  completed: {len(completed)}  errored: {len(errored)}  '
               f'no result yet: {len(pending)}  running locks: {len(running_episodes(runs, args.model))}')
    line = f'overall success: {succ}/{len(completed)} = {fmt_pct(succ, len(completed))}  95% CI [{100*lo:.1f}, {100*hi:.1f}]'
    if overall_ref is not None:
        line += f'   paper overall: {overall_ref:.2f}%'
    out.append(line)

    # Expected overall under the paper's per-task rates for the tasks actually evaluated so far.
    if reference and completed:
        expected = sum(reference.get(e['task'], 0) or 0 for e in completed) / 100
        out.append(f'expected successes if every task matched the paper rate: {expected:.1f} (observed {succ})')
        report['expected_from_paper'] = expected

    # ---- per task
    out.append('')
    out.append(f'{"task":28s} {"n":>3s} {"succ":>4s} {"rate":>7s} {"paper":>6s} {"p":>6s}  {"actions":>8s} {"sec":>6s}  flags')
    by_task = defaultdict(list)
    for e in episodes:
        by_task[e['task']].append(e)
    flagged = []
    for task in sorted(by_task):
        eps = by_task[task]
        done = [e for e in eps if e['result'] and e['result'].get('status') == 'completed']
        k, n = sum(bool(e['result']['success']) for e in done), len(done)
        paper = reference.get(task)
        pval = binom_two_sided(k, n, paper / 100) if (paper is not None and n) else None
        actions = statistics.mean(e['result'].get('actions', 0) for e in done) if done else 0
        secs = statistics.mean(e['result'].get('seconds', 0) for e in done) if done else 0
        flags = []
        if pval is not None and pval < args.alpha and n >= args.min_episodes:
            flags.append('LOW' if k / n < paper / 100 else 'HIGH')
        sig = Counter(s for e in eps for s in e['signatures'])
        for s, c in sig.items():
            flags.append(f'{s}x{c}')
        err = [e for e in eps if e['result'] and e['result'].get('status') != 'completed']
        if err:
            flags.append(f'errored x{len(err)}')
        if any(f.startswith(('LOW', 'HIGH')) for f in flags):
            flagged.append(task)
        out.append(f'{task:28s} {n:3d} {k:4d} {fmt_pct(k, n):>7s} {("%3d%%" % paper) if paper is not None else "   -":>6s} '
                   f'{(f"{pval:.3f}" if pval is not None else "-"):>6s}  {actions:8.1f} {secs:6.0f}  {" ".join(flags)}')
        report['tasks'][task] = dict(n=n, successes=k, paper=paper, p_value=pval, mean_actions=actions,
                                     mean_seconds=secs, signatures=dict(sig), errored=len(err),
                                     step_limit=limits.get(task))
    out.append(f'p = exact two-sided binomial probability of the observed count under the paper rate; '
               f'LOW/HIGH flagged when p < {args.alpha} and n >= {args.min_episodes}.')
    if flagged:
        out.append(f'tasks that deviate from the paper: {", ".join(flagged)}')

    # ---- log triage
    out.append('')
    out.append('== log triage (episodes containing each signature)')
    sig_all = Counter(s for e in episodes for s in e['signatures'])
    if not sig_all:
        out.append('no known failure signature in any episode log')
    for s, c in sig_all.most_common():
        tasks_hit = Counter(e['task'] for e in episodes if s in e['signatures'])
        fail_hit = sum(1 for e in episodes if s in e['signatures'] and e['result'] and not e['result'].get('success'))
        out.append(f'{s:18s} {c:4d} episodes ({fail_hit} of them failed): '
                   + ', '.join(f'{t}x{n}' for t, n in tasks_hit.most_common(8)))
        report['signatures'][s] = dict(episodes=c, failed=fail_hit, tasks=dict(tasks_hit))
    other = Counter()
    for e in episodes:
        other.update(e['other_errors'])
    if other:
        out.append('most common unclassified error-like log lines (numbers replaced by #):')
        for text, c in other.most_common(args.top):
            out.append(f'  {c:5d}  {text}')
    report['unclassified'] = other.most_common(args.top)

    # ---- condition factors
    out.append('')
    out.append('== physical attributes vs. outcome (all completed episodes, median split; small n = weak evidence)')
    out.append(f'{"factor":22s} {"n":>4s}  {"low half":>9s}  {"high half":>9s}  {"gap":>6s}   mean(success)  mean(fail)')
    rows = []
    for name, get in SCALARS:
        vals = [(get(e['condition']), bool(e['result']['success'])) for e in completed]
        vals = [(v, s) for v, s in vals if isinstance(v, (int, float))]
        if len(vals) < 4:
            continue
        med = statistics.median(v for v, _ in vals)
        low = [s for v, s in vals if v <= med]
        high = [s for v, s in vals if v > med]
        if not low or not high:
            continue
        rl, rh = sum(low) / len(low), sum(high) / len(high)
        ms = statistics.mean(v for v, s in vals if s) if any(s for _, s in vals) else float('nan')
        mf = statistics.mean(v for v, s in vals if not s) if any(not s for _, s in vals) else float('nan')
        rows.append((abs(rh - rl), name, len(vals), rl, rh, rh - rl, ms, mf))
    for _, name, n, rl, rh, gap, ms, mf in sorted(rows, reverse=True):
        out.append(f'{name:22s} {n:4d}  {100*rl:8.1f}%  {100*rh:8.1f}%  {100*gap:+6.1f}   {ms:12.4f}  {mf:10.4f}')
        report['factors'][name] = dict(n=n, low_half=rl, high_half=rh, mean_success=ms, mean_fail=mf)
    modes = defaultdict(list)
    for e in completed:
        modes[e['condition'].get('camera', {}).get('mode', '?')].append(bool(e['result']['success']))
    if modes:
        out.append('camera mode: ' + '  '.join(f'{m}={sum(v)}/{len(v)}' for m, v in sorted(modes.items())))
        report['factors']['camera_mode'] = {m: [sum(v), len(v)] for m, v in modes.items()}

    # ---- anomalies
    out.append('')
    out.append('== record anomalies')
    anomalies = []
    identities = Counter(json.dumps(e['result'].get('policy'), sort_keys=True) for e in completed)
    if len(identities) > 1:
        anomalies.append(f'{len(identities)} different policy identities among completed episodes')
    for e in errored:
        err = (e['result'].get('error') or '').strip().splitlines()
        anomalies.append(f'{e["task"]}/{e["stem"]}: status={e["result"].get("status")} '
                         + (err[-1] if err else ''))
    for e in completed:
        lim = limits.get(e['task'])
        if lim and not e['result'].get('success') and e['result'].get('actions', lim) < lim:
            anomalies.append(f'{e["task"]}/{e["stem"]}: failed after {e["result"]["actions"]} actions, step limit {lim}')
        if not e['result'].get('benchmark_eligible', True):
            anomalies.append(f'{e["task"]}/{e["stem"]}: not benchmark eligible')
    for e in pending:
        if e['signatures']:
            anomalies.append(f'{e["task"]}/{e["stem"]}: no result, log has {sorted(e["signatures"])}')
    out.extend(('  ' + a) for a in anomalies) if anomalies else out.append('none')
    report['anomalies'] = anomalies

    # ---- comparison with another run folder
    if args.compare:
        other_folder = Path(args.compare).expanduser()
        if (other_folder / args.protocol).exists():
            other_folder = other_folder / args.protocol
        others = {(e['task'], e['index']): e for e in load_episodes(other_folder)
                  if e['result'] and e['result'].get('status') == 'completed'}
        out.append('')
        out.append(f'== comparison with {other_folder}')
        agree = Counter()
        flips = []
        for e in completed:
            o = others.get((e['task'], e['index']))
            if o is None:
                continue
            a, b = bool(o['result']['success']), bool(e['result']['success'])
            agree[(a, b)] += 1
            if a != b:
                flips.append(f'{e["task"]}/{e["stem"]}: before={"T" if a else "F"} now={"T" if b else "F"}')
        n = sum(agree.values())
        if n:
            before = agree[(True, True)] + agree[(True, False)]
            now = agree[(True, True)] + agree[(False, True)]
            out.append(f'shared episodes: {n}  before: {before}/{n} ({fmt_pct(before, n).strip()})  '
                       f'now: {now}/{n} ({fmt_pct(now, n).strip()})')
            out.append(f'both success: {agree[(True, True)]}  both fail: {agree[(False, False)]}  '
                       f'F->T: {agree[(False, True)]}  T->F: {agree[(True, False)]}')
            out.extend('  ' + f for f in flips)
            # Same seed and initial state: a sign test on the flips is the fair comparison.
            k, m = agree[(False, True)], agree[(False, True)] + agree[(True, False)]
            if m:
                out.append(f'sign test on {m} flips: p = {binom_two_sided(k, m, 0.5):.3f}')
        else:
            out.append('no shared (task, episode index) with a completed result')
        report['comparison'] = dict(folder=str(other_folder), agreement={f'{a}->{b}': c for (a, b), c in agree.items()},
                                    flips=flips)

    # ---- error tails
    if args.errors:
        out.append('')
        out.append('== log tails of failed or errored episodes')
        for e in episodes:
            if e['result'] and (e['result'].get('status') != 'completed' or not e['result'].get('success')):
                lines = e['log'].read_text(errors='replace').splitlines() if e['log'].exists() else ['(no log)']
                picked = [l for l in lines if (ERRORISH.search(l) or any(rx.search(l) for rx in SIGNATURES.values()))
                          and not HARMLESS.search(l)]
                out.append(f'-- {e["task"]}/{e["stem"]} status={e["result"].get("status")} '
                           f'success={e["result"].get("success")} actions={e["result"].get("actions")}')
                out.extend('   ' + l[:200] for l in (picked or lines)[-args.tail:])
    text = '\n'.join(out)
    print(text)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1, default=str))
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', default=str(ROOT), help='folder containing runs/ (default: repository root)')
    p.add_argument('--experiment', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--protocol', default='physical_random')
    p.add_argument('--tasks', nargs='*')
    p.add_argument('--reference', default=str(DEFAULT_REFERENCE), help='paper per-task table JSON ("" to disable)')
    p.add_argument('--step-limits', default=str(ROOT / 'vendor/RoboTwin/task_config/_eval_step_limit.yml'))
    p.add_argument('--compare', help='another runs/<experiment>/<model> folder to compare episode by episode')
    p.add_argument('--alpha', type=float, default=0.05)
    p.add_argument('--min-episodes', type=int, default=5, help='flag LOW/HIGH only with at least this many episodes')
    p.add_argument('--top', type=int, default=15)
    p.add_argument('--errors', action='store_true', help='print error-like log lines of each failed episode')
    p.add_argument('--tail', type=int, default=12)
    p.add_argument('--json', help='also write the report as JSON')
    return analyze(p.parse_args(argv))


if __name__ == '__main__':
    main()
