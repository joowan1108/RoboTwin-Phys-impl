"""analyze_episodes summarizes every episode record, log and condition at once."""
import importlib.util
import json
from pathlib import Path

from robotwin_phys.conditions import sample

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location('analyze_episodes', ROOT / 'analysis/analyze_episodes.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def write_episode(folder, task, index, success, log='', status='completed', actions=400, friction=None):
    folder = folder / task
    folder.mkdir(parents=True, exist_ok=True)
    cond = sample(task, 100000 + index)
    if friction is not None:
        cond['friction'] = friction
    stem = f'episode-{index:04d}'
    (folder / f'{stem}.request.json').write_text(json.dumps({'mode': 'evaluate', 'episode': {'condition': cond, 'seed': 100000 + index}}))
    if status is not None:
        result = {'status': status, 'success': success, 'actions': actions, 'seconds': 50.0,
                  'policy': {'name': 'fastwam', 'config_sha256': 'c'}, 'benchmark_eligible': True}
        if status == 'error':
            result = {'status': 'error', 'error': 'Traceback ...\nRuntimeError: boom'}
        (folder / f'{stem}.result.json').write_text(json.dumps(result))
    (folder / f'{stem}.log').write_text(log)


def test_report_covers_rates_signatures_factors_and_comparison(tmp_path, capsys):
    runs = tmp_path / 'runs/exp/fastwam/physical_random'
    # adjust_bottle: 6 successes; beat_block_hammer: 6 failures with OIDN errors and low friction
    for i in range(6):
        write_episode(runs, 'adjust_bottle', i, True, log='missing pytorch3d\n', friction=0.9)
        write_episode(runs, 'beat_block_hammer', i, False, log='[svulkan2] [error] OIDN Error: invalid handle\n', friction=0.1)
    write_episode(runs, 'click_bell', 0, False, status='error', log='Traceback (most recent call last):\nRuntimeError: boom\n')
    write_episode(runs, 'click_bell', 1, False, status=None, log='TimeoutError: Process timed out\n')
    other = tmp_path / 'runs/exp/fastwam-old/physical_random'
    for i in range(6):
        write_episode(other, 'adjust_bottle', i, i < 3)  # 3 flips F->T
    (tmp_path / 'limits.yml').write_text('adjust_bottle: 400\nbeat_block_hammer: 500\n')
    report = a.main(['--root', str(tmp_path), '--experiment', 'exp', '--model', 'fastwam', '--compare', str(other.parent),
                     '--step-limits', str(tmp_path / 'limits.yml'), '--errors', '--json', str(tmp_path / 'r.json')])
    text = capsys.readouterr().out
    assert report['completed'] == 12 and report['successes'] == 6 and report['errored'] == 1 and report['pending'] == 1
    assert report['tasks']['adjust_bottle']['paper'] == 64 and report['tasks']['beat_block_hammer']['paper'] == 17
    assert report['tasks']['adjust_bottle']['p_value'] > 0.05 and 'LOW' not in text.split('adjust_bottle')[1].split('\n')[0]
    assert report['signatures']['oidn_error'] == {'episodes': 6, 'failed': 6, 'tasks': {'beat_block_hammer': 6}}
    assert report['signatures']['render_error']['episodes'] == 6 and 'timeout' in report['signatures']
    assert 'pytorch3d' not in text  # harmless lines are not reported
    assert report['factors']['friction']['low_half'] == 0.0 and report['factors']['friction']['high_half'] == 1.0
    assert any('beat_block_hammer/episode-0000: failed after 400 actions, step limit 500' in x for x in report['anomalies'])
    assert any('click_bell/episode-0000: status=error' in x for x in report['anomalies'])
    assert report['comparison']['agreement'] == {'True->True': 3, 'False->True': 3} and len(report['comparison']['flips']) == 3
    assert 'sign test on 3 flips' in text and '== log tails' in text and 'RuntimeError: boom' in text
    assert json.loads((tmp_path / 'r.json').read_text())['successes'] == 6


def test_reference_table_matches_paper_overall():
    ref = json.loads((ROOT / 'references/paper-table4-physical-random.json').read_text())
    assert len(ref['tasks']) == 50
    for model, overall in ref['overall'].items():
        mean = sum(v[model] for v in ref['tasks'].values()) / 50
        assert abs(mean - overall) < 0.11, (model, mean, overall)


def test_binomial_probability_is_exact():
    assert abs(a.binom_two_sided(0, 2, 0.5) - 0.5) < 1e-9
    assert a.binom_two_sided(3, 3, 1.0) == 1.0
    assert a.binom_two_sided(0, 10, 0.6) < 0.01
