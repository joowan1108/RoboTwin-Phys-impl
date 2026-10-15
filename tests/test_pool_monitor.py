import importlib.util
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location('pool_monitor', ROOT / 'cluster/monitor_pool.py')
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


def test_monitor_combines_tasks_then_follows_new_complete_lines_without_duplicates(tmp_path):
    first = tmp_path / 'freeze-adjust_bottle.log'
    second = tmp_path / 'freeze-beat_block_hammer.log'
    first.write_text('1/100: seed 100000\nRejected seed 100002: unstable_initialization\n')
    second.write_text('Rejected seed 100001: expert_failed\n')
    before = {p: p.read_bytes() for p in (first, second)}
    cursors = {}
    assert monitor.collect(tmp_path, cursors) == [
        ('adjust_bottle', '1/100: seed 100000'),
        ('adjust_bottle', 'Rejected seed 100002: unstable_initialization'),
        ('beat_block_hammer', 'Rejected seed 100001: expert_failed')]
    assert monitor.collect(tmp_path, cursors) == []
    assert {p: p.read_bytes() for p in before} == before
    with first.open('a') as out:
        out.write('2/100: seed ')
    assert monitor.collect(tmp_path, cursors) == []
    with first.open('a') as out:
        out.write('100003\n')
    assert monitor.collect(tmp_path, cursors) == [('adjust_bottle', '2/100: seed 100003')]
    third = tmp_path / 'freeze-click_bell.log'
    third.write_text('1/100: seed 100004\n')
    assert monitor.collect(tmp_path, cursors) == [('click_bell', '1/100: seed 100004')]


def test_replaced_or_truncated_task_log_is_not_silently_missed(tmp_path):
    path = tmp_path / 'freeze-adjust_bottle.log'
    path.write_text('Rejected seed 100002: unstable_initialization\n')
    cursors = {}
    monitor.collect(tmp_path, cursors)
    path.write_text('1/100: seed 100000\n')
    assert monitor.collect(tmp_path, cursors)[-1] == ('adjust_bottle', '1/100: seed 100000')
    replacement = tmp_path / 'replacement'
    replacement.write_text('2/100: seed 100001\n')
    replacement.replace(path)
    assert monitor.collect(tmp_path, cursors)[-1] == ('adjust_bottle', '2/100: seed 100001')


def test_monitor_output_is_a_new_file_and_never_changes_pool_state(tmp_path):
    folder = tmp_path / 'reports/parallel/phys-pro6000-100-v5/fastwam-1650'
    folder.mkdir(parents=True)
    source = folder / 'freeze-adjust_bottle.log'
    source.write_text('1/100: seed 100000\n')
    manifest = tmp_path / 'manifests/phys-pro6000-100-v5/physical_random/adjust_bottle.progress.json'
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"status":"building"}')
    original = source.read_bytes(), manifest.read_bytes()
    result = subprocess.run([sys.executable, str(ROOT / 'cluster/monitor_pool.py'), '--data', str(tmp_path),
                             '--job', '1650', '--once'], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert '[adjust_bottle] 1/100: seed 100000' in result.stdout
    assert (folder / 'pool-progress.log').read_text() == result.stdout
    assert (source.read_bytes(), manifest.read_bytes()) == original
    assert not (manifest.parent / 'adjust_bottle.json').exists()


def test_self_contained_ood_launcher_needs_no_new_zip_or_gpu(tmp_path):
    text = (ROOT / 'cluster/ood-pool-monitor-v5.txt').read_text()
    assert text == (ROOT / 'cluster/job-pool-monitor-v5.sbatch').read_text()
    embedded = text.split("<<'ROBOTWIN_MONITOR'\n")[1].removesuffix('ROBOTWIN_MONITOR\n')
    assert embedded == (ROOT / 'cluster/monitor_pool.py').read_text()
    assert '#SBATCH --gres' not in text and 'apptainer exec' not in text
    assert '.zip' not in text
    folder = tmp_path / 'reports/parallel/phys-pro6000-100-v5/fastwam-1650'
    folder.mkdir(parents=True)
    (folder / 'freeze-click_bell.log').write_text('3/100: seed 100007\n')
    script = tmp_path / 'job.sh'
    text = text.replace('DATA="$HOME/robotwin-data"', f'DATA="{tmp_path}"')
    text = text.replace('POOL_JOB_ID="CHANGE_ME"', 'POOL_JOB_ID="1650"')
    text = text.replace('--job "$POOL_JOB_ID"', '--job "$POOL_JOB_ID" --once')
    script.write_text(text)
    subprocess.run(['bash', '-n', str(script)], check=True)
    result = subprocess.run(['bash', str(script)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert '[click_bell] 3/100: seed 100007' in result.stdout


def test_job_completion_observation_uses_pool_id_and_handles_scheduler_errors(monkeypatch):
    seen = []
    monkeypatch.setattr(monitor.shutil, 'which', lambda name: '/usr/bin/squeue')
    def run(argv, **kwargs):
        seen.append(argv)
        return subprocess.CompletedProcess(argv, 0, 'RUNNING\n', '')
    monkeypatch.setattr(monitor.subprocess, 'run', run)
    assert monitor.job_active('1650') is True
    assert seen == [['squeue', '-h', '-j', '1650', '-o', '%T']]
    monkeypatch.setattr(monitor.subprocess, 'run',
                        lambda argv, **kw: subprocess.CompletedProcess(argv, 0, '', ''))
    assert monitor.job_active('1650') is False
    monkeypatch.setattr(monitor.subprocess, 'run',
                        lambda argv, **kw: subprocess.CompletedProcess(argv, 1, '', 'Scheduler unavailable'))
    assert monitor.job_active('1650') is None
