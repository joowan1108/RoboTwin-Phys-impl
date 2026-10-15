#!/usr/bin/env python3
"""Read existing task logs into one stream; never launch or modify episode work."""
import argparse
from datetime import datetime
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def collect(directory, cursors):
    lines = []
    for path in sorted(directory.glob('freeze-*.log')):
        task = path.stem.removeprefix('freeze-')
        with path.open('rb') as source:
            stat = os.fstat(source.fileno())
            identity = (stat.st_dev, stat.st_ino)
            cursor = cursors.get(path)
            if cursor is None or cursor['identity'] != identity or stat.st_size < cursor['offset']:
                if cursor is not None:
                    lines.append((task, 'LOG_RESTART: reading the replaced/truncated task log'))
                cursor = dict(identity=identity, offset=0, partial=b'')
            source.seek(cursor['offset'])
            data = cursor['partial'] + source.read()
            cursor['offset'] = source.tell()
            chunks = data.split(b'\n')
            cursor['partial'] = chunks.pop()
            lines.extend((task, chunk.decode('utf-8', errors='replace').rstrip('\r')) for chunk in chunks)
            cursors[path] = cursor
    return lines


def job_active(job):
    if shutil.which('squeue') is None:
        return None
    result = subprocess.run(['squeue', '-h', '-j', job, '-o', '%T'],
                            capture_output=True, text=True, timeout=15)
    if result.returncode:
        return None
    return bool(result.stdout.strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--experiment', default='phys-pro6000-100-v5')
    parser.add_argument('--job', required=True, help='Original pool job ID, not this monitor job ID')
    parser.add_argument('--poll-seconds', type=float, default=5.)
    parser.add_argument('--once', action='store_true', help='Collect current task logs and exit')
    args = parser.parse_args()
    if not args.job.isdecimal() or int(args.job) < 1 or args.poll_seconds <= 0:
        parser.error('Use a positive pool job ID and polling interval')
    if '/' in args.experiment or args.experiment in ('.', '..'):
        parser.error('Experiment must be a directory name')
    directory = args.data / 'reports/parallel' / args.experiment / ('fastwam-' + args.job)
    deadline = time.monotonic() + 600
    while not directory.is_dir():
        if args.once or job_active(args.job) is False or time.monotonic() >= deadline:
            raise SystemExit(f'Pool log directory is not available: {directory}; check the original pool job ID')
        print(f'Waiting for pool logs: {directory}', flush=True)
        time.sleep(args.poll_seconds)
    # One observer owns this output. Benchmark manifests, seeds and code stay untouched.
    with (directory / '.pool-monitor.lock').open('a') as guard:
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('A monitor for this pool job is already running')
        with (directory / 'pool-progress.log').open('w', buffering=1) as output:
            def emit(task, line):
                record = f'{datetime.now().astimezone().isoformat(timespec="seconds")} [{task}] {line}'
                print(record, file=output, flush=True)
                print(record, flush=True)
            emit('monitor', f'POOL_LOGS={directory}; OUTPUT={output.name}; replay existing logs, then follow updates')
            cursors = {}
            ended = 0
            try:
                while True:
                    for task, line in collect(directory, cursors):
                        emit(task, line)
                    if args.once:
                        break
                    try:
                        active = job_active(args.job)
                    except subprocess.TimeoutExpired:
                        active = None
                    ended = ended + 1 if active is False else 0
                    if ended >= 2:
                        emit('monitor', 'POOL_JOB_ENDED: this does not imply every task completed successfully')
                        break
                    time.sleep(args.poll_seconds)
            except KeyboardInterrupt:
                emit('monitor', 'MONITOR_STOPPED: original pool job is unaffected')
            # Preserve an incomplete final line when the producer has exited.
            for path, cursor in cursors.items():
                if cursor['partial']:
                    emit(path.stem.removeprefix('freeze-'),
                         cursor['partial'].decode('utf-8', errors='replace'))


if __name__ == '__main__':
    main()
