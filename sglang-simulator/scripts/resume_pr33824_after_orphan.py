#!/usr/bin/env python3
"""Resume downstream stages only after an orphaned native matrix has finished.

Never stop or relaunch its live collector, delete outputs, or weaken completion
gates. Run detached, with the original child's environment captured while live.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def identity(pid: int) -> dict | None:
    try:
        proc = Path('/proc') / str(pid)
        stat = (proc / 'stat').read_text()
        fields = stat[stat.rfind(')') + 2:].split()
        if fields[0] == 'Z':
            return None
        return {'pid': pid, 'start_ticks': int(fields[19]),
                'argv': (proc / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')}
    except (FileNotFoundError, ProcessLookupError):
        return None


def complete(path: Path) -> bool:
    return path.is_file() and path.read_text().strip() == 'complete'


def native_complete(root: Path, models: list[str]) -> bool:
    return (complete(root / 'REAL_ML_ALL_MODELS_COMPLETE')
            and all(complete(root / 'models' / model / 'REAL_ML_COMPLETE') for model in models))


def decision(owner: dict, observed: dict | None, ready: bool) -> str:
    if observed == owner:
        return 'wait'
    if observed is not None:
        raise RuntimeError('Native PID was reused; inspect before any continuation')
    if not ready:
        raise RuntimeError('Native child ended without all model completion markers; outputs retained')
    return 'resume'


def matching_runners(runner: Path, root: Path) -> list[dict]:
    found = []
    for entry in Path('/proc').glob('[0-9]*'):
        observed = identity(int(entry.name))
        if observed is None:
            continue
        argv = observed['argv']
        if not any(arg.startswith('/') and Path(arg).resolve() == runner for arg in argv):
            continue
        if '--results-root' in argv:
            index = argv.index('--results-root')
            if index + 1 < len(argv) and Path(argv[index + 1]).resolve() == root:
                found.append(observed)
    return found


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, required=True)
    parser.add_argument('--native-pid', type=int, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--interval', type=int, default=600)
    args = parser.parse_args()
    if args.interval < 10:
        parser.error('interval must be at least ten seconds')
    root, out = args.results_root.resolve(), args.out_dir.resolve()
    sim = Path(__file__).resolve().parents[1]
    runner = sim / 'scripts/run_pr33824_comprehensive_predictor_matrix.sh'
    native = sim / 'scripts/run_pr33824_data_tools_same_distribution_matrix.sh'
    owner = identity(args.native_pid)
    if owner is None or not any(arg.startswith('/') and Path(arg).resolve() == native for arg in owner['argv']):
        raise RuntimeError('Expected original native child is not live')
    argv = owner['argv']
    if '--results-root' not in argv or Path(argv[argv.index('--results-root') + 1]).resolve() != root:
        raise RuntimeError('Original child belongs to another output root')
    environment = dict(item.split('=', 1) for item in (Path('/proc') / str(args.native_pid) / 'environ').read_bytes().decode().split('\0') if '=' in item)
    if identity(args.native_pid) != owner:
        raise RuntimeError('Native identity changed while capturing its environment')
    metadata = dict(line.split('=', 1) for line in (root / 'run_metadata.txt').read_text().splitlines() if '=' in line)
    models = metadata['models'].split(',')
    if len(models) != 13 or len(set(models)) != 13:
        raise RuntimeError('Recovery must retain the full thirteen-model scope')
    for key, expected in {'training_requests_total': '800', 'heldout_requests_total': '1280', 'training_real_repeats': '5', 'heldout_real_repeats': '5', 'prompt_multiplier': '2'}.items():
        if metadata.get(key) != expected:
            raise RuntimeError(f'Unexpected immutable protocol {key}')
    if matching_runners(runner, root):
        raise RuntimeError('A comprehensive runner is already live; do not duplicate it')
    out.mkdir(parents=True, exist_ok=True)
    command = ['/bin/bash', str(runner), '--resume', '--results-root', str(root), '--models', metadata['models'], '--train-repeats', '5', '--holdout-repeats', '5', '--prompt-multiplier', '2']
    pinned = [runner, native, sim / 'scripts/run_pr33824_data_tools_baseline.sh', sim / 'tools/train_latency_model.py', sim / 'tools/append_pr33824_expanded_report.py', root / 'run_metadata.txt', Path(__file__).resolve()]
    pinned.extend(sorted((root / 'models').glob('*/calibration/training/*_p50.pkl')))
    hashes = {str(path): digest(path) for path in pinned}
    now = lambda: datetime.now().astimezone().isoformat()
    state = {'started_at': now(), 'pid': os.getpid(), 'native_owner': owner, 'command': command,
             'interval_seconds': args.interval, 'source_sha256': hashes,
             'environment_policy': 'Exact original live native-child environment kept only in memory; no runtime setting changed',
             'settings': {key: value for key, value in environment.items() if key.startswith('SGLANG_') and ('KEY' not in key and 'TOKEN' not in key)},
             'state': 'waiting_for_original_native'}
    with (out / 'supervisor.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (out / 'launch.json').open('x') as stream:
            json.dump(state, stream, indent=2)
        try:
            while True:
                state['updated_at'] = now()
                state['decision'] = decision(owner, identity(args.native_pid), native_complete(root, models))
                (out / 'state.json').write_text(json.dumps(state, indent=2) + '\n')
                print(json.dumps({'at': state['updated_at'], 'state': state['decision']}), flush=True)
                if state['decision'] == 'resume':
                    break
                time.sleep(args.interval)
            if matching_runners(runner, root):
                raise RuntimeError('Another comprehensive runner appeared; continuation refused')
            if any(digest(Path(path)) != value for path, value in hashes.items()):
                raise RuntimeError('Pinned source/protocol/frozen bundles changed; continuation refused')
            with (root / '.same-distribution.lock').open('a+') as native_lock:
                fcntl.flock(native_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(native_lock, fcntl.LOCK_UN)
            with (out / 'continuation.log').open('x') as log:
                process = subprocess.Popen(command, cwd=sim, env=environment, stdout=log, stderr=subprocess.STDOUT)
                state.update(state='continuation_running', continuation_pid=process.pid, continuation_started_at=now())
                (out / 'state.json').write_text(json.dumps(state, indent=2) + '\n')
                result = process.wait()
            state.update(exit_code=result, finished_at=now())
            if result != 0 or not complete(root / 'FINAL_REPORT_COMPLETE'):
                raise RuntimeError(f'Continuation incomplete: exit={result}; logs and outputs retained')
            state['state'] = 'complete'
        except BaseException as error:
            state.update(state='failed', updated_at=now(), error=repr(error))
            raise
        finally:
            (out / 'state.json').write_text(json.dumps(state, indent=2) + '\n')


if __name__ == '__main__':
    main()
