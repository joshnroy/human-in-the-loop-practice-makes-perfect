"""Quiet completion-only refresh; never launches or changes experiment jobs."""
import fcntl
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from arms import ARMS

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/josh/Documents/repos/research/hitl-pmp-results')
DATA = HERE / 'data'
RUNS = [ROOT / 'exp22c' / group / 'results' / arm /
        ('ees' if arm.startswith('ees') else 'pomdp') / str(seed)
        for group, arm, *_ in ARMS for seed in range(3)]


def completed(run):
    try:
        stats = json.loads((run / 'stats.json').read_text())
        timing = json.loads((run / 'timing.json').read_text())
        config = json.loads((run / 'config_snapshot.json').read_text())
        return (len(stats['evaluations']) == 51
                and len(stats['practice_session_ends']) == 50
                and timing.get('returncode') == 0
                and timing.get('succeeded') is not False
                and config['git_commit'].startswith('76732331'))
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def available_gib():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / 1024 ** 2
    raise RuntimeError('Missing MemAvailable')


def main():
    assert len(RUNS) == len(set(RUNS)) == 57
    with (DATA / 'watch-final.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = None
        while True:
            count = sum(completed(run) for run in RUNS)
            if count != previous:
                print(f'{datetime.now().astimezone().isoformat()} validated {count}/57', flush=True)
                previous = count
            if count == 57 and available_gib() >= 14:
                break
            time.sleep(45)
        subprocess.run([sys.executable, str(HERE / 'run.py')], check=True)
        rows = json.loads((DATA / 'rows.json').read_text())
        ws = [r for r in rows if r['src'] == 'ws']
        assert len(ws) == 57 and all(r['scan_valid'] for r in ws)
        timestamp = datetime.now().astimezone().isoformat()
        ready = {'timestamp': timestamp, 'workstation_complete': '57/57',
                 'practice_refused': sum(r['refused'] for r in ws),
                 'practice_picks': sum(r['picks'] for r in ws),
                 'eval_refused': sum(r['eval_refused'] for r in ws),
                 'eval_picks': sum(r['eval_picks'] for r in ws),
                 'report': 'http://agni:8766/exp22c/'}
        (DATA / 'final-ready.json').write_text(json.dumps(ready, indent=2) + '\n')
        with (ROOT / 'HANDOFF.md').open('a') as handoff:
            fcntl.flock(handoff, fcntl.LOCK_EX)
            handoff.write(f'\n\n## EXP-22c final workstation analysis — {timestamp}\n\n'
                          f'Completion-only watcher verified 57/57 workstation runs: '
                          f'51/51 evaluations, 50/50 practice cycles, commit 76732331, '
                          f'successful timing metadata, and per-run scan/accounting checks. '
                          f'Regenerated http://agni:8766/exp22c/ with separate machine tables, '
                          f'paired tests and plots. Workstation refused picks: '
                          f'{ready["practice_refused"]}/{ready["practice_picks"]} practice; '
                          f'{ready["eval_refused"]}/{ready["eval_picks"]} evaluation. '
                          f'No experiments launched or relaunched. Durable marker: '
                          f'{DATA / "final-ready.json"}. Main-page status requires its owner to refresh.\n')
        renderer = HERE.parents[2] / 'exp22c-checkpoint' / 'render_handoff.py'
        subprocess.run([sys.executable, str(renderer)], check=True)
        print(json.dumps(ready), flush=True)


if __name__ == '__main__':
    main()
