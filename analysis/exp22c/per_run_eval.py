"""Every planned machine/arm/seed, including explicit unavailable placeholders."""
import html
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from arms import ARMS

ROOT = Path('/home/josh/Documents/repos/research/hitl-pmp-results')
OUT = Path('/home/josh/Documents/repos/research/hitl-pmp-site/tossing3d-baseline/exp22c/per-run-eval')
COLORS = {'A': '#56B4E9', 'B': '#0072B2', 'ees': '#009E73'}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().isoformat()
    records, cards = [], []
    for machine, folder in [('ws', 'exp22c'), ('della', 'exp22c-della')]:
        for group, arm, label, color, reset in ARMS:
            for seed in range(3):
                method = 'ees' if arm.startswith('ees') else 'pomdp'
                run = ROOT / folder / group / 'results' / arm / method / str(seed)
                slug = f'{machine}-{group}-{arm}-seed{seed}'
                stats = json.loads((run / 'stats.json').read_text()) if (run / 'stats.json').exists() else {}
                timing = json.loads((run / 'timing.json').read_text()) if (run / 'timing.json').exists() else {}
                config = json.loads((run / 'config_snapshot.json').read_text()) if (run / 'config_snapshot.json').exists() else {}
                evaluations = stats.get('evaluations', [])
                assert all(e[2] == 10 and 0 <= e[1] <= 10 for e in evaluations), run
                done = len(evaluations) == 51 and len(stats.get('practice_session_ends', [])) == 50 and timing.get('returncode') == 0 and timing.get('succeeded') is not False
                status = 'complete' if done else 'partial' if evaluations else 'not started'
                if config:
                    assert config['git_commit'].startswith('76732331'), run
                values = [e[1] for e in evaluations]
                fig, ax = plt.subplots(figsize=(7.2, 4.2), layout='constrained')
                ax.plot(range(len(values)), values, color=COLORS[color], linestyle='-' if reset else '--', linewidth=1.8)
                ax.set(xlim=(0, 50), ylim=(-.15, 10.15), xlabel='Completed practice episodes (0 = initial evaluation)', ylabel='Solved evaluation tasks /10', title=f'{machine.upper()} · {label} · seed {seed}\n{status}: {len(values)}/51 evaluations')
                ax.set_yticks(range(0, 11, 2))
                ax.grid(alpha=.18)
                if not values:
                    ax.text(.5, .5, 'No recorded evaluation', transform=ax.transAxes, ha='center')
                for suffix in ('png', 'svg', 'pdf'):
                    fig.savefig(OUT / f'{slug}.{suffix}', dpi=240)
                plt.close(fig)
                record = {'machine': machine, 'group': group, 'arm': arm, 'method': method, 'seed': seed, 'status': status, 'evaluations_recorded': len(values), 'evaluations_planned': 51, 'values': values, 'run': str(run), 'commit': config.get('git_commit'), 'generated_at': timestamp, 'stem': slug}
                records.append(record)
                title = html.escape(f'{machine.upper()} · {label} · seed {seed}')
                cards.append(f'<article data-machine="{machine}" data-status="{status}"><h2>{title}</h2><p>{status}: {len(values)}/51 evaluations · <a href="{slug}.svg">SVG</a> · <a href="{slug}.pdf">PDF</a> · <a href="{slug}.png">PNG</a></p><img loading="lazy" src="{slug}.png" alt="{title} evaluation learning curve"></article>')
                print(f'{len(records)}/114 {slug} {status}', flush=True)
    assert len(records) == 114 and len({r['stem'] for r in records}) == 114
    counts = Counter(r['status'] for r in records)
    (OUT / 'manifest.json').write_text(json.dumps({'generated_at': timestamp, 'coverage': '114/114 planned runs', 'status_counts': dict(counts), 'runs': records}, indent=2))
    (OUT / 'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>EXP-22c · every run</title><style>body{font:16px system-ui;margin:2rem;background:#fafafa;color:#17212b}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:1rem}article{background:white;padding:1rem;border:1px solid #ddd;border-radius:8px}h2{font-size:1rem}img{width:100%}a{color:#0072B2}button{margin:.5rem;padding:.6rem}</style><h1>EXP-22c: every evaluation curve</h1><p><a href="../">Full results</a> · <a href="manifest.json">Provenance and values</a></p>'''+f'<p>114/114 planned runs shown; {counts["complete"]}/114 complete. Each trace is one machine–arm–seed run, scored on 10/10 fixed evaluation tasks at each checkpoint. No seed or machine averaging. Dashed = no human reset. Snapshot {html.escape(timestamp)}.</p>'+'''<p><button onclick="filter('all')">All 114</button><button onclick="filter('ws')">Workstation 57</button><button onclick="filter('della')">Della 57</button></p><main>'''+''.join(cards)+'''</main><script>function filter(m){document.querySelectorAll('article').forEach(a=>a.hidden=m!=='all'&&a.dataset.machine!==m)}</script></html>''')
    print(json.dumps({'coverage': len(records), 'statuses': dict(counts)}), flush=True)


if __name__ == '__main__':
    main()
