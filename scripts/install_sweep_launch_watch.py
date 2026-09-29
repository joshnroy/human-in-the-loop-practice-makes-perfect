"""Install the ready-only launcher independently of the interactive agent."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = Path("/home/josh/Documents/repos/research/hitl-pmp-results/sweep-launch")
UNITS = Path.home() / ".config/systemd/user"
STATE.mkdir(parents=True, exist_ok=True)
UNITS.mkdir(parents=True, exist_ok=True)
(UNITS / "sweep-ready-launch.service").write_text(f"""[Unit]
Description=Submit frozen READY Sweep batch exactly once
[Service]
Type=oneshot
MemoryMax=6G
MemorySwapMax=0
OOMPolicy=continue
TimeoutStartSec=1800
WorkingDirectory={ROOT}
ExecStart={ROOT}/scripts/with_env.sh python {ROOT}/scripts/sweep_launch_watch.py \\
 --readiness {STATE}/readiness.json --state-dir {STATE}
""")
(UNITS / "sweep-ready-launch.timer").write_text("""[Unit]
Description=Check approved Sweep readiness every minute
[Timer]
OnBootSec=30s
OnUnitInactiveSec=60s
Unit=sweep-ready-launch.service
[Install]
WantedBy=timers.target
""")
subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
subprocess.run(["systemctl", "--user", "enable", "--now", "sweep-ready-launch.timer"], check=True)
subprocess.run(["systemctl", "--user", "start", "sweep-ready-launch.service"], check=True)
