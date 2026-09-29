"""Install the authorized30-minute AI continuation timer, independent of launch gates."""

import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = args.config.resolve()
    if not config.is_file():
        raise ValueError("Explicit supervisor task configuration is required")
    state = args.state_dir.resolve()
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    units = Path.home() / ".config/systemd/user"
    units.mkdir(parents=True, exist_ok=True)
    (units / "sweep-ai-wakeup.service").write_text(f"""[Unit]
Description=Resume authorized Sweep implementation after stale owner leases
[Service]
Type=oneshot
MemoryMax=2G
MemorySwapMax=0
OOMPolicy=continue
TimeoutStartSec=2700
KillMode=control-group
UMask=0077
WorkingDirectory={root}
ExecStart={root}/scripts/with_env.sh python {root}/scripts/sweep_ai_wakeup.py \\
 --config {config} --state-dir {state}
StandardOutput=append:{state}/service.log
StandardError=append:{state}/service.log
""")
    (units / "sweep-ai-wakeup.timer").write_text("""[Unit]
Description=Wake a continuation agent every30minutes when an authorized work lease is stale
[Timer]
OnBootSec=2min
OnCalendar=*-*-* *:00,30:00
Persistent=true
Unit=sweep-ai-wakeup.service
[Install]
WantedBy=timers.target
""")
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", "sweep-ai-wakeup.timer"], check=True)
    subprocess.run(["systemctl", "--user", "start", "sweep-ai-wakeup.service"], check=True)


if __name__ == "__main__":
    main()
