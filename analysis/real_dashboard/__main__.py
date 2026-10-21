"""A live, read-only browser dashboard for one real-robot practice run.

    scripts/with_env.sh python -m analysis.real_dashboard --run-dir <output-dir> \\
        [--skill-calls <skill_calls.jsonl>] --port 8770

It reads the run's `--output-dir` (and, optionally, the skill server's log pulled from
the NUC) every time the page polls, and writes nothing anywhere. Prompts stay in the
run's terminal; the page only shows the one that is waiting.
"""

import argparse
from pathlib import Path

from .dashboard import RealRunDashboard
from .server import DashboardServer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m analysis.real_dashboard",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="the run's --output-dir")
    parser.add_argument(
        "--skill-calls",
        type=Path,
        default=None,
        help="the skill server's skill_calls.jsonl (toss landings, in_bin, holds, timings)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="address to bind (default: all)")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument(
        "--landings", type=int, default=30, help="how many recent toss landings the map shows"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.run_dir.is_dir():
        raise SystemExit(f"--run-dir {args.run_dir} is not a directory")
    dashboard = RealRunDashboard(
        run_dir=args.run_dir, skill_calls=args.skill_calls, landings_shown=args.landings
    )
    dashboard.refresh()  # Fail here, not in the first poll, on an unreadable run.
    server = DashboardServer.make(dashboard=dashboard, host=args.host, port=args.port)
    print(f"serving {args.run_dir} on http://{args.host}:{server.server_address[1]}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
