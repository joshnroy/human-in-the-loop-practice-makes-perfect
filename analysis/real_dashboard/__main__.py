"""A live, read-only browser dashboard for one real-robot practice run.

    scripts/with_env.sh python -m analysis.real_dashboard --run-dir <output-dir> \\
        [--skill-calls <skill_calls.jsonl>] --port 8770 \\
        [--camera-python <prpl-tidybot python> --camera-pythonpath <prpl src>]

It reads the run's `--output-dir` (and, optionally, the skill server's log pulled from
the NUC) every time the page polls, and writes nothing anywhere. Prompts stay in the
run's terminal; the page only shows the one that is waiting.
"""

import argparse
from pathlib import Path

from .camera import CameraFeed
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
    parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help="JSONL of human corrections to evaluation tasks, shown beside the recorded "
        "numbers on the eval curve (never in place of them)",
    )
    camera = parser.add_argument_group(
        "live ceiling camera (off unless --camera-python is given)",
        "ceiling_helper.py runs under prpl-tidybot's interpreter and subscribes to the "
        "perception PC's ceiling-image publisher; without it the page shows the plain map.",
    )
    camera.add_argument("--camera-python", type=Path, default=None, help="prpl-tidybot's python")
    camera.add_argument(
        "--camera-pythonpath", default=None, help="PYTHONPATH for that python (prpl_tidybot)"
    )
    camera.add_argument("--camera-host", default="192.168.0.131")
    camera.add_argument("--camera-period", type=float, default=2.0, help="seconds per frame")
    camera.add_argument(
        "--camera-dir",
        type=Path,
        default=None,
        help="where the helper writes frames (default: a fresh temporary directory). "
        "Given without --camera-python, an existing directory is only read.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.run_dir.is_dir():
        raise SystemExit(f"--run-dir {args.run_dir} is not a directory")
    feed = None
    if args.camera_python is not None or args.camera_dir is not None:
        feed = CameraFeed(
            directory=args.camera_dir or CameraFeed.scratch_directory(),
            python=args.camera_python,
            pythonpath=args.camera_pythonpath,
            host=args.camera_host,
            period=args.camera_period,
        )
        feed.start()
    dashboard = RealRunDashboard(
        run_dir=args.run_dir,
        skill_calls=args.skill_calls,
        landings_shown=args.landings,
        camera=feed,
        annotations=args.annotations,
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
        if feed is not None:
            feed.stop()


if __name__ == "__main__":
    main()
