"""Persistent, idempotent launch of the authorized four-arm workstation fallback."""

import argparse
import fcntl
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    raw = args.manifest.read_bytes()
    manifest = json.loads(raw)
    root = Path(__file__).resolve().parent.parent
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if manifest["status"] != "READY" or manifest["source_revision"] != revision:
        raise RuntimeError("Requires ready manifest pinned to this source")
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True
    )
    if dirty:
        raise RuntimeError("Tracked source changed after readiness")
    token = hashlib.sha256(raw).hexdigest()[:20]
    directory = args.manifest.parent / ("workstation-" + token)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for arm in manifest["launch_order"][0]:
            unit = "sweep-science-" + token + "-" + arm
            record = directory / (arm + ".json")
            if record.exists():
                print(record.read_text(), flush=True)
                continue
            output = Path(manifest["workstation_results_root"]) / arm
            item = {
                "unit": unit,
                "arm": arm,
                "source_revision": revision,
                "status": "launch_intent",
                "output": str(output),
            }
            record.write_text(json.dumps(item, indent=2) + "\n")
            command = [
                "systemd-run",
                "--user",
                "--unit=" + unit,
                "-p",
                "MemoryMax=6G",
                "-p",
                "MemorySwapMax=0",
                "-p",
                "OOMPolicy=continue",
                "--working-directory=" + str(root),
                str(root / "scripts/with_sweep_env.sh"),
                "python",
                str(root / "scripts/run_sweep_manifest_arm.py"),
                "--manifest",
                str(args.manifest.resolve()),
                "--arm",
                arm,
                "--seed",
                str(manifest["first_seed"]),
                "--output",
                str(output),
            ]
            completed = subprocess.run(command, text=True, capture_output=True)
            item.update(
                status="submitted" if completed.returncode == 0 else "launch_failed",
                returncode=completed.returncode,
                output_message=completed.stdout + completed.stderr,
            )
            record.write_text(json.dumps(item, indent=2) + "\n")
            print(json.dumps(item), flush=True)
            completed.check_returncode()


if __name__ == "__main__":
    main()
