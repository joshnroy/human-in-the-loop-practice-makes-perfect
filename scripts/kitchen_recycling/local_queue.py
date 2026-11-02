"""Prepare a local fallback; execution additionally requires explicit local authorization."""

import argparse
import json
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scripts.kitchen_recycling.worker import REPO, binding, read, verify


def execute(*, command: list[str]) -> int:
    return subprocess.run(command, cwd=REPO, check=False).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--max-parallel", type=int, choices=range(1, 10), default=3)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    launch = args.launch.resolve()
    manifest = read(path=launch / "manifest.json")
    commands = [
        [sys.executable, "-m", "scripts.kitchen_recycling.worker", "full",
         "--launch", str(launch), "--task", str(row["index"])]
        for row in manifest["runs"]
    ]
    if not args.execute:
        print("PREPARATION ONLY: set PYTHONPATH to the frozen source and validated local overlay.")
        print("Local and Della runtime locks differ; never reuse an unvalidated remote lock.")
        print("\n".join(shlex.join(command) for command in commands))
        return
    authorization = read(path=launch / "local-authorization.json")
    if authorization != {"binding": binding(launch=launch), "authorized_by": "main",
                         "run_local": True, "max_parallel": args.max_parallel}:
        raise ValueError("Main must explicitly authorize this exact local fallback and concurrency")
    verify(launch=launch, full=True)
    lock = launch / "local-submission.lock"
    lock.mkdir()
    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        tasks = {executor.submit(execute, command=command): index
                 for index, command in enumerate(commands)}
        failures = {}
        for future in as_completed(tasks):
            result = future.result()
            if result:
                failures[tasks[future]] = result
    (launch / "local-receipt.json").write_text(json.dumps({"failures": failures}) + "\n")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
