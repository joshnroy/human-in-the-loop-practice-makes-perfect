"""Resume authorized Sweep work only after its owner's durable lease expires."""

import argparse
import datetime as dt
import fcntl
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def atomic_json(*, path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def timestamp(*, text):
    return dt.datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


class Wakeup:
    def __init__(self, *, config, state_dir):
        self.config = config
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def event(self, *, task, status, **details):
        value = {
            "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "task": task,
            "status": status,
            **details,
        }
        with (self.state_dir / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        atomic_json(path=self.state_dir / "latest.json", value=value)
        return value

    @staticmethod
    def unit_active(*, unit):
        result = subprocess.run(
            ["systemctl", "--user", "show", unit, "--property=ActiveState", "--value"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        return result.stdout.strip() in {"active", "activating", "reloading"}

    @staticmethod
    def execute(*, argv, prompt, log, timeout):
        with log.open("w") as output:
            process = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=output,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            try:
                process.communicate(prompt, timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                return "timeout"
            return "complete" if process.returncode == 0 else f"exit-{process.returncode}"

    def tick(self):
        with (self.state_dir / "supervisor.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return {"status": "already-running"}
            results = []
            for task in self.config["tasks"]:
                try:
                    results.append(self.task(task=task))
                except Exception as error:  # retain evidence; never restart science
                    results.append(
                        self.event(
                            task=task["name"],
                            status="supervisor-error",
                            error=f"{type(error).__name__}: {error}",
                        )
                    )
            return results

    def task(self, *, task):
        name = task["name"]
        owner_path = Path(task["owner_status"])
        owner = json.loads(owner_path.read_text())
        now = time.time()
        progress = timestamp(text=owner["last_progress_utc"])
        expires = (
            timestamp(text=owner["lease_expires_utc"])
            if owner.get("lease_expires_utc")
            else progress + self.config.get("lease_seconds", 2700)
        )
        if expires > now:
            return self.event(task=name, status="owner-active", lease_expires=expires)
        if owner.get("status", "").startswith(("blocked-pending", "complete", "done")):
            return self.event(
                task=name, status="waiting-authorized-next-step", owner_status=owner.get("status")
            )
        active = [unit for unit in owner.get("current_units", []) if self.unit_active(unit=unit)]
        observed = [
            Path(path).stat().st_mtime
            for path in owner.get("progress_paths", [])
            if Path(path).exists()
        ]
        latest_progress = max([progress, *observed])
        read_only = False
        if active:
            if now - latest_progress < self.config.get("stalled_seconds", 7200):
                return self.event(task=name, status="validation-running", units=active)
            # A stale running process gets diagnosis, never automatic termination.
            read_only = True
        worktree = Path(owner["worktree"]).resolve()
        allowed = Path(task["worktree"]).resolve()
        if worktree != allowed or not worktree.is_dir():
            raise ValueError("Owner worktree must exactly match configured isolated worktree")
        free_kib = int(
            next(
                line.split()[1]
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:")
            )
        )
        if free_kib < 12 * 1024 * 1024:
            return self.event(task=name, status="waiting-memory", available_kib=free_kib)
        run_id = f"{name}-{time.time_ns()}"
        timeout = self.config.get("max_turn_seconds", 1200)
        if not read_only:
            owner.update(
                owner=f"codex-cli:{run_id}",
                status="active-cli-continuation",
                lease_expires_utc=dt.datetime.fromtimestamp(
                    now + timeout + 300, dt.timezone.utc
                ).isoformat(),
            )
            atomic_json(path=owner_path, value=owner)
        mode = (
            (
                "READ ONLY: diagnose stale running validation; "
                "do not edit, stop, restart or launch anything."
            )
            if read_only
            else (
                "You own the expired work lease. Continue unfinished authorized implementation "
                "and validation; do not stop at monitoring."
            )
        )
        prompt = (
            Path(task["prompt_file"]).read_text()
            + "\n\n"
            + mode
            + "\nOwner status: "
            + str(owner_path)
            + "\nOwner snapshot:\n"
            + json.dumps(owner, indent=2)
        )
        argv = [
            self.config.get("codex", "/home/josh/.local/bin/codex"),
            "exec",
            "--ephemeral",
            "--json",
            "-C",
            str(worktree),
            "--output-last-message",
            str(self.state_dir / f"{run_id}.last.txt"),
        ]
        if read_only:
            argv.extend(["--sandbox", "read-only"])
        else:
            # This flag selects workspace-write plus automatic review.
            argv.append("--approve-for-me")
            for path in task.get("additional_write_dirs", []):
                argv.extend(["--add-dir", path])
        argv.append("-")
        self.event(
            task=name,
            status="diagnosing-stall" if read_only else "continuing",
            run_id=run_id,
            argv=argv,
        )
        outcome = self.execute(
            argv=argv, prompt=prompt, log=self.state_dir / f"{run_id}.jsonl", timeout=timeout
        )
        if not read_only:
            current = json.loads(owner_path.read_text())
            if current.get("owner") == f"codex-cli:{run_id}":
                current["lease_expires_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
                if current.get("status") == "active-cli-continuation":
                    current["status"] = "idle-continuation-returned"
                current["last_cli_outcome"] = outcome
                atomic_json(path=owner_path, value=current)
        return self.event(task=name, status=outcome, run_id=run_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            Wakeup(config=json.loads(args.config.read_text()), state_dir=args.state_dir).tick()
        )
    )


if __name__ == "__main__":
    main()
