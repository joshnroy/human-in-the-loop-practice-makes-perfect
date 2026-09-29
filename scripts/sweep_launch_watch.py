"""Durable, fail-closed submission of a frozen, approved Sweep batch."""

import argparse
import datetime
import fcntl
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path


class LaunchWatch:
    def __init__(self, *, readiness, state_dir):
        self.readiness = Path(readiness)
        self.root = Path(state_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}

    @staticmethod
    def run(*, argv, timeout=60):
        return subprocess.check_output(
            argv, text=True, stderr=subprocess.STDOUT, timeout=timeout
        ).strip()

    @staticmethod
    def digest(*, path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def save(self, *, status, reason, **fields):
        old = (self.state.get("status"), self.state.get("reason"))
        self.state.update(
            fields,
            status=status,
            reason=reason,
            updated=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
        temporary = self.state_path.with_suffix(".tmp")
        with temporary.open("w") as stream:
            json.dump(self.state, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.state_path)
        descriptor = os.open(self.root, os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if old != (status, reason):
            with (self.root / "transitions.jsonl").open("a") as stream:
                stream.write(json.dumps(self.state) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        print(json.dumps(self.state), flush=True)

    def reconcile(self, *, token):
        name = "sweep-" + token
        # Both commands must succeed: an empty queue alone does not rule out a finished job.
        command = (
            "squeue -u jr2860 -h -o '%i|%j'; "
            "a=$?; sacct -u jr2860 -S 2026-09-28 -n -P "
            "--format=JobIDRaw,JobName%128; b=$?; test $a -eq 0 -a $b -eq 0"
        )
        output = self.run(
            argv=["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "della", command]
        )
        return sorted({
            line.split("|")[0].strip().split("_")[0]
            for line in output.splitlines()
            if len(line.split("|")) >= 2 and line.split("|")[1].strip() == name
        })

    def tick(self):
        with (self.root / "launch.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
            # Load inside the lock, avoiding stale state in concurrent invocations.
            self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
            if self.state.get("job_ids"):
                self.save(status="submitted", reason="Recorded Slurm job IDs; never resubmit")
                return
            if self.state.get("submission_intent"):
                try:
                    jobs = self.reconcile(token=self.state["token"])
                except Exception as error:
                    self.save(
                        status="blocked",
                        reason="Submission uncertain; reconciliation failed: " + str(error),
                    )
                    return
                if jobs:
                    self.save(
                        status="submitted", reason="Recovered submission from Slurm", job_ids=jobs
                    )
                else:
                    self.save(
                        status="blocked",
                        reason=(
                            "Submission uncertain; no matching Slurm record yet. "
                            "No automatic retry."
                        ),
                    )
                return
            if not self.readiness.exists():
                self.save(status="waiting", reason="Readiness evidence has not been published")
                return
            try:
                ready = json.loads(self.readiness.read_text())
                if ready.get("status") != "READY":
                    self.save(
                        status="blocked",
                        reason=ready.get("reason", "Scientific readiness is not READY"),
                    )
                    return
                revision = ready["revision"]
                if not re.fullmatch("[0-9a-f]{40}", revision):
                    raise ValueError("Revision must be a full commit SHA")
                source = Path(ready["source_dir"])
                if not source.is_absolute():
                    raise ValueError("Source directory must be absolute")
                if self.run(argv=["git", "-C", str(source), "rev-parse", "HEAD"]) != revision:
                    raise ValueError("Source revision changed")
                if self.run(
                    argv=["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"]
                ):
                    raise ValueError("Tracked source or dependencies are dirty")
                if not ready["evidence"]:
                    raise ValueError("Readiness needs hashed evidence")
                for artifact in [ready["manifest"], *ready["evidence"]]:
                    if (
                        not Path(artifact["path"]).is_absolute()
                        or self.digest(path=artifact["path"]) != artifact["sha256"]
                    ):
                        raise ValueError(
                            "Readiness artifact missing or changed: " + artifact["path"]
                        )
                token = self.digest(path=self.readiness)[:20]
                argv = ready["submit_argv"]
                if (
                    not isinstance(argv, list)
                    or not all(isinstance(v, str) for v in argv)
                    or "{token}" not in argv
                ):
                    raise ValueError("Submission argv must contain a token placeholder")
                argv = [v.replace("{token}", token) for v in argv]
                self.save(
                    status="ready",
                    reason="Frozen revision and readiness hashes verified",
                    token=token,
                    revision=revision,
                )
                jobs = self.reconcile(token=token)
                if jobs:
                    self.save(
                        status="submitted", reason="Matching batch already exists", job_ids=jobs
                    )
                    return
                # Persist intent before sbatch; uncertain intent is never retried.
                self.save(
                    status="submitting",
                    reason="Submission intent persisted",
                    submission_intent=True,
                    submit_argv=argv,
                )
                output = self.run(argv=argv, timeout=1500)
                result = json.loads(output.splitlines()[-1])
                jobs = result["job_ids"]
                if not jobs or not all(re.fullmatch(r"[0-9]+", str(job)) for job in jobs):
                    raise ValueError("Submission did not return numeric job IDs")
                self.save(status="submitted", reason="Submission acknowledged", job_ids=jobs)
            except Exception as error:
                self.save(status="blocked", reason=str(error))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--readiness", required=True)
    parser.add_argument("--state-dir", required=True)
    arguments = parser.parse_args()
    LaunchWatch(readiness=arguments.readiness, state_dir=arguments.state_dir).tick()
