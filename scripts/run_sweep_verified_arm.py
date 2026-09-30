"""Run one frozen science arm once and attest completed, reconciled output only."""

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path


class SweepCompletion:
    @staticmethod
    def require(*, condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    @staticmethod
    def digest(*, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def lines(*, path: Path) -> list[dict]:
        with path.open() as stream:
            return [json.loads(line) for line in stream if line.strip()]

    @classmethod
    def validate(
        cls,
        *,
        output: Path,
        manifest_path: Path,
        arm_name: str,
        seed: int,
        revision: str,
        job_id: str,
    ) -> dict:
        manifest = json.loads(manifest_path.read_text())
        arm = next(a for a in manifest["arms"] if a["name"] == arm_name)
        root = output / arm["method"] / str(seed)
        paths = {
            name: root / name
            for name in (
                "stats.json",
                "timing.json",
                "config_snapshot.json",
                "episode_traces.jsonl",
                "practice/sweep_events.jsonl",
                "evaluation/sweep_events.jsonl",
            )
        }
        cls.require(
            condition=all(p.is_file() for p in paths.values()),
            message="Missing mandatory completion evidence",
        )
        stats = json.loads(paths["stats.json"].read_text())
        timing = json.loads(paths["timing.json"].read_text())
        config = json.loads(paths["config_snapshot.json"].read_text())
        cls.require(
            condition=timing.get("succeeded") is True and timing.get("returncode") == 0,
            message="Runner did not report a successful exit",
        )
        cls.require(
            condition=config.get("git_commit") == revision and config.get("git_dirty") is False,
            message="Output was produced by a different or dirty revision",
        )
        for key in ("kindergarden_dirty", "kinder_models_dirty"):
            cls.require(condition=config.get(key) is False, message=f"Unclean dependency: {key}")
        args = config["args"]
        expected = dict(
            num_cycles="50",
            max_steps_per_interaction="20",
            num_test_tasks="10",
            practice_reset_policy="never",
            goal_pursuit_horizon="0",
            method=arm["method"],
            canonical_seed=str(seed),
            seed=str(seed),
            no_human_reset=str(not arm["human_reset"]),
        )
        for key, value in expected.items():
            cls.require(condition=str(args.get(key)) == value, message=f"Unexpected argument {key}")
        cls.require(
            condition=Path(args["sweep_manifest"]).resolve() == manifest_path.resolve(),
            message="Output refers to another manifest",
        )
        cls.require(
            condition=math.isclose(
                float(args["human_reset_practice_cost"]),
                arm["human_reset_practice_cost"],
                rel_tol=1e-12,
            ),
            message="Human cost differs from frozen arm",
        )
        evaluations = stats["evaluations"]
        ends = stats["practice_session_ends"]
        cls.require(
            condition=len(evaluations) == 51 and all(e[2] == 10 for e in evaluations),
            message="Expected 51 complete 10-task evaluation checkpoints",
        )
        cls.require(
            condition=[e["cycle_index"] for e in ends] == list(range(50)),
            message="Expected exactly 50 ordered practice cycles",
        )
        cls.require(
            condition=all(
                e["action_limit"] == 20 and 0 <= e["actions_executed"] <= 20 for e in ends
            ),
            message="Practice action budget violated",
        )
        cls.require(condition=stats["num_practice_resets"] == 0, message="Automatic practice reset")
        breakdowns = stats["breakdowns"]
        cls.require(
            condition=len(breakdowns) == 51
            and all(
                sorted(o["task_index"] for o in b["outcomes"]) == list(range(10))
                for b in breakdowns
            ),
            message="Missing evaluation task outcomes",
        )
        practice = cls.lines(path=paths["practice/sweep_events.jsonl"])
        cls.require(
            condition=all(
                r["kind"] in {"episode_start", "initial_validation", "action", "human_reset"}
                for r in practice
            ),
            message="Unrecognized practice event; audit required",
        )
        starts = [r for r in practice if r["kind"] == "episode_start"]
        cls.require(
            condition=len(starts) == 1
            and starts[0]["seed"] == seed
            and starts[0]["evaluation"] is False,
            message="Practice physical state was restarted",
        )
        actions = [r for r in practice if r["kind"] == "action"]
        humans = [r for r in practice if r["kind"] == "human_reset"]
        for rows in (actions, humans):
            cls.require(
                condition=[r["index"] for r in rows] == list(range(1, len(rows) + 1)),
                message="Action log indices are missing or duplicated",
            )
        cls.require(
            condition=all(
                r["success"] is True and r["validation"]["valid"] is True for r in humans
            ),
            message="Human restoration failed",
        )
        cls.require(
            condition=arm["human_reset"] or not humans, message="Human used in no-human arm"
        )
        cls.require(
            condition=stats["num_human_interventions_recorded"] == len(humans),
            message="Human intervention counter disagrees with physical events",
        )
        cls.require(
            condition=math.isclose(
                stats["summed_human_cost_recorded"],
                len(humans) * arm["human_reset_practice_cost"],
                rel_tol=1e-9,
                abs_tol=1e-9,
            ),
            message="Human cost counter disagrees with physical events",
        )
        cls.require(
            condition=sum(e["actions_executed"] for e in ends) == len(actions) + len(humans),
            message="Practice budget omits or duplicates physical/human actions",
        )
        cls.require(
            condition=evaluations[-1][0] == len(actions),
            message="Online transition counter disagrees with physical actions",
        )
        evaluation_events = cls.lines(path=paths["evaluation/sweep_events.jsonl"])
        cls.require(
            condition=sum(r["kind"] == "episode_start" for r in evaluation_events) >= 510
            and not any(r["kind"] == "human_reset" for r in evaluation_events),
            message="Evaluation lacks isolated physical task starts or used human resets",
        )
        for rows in (practice, evaluation_events):
            validations = [
                r.get("validation", r)
                for r in rows
                if r["kind"] == "initial_validation"
                or (r["kind"] == "episode_start" and "validation" in r)
            ]
            cls.require(
                condition=bool(validations) and all(r["valid"] is True for r in validations),
                message="Missing or invalid native start checks",
            )
        traces = cls.lines(path=paths["episode_traces.jsonl"])
        horizon = manifest["deployment_horizon"]
        cls.require(
            condition=all(
                0 <= r["checkpoint"] <= 50
                and 0 <= r["task_index"] < 10
                and 0 <= r["step_index"] < horizon
                for r in traces
            ),
            message="Evaluation trace violates frozen horizon/checkpoint bounds",
        )
        artifacts = [manifest_path, *paths.values()]
        return dict(
            status="PASS",
            job_id=job_id,
            revision=revision,
            practice_cycles=50,
            evaluation_checkpoints=51,
            practice_resets=0,
            human_accounting_matches=True,
            robot_actions=len(actions),
            human_resets=len(humans),
            summed_human_cost=stats["summed_human_cost_recorded"],
            final_evaluation=evaluations[-1],
            artifacts=[dict(path=str(p.resolve()), sha256=cls.digest(path=p)) for p in artifacts],
        )

    @classmethod
    def run(cls) -> int:
        from check_slurm_memory import main as check_memory

        check_memory()
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--manifest", type=Path, required=True)
        parser.add_argument("--arm", required=True)
        parser.add_argument("--seed", type=int, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--completion-record", type=Path, required=True)
        parser.add_argument("--job-id", required=True)
        parser.add_argument("--revision", required=True)
        args = parser.parse_args()
        source = Path(__file__).resolve().parents[1]
        cls.require(
            condition=re.fullmatch(r"[0-9a-f]{40}", args.revision) is not None,
            message="Expected full frozen revision",
        )
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True)
        cls.require(
            condition=head == args.revision and not dirty, message="Frozen source mismatch or dirty"
        )
        cls.require(
            condition=not args.completion_record.exists(),
            message="Completion record already exists",
        )
        manifest = json.loads(args.manifest.read_text())
        for key, value in dict(
            num_cycles=50,
            max_steps_per_interaction=20,
            num_test_tasks=10,
            practice_reset_policy="never",
        ).items():
            cls.require(condition=manifest.get(key) == value, message=f"Unapproved protocol {key}")
        solver = manifest["pomdp"]
        cls.require(
            condition=solver["pomdp_solver"] == "determinized_astar"
            and int(solver["pomdp_max_search_iterations"]) == 1000
            and int(solver["pomdp_search_depth"]) == 20,
            message="Unapproved planning approximation/budget",
        )
        cls.require(
            condition=manifest.get("source_revision") == args.revision,
            message="Manifest is not frozen to this revision",
        )
        if manifest.get("environment") == "sweep_simple3d":
            cls.require(
                condition=manifest["deployment_horizon"] == 10,
                message="Simple evaluation horizon must be 10",
            )
        manifest_hash = cls.digest(path=args.manifest)
        command = [
            sys.executable,
            str(source / "scripts/run_sweep_manifest_arm.py"),
            "--manifest",
            str(args.manifest.resolve()),
            "--arm",
            args.arm,
            "--seed",
            str(args.seed),
            "--output",
            str(args.output.resolve()),
        ]
        subprocess.run(command, cwd=source, check=True)
        cls.require(
            condition=cls.digest(path=args.manifest) == manifest_hash,
            message="Manifest changed while run was active",
        )
        report = cls.validate(
            output=args.output,
            manifest_path=args.manifest,
            arm_name=args.arm,
            seed=args.seed,
            revision=args.revision,
            job_id=args.job_id,
        )
        args.completion_record.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.completion_record.with_name(
            args.completion_record.name + f".{os.getpid()}.tmp"
        )
        with temporary.open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, args.completion_record)
        descriptor = os.open(args.completion_record.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return 0


if __name__ == "__main__":
    raise SystemExit(SweepCompletion.run())
