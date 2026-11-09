"""Evaluate a saved deployment snapshot of a step-protocol run again, later.

    python -m hitl_pmp.evaluate_snapshot --run <run> \\
        --snapshot <run>/snapshots/after-update-0006.pickle --num-test-tasks 1 \\
        --reason "eval 2 task 2 ended on a pick outlier" [--sampling-seed S] [run flags]

The run's settings are its `config_snapshot.json`'s, replayed through the same parser
(`ConfigReplay`), so nothing can drift from what the run used. A run flag may be
repeated (`--robot real`), and is refused if it disagrees with the recorded value.

The evaluation is the one a run takes in process (`StepEvaluation.run`): the run's
first N test tasks, each a scene the person sets (on the real robot, driving home before
the first task only), the policy acting greedily until the goal or the task's step
budget. It goes to a new folder, `<run>/evaluations-extra/<snapshot-stem>-<k>/`, and a
line in `<run>/evaluations-extra/events.jsonl` says which snapshot, how many tasks, the
code's commit, the sampling seed and why. Nothing else in the run is written to.

`--sampling-seed` reseeds every RNG the deployment draws from (its own, each sampler's
and the process-global ones) before the first task. By default a fresh seed is taken
from the clock, and it is always written down. `--sampling-seed snapshot` keeps the
snapshot's RNG state instead, which repeats the run's own evaluation of that policy.

With `--robot real` it asks for a y before anything moves, as the run's launch did.
"""

import argparse
import ast
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, ClassVar

from hitl_pmp.cli import STEP_PROTOCOL_ENVIRONMENTS, Cli
from hitl_pmp.config_snapshot import ConfigSnapshot
from hitl_pmp.step_protocol import PolicyFileHash, StepEvaluation, StepFiles, StepProtocolResume


class SnapshotEvaluation:
    """A static-method container, never instantiated."""

    EXTRA_DIRECTORY: ClassVar[str] = "evaluations-extra"
    EVENTS_FILENAME: ClassVar[str] = "events.jsonl"
    REQUEST_FILENAME: ClassVar[str] = "request.json"
    SNAPSHOT_RNG: ClassVar[str] = "snapshot"

    @staticmethod
    def parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            allow_abbrev=False,
        )
        parser.add_argument("--run", type=Path, required=True, help="The run's output folder.")
        parser.add_argument(
            "--snapshot",
            type=Path,
            required=True,
            help="A deployment snapshot in the run's snapshots/ folder.",
        )
        parser.add_argument(
            "--num-test-tasks",
            type=lambda value: Cli.parse_positive_int(value=value),
            required=True,
            help="How many of the run's test tasks to evaluate, from the first.",
        )
        parser.add_argument(
            "--reason", required=True, help="Why this re-run, in words; written down."
        )
        parser.add_argument(
            "--sampling-seed",
            default=None,
            help="An integer seed for the deployment's draws, or 'snapshot' for the "
            "snapshot's own RNG state. Default: a fresh seed from the clock.",
        )
        return parser

    @staticmethod
    def main(*, argv: list[str] | None = None) -> Path:
        """Run the evaluation; return its folder."""
        own, run_flags = SnapshotEvaluation.parser().parse_known_args(argv)
        run = Path(own.run)
        snapshot = SnapshotEvaluation.checked_snapshot(run=run, snapshot=Path(own.snapshot))
        recorded = json.loads((run / "config_snapshot.json").read_text())
        args = ConfigReplay.args(recorded=recorded["args"], flags=run_flags)
        seed, source = SnapshotEvaluation.sampling_seed(value=own.sampling_seed)
        # The run's launch asked before the robot first moved; with --robot-no-confirm-home
        # nothing else would before this evaluation drives home.
        if getattr(args, "robot", None) == "real":
            question = (
                f"Evaluate {snapshot.name} on the robot for {own.num_test_tasks} task(s)? "
                "It drives home first. [y/n] "
            )
            if input(question).strip() != "y":
                raise SystemExit("not started")
        out = SnapshotEvaluation.next_directory(run=run, snapshot=snapshot)
        configuration = {
            **vars(args),
            "num_test_tasks": own.num_test_tasks,
            # The robot's session log goes beside this evaluation, not into the run's.
            "output_dir": str(out),
        }
        code = ConfigSnapshot.collect(args=args, fd_exec_path=os.environ.get("FD_EXEC_PATH"))
        raw = snapshot.read_bytes()
        request = {
            "snapshot": str(snapshot),
            **PolicyFileHash.record(raw=raw),
            "num_test_tasks": own.num_test_tasks,
            "sampling_seed": seed,
            "sampling_seed_source": source,
            "reason": own.reason,
            "output": str(out),
            "git_commit": code.git_commit,
            "git_dirty": code.git_dirty,
            "run_git_commit": recorded.get("git_commit"),
            "argv": list(sys.argv[1:] if argv is None else argv),
            "started_unix": time.time(),
        }
        if code.git_commit != recorded.get("git_commit"):
            print(
                f"note: this code is {code.git_commit}; the run was {recorded.get('git_commit')}",
                flush=True,
            )
        events = run / SnapshotEvaluation.EXTRA_DIRECTORY / SnapshotEvaluation.EVENTS_FILENAME
        StepFiles.event(path=events, event="extra_evaluation", **request)
        try:
            result = StepEvaluation.run(
                snapshot=str(snapshot),
                configuration=configuration,
                output=str(out),
                environment=STEP_PROTOCOL_ENVIRONMENTS[args.env],
                label=f"re-run of {snapshot.stem}",
                sampling_seed=seed,
            )
        except BaseException as error:
            StepFiles.event(
                path=events, event="extra_evaluation_failed", output=str(out), error=repr(error)
            )
            raise
        finally:
            if out.exists():
                StepFiles.json(path=out / SnapshotEvaluation.REQUEST_FILENAME, value=request)
        StepFiles.event(
            path=events,
            event="extra_evaluation_done",
            output=str(out),
            num_solved=result["num_solved"],
            num_total=result["num_total"],
        )
        print(f"{result['num_solved']}/{result['num_total']} solved; written to {out}", flush=True)
        return out

    @staticmethod
    def checked_snapshot(*, run: Path, snapshot: Path) -> Path:
        """The snapshot, refused unless it is one of this run's."""
        if not (run / "config_snapshot.json").is_file():
            raise FileNotFoundError(f"{run} has no config_snapshot.json: not a run's folder")
        resolved = snapshot.resolve()
        if resolved.parent != (run / "snapshots").resolve():
            raise ValueError(
                f"{snapshot} is not in {run / 'snapshots'}: only a run's own snapshots "
                "are evaluated with its settings"
            )
        if not resolved.is_file():
            raise FileNotFoundError(f"no snapshot at {snapshot}")
        return resolved

    @staticmethod
    def sampling_seed(*, value: str | None) -> tuple[int | None, str]:
        """The seed and where it came from: given, the clock, or none (the snapshot's)."""
        if value == SnapshotEvaluation.SNAPSHOT_RNG:
            return None, SnapshotEvaluation.SNAPSHOT_RNG
        if value is None:
            return time.time_ns() % 2**32, "time"
        try:
            seed = int(value)
        except ValueError:
            raise ValueError(
                f"--sampling-seed takes an integer or '{SnapshotEvaluation.SNAPSHOT_RNG}', "
                f"got {value!r}"
            ) from None
        if seed < 0:
            raise ValueError(f"--sampling-seed must not be negative, got {seed}")
        return seed, "given"

    @staticmethod
    def next_directory(*, run: Path, snapshot: Path) -> Path:
        """`<snapshot-stem>-<k>` with the first k from 1 not yet taken; never an
        existing folder."""
        parent = run / SnapshotEvaluation.EXTRA_DIRECTORY
        parent.mkdir(exist_ok=True)
        k = 1
        while (parent / f"{snapshot.stem}-{k}").exists():
            k += 1
        return parent / f"{snapshot.stem}-{k}"


class ConfigReplay:
    """A run's recorded `config_snapshot.json` arguments turned back into the namespace
    `Cli.parse_args` resolved, through the same parser, and checked to be exactly it."""

    # Flags whose recorded value is not replayed: the output folder is this re-run's,
    # and the number of test tasks is chosen for it.
    NOT_REPLAYED: ClassVar[frozenset[str]] = frozenset({"output_dir", "resume"})

    @staticmethod
    def args(*, recorded: dict[str, str], flags: list[str]) -> argparse.Namespace:
        """The run's namespace; `flags` may repeat any run flag, and one that changes a
        recorded value is refused, naming it."""
        argv = ConfigReplay.argv(recorded=recorded)
        replayed = Cli.parse_args(argv=argv)
        ConfigReplay.refuse_differences(
            before=recorded,
            after=ConfigReplay.strings(args=replayed),
            reason="the run's recorded config cannot be reproduced by this code's parser",
        )
        if flags:
            given = Cli.parse_args(argv=[*argv, *flags])
            ConfigReplay.refuse_differences(
                before=ConfigReplay.strings(args=replayed),
                after=ConfigReplay.strings(args=given),
                reason="a passed flag disagrees with the run's config_snapshot.json",
                compare_all=True,
            )
        return replayed

    @staticmethod
    def strings(*, args: argparse.Namespace) -> dict[str, str]:
        return {key: str(value) for key, value in vars(args).items()}

    @staticmethod
    def refuse_differences(
        *, before: dict[str, str], after: dict[str, str], reason: str, compare_all: bool = False
    ) -> None:
        if compare_all:
            left, right = before, after
        else:
            left = StepProtocolResume.comparable(arguments=before)
            right = StepProtocolResume.comparable(arguments=after)
        differing = sorted(
            key for key in left.keys() | right.keys() if left.get(key) != right.get(key)
        )
        if compare_all:
            differing = [key for key in differing if key not in ConfigReplay.NOT_REPLAYED]
        if differing:
            details = "; ".join(
                f"{key}: {left.get(key, '<absent>')} -> {right.get(key, '<absent>')}"
                for key in differing
            )
            raise ValueError(
                f"refusing to evaluate: {reason}, in {', '.join(differing)} ({details})"
            )

    @staticmethod
    def argv(*, recorded: dict[str, str]) -> list[str]:
        """The command line that resolves to `recorded`: every recorded flag with a
        value, in the parser's own spelling."""
        discovery = ["--env", recorded["env"], "--method", recorded["method"]]
        if recorded.get("preset", "None") != "None":
            discovery += ["--preset", recorded["preset"]]
        parser = Cli.build_parser(argv=discovery)
        argv: list[str] = []
        for action in parser._actions:  # noqa: SLF001 -- argparse exposes no public listing
            if not action.option_strings or action.dest in ConfigReplay.NOT_REPLAYED:
                continue
            if action.dest not in recorded or isinstance(action, argparse._HelpAction):  # noqa: SLF001
                continue
            argv += ConfigReplay.tokens(action=action, value=recorded[action.dest])
        return argv

    @staticmethod
    def tokens(*, action: argparse.Action, value: str) -> list[str]:
        flag = max(action.option_strings, key=len)
        if isinstance(action, argparse.BooleanOptionalAction):
            if value == "None":
                return []
            positive = next(o for o in action.option_strings if not o.startswith("--no-"))
            negative = next(o for o in action.option_strings if o.startswith("--no-"))
            return [positive if value == "True" else negative]
        if isinstance(action, argparse._StoreTrueAction):  # noqa: SLF001
            return [flag] if value == "True" else []
        if isinstance(action, argparse._StoreFalseAction):  # noqa: SLF001
            return [flag] if value == "False" else []
        if value == "None":
            return []
        if isinstance(action, argparse._AppendAction):  # noqa: SLF001
            return [f"{flag}={item}" for item in ConfigReplay.items(value=value)]
        if type(action) is not argparse._StoreAction:  # noqa: SLF001
            raise ValueError(f"cannot replay {flag}: an unsupported {type(action).__name__}")
        if action.nargs in {"+", "*"} or isinstance(action.nargs, int):
            items = ConfigReplay.items(value=value)
            return [flag, *map(str, items)] if items else []
        return [f"{flag}={value}"]

    @staticmethod
    def items(*, value: str) -> list[Any]:
        parsed = ast.literal_eval(value)
        if not isinstance(parsed, list | tuple):
            raise ValueError(f"expected a recorded list, got {value!r}")
        return list(parsed)


if __name__ == "__main__":
    SnapshotEvaluation.main()
