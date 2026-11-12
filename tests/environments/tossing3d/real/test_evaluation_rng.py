"""An in-process evaluation draws from its own RNG stream, never practice's.

Seed 7 (2026-11-09) and the October run: a real run's first practice cycle threw the
same parameters, in the same order, as its step-0 evaluation, because the evaluation's
deployment was restored with the snapshot's RNG state and practice carried on from that
same state. Each evaluation's draws still depend only on its snapshot, so a resumed
evaluation re-throws what the interrupted one threw.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from hitl_pmp.cli import Cli
from hitl_pmp.environments.tossing3d.real import cli as real_cli
from hitl_pmp.environments.tossing3d.real.fake_robot import FakeTossingRobot
from hitl_pmp.step_protocol import StepFiles

pytestmark = pytest.mark.xdist_group("tossing3d_real_evaluation_rng")

Throw = tuple[float, float]


def _argv(*, output: Path, resume: bool = False) -> list[str]:
    return [
        *("--env", "tossing3d_real", "--method", "pomdp", "--preset", "tossing3d-canonical"),
        *("--robot", "fake", "--num-test-tasks", "3", "--seed", "0"),
        *("--practice-step-budget", "4500", "--max-steps-per-interaction", "4"),
        *("--measure-every-cycles", "1", "--output-dir", str(output)),
        *(["--resume", str(output)] if resume else []),
    ]


class Phases:
    """The room's throws, by phase: each `measurement` event opens an evaluation
    (keyed by its index), every other run event opens practice."""

    def __init__(self, *, robot: FakeTossingRobot, monkeypatch: pytest.MonkeyPatch) -> None:
        self.robot = robot
        original = StepFiles.event

        def marking(*, path: Path, **record: Any) -> None:
            if Path(path).name == "step_events.jsonl":
                robot.calls.append(("event", (record["event"], record.get("index"))))
            original(path=path, **record)

        monkeypatch.setattr(StepFiles, "event", staticmethod(marking))

    def segments(self) -> list[tuple[str, list[Throw]]]:
        """(`eval N` or `practice`, the throws in it as (speed, release ms)), in order."""
        segments: list[tuple[str, list[Throw]]] = [("practice", [])]
        for name, payload in self.robot.calls:
            if name == "event":
                kind, index = payload
                phase = f"eval {index}" if kind == "measurement" else "practice"
                if phase != segments[-1][0]:
                    segments.append((phase, []))
            elif name == "toss":
                _, speed, release_ms, _ = payload
                segments[-1][1].append((round(speed, 6), round(release_ms, 6)))
        return [(phase, throws) for phase, throws in segments if throws]


class InterruptedInEvaluation(FakeTossingRobot):
    """Ctrl-C once `tosses_left` more throws have landed; None never interrupts."""

    tosses_left: int | None = None

    def toss(
        self, *, standoff: float, speed: float, release_ms: float, direction: float
    ) -> dict[str, Any]:
        reply = super().toss(
            standoff=standoff, speed=speed, release_ms=release_ms, direction=direction
        )
        if self.tosses_left is not None:
            self.tosses_left -= 1
            if self.tosses_left == 0:
                raise KeyboardInterrupt
        return reply


def test_the_first_practice_cycle_does_not_replay_the_step_zero_evaluations_throws(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    robot = FakeTossingRobot()
    monkeypatch.setattr(real_cli, "FakeTossingRobot", lambda: robot)
    phases = Phases(robot=robot, monkeypatch=monkeypatch)
    Cli.main(argv=_argv(output=tmp_path / "run"))
    segments = phases.segments()
    (phase, evaluation), (next_phase, practice) = segments[:2]
    assert phase == "eval 0" and next_phase == "practice"
    assert len(evaluation) == 3 and practice
    shared = min(len(evaluation), len(practice))
    assert practice[:shared] != evaluation[:shared]
    assert practice[0] != evaluation[0]
    # Nor does practice after a later evaluation replay that evaluation's throws.
    for (before, thrown), (after, practised) in zip(segments, segments[1:], strict=False):
        if before.startswith("eval") and after == "practice":
            assert practised[0] != thrown[0], before


def test_a_resumed_evaluation_re_throws_what_the_interrupted_one_threw(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Interrupted after its second task, evaluation 2 is re-run from its first task on
    resume, and throws what it threw before and what an uninterrupted run throws."""
    alone_robot = FakeTossingRobot()
    with monkeypatch.context() as patch:
        patch.setattr(real_cli, "FakeTossingRobot", lambda: alone_robot)
        alone = Phases(robot=alone_robot, monkeypatch=patch)
        Cli.main(argv=_argv(output=tmp_path / "alone"))
    expected = dict(alone.segments()[::-1])  # the first of any repeated phase wins
    assert len(expected["eval 2"]) == 3

    robot = InterruptedInEvaluation()
    monkeypatch.setattr(real_cli, "FakeTossingRobot", lambda: robot)
    phases = Phases(robot=robot, monkeypatch=monkeypatch)
    output = tmp_path / "resumed"
    with monkeypatch.context() as patch:
        original = StepFiles.event

        def event(*, path: Path, **record: Any) -> None:
            original(path=path, **record)
            if record.get("event") == "measurement" and record.get("index") == 2:
                robot.tosses_left = 2

        patch.setattr(StepFiles, "event", staticmethod(event))
        with pytest.raises(KeyboardInterrupt):
            Cli.main(argv=_argv(output=output))
    robot.tosses_left = None
    Cli.main(argv=_argv(output=output, resume=True))

    evaluations = [(p, t) for p, t in phases.segments() if p == "eval 2"]
    (_, interrupted), (_, rerun) = evaluations
    assert interrupted == expected["eval 2"][:2]
    assert rerun == expected["eval 2"]
    for phase, throws in phases.segments():
        if phase.startswith("eval") and phase != "eval 2":
            assert throws == expected[phase], phase
    assert json.loads((output / "step_evaluations.json").read_text())
