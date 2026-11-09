"""`python -m hitl_pmp.evaluate_snapshot`: a saved deployment snapshot of a finished (or
stopped) run evaluated again later, on a fake robot here. The run's own settings come
from its `config_snapshot.json`; the output goes beside the run's evaluations, never
over them."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from hitl_pmp.cli import Cli
from hitl_pmp.environments.tossing3d.real import cli as real_cli
from hitl_pmp.environments.tossing3d.real.backend import PlacementRequest
from hitl_pmp.environments.tossing3d.real.fake_robot import FakePerson, FakeTossingRobot
from hitl_pmp.evaluate_snapshot import SnapshotEvaluation
from hitl_pmp.step_protocol import PolicyFileHash

pytestmark = pytest.mark.xdist_group("evaluate_snapshot")


def _run_argv(*, output: Path) -> list[str]:
    return [
        *("--env", "tossing3d_real", "--method", "pomdp", "--preset", "tossing3d-canonical"),
        *("--robot", "fake", "--num-test-tasks", "2", "--seed", "0"),
        *("--practice-step-budget", "3000", "--max-steps-per-interaction", "4"),
        *("--measure-every-cycles", "1", "--no-initial-measurement"),
        *("--wall-clock-cost-per-second", "0.1", "--output-dir", str(output)),
    ]


@pytest.fixture(scope="module")
def run(*, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """A fresh fake run that evaluated in process after its first cycle."""
    output = tmp_path_factory.mktemp("evaluate_snapshot") / "run"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(real_cli, "FakeTossingRobot", FakeTossingRobot)
        Cli.main(argv=_run_argv(output=output))
    yield output


def _measured(*, run: Path) -> tuple[Path, Path]:
    """The first in-process evaluation and the after-update snapshot it measured."""
    record = json.loads((run / "measurements.json").read_text())[0]
    snapshot = run / "snapshots" / f"after-update-{record['learning_updates']:04d}.pickle"
    return run / "evaluations" / f"{record['index']:04d}", snapshot


def _actions(*, directory: Path) -> list[tuple[Any, ...]]:
    return [
        (a["task"], a["label"], a["start_step"], a["end_step"], a["skill_error"])
        for a in map(json.loads, (directory / "actions.jsonl").read_text().splitlines())
    ]


def _tree(*, root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def _evaluate(*, run: Path, snapshot: Path, extra: tuple[str, ...] = ()) -> Path:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(real_cli, "FakeTossingRobot", FakeTossingRobot)
        return SnapshotEvaluation.main(
            argv=[
                *("--run", str(run), "--snapshot", str(snapshot)),
                *("--reason", "a test re-run"),
                *extra,
            ]
        )


def test_the_snapshot_rng_reproduces_the_in_process_evaluation_action_for_action(
    *, run: Path
) -> None:
    evaluation, snapshot = _measured(run=run)
    out = _evaluate(
        run=run,
        snapshot=snapshot,
        extra=("--num-test-tasks", "2", "--sampling-seed", "snapshot"),
    )
    assert _actions(directory=out) == _actions(directory=evaluation)
    assert _actions(directory=out)
    original = json.loads((evaluation / "results.json").read_text())
    again = json.loads((out / "results.json").read_text())
    assert [t | {"seed": None} for t in again["tasks"]] == [
        t | {"seed": None} for t in original["tasks"]
    ]
    assert again[PolicyFileHash.KEY] == original[PolicyFileHash.KEY]


def test_one_sampling_seed_gives_one_greedy_sequence_and_it_is_recorded(*, run: Path) -> None:
    _, snapshot = _measured(run=run)
    first = _evaluate(
        run=run, snapshot=snapshot, extra=("--num-test-tasks", "1", "--sampling-seed", "7")
    )
    second = _evaluate(
        run=run, snapshot=snapshot, extra=("--num-test-tasks", "1", "--sampling-seed", "7")
    )
    assert first != second
    assert _actions(directory=first) == _actions(directory=second)
    events = [
        json.loads(line)
        for line in (run / "evaluations-extra" / "events.jsonl").read_text().splitlines()
    ]
    started = [e for e in events if e["event"] == "extra_evaluation" and e["output"] == str(first)]
    assert len(started) == 1
    record = started[0]
    assert record["sampling_seed"] == 7
    assert record["sampling_seed_source"] == "given"
    assert record["snapshot"] == str(snapshot.resolve())
    assert record["num_test_tasks"] == 1
    assert record["reason"] == "a test re-run"
    assert len(record["git_commit"]) == 40
    assert (
        record["run_git_commit"]
        == json.loads((run / "config_snapshot.json").read_text())["git_commit"]
    )
    finished = [
        e for e in events if e["event"] == "extra_evaluation_done" and e["output"] == str(first)
    ]
    assert [e["num_total"] for e in finished] == [1]


def test_with_no_sampling_seed_a_fresh_one_is_drawn_and_written_down(*, run: Path) -> None:
    _, snapshot = _measured(run=run)
    out = _evaluate(run=run, snapshot=snapshot, extra=("--num-test-tasks", "1"))
    request = json.loads((out / "request.json").read_text())
    assert request["sampling_seed_source"] == "time"
    assert isinstance(request["sampling_seed"], int)


def test_output_goes_to_evaluations_extra_and_existing_evaluations_are_untouched(
    *, run: Path
) -> None:
    _, snapshot = _measured(run=run)
    before = _tree(root=run / "evaluations")
    snapshots = _tree(root=run / "snapshots")
    top = {p.name: p.read_bytes() for p in run.iterdir() if p.is_file()}
    one = _evaluate(
        run=run, snapshot=snapshot, extra=("--num-test-tasks", "1", "--sampling-seed", "1")
    )
    two = _evaluate(
        run=run, snapshot=snapshot, extra=("--num-test-tasks", "1", "--sampling-seed", "1")
    )
    assert one.parent == two.parent == run / "evaluations-extra"
    assert one.name.startswith(f"{snapshot.stem}-") and two.name.startswith(f"{snapshot.stem}-")
    assert one != two
    for out in (one, two):
        assert {"actions.jsonl", "progress.json", "recording.jsonl", "results.json"} <= {
            p.name for p in out.iterdir()
        }
    assert _tree(root=run / "evaluations") == before
    assert _tree(root=run / "snapshots") == snapshots
    assert {p.name: p.read_bytes() for p in run.iterdir() if p.is_file()} == top


@pytest.mark.parametrize(
    ("flags", "named"),
    [
        (("--robot", "real", "--robot-host", "127.0.0.1"), "robot"),
        (("--seed", "4"), "seed"),
        (("--evaluation-control-steps", "50"), "evaluation_control_steps"),
        (("--wall-clock-cost-per-second", "0.2"), "wall_clock_cost_per_second"),
    ],
)
def test_a_flag_that_disagrees_with_the_run_is_refused(
    *, run: Path, flags: tuple[str, ...], named: str
) -> None:
    _, snapshot = _measured(run=run)
    before = (
        sorted((run / "evaluations-extra").glob("*"))
        if (run / "evaluations-extra").exists()
        else []
    )
    with pytest.raises(ValueError, match=named):
        _evaluate(run=run, snapshot=snapshot, extra=("--num-test-tasks", "1", *flags))
    after = (
        sorted((run / "evaluations-extra").glob("*"))
        if (run / "evaluations-extra").exists()
        else []
    )
    assert after == before


def test_a_flag_that_agrees_with_the_run_is_accepted(*, run: Path) -> None:
    _, snapshot = _measured(run=run)
    out = _evaluate(
        run=run,
        snapshot=snapshot,
        extra=("--num-test-tasks", "1", "--sampling-seed", "3", "--robot", "fake", "--seed", "0"),
    )
    assert (out / "results.json").exists()


def test_a_snapshot_from_outside_the_run_is_refused(*, run: Path, tmp_path: Path) -> None:
    _, snapshot = _measured(run=run)
    elsewhere = tmp_path / snapshot.name
    elsewhere.write_bytes(snapshot.read_bytes())
    with pytest.raises(ValueError, match="snapshots"):
        _evaluate(run=run, snapshot=elsewhere, extra=("--num-test-tasks", "1"))


def test_n_tasks_run_each_with_its_own_scene_prompt_and_one_drive_home(
    *, run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, snapshot = _measured(run=run)
    robots: list[FakeTossingRobot] = []
    prompts: list[str] = []
    original = FakePerson.place

    def place(self: FakePerson, *, request: PlacementRequest) -> None:  # noqa: PLR0917 -- a method
        prompts.append(request.kind)
        original(self, request=request)

    def robot() -> FakeTossingRobot:
        robots.append(FakeTossingRobot())
        return robots[-1]

    monkeypatch.setattr(FakePerson, "place", place)
    monkeypatch.setattr(real_cli, "FakeTossingRobot", robot)
    out = SnapshotEvaluation.main(
        argv=[
            *("--run", str(run), "--snapshot", str(snapshot), "--reason", "three tasks"),
            *("--num-test-tasks", "3", "--sampling-seed", "11"),
        ]
    )
    results = json.loads((out / "results.json").read_text())
    assert results["num_total"] == 3
    assert [t["task"] for t in results["tasks"]] == [0, 1, 2]
    assert prompts.count("scene") == 3
    assert len(robots) == 1
    assert [name for name, _ in robots[0].calls].count("go_home") == 1


def test_a_real_robot_run_asks_before_anything_moves_and_a_no_starts_nothing(
    *, run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, snapshot = _measured(run=run)
    real = tmp_path / "real-run"
    (real / "snapshots").mkdir(parents=True)
    (real / "snapshots" / snapshot.name).write_bytes(snapshot.read_bytes())
    config = json.loads((run / "config_snapshot.json").read_text())
    config["args"] |= {"robot": "real", "robot_host": "192.0.2.1", "output_dir": str(real)}
    (real / "config_snapshot.json").write_text(json.dumps(config))
    asked: list[str] = []

    def answer(prompt: str) -> str:  # noqa: PLR0917 -- `input`'s signature
        asked.append(prompt)
        return "n"

    def no_robot(**kwargs: Any) -> None:
        raise AssertionError(f"connected to a robot: {kwargs}")

    monkeypatch.setattr("builtins.input", answer)
    monkeypatch.setattr(real_cli, "SkillServerClient", no_robot)
    with pytest.raises(SystemExit, match="not started"):
        SnapshotEvaluation.main(
            argv=[
                *("--run", str(real), "--snapshot", str(real / "snapshots" / snapshot.name)),
                *("--num-test-tasks", "1", "--reason", "declined", "--robot", "real"),
            ]
        )
    assert len(asked) == 1 and "drives home" in asked[0]
    assert not (real / "evaluations-extra").exists()
