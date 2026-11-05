"""A checkpoint's test tasks run in parallel: `--evaluation-workers` and
`--evaluation-task-rng`.

Today's evaluation (`--evaluation-task-rng shared`) draws every task's parameters from one
deployment RNG carried from task to task, so a task's outcome depends on the tasks
before it. `per-task` reseeds every RNG and clears every carried state from (snapshot,
task) at the start of each task, which is what lets the tasks run in separate processes
and still produce exactly what one process running them in order produces."""

import json
import os
import time
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import pytest

from hitl_pmp.step_protocol import (
    EvaluationTaskFailures,
    FreshProcessEvaluations,
    StepEvaluation,
    StepPracticeRunner,
    StepProtocolResume,
    StepProtocolValidation,
    TaskParallelEvaluation,
)
from tests.practice_cost_fixtures import CostFixtures
from tests.step_evaluation_fakes import LightSwitchEvaluationProtocol
from tests.test_step_protocol import FakeProblem, FakeStepProtocol, _scripted_method, _step_args

NUM_TASKS = 4


def _configuration(**extra: Any) -> dict[str, Any]:
    return {"seed": 0, "num_test_tasks": NUM_TASKS, "evaluation_control_steps": 12, **extra}


def _evaluate(
    *, tmp_path: Path, name: str, task_indices: list[int] | None = None, **extra: Any
) -> Path:
    snapshot = LightSwitchEvaluationProtocol.write_snapshot(path=tmp_path / "snapshot.pickle")
    output = tmp_path / name
    StepEvaluation.run(
        snapshot=str(snapshot),
        configuration=_configuration(**extra),
        output=str(output),
        environment=LightSwitchEvaluationProtocol,
        task_indices=task_indices,
    )
    return output


def _actions(*, directory: Path, task: int | None = None) -> list[dict[str, Any]]:
    """`actions.jsonl` without its wall-clock fields."""
    lines = (directory / "actions.jsonl").read_text().splitlines()
    actions = [json.loads(line) for line in lines]
    for action in actions:
        del action["timestamp"], action["elapsed_seconds"]
    return [a for a in actions if task is None or a["task"] == task]


def _untimed_lines(*, path: Path) -> list[dict[str, Any]]:
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    for entry in entries:
        entry.pop("timestamp", None)
        entry.pop("elapsed_seconds", None)
    return entries


def _untimed_results(*, directory: Path) -> str:
    result = json.loads((directory / "results.json").read_text())
    del result["elapsed_seconds"]
    return json.dumps(result, indent=2)


# --- shared: today's evaluation, unchanged ----------------------------------------------


def test_the_shared_default_evaluates_exactly_as_before(*, tmp_path):
    """Pinned against the code before `--evaluation-task-rng` existed: one RNG carried
    through the tasks in order, and environment state carried with it."""
    output = _evaluate(tmp_path=tmp_path, name="shared")
    result = json.loads((output / "results.json").read_text())
    assert [(t["solved"], t["steps"], t["reason"]) for t in result["tasks"]] == [
        (False, 12, "step_budget"),
        (True, 6, "goal"),
        (False, 12, "step_budget"),
        (True, 3, "goal"),
    ]
    toggles = [a["label"] for a in _actions(directory=output) if "params" in a["label"]]
    assert toggles[:3] == [
        "TurnOnLight(robot, cell2, light), params=[-0.41]",
        "TurnOnLight(robot, cell2, light), params=[-0.22]",
        "TurnOnLight(robot, cell2, light), params=[-0.2]",
    ]
    assert [a["skill_trace"]["carried"] for a in _actions(directory=output, task=3)] == [
        31,
        32,
        33,
    ]


def test_under_shared_a_task_depends_on_the_tasks_before_it(*, tmp_path):
    """Why `per-task` exists: task 2 alone is not task 2 after tasks 0 and 1."""
    in_order = _evaluate(tmp_path=tmp_path, name="in_order")
    alone = _evaluate(tmp_path=tmp_path, name="alone", task_indices=[2])
    assert _actions(directory=alone, task=2) != _actions(directory=in_order, task=2)


# --- per-task: a task's outcome is a function of (snapshot, task) -----------------------


def test_under_per_task_a_task_alone_is_the_task_in_order(*, tmp_path):
    in_order = _evaluate(tmp_path=tmp_path, name="in_order", evaluation_task_rng="per-task")
    for task in range(NUM_TASKS):
        alone = _evaluate(
            tmp_path=tmp_path,
            name=f"alone{task}",
            task_indices=[task],
            evaluation_task_rng="per-task",
        )
        assert _actions(directory=alone) == _actions(directory=in_order, task=task)
        assert json.loads((alone / "results.json").read_text())["tasks"] == [
            json.loads((in_order / "results.json").read_text())["tasks"][task]
        ]


def test_per_task_reseeds_the_global_rngs_and_clears_carried_environment_state(*, tmp_path):
    output = _evaluate(tmp_path=tmp_path, name="per_task", evaluation_task_rng="per-task")
    firsts = [_actions(directory=output, task=task)[0]["skill_trace"] for task in range(4)]
    assert [first["carried"] for first in firsts] == [1, 1, 1, 1]
    # Every task's first draws are its own, not the continuation of the last task's.
    assert len({(f["numpy"], f["random"], f["torch"]) for f in firsts}) == NUM_TASKS
    again = _evaluate(tmp_path=tmp_path, name="again", evaluation_task_rng="per-task")
    assert _actions(directory=again) == _actions(directory=output)


def test_per_task_seeds_depend_on_the_snapshot(*, tmp_path):
    one = _evaluate(tmp_path=tmp_path, name="one", evaluation_task_rng="per-task")
    LightSwitchEvaluationProtocol.write_snapshot(path=tmp_path / "other.pickle", seed=4)
    other = tmp_path / "other"
    StepEvaluation.run(
        snapshot=str(tmp_path / "other.pickle"),
        configuration=_configuration(evaluation_task_rng="per-task"),
        output=str(other),
        environment=LightSwitchEvaluationProtocol,
    )
    assert _actions(directory=one) != _actions(directory=other)


# --- parallel == sequential -------------------------------------------------------------


def _parallel(*, tmp_path: Path, name: str, workers: int, **extra: Any) -> tuple[Path, Future]:
    snapshot = LightSwitchEvaluationProtocol.write_snapshot(path=tmp_path / "snapshot.pickle")
    output = tmp_path / name
    output.mkdir()
    pool = FreshProcessEvaluations(workers=workers, timeout_seconds=extra.pop("timeout", None))
    future = TaskParallelEvaluation.submit(
        pool=pool,
        snapshot=str(snapshot),
        configuration=_configuration(evaluation_task_rng="per-task", **extra),
        output=str(output),
        environment=LightSwitchEvaluationProtocol,
        checkpoint=0,
        label="(final)",
    )
    pool.finish()
    return output, future


def test_parallel_tasks_reproduce_per_task_sequential_evaluation_byte_for_byte(*, tmp_path):
    # The sequential reference also runs in a fresh process, as a real evaluation does.
    snapshot = LightSwitchEvaluationProtocol.write_snapshot(path=tmp_path / "snapshot.pickle")
    sequential = tmp_path / "sequential"
    pool = FreshProcessEvaluations()
    reference = pool.submit(
        fn=StepEvaluation.run,
        snapshot=str(snapshot),
        configuration=_configuration(evaluation_task_rng="per-task"),
        output=str(sequential),
        environment=LightSwitchEvaluationProtocol,
        checkpoint=0,
        label="(final)",
    )
    pool.finish()
    parallel, future = _parallel(tmp_path=tmp_path, name="parallel", workers=3)

    expected, actual = reference.result(), future.result()
    del expected["elapsed_seconds"], actual["elapsed_seconds"]
    assert actual == expected
    assert _untimed_results(directory=parallel) == _untimed_results(directory=sequential)
    assert (parallel / "progress.json").read_bytes() == (sequential / "progress.json").read_bytes()
    assert _actions(directory=parallel) == _actions(directory=sequential)
    assert _untimed_lines(path=parallel / "recording.jsonl") == _untimed_lines(
        path=sequential / "recording.jsonl"
    )
    assert sorted(p.name for p in (parallel / "tasks").iterdir()) == ["00", "01", "02", "03"]


def test_a_failing_task_fails_the_evaluation_by_name_and_the_others_still_finish(*, tmp_path):
    output, future = _parallel(tmp_path=tmp_path, name="failing", workers=2, fake_fail_task=1)
    with pytest.raises(EvaluationTaskFailures, match=r"task 1 .*scripted failure in task 1"):
        future.result(timeout=120)
    for task in (0, 2, 3):
        assert (output / "tasks" / f"{task:02d}" / "results.json").exists()
    assert (output / "tasks" / "01" / "failure.txt").exists()
    assert "task 1" in (output / "failure.txt").read_text()
    assert not (output / "results.json").exists()


def test_a_task_past_its_timeout_is_terminated_and_named(*, tmp_path):
    started = time.monotonic()
    output, future = _parallel(
        tmp_path=tmp_path, name="slow", workers=2, fake_sleep_task=0, timeout=20
    )
    with pytest.raises(EvaluationTaskFailures, match=r"task 0 .*timed out after 20"):
        future.result(timeout=120)
    assert time.monotonic() - started < 120
    assert (output / "tasks" / "01" / "results.json").exists()


def _stamp(*, delay: float) -> tuple[int, float, float]:
    started = time.monotonic()
    time.sleep(delay)
    return os.getpid(), started, time.monotonic()


def test_parts_run_in_their_own_processes_at_most_workers_at_once():
    pool = FreshProcessEvaluations(workers=2)
    future = pool.submit_parts(
        fn=_stamp, parts=[{"delay": 3.0}] * 4, combine=lambda *, results, elapsed_seconds: results
    )
    pool.finish()
    stamps = future.result()
    assert len({pid for pid, _, _ in stamps}) == 4
    assert os.getpid() not in {pid for pid, _, _ in stamps}
    for _, started, _ in stamps:
        running = sum(s <= started < e for _, s, e in stamps)
        assert 1 <= running <= 2
    assert max(sum(s <= t < e for _, s, e in stamps) for _, t, _ in stamps) == 2


# --- configuration ----------------------------------------------------------------------


def _parse(*extra: str):
    return CostFixtures.parse(
        argv=["--env", "pouring3d_kitchen", "--method", "ees", "--output-dir", "/x", *extra]
    )


def test_the_flags_default_to_todays_evaluation():
    args = _parse()
    assert args.evaluation_workers == 1
    assert args.evaluation_task_rng == "shared"
    assert args.evaluation_task_timeout_seconds is None


@pytest.mark.parametrize("value", ["0", "-1"])
def test_evaluation_workers_must_be_positive(*, value):
    with pytest.raises(SystemExit):
        _parse("--evaluation-workers", value)


def test_parallel_workers_need_per_task_rngs(*, tmp_path):
    args = _step_args(output_dir=tmp_path / "run", budget=5, interval=5, evaluation_workers=3)
    with pytest.raises(
        ValueError, match="--evaluation-workers 3 needs --evaluation-task-rng per-task"
    ):
        StepProtocolValidation.check(args=args)
    args.evaluation_task_rng = "per-task"
    StepProtocolValidation.check(args=args)


def test_a_run_recorded_before_these_flags_existed_still_resumes():
    before = {"seed": "0", "env": "tossing3d_real"}
    now = {
        **before,
        "evaluation_workers": "1",
        "evaluation_task_rng": "shared",
        "evaluation_task_timeout_seconds": "None",
    }
    assert StepProtocolResume.comparable(arguments=before) == StepProtocolResume.comparable(
        arguments=now
    )


class _PartsPool:
    """Records what the runner submits; every evaluation completes at once."""

    def __init__(self) -> None:
        self.parts: list[list[dict[str, Any]]] = []

    def submit(self, **kwargs: Any) -> Future:
        raise AssertionError("parallel evaluation must not submit one whole evaluation")

    def submit_parts(self, *, fn, parts, combine, on_failure=None) -> Future:
        self.parts.append(parts)
        future: Future = Future()
        future.set_result({
            "complete": True,
            "tasks": [{"task": i, "goal": "g", "solved": True} for i in range(len(parts))],
            "num_solved": len(parts),
            "num_total": len(parts),
        })
        return future

    def finish(self) -> None:
        pass

    def abort(self) -> None:
        pass


def test_the_runner_submits_one_part_per_test_task(*, tmp_path):
    output = tmp_path / "run"
    pool = _PartsPool()
    StepPracticeRunner.run(
        environment=FakeStepProtocol,
        args=_step_args(
            output_dir=output,
            budget=3,
            interval=100,
            num_test_tasks=3,
            evaluation_workers=3,
            evaluation_task_rng="per-task",
        ),
        method=_scripted_method(script=[]),
        problem=FakeProblem(),
        evaluations=pool,  # type: ignore[arg-type]
    )
    assert pool.parts, "no evaluation was submitted"
    for parts in pool.parts:
        assert [part["task_indices"] for part in parts] == [[0], [1], [2]]
        index = Path(parts[0]["output"]).parent.parent.name
        assert [Path(part["output"]).relative_to(output) for part in parts] == [
            Path("evaluations") / index / "tasks" / f"{task:02d}" for task in range(3)
        ]
    protocol = json.loads((output / "step_protocol.json").read_text())
    assert protocol["evaluation_workers"] == 3
    assert protocol["evaluation_task_rng"] == "per-task"


def test_the_default_step_protocol_record_is_unchanged(*, tmp_path):
    output = tmp_path / "run"
    from tests.test_step_protocol import FakeEvaluations

    StepPracticeRunner.run(
        environment=FakeStepProtocol,
        args=_step_args(output_dir=output, budget=3, interval=100),
        method=_scripted_method(script=[]),
        problem=FakeProblem(),
        evaluations=FakeEvaluations(),
    )
    protocol = json.loads((output / "step_protocol.json").read_text())
    assert protocol["evaluation_process"] == "spawn, one fresh process per evaluation"
    assert not {"evaluation_workers", "evaluation_task_rng"} & protocol.keys()
