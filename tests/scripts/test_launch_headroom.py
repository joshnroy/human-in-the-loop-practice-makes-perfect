"""`scripts/launch_headroom.py`: does the untrained policy leave practice anything to do?"""

import json
from pathlib import Path

import pytest

from scripts.launch_headroom import LaunchHeadroom

SWEEP = "SweepShelf(robot, wiper_0, box_0, cube_0, cube_1, cube_2, cube_3, cube_4)"
PICK = "PickWiper(robot, wiper_0)"


def _action(
    *, task: int, label: str, start: int, end: int, success: bool, goal: bool, outcome: str | None
) -> dict:
    return {
        "kind": "action",
        "phase": "evaluation",
        "checkpoint": 0,
        "task_index": task,
        "actor": "robot",
        "skill": label.split("(")[0],
        "objects": [],
        "params": [],
        "label": label,
        "start_step": start,
        "end_step": end,
        "success": success,
        "goal_reached": goal,
        "outcome": outcome,
    }


def _sweep(*, params: list[float]) -> str:
    return f"{SWEEP}, params={params}"


def _episode(*, task: int, calls: list[tuple[str, str]]) -> list[dict]:
    """`calls` is (label, result), result one of success / failure / goal / budget."""
    records: list[dict] = [
        {"kind": "segment", "phase": "evaluation", "checkpoint": 0, "task_index": task,
         "task_seed": 1000 + task},
    ]  # fmt: skip
    step = 0
    for label, result in calls:
        records.append(
            _action(
                task=task,
                label=label,
                start=step,
                end=step + 100,
                success=result == "success",
                goal=result == "goal",
                outcome={
                    "success": None,
                    "failure": "sweep_shelf: 5 cube(s) still on the shelf",
                    "goal": "stopped: goal reached",
                    "budget": "interrupted: evaluation step budget",
                }[result],
            )
        )
        step += 100
    solved = bool(calls) and calls[-1][1] == "goal"
    records.append({
        "kind": "segment_end",
        "phase": "evaluation",
        "checkpoint": 0,
        "task_index": task,
        "success": solved,
        "outcome": "goal" if solved else "step_budget",
        "steps": step,
    })
    return records


def write_evaluation(*, path: Path, episodes: list[list[tuple[str, str]]]) -> Path:
    path.mkdir(parents=True)
    records: list[dict] = [{"kind": "header", "schema_version": 1}]
    for task, calls in enumerate(episodes):
        records.extend(_episode(task=task, calls=calls))
    (path / "recording.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def _one_call_macro(*, params: list[float]) -> list[tuple[str, str]]:
    return [(PICK, "success"), (_sweep(params=params), "goal")]


# Launch shelf-20261023b, seed 0, step 0, as its recording logged it: 8/10 solved, seven
# of them by the first SweepShelf call.
SHELF_20261023B = [
    _one_call_macro(params=[0.01, 0.25]),
    _one_call_macro(params=[0.01, 0.18]),
    [(PICK, "success"), (_sweep(params=[0.04, 0.1]), "failure"),
     (_sweep(params=[0.03, 0.08]), "failure"), (_sweep(params=[0.01, 0.18]), "goal")],
    _one_call_macro(params=[0.0, 0.1]),
    _one_call_macro(params=[0.0, 0.15]),
    _one_call_macro(params=[0.02, 0.15]),
    _one_call_macro(params=[0.01, 0.07]),
    _one_call_macro(params=[0.0, 0.1]),
    [(PICK, "success"), (_sweep(params=[0.03, 0.15]), "budget")],
    [(PICK, "success"), (_sweep(params=[0.03, 0.06]), "failure"),
     (_sweep(params=[0.03, 0.08]), "budget")],
]  # fmt: skip


def test_the_shelf_20261023b_step0_evaluation_is_counted_from_its_recording(
    *, tmp_path: Path
) -> None:
    headroom = LaunchHeadroom.analyse(
        evaluation=write_evaluation(path=tmp_path / "0000", episodes=SHELF_20261023B)
    )
    assert (headroom.solved, headroom.total) == (8, 10)
    assert headroom.task_seeds == list(range(1000, 1010))
    assert headroom.calls_to_goal == [2, 2, 4, 2, 2, 2, 2, 2, None, None]
    assert headroom.continuous_calls_to_goal == [1, 1, 3, 1, 1, 1, 1, 1, None, None]
    sweep = headroom.skills["SweepShelf"]
    # A call the goal latch stopped counts as a success; one the budget cut off counts
    # apart, as neither.
    assert (sweep.calls, sweep.successes, sweep.goal_reached) == (13, 8, 8)
    assert (sweep.failures, sweep.interrupted) == (3, 2)
    assert headroom.skills["PickWiper"].successes == headroom.skills["PickWiper"].calls == 10


def test_the_shelf_20261023b_step0_evaluation_fails_both_headroom_checks(*, tmp_path: Path) -> None:
    headroom = LaunchHeadroom.analyse(
        evaluation=write_evaluation(path=tmp_path / "0000", episodes=SHELF_20261023B)
    )
    problems = LaunchHeadroom.problems(headroom=headroom, max_solved=7)
    assert len(problems) == 2
    assert "8/10 held-out tasks" in problems[0] and "at most 7/10" in problems[0]
    assert "7/10 episodes" in problems[1] and "one continuous-parameter" in problems[1]


def test_a_policy_that_needs_practice_passes(*, tmp_path: Path) -> None:
    failing = [(PICK, "success"), (_sweep(params=[0.0, 0.1]), "failure"),
               (_sweep(params=[0.0, 0.1]), "budget")]  # fmt: skip
    multi_step = [(PICK, "success"), (_sweep(params=[0.0, 0.1]), "success"),
                  (_sweep(params=[0.0, 0.2]), "goal")]  # fmt: skip
    headroom = LaunchHeadroom.analyse(
        evaluation=write_evaluation(
            path=tmp_path / "0000", episodes=[*[failing] * 6, *[multi_step] * 4]
        )
    )
    assert (headroom.solved, headroom.total) == (4, 10)
    assert LaunchHeadroom.problems(headroom=headroom, max_solved=7) == []


@pytest.mark.parametrize(("max_solved", "blocked"), [(7, False), (6, True)])
def test_the_solved_threshold_is_configurable(
    *, tmp_path: Path, max_solved: int, blocked: bool
) -> None:
    two_calls = [(PICK, "success"), (_sweep(params=[0.0, 0.1]), "failure"),
                 (_sweep(params=[0.0, 0.1]), "goal")]  # fmt: skip
    unsolved = [(PICK, "success"), (_sweep(params=[0.0, 0.1]), "budget")]
    headroom = LaunchHeadroom.analyse(
        evaluation=write_evaluation(
            path=tmp_path / "0000", episodes=[*[two_calls] * 7, *[unsolved] * 3]
        )
    )
    problems = LaunchHeadroom.problems(headroom=headroom, max_solved=max_solved)
    assert bool(problems) is blocked


def test_half_the_episodes_in_one_call_is_not_yet_the_median(*, tmp_path: Path) -> None:
    unsolved = [(PICK, "success"), (_sweep(params=[0.0, 0.1]), "budget")]
    episodes = [*[_one_call_macro(params=[0.0, 0.1])] * 5, *[unsolved] * 5]
    headroom = LaunchHeadroom.analyse(
        evaluation=write_evaluation(path=tmp_path / "a", episodes=episodes)
    )
    assert LaunchHeadroom.problems(headroom=headroom, max_solved=7) == []
    episodes = [*[_one_call_macro(params=[0.0, 0.1])] * 6, *[unsolved] * 4]
    headroom = LaunchHeadroom.analyse(
        evaluation=write_evaluation(path=tmp_path / "b", episodes=episodes)
    )
    assert len(LaunchHeadroom.problems(headroom=headroom, max_solved=7)) == 1


def test_success_is_binned_per_parameter_dimension(*, tmp_path: Path) -> None:
    headroom = LaunchHeadroom.analyse(
        evaluation=write_evaluation(path=tmp_path / "0000", episodes=SHELF_20261023B)
    )
    bins = headroom.parameter_bins["SweepShelf"]
    height = [b for b in bins if b.dimension == 0]
    assert [(b.low, b.high) for b in height] == pytest.approx([
        (0.0, 4 / 300),
        (4 / 300, 8 / 300),
        (8 / 300, 0.04),
    ])
    # Interrupted calls are in no bin: they have no verdict.
    assert sum(b.total for b in height) == 11
    assert [f"{b.successes}/{b.total}" for b in height] == ["7/7", "1/1", "0/3"]
    assert "PickWiper" not in headroom.parameter_bins


def test_the_rendering_reports_counts_never_bare_percentages(*, tmp_path: Path) -> None:
    headroom = LaunchHeadroom.analyse(
        evaluation=write_evaluation(path=tmp_path / "0000", episodes=SHELF_20261023B)
    )
    text = LaunchHeadroom.render(arm="ees", headroom=headroom, max_solved=7)
    assert "ees: step-0 policy solved 8/10" in text
    assert "SweepShelf" in text and "8/11" in text
    assert "%" not in text
    assert "NO HEADROOM" in text


def test_a_missing_recording_is_an_error_not_a_pass(*, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        LaunchHeadroom.analyse(evaluation=tmp_path)


def test_the_cli_exits_nonzero_on_no_headroom(
    *, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    evaluation = write_evaluation(path=tmp_path / "0000", episodes=SHELF_20261023B)
    assert LaunchHeadroom.main(argv=[str(evaluation)]) == 1
    assert "8/10" in capsys.readouterr().out
    assert LaunchHeadroom.main(argv=[str(evaluation), "--max-solved", "9"]) == 1
    assert LaunchHeadroom.main(argv=[str(evaluation), "--max-solved", "9", "--report-only"]) == 0
