"""`scripts/skill_health.py`: success/total per robot skill from any run's logs."""

import json
from pathlib import Path

from scripts.skill_health import SkillHealth


def _robot(*, label: str, start: int, end: int, error: str | None, **extra: object) -> dict:
    return {
        "event": "robot",
        "label": label,
        "start_step": start,
        "end_step": end,
        "interrupted": False,
        "skill_error": error,
        **extra,
    }


def _write(*, path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def _trace(*, outcome: str, steps: int) -> dict:
    return {"outcome": outcome, "controller_steps": steps, "notes": []}


def test_counts_split_by_who_chose_the_skill_and_how_it_failed(*, tmp_path: Path) -> None:
    _write(
        path=tmp_path / "run" / "pomdp" / "0" / "step_events.jsonl",
        records=[
            {"event": "measurement", "index": 0},
            _robot(label="PickWiper(robot, wiper_0)", start=0, end=743, error=None),
            _robot(label="random: PickWiper(robot, wiper_0)", start=743, end=743, error="no plan"),
            _robot(
                label="PickWiper(robot, wiper_0)", start=743, end=900, error="did not come along"
            ),
            _robot(
                label="SweepShelf(robot, wiper_0, box_0, cube_0), params=[-0.06, 0.25]",
                start=900,
                end=3000,
                error=None,
            ),
            {
                "event": "robot",
                "label": "ReturnWiper(robot, wiper_0)",
                "start_step": 3000,
                "end_step": 3100,
                "interrupted": True,
                "skill_error": "interrupted: practice_step_budget",
            },
        ],
    )
    report = SkillHealth.collect(paths=[tmp_path])
    wiper = report.skills["PickWiper"]
    assert (wiper.planner.successes, wiper.planner.total) == (1, 2)
    assert (wiper.random.successes, wiper.random.total) == (0, 1)
    assert (wiper.plan_failures, wiper.execution_failures) == (1, 1)
    assert report.skills["ReturnWiper"].interrupted == 1
    assert report.skills["ReturnWiper"].total == 0


def test_a_logged_trace_decides_success_over_the_bare_error(*, tmp_path: Path) -> None:
    _write(
        path=tmp_path / "step_events.jsonl",
        records=[
            _robot(
                label="PlaceCubeInBox(robot, cube_1, box_0)",
                start=0,
                end=500,
                error=None,
                skill_trace=_trace(outcome="effects_missing", steps=500),
            ),
            _robot(
                label="PlaceCubeInBox(robot, cube_1, box_0)",
                start=500,
                end=900,
                error=None,
                skill_trace=_trace(outcome="success", steps=400),
            ),
        ],
    )
    skill = SkillHealth.collect(paths=[tmp_path / "step_events.jsonl"]).skills["PlaceCubeInBox"]
    assert (skill.successes, skill.total) == (1, 2)
    assert skill.execution_failures == 1
    assert skill.traced == 2


def test_skills_below_their_benchmark_target_are_flagged(*, tmp_path: Path) -> None:
    _write(
        path=tmp_path / "step_events.jsonl",
        records=[
            _robot(label="PickWiper(robot, wiper_0)", start=0, end=10, error=None),
            _robot(label="PickCubeFromShelf(robot, cube_0)", start=10, end=10, error="no grasp"),
            _robot(label="PickCubeFromShelf(robot, cube_1)", start=10, end=20, error=None),
            _robot(
                label="SweepShelf(robot, wiper_0, box_0, cube_1), params=[0.03, 0.2]",
                start=20,
                end=30,
                error="x",
            ),
            _robot(
                label="SweepShelf(robot, wiper_0, box_0, cube_1), params=[0.0, 0.05]",
                start=30,
                end=40,
                error=None,
            ),
        ],
    )
    report = SkillHealth.collect(paths=[tmp_path])
    flagged = SkillHealth.flagged(report=report)
    # Without continuous parameters the target is every attempt; SweepShelf's is that
    # some setting works per state and target cube, which a run's random draws cannot
    # judge -- it is
    # flagged only when no attempt ever succeeded.
    assert flagged == ["PickCubeFromShelf"]
    text = SkillHealth.render(report=report)
    assert "PickCubeFromShelf" in text and "1/2" in text
    assert "%" not in text


def test_evaluation_actions_are_counted_on_their_own(*, tmp_path: Path) -> None:
    _write(
        path=tmp_path / "evaluations" / "0000" / "actions.jsonl",
        records=[
            {"task": 0, "label": "PickWiper(robot, wiper_0)", "start_step": 0, "end_step": 700},
            {
                "task": 0,
                "label": "SweepShelf(robot, wiper_0, box_0, cube_0), params=[-0.06, 0.05]",
                "start_step": 700,
                "end_step": 700,
                "skill_error": "sweep_shelf: no route",
            },
        ],
    )
    report = SkillHealth.collect(paths=[tmp_path])
    assert report.skills["PickWiper"].evaluation.total == 1
    sweep = report.skills["SweepShelf"]
    assert (sweep.evaluation.successes, sweep.evaluation.total) == (0, 1)
    assert sweep.plan_failures == 1


def test_the_command_line_exits_nonzero_only_when_asked_to_and_a_skill_is_flagged(
    *, tmp_path: Path
) -> None:
    _write(
        path=tmp_path / "step_events.jsonl",
        records=[_robot(label="PickWiper(robot, wiper_0)", start=0, end=0, error="no plan")],
    )
    assert SkillHealth.main(argv=[str(tmp_path)]) == 0
    assert SkillHealth.main(argv=[str(tmp_path), "--fail-on-flag"]) == 1
    out = tmp_path / "health.json"
    SkillHealth.main(argv=[str(tmp_path), "--json", str(out)])
    assert json.loads(out.read_text())["flagged"] == ["PickWiper"]
