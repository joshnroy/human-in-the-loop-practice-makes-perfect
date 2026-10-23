"""Per-skill wall-clock timers: `SkillTiming`, the backend phases it splits a skill or a
reset into, and `skill_timing.jsonl` beside a step-protocol run.

The timers must not change a run's results, so the end-to-end test compares
`stats.json` byte for byte between a timed and an untimed run of the same seed.
"""

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pytest

from hitl_pmp.environments.tossing3d.skill_timing import SkillTiming

_NEEDS_KINDER = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None, reason="KINDER simulator dependency"
)


def test_a_phase_excludes_the_phases_timed_inside_it() -> None:
    timing = SkillTiming(kind="skill")
    with timing.phase(name="outer"):
        time.sleep(0.02)
        with timing.phase(name="inner"):
            time.sleep(0.03)
    timing.finish()
    assert timing.counts == {"outer": 1, "inner": 1}
    assert 0.015 < timing.seconds["outer"] < 0.03
    assert timing.seconds["inner"] >= 0.03
    assert timing.total_seconds >= sum(timing.seconds.values())


def test_a_repeated_phase_accumulates_seconds_and_calls() -> None:
    timing = SkillTiming(kind="skill")
    for _ in range(3):
        with timing.phase(name="mujoco_step"):
            pass
    timing.count(name="pick_plan_cache_hit")
    timing.finish()
    assert timing.counts == {"mujoco_step": 3, "pick_plan_cache_hit": 1}
    assert "pick_plan_cache_hit" not in timing.seconds


def test_a_phase_that_raises_is_still_recorded() -> None:
    timing = SkillTiming(kind="skill")
    with pytest.raises(RuntimeError), timing.phase(name="controller_reset"):
        raise RuntimeError("planning failed")
    assert timing.counts == {"controller_reset": 1}


def test_without_a_timing_the_phase_helper_is_a_no_op() -> None:
    with SkillTiming.phase_of(timing=None, name="mujoco_step"):
        pass


def test_the_record_carries_the_untimed_remainder() -> None:
    timing = SkillTiming(kind="movables_reset")
    with timing.phase(name="reset_cube_and_bin"):
        pass
    timing.finish()
    record = timing.record()
    assert set(record) == {"kind", "total_seconds", "untimed_seconds", "seconds", "counts"}
    assert record["kind"] == "movables_reset"
    assert record["untimed_seconds"] == pytest.approx(
        record["total_seconds"] - sum(record["seconds"].values())
    )


def test_the_environment_keeps_no_timing_when_disabled(*, no_kinder_import) -> None:
    from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment

    env = Tossing3DEnvironment(skill_timing=False)
    assert env.drain_execution_timing() is None


@_NEEDS_KINDER
def test_a_pick_and_a_reset_are_split_into_their_phases() -> None:
    from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment

    env = Tossing3DEnvironment(scene_bg=False)
    try:
        env.reset_to_seed(seed=0)
        assert env.drain_execution_timing()["kind"] == "scene_reset"
        assert env.drain_execution_timing() is None
        env.take_action(action=np.array([float(env.pick_cube_id), 0, 0, 0, 0]))
        pick = env.drain_execution_timing()
        ticks = sum(env.last_controller_steps())
        assert ticks > 0
        assert pick["kind"] == "skill"
        assert pick["counts"]["mujoco_step"] == ticks
        assert pick["counts"]["controller_reset"] == 1
        assert pick["counts"]["pick_collision_sim"] == 1
        assert pick["counts"]["controller_tick"] == ticks
        assert pick["counts"]["abstract_atoms"] >= 1
        for phase in ("controller_reset", "mujoco_step", "abstract_atoms"):
            assert pick["seconds"][phase] > 0
        assert pick["untimed_seconds"] >= 0
        assert env.reset_movables(destination="robot_side")
        reset = env.drain_execution_timing()
        assert reset["kind"] == "movables_reset"
        assert reset["counts"]["reset_cube_and_bin"] == 1
        assert reset["counts"]["abstract_atoms"] == 1
        # A robot-side reset leaves the cube on the robot's side: the dry run is made
        # (or answered from the cache), never skipped by the barrier gate.
        dry_runs = reset["counts"].get("pick_plan_dry_run", 0)
        assert dry_runs + reset["counts"].get("pick_plan_cache_hit", 0) == 1
    finally:
        env.close()


def _run(*, output: Path, timed: bool) -> None:
    from hitl_pmp.cli import Cli

    Cli.main(
        argv=[
            *("--env", "tossing3d", "--method", "pomdp", "--preset", "tossing3d-canonical"),
            *("--pomdp-solver", "determinized_astar", "--seed", "0", "--no-scene-bg"),
            *("--practice-step-budget", "400", "--num-test-tasks", "1"),
            *("--evaluation-control-steps", "5", "--no-initial-measurement"),
            "--skill-timing" if timed else "--no-skill-timing",
            *("--output-dir", str(output)),
        ]
    )


@_NEEDS_KINDER
def test_timers_leave_stats_byte_identical_and_write_one_record_per_execution(
    *, tmp_path: Path
) -> None:
    timed, untimed = tmp_path / "timed", tmp_path / "untimed"
    _run(output=timed, timed=True)
    _run(output=untimed, timed=False)

    assert (timed / "stats.json").read_bytes() == (untimed / "stats.json").read_bytes()
    assert not (untimed / "skill_timing.jsonl").exists()

    events = [json.loads(line) for line in (timed / "step_events.jsonl").read_text().splitlines()]
    assert all("timing" not in event and "seconds" not in event for event in events)
    executions = [event for event in events if event["event"] in ("robot", "human")]
    records = [json.loads(line) for line in (timed / "skill_timing.jsonl").read_text().splitlines()]
    assert executions, "the run made no robot skill or human reset to time"
    assert [(r["event"], r["start_step"], r["end_step"]) for r in records] == [
        (e["event"], e["start_step"], e["end_step"]) for e in executions
    ]
    for record, event in zip(records, executions, strict=True):
        assert record["cycle"] == event["cycle"]
        assert record["label"] == (event.get("label") or event["skill"])
        assert record["kind"] == ("skill" if event["event"] == "robot" else "movables_reset")
        assert record["total_seconds"] >= sum(record["seconds"].values())
        if event["event"] == "robot" and event["end_step"] > event["start_step"]:
            assert record["counts"]["mujoco_step"] == event["end_step"] - event["start_step"]
            assert record["counts"]["controller_reset"] == 1
        if event["event"] == "human":
            assert record["counts"]["reset_cube_and_bin"] == 1
