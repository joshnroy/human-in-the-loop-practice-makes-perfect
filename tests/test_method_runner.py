import argparse
from pathlib import Path

import imageio
import pytest

from hitl_pmp.config_snapshot import ConfigSnapshot
from hitl_pmp.core.method.types import Policy, SkillPracticeTally
from hitl_pmp.core.metrics.metrics import Metrics
from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.problem import LightSwitchProblem
from hitl_pmp.environments.lightswitch.renderer import LightSwitchRenderer
from hitl_pmp.environments.lightswitch.skill_provider import LightSwitchOracle
from hitl_pmp.environments.lightswitch.tasks import LightSwitchTasks
from hitl_pmp.method_runner import MethodRunner
from hitl_pmp.methods.oracle.skill_oracle_method import SkillOracleMethod
from hitl_pmp.practice_loop import PracticeResetPolicy


def _args(*, num_test_tasks: int = 5, output_dir: Path | None = None) -> argparse.Namespace:
    return argparse.Namespace(num_test_tasks=num_test_tasks, output_dir=output_dir)


def _build_problem() -> LightSwitchProblem:
    env = LightSwitchEnvironment()
    return LightSwitchProblem(env=env, tasks=LightSwitchTasks(env=env))


def test_run_prints_success_rate(*, capsys: pytest.CaptureFixture[str]) -> None:
    problem = _build_problem()
    MethodRunner.run(
        args=_args(num_test_tasks=5),
        method=SkillOracleMethod(env=problem.env, oracle=LightSwitchOracle(env=problem.env)),
        problem=problem,
        num_cycles=0,
        max_steps_per_interaction=0,
        renderer=None,
        render_fps=2,
    )
    assert "success rate: 5/5 (100%)" in capsys.readouterr().out


def test_run_records_one_evaluation_per_cycle_plus_the_initial_one() -> None:
    problem = _build_problem()
    metrics = MethodRunner.run(
        args=_args(num_test_tasks=3),
        method=SkillOracleMethod(env=problem.env, oracle=LightSwitchOracle(env=problem.env)),
        problem=problem,
        num_cycles=2,
        max_steps_per_interaction=2,
        renderer=None,
        render_fps=2,
    )
    # num_cycles/max_steps_per_interaction are forwarded to PracticeLoop.run, not
    # hardcoded inside MethodRunner -- one initial evaluation plus one per cycle.
    assert len(metrics.evaluations) == 3


def test_run_without_output_dir_writes_no_files(*, tmp_path: Path) -> None:
    problem = _build_problem()
    MethodRunner.run(
        args=_args(output_dir=None),
        method=SkillOracleMethod(env=problem.env, oracle=LightSwitchOracle(env=problem.env)),
        problem=problem,
        num_cycles=0,
        max_steps_per_interaction=0,
        renderer=LightSwitchRenderer,
        render_fps=2,
    )
    assert list(tmp_path.iterdir()) == []


def test_run_without_output_dir_writes_no_stats_json(*, tmp_path: Path) -> None:
    problem = _build_problem()
    MethodRunner.run(
        args=_args(output_dir=None),
        method=SkillOracleMethod(env=problem.env, oracle=LightSwitchOracle(env=problem.env)),
        problem=problem,
        num_cycles=0,
        max_steps_per_interaction=0,
        renderer=None,
        render_fps=2,
    )
    assert not (tmp_path / "stats.json").exists()


def test_run_with_output_dir_and_renderer_writes_a_video_file(*, tmp_path: Path) -> None:
    problem = _build_problem()
    MethodRunner.run(
        args=_args(output_dir=tmp_path),
        method=SkillOracleMethod(env=problem.env, oracle=LightSwitchOracle(env=problem.env)),
        problem=problem,
        num_cycles=0,
        max_steps_per_interaction=0,
        renderer=LightSwitchRenderer,
        render_fps=2,
    )
    video_path = tmp_path / "episode.mp4"
    assert video_path.exists()
    assert video_path.stat().st_size > 0


def test_run_with_output_dir_writes_stats_json_that_round_trips(*, tmp_path: Path) -> None:
    problem = _build_problem()
    metrics = MethodRunner.run(
        args=_args(num_test_tasks=3, output_dir=tmp_path),
        method=SkillOracleMethod(env=problem.env, oracle=LightSwitchOracle(env=problem.env)),
        problem=problem,
        num_cycles=2,
        max_steps_per_interaction=2,
        renderer=LightSwitchRenderer,
        render_fps=2,
    )
    stats_path = tmp_path / "stats.json"
    assert stats_path.exists()

    loaded = Metrics.model_validate_json(stats_path.read_text())
    assert loaded == metrics
    assert loaded.evaluations == metrics.evaluations
    assert loaded.task_name == metrics.task_name


def test_run_with_output_dir_writes_config_snapshot_json_beside_stats_json(
    *, tmp_path: Path
) -> None:
    """Beside stats.json, never inside it: stats.json's byte-stability is what
    verifies a change did not alter results, and a SHA in it would break that."""
    problem = _build_problem()
    MethodRunner.run(
        args=_args(num_test_tasks=1, output_dir=tmp_path),
        method=SkillOracleMethod(env=problem.env, oracle=LightSwitchOracle(env=problem.env)),
        problem=problem,
        num_cycles=0,
        max_steps_per_interaction=0,
        renderer=None,
        render_fps=2,
    )
    snapshot_path = tmp_path / "config_snapshot.json"
    assert snapshot_path.exists()

    snapshot = ConfigSnapshot.model_validate_json(snapshot_path.read_text())
    # Recorded post-argparse, so a flag that was defaulted rather than passed is
    # still visible afterwards.
    assert snapshot.args["num_test_tasks"] == "1"
    assert "num_test_tasks" not in (tmp_path / "stats.json").read_text()


def test_run_without_output_dir_writes_no_config_snapshot_json(*, tmp_path: Path) -> None:
    problem = _build_problem()
    MethodRunner.run(
        args=_args(output_dir=None),
        method=SkillOracleMethod(env=problem.env, oracle=LightSwitchOracle(env=problem.env)),
        problem=problem,
        num_cycles=0,
        max_steps_per_interaction=0,
        renderer=None,
        render_fps=2,
    )
    assert not (tmp_path / "config_snapshot.json").exists()


def test_run_does_not_leak_evaluations_between_calls() -> None:
    """MethodRunner constructs a fresh Metrics() per call now (see its own
    docstring) -- there's no reset() step to forget, and no shared state two
    back-to-back calls could leak through."""
    problem = _build_problem()
    first = MethodRunner.run(
        args=_args(num_test_tasks=2),
        method=SkillOracleMethod(env=problem.env, oracle=LightSwitchOracle(env=problem.env)),
        problem=problem,
        num_cycles=0,
        max_steps_per_interaction=0,
        renderer=None,
        render_fps=2,
    )
    assert len(first.evaluations) == 1

    second = MethodRunner.run(
        args=_args(num_test_tasks=2),
        method=SkillOracleMethod(env=problem.env, oracle=LightSwitchOracle(env=problem.env)),
        problem=problem,
        num_cycles=0,
        max_steps_per_interaction=0,
        renderer=None,
        render_fps=2,
    )
    assert len(second.evaluations) == 1
    assert first is not second


def test_run_writes_one_clip_per_render_checkpoint(*, tmp_path: Path) -> None:
    """A set of clips across training is the point -- named by the transition
    count each depicts, so they sort into a progression."""
    problem = _build_problem()
    MethodRunner.run(
        args=_args(num_test_tasks=1, output_dir=tmp_path),
        method=SkillOracleMethod(env=problem.env, oracle=LightSwitchOracle(env=problem.env)),
        problem=problem,
        num_cycles=4,
        max_steps_per_interaction=2,
        renderer=LightSwitchRenderer,
        render_fps=2,
        num_render_checkpoints=3,
    )
    clips = sorted(path.name for path in tmp_path.glob("episode_*.mp4"))
    assert clips == ["episode_000000.mp4", "episode_000004.mp4", "episode_000008.mp4"]
    # The final one is also written under the plain name callers already use.
    assert (tmp_path / "episode.mp4").exists()


def test_run_defaults_to_a_single_final_clip(*, tmp_path: Path) -> None:
    problem = _build_problem()
    MethodRunner.run(
        args=_args(num_test_tasks=1, output_dir=tmp_path),
        method=SkillOracleMethod(env=problem.env, oracle=LightSwitchOracle(env=problem.env)),
        problem=problem,
        num_cycles=3,
        max_steps_per_interaction=2,
        renderer=LightSwitchRenderer,
        render_fps=2,
    )
    assert len(list(tmp_path.glob("episode_*.mp4"))) == 1
    assert (tmp_path / "episode.mp4").exists()


def test_run_reports_the_final_evaluation_not_the_first(
    *, capsys: pytest.CaptureFixture[str]
) -> None:
    """With num_cycles > 0 the first sweep runs before any practice, so reporting
    evaluations[0] would always print the untrained score. The oracle solves
    everything at every checkpoint, so this pins the index rather than the value:
    the printed denominator must match num_test_tasks of the LAST sweep."""
    problem = _build_problem()
    metrics = MethodRunner.run(
        args=_args(num_test_tasks=3),
        method=SkillOracleMethod(env=problem.env, oracle=LightSwitchOracle(env=problem.env)),
        problem=problem,
        num_cycles=2,
        max_steps_per_interaction=2,
        renderer=None,
        render_fps=2,
    )
    assert len(metrics.evaluations) == 3
    _transitions, num_solved, num_total = metrics.evaluations[-1]
    assert f"success rate: {num_solved}/{num_total}" in capsys.readouterr().out


def test_run_forwards_the_practice_reset_interval_from_args() -> None:
    """--practice-reset-interval is a global flag read off args here, the same way
    --num-render-checkpoints is, rather than threaded through every environment's
    run_method. Pinned because a silently dropped flag would produce a sweep whose
    arms are identical and whose null looks real."""
    problem = _build_problem()
    args = _args(num_test_tasks=1)
    args.practice_reset_interval = 2
    metrics = MethodRunner.run(
        args=args,
        method=SkillOracleMethod(env=problem.env, oracle=LightSwitchOracle(env=problem.env)),
        problem=problem,
        num_cycles=2,
        max_steps_per_interaction=6,
        renderer=None,
        render_fps=2,
    )
    # 6 // 2 = 3 resets per period, over 2 cycles.
    assert metrics.num_practice_resets == 6


def test_run_without_a_practice_reset_interval_resets_only_per_cycle() -> None:
    problem = _build_problem()
    metrics = MethodRunner.run(
        args=_args(num_test_tasks=1),
        method=SkillOracleMethod(env=problem.env, oracle=LightSwitchOracle(env=problem.env)),
        problem=problem,
        num_cycles=2,
        max_steps_per_interaction=6,
        renderer=None,
        render_fps=2,
    )
    assert metrics.num_practice_resets == 2


def test_run_forwards_the_practice_reset_policy_from_args() -> None:
    """Same reasoning as the interval above: a silently dropped flag produces a
    sweep whose two arms are identical and whose null result looks real."""
    problem = _build_problem()
    args = _args(num_test_tasks=1)
    args.practice_reset_policy = PracticeResetPolicy.NEVER
    metrics = MethodRunner.run(
        args=args,
        method=SkillOracleMethod(env=problem.env, oracle=LightSwitchOracle(env=problem.env)),
        problem=problem,
        # Required under `never`: without its own evaluation environment the arm
        # would be reset once per test task per sweep and would not be reset-free.
        evaluation_problem=_build_problem(),
        num_cycles=2,
        max_steps_per_interaction=6,
        renderer=None,
        render_fps=2,
    )
    assert metrics.num_practice_resets == 0


def test_run_with_an_args_namespace_predating_the_policy_flag_still_resets() -> None:
    """Every archived driver's namespace lacks the attribute entirely; the fallback
    has to be the scheduled behaviour, not an accidental reset-free run."""
    problem = _build_problem()
    args = _args(num_test_tasks=1)
    assert not hasattr(args, "practice_reset_policy")
    metrics = MethodRunner.run(
        args=args,
        method=SkillOracleMethod(env=problem.env, oracle=LightSwitchOracle(env=problem.env)),
        problem=problem,
        num_cycles=2,
        max_steps_per_interaction=6,
        renderer=None,
        render_fps=2,
    )
    assert metrics.num_practice_resets == 2


def test_run_without_the_record_full_loop_flag_writes_no_video(*, tmp_path: Path) -> None:
    """The flag defaults to off, and an args namespace that predates it entirely
    (as every archived driver's does) must still run."""
    problem = _build_problem()
    args = _args(num_test_tasks=1)
    assert not hasattr(args, "record_full_loop")
    MethodRunner.run(
        args=args,
        method=SkillOracleMethod(env=problem.env, oracle=LightSwitchOracle(env=problem.env)),
        problem=problem,
        num_cycles=1,
        max_steps_per_interaction=2,
        renderer=LightSwitchRenderer,
        render_fps=2,
    )
    assert list(tmp_path.iterdir()) == []


def test_run_with_record_full_loop_writes_one_seekable_video(*, tmp_path: Path) -> None:
    problem = _build_problem()
    args = _args(num_test_tasks=1)
    args.record_full_loop = tmp_path / "full_loop.mp4"
    MethodRunner.run(
        args=args,
        method=SkillOracleMethod(env=problem.env, oracle=LightSwitchOracle(env=problem.env)),
        problem=problem,
        num_cycles=1,
        max_steps_per_interaction=2,
        renderer=LightSwitchRenderer,
        render_fps=2,
    )
    assert (tmp_path / "full_loop.mp4").exists()
    decoded = imageio.mimread(tmp_path / "full_loop.mp4", memtest=False)
    # One continuous timeline: the practice period's own frames are in here too,
    # so it is longer than the single evaluation episode --output-dir would write.
    assert len(decoded) > 10


def test_run_with_record_full_loop_leaves_stats_json_byte_identical(*, tmp_path: Path) -> None:
    """The recording must be a pure observer. This is the in-suite form of the
    same check run at full scale on Tossing Room: same seed, one run with the flag
    and one without, compared byte for byte."""
    outputs: dict[str, bytes] = {}
    for name, record in (("plain", None), ("recorded", tmp_path / "full_loop.mp4")):
        output_dir = tmp_path / name
        problem = _build_problem()
        args = _args(num_test_tasks=3, output_dir=output_dir)
        args.record_full_loop = record
        MethodRunner.run(
            args=args,
            method=SkillOracleMethod(env=problem.env, oracle=LightSwitchOracle(env=problem.env)),
            problem=problem,
            num_cycles=2,
            max_steps_per_interaction=3,
            renderer=LightSwitchRenderer,
            render_fps=2,
        )
        outputs[name] = (output_dir / "stats.json").read_bytes()
    # Non-vacuity: a comparison of two runs that both ignored the flag would pass
    # trivially, so the recorded arm must actually have recorded something.
    assert (tmp_path / "full_loop.mp4").exists()
    assert outputs["recorded"] == outputs["plain"]


def test_run_rejects_record_full_loop_when_the_domain_has_no_renderer(*, tmp_path: Path) -> None:
    """Failing up front beats a run that completes and silently writes no video --
    Ball-Ring has no renderer.py yet, so asking to record one is a mistake."""
    problem = _build_problem()
    args = _args(num_test_tasks=1)
    args.record_full_loop = tmp_path / "full_loop.mp4"
    with pytest.raises(ValueError, match="renderer"):
        MethodRunner.run(
            args=args,
            method=SkillOracleMethod(env=problem.env, oracle=LightSwitchOracle(env=problem.env)),
            problem=problem,
            num_cycles=0,
            max_steps_per_interaction=0,
            renderer=None,
            render_fps=2,
        )


class _CountingPlanMethod(SkillOracleMethod):
    """Reports a planning attempt per policy call, so a test can decode which calls
    landed in which bucket. Practice and evaluation are charged different amounts, so
    the two are separable in the recorded totals -- 1 per evaluation policy, 100 per
    practice policy, and one failure per ten attempts."""

    attempts: int = 0

    def get_task_policy(self, *, task: Task) -> Policy:
        self.attempts += 1
        return super().get_task_policy(task=task)

    def get_practice_policy(self, *, task: Task) -> Policy:
        self.attempts += 100
        return super().get_task_policy(task=task)

    def planning_outcomes(self) -> tuple[int, int]:
        return (self.attempts // 10, self.attempts)


def test_each_planning_bucket_covers_one_evaluation_sweep_and_the_practice_after_it() -> None:
    """Pins the window, which is not the obvious one and is easy to misread as
    "cycle i's practice". `on_cycle_end` fires after cycle i's practice and *before*
    cycle i's sweep, so bucket i is (sweep i, practice i) and the trailing bucket is
    the final sweep alone -- a different kind of window, not a short cycle.

    Decoded from a Method charging 1 per evaluation policy and 100 per practice
    policy, with 2 test tasks: 2 + 100 for each of the first two buckets, 2 for the
    last. Recorded here rather than only in prose because anyone plotting these
    against `evaluations`' transition counts needs to know about the offset."""
    problem = _build_problem()
    metrics = MethodRunner.run(
        args=_args(num_test_tasks=2),
        method=_CountingPlanMethod(env=problem.env, oracle=LightSwitchOracle(env=problem.env)),
        problem=problem,
        num_cycles=2,
        max_steps_per_interaction=2,
        renderer=None,
        render_fps=2,
    )
    assert len(metrics.planning_attempts_per_cycle) == len(metrics.evaluations) == 3
    assert metrics.planning_attempts_per_cycle == [102, 102, 2]
    assert metrics.planning_failures_per_cycle == [10, 10, 0]
    assert metrics.total_planning_outcomes() == (20, 206)


def test_a_method_that_never_plans_reports_zero_out_of_zero() -> None:
    """The oracle carries no planner, so its buckets must be present-and-zero rather
    than absent -- "planned fine" and "did not plan" have to stay distinguishable from
    a run's own stats.json."""
    problem = _build_problem()
    metrics = MethodRunner.run(
        args=_args(num_test_tasks=2),
        method=SkillOracleMethod(env=problem.env, oracle=LightSwitchOracle(env=problem.env)),
        problem=problem,
        num_cycles=0,
        max_steps_per_interaction=0,
        renderer=None,
        render_fps=2,
    )
    assert metrics.planning_failures_per_cycle == [0]
    assert metrics.planning_attempts_per_cycle == [0]
    assert metrics.total_planning_outcomes() == (0, 0)


def test_planning_outcomes_reach_stats_json(*, tmp_path: Path) -> None:
    """The fields are only useful if they survive to the file a sweep reads back."""
    problem = _build_problem()
    MethodRunner.run(
        args=_args(num_test_tasks=2, output_dir=tmp_path),
        method=SkillOracleMethod(env=problem.env, oracle=LightSwitchOracle(env=problem.env)),
        problem=problem,
        num_cycles=0,
        max_steps_per_interaction=0,
        renderer=None,
        render_fps=2,
    )
    restored = Metrics.model_validate_json((tmp_path / "stats.json").read_text())
    assert restored.planning_failures_per_cycle == [0]
    assert restored.planning_attempts_per_cycle == [0]


class _AlwaysFailsToPlanMethod(SkillOracleMethod):
    """Every planner call found no plan -- the shape of the defect that motivated
    counting them at all. Counters start at zero and rise during the run, like a real
    Method's: a fake returning a constant would be differenced away to nothing, which
    is exactly the reused-instance protection working."""

    attempts: int = 0

    def get_task_policy(self, *, task: Task) -> Policy:
        self.attempts += 1
        return super().get_task_policy(task=task)

    def planning_outcomes(self) -> tuple[int, int]:
        return (self.attempts, self.attempts)


def test_planning_failures_are_printed_beside_the_success_rate(
    *, capsys: pytest.CaptureFixture[str]
) -> None:
    """The original defect was a clean-looking 0/5 with nothing on the console saying
    planning had failed. As an x/y, never a bare count: against EES's speculative
    planning a numerator alone does not distinguish a healthy run from a broken one."""
    problem = _build_problem()
    MethodRunner.run(
        args=_args(num_test_tasks=2),
        method=_AlwaysFailsToPlanMethod(env=problem.env, oracle=LightSwitchOracle(env=problem.env)),
        problem=problem,
        num_cycles=0,
        max_steps_per_interaction=0,
        renderer=None,
        render_fps=2,
    )
    assert "planning failures: 2/2 planner calls found no plan" in capsys.readouterr().out


def test_a_run_whose_planner_never_failed_prints_nothing_extra(
    *, capsys: pytest.CaptureFixture[str]
) -> None:
    """A healthy run's console output is exactly what it was before this change."""
    problem = _build_problem()
    MethodRunner.run(
        args=_args(num_test_tasks=2),
        method=SkillOracleMethod(env=problem.env, oracle=LightSwitchOracle(env=problem.env)),
        problem=problem,
        num_cycles=0,
        max_steps_per_interaction=0,
        renderer=None,
        render_fps=2,
    )
    assert "planning failures" not in capsys.readouterr().out


class _CountingPracticeMethod(SkillOracleMethod):
    """Charges one practice attempt of `Toggle` per practice policy handed out, and
    one of `Move` per evaluation policy, so a test can decode which executions landed
    in which window. Counters are cumulative and rise during the run, like a real
    Method's -- a fake returning a constant would be differenced away to nothing,
    which is exactly the reused-instance protection working."""

    toggles: int = 0
    moves: int = 0

    def get_task_policy(self, *, task: Task) -> Policy:
        self.moves += 1
        return super().get_task_policy(task=task)

    def get_practice_policy(self, *, task: Task) -> Policy:
        self.toggles += 1
        return super().get_task_policy(task=task)

    def practice_outcomes(self) -> dict[str, SkillPracticeTally]:
        return {
            "Toggle": SkillPracticeTally(
                num_attempts=self.toggles,
                num_successes=self.toggles,
                num_informed_attempts=self.toggles,
                num_informed_successes=self.toggles,
            ),
            "Move": SkillPracticeTally(num_attempts=self.moves),
        }


def test_each_practice_bucket_covers_one_evaluation_sweep_and_the_practice_after_it() -> None:
    """The same window as the planning counters, and the same trap: `on_cycle_end`
    fires after cycle i's practice and *before* cycle i's sweep, so bucket i is
    (sweep i, practice i) and the trailing bucket is the final sweep alone.

    Decoded from a Method charging one `Move` per evaluation policy and one `Toggle`
    per practice policy, with 2 test tasks: 2 Moves + 1 Toggle in each of the first
    two buckets, 2 Moves in the last."""
    problem = _build_problem()
    metrics = MethodRunner.run(
        args=_args(num_test_tasks=2),
        method=_CountingPracticeMethod(env=problem.env, oracle=LightSwitchOracle(env=problem.env)),
        problem=problem,
        num_cycles=2,
        max_steps_per_interaction=2,
        renderer=None,
        render_fps=2,
    )
    assert len(metrics.practice_outcomes_per_cycle) == len(metrics.evaluations) == 3
    assert [window["Move"].num_attempts for window in metrics.practice_outcomes_per_cycle] == [
        2,
        2,
        2,
    ]
    assert [window["Toggle"].num_attempts for window in metrics.practice_outcomes_per_cycle] == [
        1,
        1,
        0,
    ]
    totals = metrics.total_practice_outcomes()
    assert (totals["Toggle"].num_successes, totals["Toggle"].num_attempts) == (2, 2)
    assert (totals["Move"].num_successes, totals["Move"].num_attempts) == (0, 6)


def test_a_method_that_scores_no_skills_records_an_empty_window_per_cycle() -> None:
    """Present-and-empty rather than absent, so the buckets stay index-aligned with
    `evaluations` for a reader plotting the two together."""
    problem = _build_problem()
    metrics = MethodRunner.run(
        args=_args(num_test_tasks=2),
        method=SkillOracleMethod(env=problem.env, oracle=LightSwitchOracle(env=problem.env)),
        problem=problem,
        num_cycles=0,
        max_steps_per_interaction=0,
        renderer=None,
        render_fps=2,
    )
    assert metrics.practice_outcomes_per_cycle == [{}]
    assert metrics.total_practice_outcomes() == {}


def test_practice_outcomes_reach_stats_json(*, tmp_path: Path) -> None:
    """The fields are only useful if they survive to the file a sweep reads back --
    which is the whole point: the previous route to these numbers was a bespoke
    per-domain collector script."""
    problem = _build_problem()
    MethodRunner.run(
        args=_args(num_test_tasks=2, output_dir=tmp_path),
        method=_CountingPracticeMethod(env=problem.env, oracle=LightSwitchOracle(env=problem.env)),
        problem=problem,
        num_cycles=1,
        max_steps_per_interaction=2,
        renderer=None,
        render_fps=2,
    )
    restored = Metrics.model_validate_json((tmp_path / "stats.json").read_text())
    assert restored.total_practice_outcomes()["Toggle"].num_attempts == 1


def test_practice_outcomes_are_printed_beside_the_success_rate(
    *, capsys: pytest.CaptureFixture[str]
) -> None:
    """Visible without opening stats.json, and as x/y counts throughout: the question
    these exist to answer -- was the sampler starved or is it unable -- is asked while
    watching a run, not only afterwards."""
    problem = _build_problem()
    MethodRunner.run(
        args=_args(num_test_tasks=2),
        method=_CountingPracticeMethod(env=problem.env, oracle=LightSwitchOracle(env=problem.env)),
        problem=problem,
        num_cycles=1,
        max_steps_per_interaction=2,
        renderer=None,
        render_fps=2,
    )
    out = capsys.readouterr().out
    assert "practice Toggle: 1/1 succeeded" in out
    assert "1/1 informed" in out


def test_a_run_that_practiced_nothing_prints_nothing_extra(
    *, capsys: pytest.CaptureFixture[str]
) -> None:
    """A non-learning baseline's console output is exactly what it was before."""
    problem = _build_problem()
    MethodRunner.run(
        args=_args(num_test_tasks=2),
        method=SkillOracleMethod(env=problem.env, oracle=LightSwitchOracle(env=problem.env)),
        problem=problem,
        num_cycles=0,
        max_steps_per_interaction=0,
        renderer=None,
        render_fps=2,
    )
    assert "practice " not in capsys.readouterr().out
