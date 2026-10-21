import argparse
import functools
import hashlib
import json
import os
import pickle
import time
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

from hitl_pmp.core.control_steps import ControlStepLimitReached
from hitl_pmp.core.method.method import HumanCubeBinResetRequested
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.skill_provider import LightSwitchSkillProvider
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod
from hitl_pmp.practice_loop import PracticeResetPolicy
from hitl_pmp.recording.run_record.types import RecordingHeader
from hitl_pmp.recording.run_record.writer import RunRecordWriter
from hitl_pmp.step_protocol import (
    CHECKPOINT_FILENAME,
    DeploymentSnapshot,
    PolicyFileHash,
    PracticeClock,
    StepPracticeRunner,
    StepProtocolEntry,
)


def test_clock_uses_prepared_per_skill_duration_and_actual_robot_cost():
    from hitl_pmp.core.practice_costs import (
        ChargeFunction,
        HumanCharge,
        PracticeAccounting,
        PracticeCosts,
    )

    ledger = PracticeAccounting(
        costs=PracticeCosts(
            robot_step=ChargeFunction(value=2),
            human_skills={
                "near": HumanCharge(cost=ChargeFunction(value=7), duration=ChargeFunction(value=3))
            },
        )
    )
    seen = []
    clock = PracticeClock(
        budget=5,
        interval=2,
        accounting=ledger,
        on_measure=lambda: seen.append((
            clock.measurement_steps,
            clock.steps,
            ledger.total_cost,
        )),
    )
    clock.robot_active = True
    clock.before_robot_step()
    clock.after_robot_step()
    clock.robot_active = False
    charge = clock.before_human(skill="near")
    clock.human_invoked(charge=charge)
    # The 3-step human jump from 1 to 4 crosses thresholds 2 and 4: one measurement
    # each, at its own x, both after the whole charge is recorded.
    assert seen == [(2, 4, 9), (4, 4, 9)]
    assert clock.steps == ledger.practice_steps == 4
    with pytest.raises(ControlStepLimitReached):
        clock.before_human(skill="near")
    assert ledger.total_cost == 9


def test_tossing_snapshot_with_observed_competence_round_trips():
    from hitl_pmp.core.method.types import GroundSkill
    from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
    from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider

    env = Tossing3DEnvironment()
    provider = Tossing3DSkillProvider(env=env)
    method = EesMethod(env=env, skill_provider=provider)
    skill = provider.skills()[0]
    objects = tuple(
        next(o for o in provider.objects() if o.type == p.type) for p in skill.parameters
    )
    ground = GroundSkill(skill=skill, objects=objects)
    method.competence_model(ground_skill=ground).observe(success=True)
    raw = DeploymentSnapshot.encode(method=method)
    restored = DeploymentSnapshot.restore(raw=raw, env=env, provider=provider)
    assert restored.current_competences() == method.current_competences()


def test_evaluation_clock_stops_at_goal_or_exact_budget():
    from hitl_pmp.step_protocol import EvaluationClock

    clock = EvaluationClock(budget=2, check_goal=lambda: False)
    for _ in range(2):
        clock.before()
        clock.after()
    with pytest.raises(ControlStepLimitReached):
        clock.before()
    assert clock.steps == 2
    successful = EvaluationClock(budget=20, check_goal=lambda: True)
    successful.before()
    successful.after()
    with pytest.raises(ControlStepLimitReached):
        successful.before()
    assert successful.solved and successful.steps == 1


def test_a_skill_the_goal_stopped_is_recorded_as_reaching_the_goal_not_the_budget():
    """The recording named every evaluation cut-off the step budget, including a sweep
    the goal latch stopped at ~1,800 of 3,000 steps on a task it solved."""
    from hitl_pmp.step_protocol import EvaluationClock

    reached = EvaluationClock(budget=3000, check_goal=lambda: True)
    reached.before()
    reached.after()
    with pytest.raises(ControlStepLimitReached):
        reached.before()
    assert reached.interruption() == "stopped: goal reached"
    exhausted = EvaluationClock(budget=1, check_goal=lambda: False)
    exhausted.before()
    exhausted.after()
    with pytest.raises(ControlStepLimitReached):
        exhausted.before()
    assert exhausted.interruption() == "interrupted: evaluation step budget"


def test_measurement_inside_skill_does_not_end_it():
    measurements = []
    clock = PracticeClock(budget=9, interval=3, on_measure=lambda: measurements.append(clock.steps))
    clock.robot_active = True
    for _ in range(8):
        clock.before_robot_step()
        clock.after_robot_step()
    assert measurements == [3, 6]
    assert clock.robot_active
    assert clock.steps == 8


def test_human_counts_once_and_reset_physics_does_not_count():
    measurements = []
    clock = PracticeClock(budget=9, interval=3, on_measure=lambda: measurements.append(clock.steps))
    clock.robot_active = True
    for _ in range(2):
        clock.after_robot_step()
    clock.robot_active = False
    for _ in range(10):
        clock.after_robot_step()
    clock.before_human()
    clock.human_invoked()
    assert (clock.robot_steps, clock.human_invocations, clock.steps) == (2, 1, 3)
    assert measurements == [3]


def test_budget_stops_before_extra_physics():
    clock = PracticeClock(budget=3, interval=2, on_measure=lambda: None)
    clock.robot_active = True
    for _ in range(3):
        clock.before_robot_step()
        clock.after_robot_step()
    with pytest.raises(ControlStepLimitReached):
        clock.before_robot_step()
    with pytest.raises(ControlStepLimitReached):
        clock.before_human()
    assert clock.steps == 3


def test_custom_human_duration_checks_remaining_budget():
    clock = PracticeClock(budget=3, interval=2, human_steps=2, on_measure=lambda: None)
    clock.before_human()
    clock.human_invoked()
    with pytest.raises(ControlStepLimitReached):
        clock.before_human()
    assert clock.steps == 2


def test_snapshot_freezes_sampler_without_refitting_or_sharing_rng():
    env = LightSwitchEnvironment(grid_size=4)
    provider = LightSwitchSkillProvider(env=env)
    method = EesMethod(env=env, skill_provider=provider, seed=7)
    sampler = method.sampler(skill_name="test", param_dim=1)
    sampler.observe(sampler_input=[1.0, 0.3], success=True)
    sampler.fit()
    before_rng = pickle.dumps(method._rng.bit_generator.state)
    raw = DeploymentSnapshot.encode(method=method)
    sampler.observe(sampler_input=[1.0, 0.8], success=False)
    fresh = LightSwitchEnvironment(grid_size=4)
    restored = DeploymentSnapshot.restore(
        raw=raw, env=fresh, provider=LightSwitchSkillProvider(env=fresh)
    )
    assert restored.env is fresh
    assert restored._samplers["test"].num_observations == 1
    assert sampler.num_observations == 2
    assert restored._samplers["test"].score_inputs(sampler_inputs=[[1.0, 0.2]]) == [1.0]
    restored._rng.random(10)
    assert pickle.dumps(method._rng.bit_generator.state) == before_rng
    assert np.array_equal(
        restored._samplers["test"]._classifier._input_shift,
        sampler._classifier._input_shift,
    )


# --- Runner-level fakes -----------------------------------------------------------------
# The runner is driven against a scripted problem whose `take_action` fires the control-
# step observers once per requested tick, so every boundary below is exercised exactly as
# a real controller would exercise it, without a simulator.


class FakeBackend:
    def __init__(self) -> None:
        self.before = None
        self.after = None

    def set_control_step_observers(self, *, before, after) -> None:
        self.before = before
        self.after = after


class FakeEnv:
    def __init__(self) -> None:
        self._backend = FakeBackend()
        self.error = None
        self.adopted: list[tuple[float, str]] = []
        self.problem = None

    def backend(self) -> FakeBackend:
        return self._backend

    def set_control_step_observers(self, *, before, after) -> None:
        self._backend.set_control_step_observers(before=before, after=after)

    def _log_skill_ticks(self, *, action) -> None:
        pass

    def last_skill_error(self):
        return self.error

    def adopt_interrupted_skill(self, *, action, reason: str):
        self.error = reason
        self.adopted.append((float(action[0]), reason))
        self.problem.state = ("adopted", float(action[0]))
        return self.problem.state

    def close(self) -> None:
        pass


class FakeProblem:
    def __init__(self) -> None:
        self.env = FakeEnv()
        self.env.problem = self
        self.state: object = ("initial",)
        self.tasks = self

    def hard_reset(self) -> None:
        pass

    def get_current_state(self):
        return self.state

    def sample_train_task_in_place(self):
        return SimpleNamespace(name="task")

    def take_action(self, *, action):
        self.env.error = None
        backend = self.env.backend()
        for _ in range(int(action[0])):
            backend.before()
            backend.after()
        self.state = ("after", float(action[0]))
        return self.state

    def execute_movables_reset(self, *, destination) -> None:
        self.state = ("reset", destination)


class FakeStepProtocol:
    """The domain protocol over `FakeProblem`: Tossing3D's two human skills, and every
    per-domain hook delegated to the fake environment."""

    @staticmethod
    def human_skill_names() -> dict[str, str]:
        return {"robot_side": "reset_cube_and_bin_near", "opposite_side": "reset_cube_far"}

    @staticmethod
    def adopt_interrupted_skill(*, problem, action, reason: str):
        return problem.env.adopt_interrupted_skill(action=action, reason=reason)

    @staticmethod
    def last_skill_error(*, problem):
        return problem.env.last_skill_error()

    @staticmethod
    def last_skill_trace(*, problem):
        del problem
        return None

    @staticmethod
    def interrupted(*, problem, action) -> None:
        pass

    @staticmethod
    def practice_progress(**progress) -> None:
        del progress

    @staticmethod
    def evaluation_progress(**progress) -> None:
        del progress

    @staticmethod
    def evaluation_pool() -> None:
        return None  # Evaluations beside practice, as in every simulated domain.


class ScriptedMethod(EesMethod):
    """A real `EesMethod` (so deployment snapshots encode), with a scripted practice
    policy: an int is a robot skill of that many control steps, an exception is raised.
    Once the script runs out every action is a one-step skill, so the budget is reached."""

    def model_post_init(self, context, /) -> None:  # noqa: PLR0917
        super().model_post_init(context)
        self._script: list = []

    def get_practice_policy(self, *, task):
        def policy(state):  # noqa: PLR0917 -- Policy is a positional callable
            item = self._script.pop(0) if self._script else 1
            if isinstance(item, BaseException):
                raise item
            return SimpleNamespace(action=np.array([float(item)]), label=f"skill{item}")

        return policy

    def observe_practice_action_budget(self, *, remaining_actions: int) -> None:
        pass

    def observe_help_granted(self, *, state) -> None:
        pass

    def observe_environment_reset(self, *, state) -> None:
        pass

    def end_cycle(self) -> None:
        pass


class FakeEvaluations:
    """Records how the runner closes the evaluation queue. `complete=False` leaves every
    future pending, as a long queue behind a single worker would be."""

    def __init__(self, *, complete: bool = True) -> None:
        self.complete = complete
        self.futures: list[Future] = []
        self.finished = False
        self.aborted = False

    def submit(self, *, fn, **kwargs) -> Future:
        future: Future = Future()
        if self.complete:
            future.set_result({
                "complete": True,
                "tasks": [{"task": 0, "goal": "goal", "solved": False}],
                "num_solved": 0,
                "num_total": 1,
            })
        self.futures.append(future)
        return future

    def finish(self) -> None:
        self.finished = True
        pending = sum(not future.done() for future in self.futures)
        if pending:
            # The real queue would block here until every evaluation completed.
            raise AssertionError(f"finish() would wait on {pending} pending evaluations")

    def abort(self) -> None:
        self.aborted = True
        for future in self.futures:
            future.cancel()


def _step_args(*, output_dir, budget: int, interval: int, human_steps: int = 1, **extra):
    values = {
        "env": "tossing3d",
        "method": "ees",
        "practice_reset_policy": PracticeResetPolicy.NEVER,
        "practice_reset_interval": None,
        "output_dir": output_dir,
        "defer_rendering": True,
        "practice_step_budget": budget,
        "measurement_interval_steps": interval,
        "human_skill_steps": human_steps,
        "evaluation_control_steps": 5,
        "num_test_tasks": 10,
        "seed": 0,
        "max_steps_per_interaction": 20,
        "num_cycles": 1,
    }
    values.update(extra)
    return argparse.Namespace(**values)


def _scripted_method(*, script: list) -> ScriptedMethod:
    env = LightSwitchEnvironment(grid_size=4)
    method = ScriptedMethod(env=env, skill_provider=LightSwitchSkillProvider(env=env), seed=0)
    method._script = list(script)
    return method


def _events(*, output) -> list[dict]:
    return [json.loads(line) for line in (output / "step_events.jsonl").read_text().splitlines()]


def _measured(*, output) -> list[int]:
    records = json.loads((output / "measurements.json").read_text())
    return [m["practice_steps"] for m in records]


def test_a_human_request_that_does_not_fit_ends_practice_at_the_budget(*, tmp_path):
    """Finding 2: the budget reached on a human request ends practice like the budget
    reached on a robot action -- not a crash -- and the final measurement still runs."""
    output = tmp_path / "run"
    evaluations = FakeEvaluations()
    StepPracticeRunner.run(
        environment=FakeStepProtocol,
        args=_step_args(output_dir=output, budget=5, interval=100, human_steps=2),
        method=_scripted_method(
            script=[4, HumanCubeBinResetRequested(cost=1.0, destination="robot_side")]
        ),
        problem=FakeProblem(),
        evaluations=evaluations,
    )
    assert not (output / "failure.txt").exists()
    end = [e for e in _events(output=output) if e["event"] == "practice_end"]
    assert [e["reason"] for e in end] == ["practice_step_budget"]
    assert _measured(output=output) == [0, 4]
    assert json.loads((output / "status.json").read_text())["phase"] == "complete"
    assert evaluations.finished and not evaluations.aborted


def test_measurements_are_recorded_on_the_practice_step_axis(*, tmp_path):
    """Finding 6: stats.json's x is the counted practice step of each measurement --
    unique and strictly increasing, even when two thresholds fall inside one skill, and
    a human request advances it."""
    output = tmp_path / "run"
    StepPracticeRunner.run(
        environment=FakeStepProtocol,
        args=_step_args(output_dir=output, budget=7, interval=1, human_steps=1),
        method=_scripted_method(
            script=[2, HumanCubeBinResetRequested(cost=1.0, destination="opposite_side"), 3]
        ),
        problem=FakeProblem(),
        evaluations=FakeEvaluations(),
    )
    stats = json.loads((output / "stats.json").read_text())
    xs = [x for x, _, _ in stats["evaluations"]]
    assert xs == list(range(8))
    assert xs == _measured(output=output)
    assert [b["num_online_transitions"] for b in stats["breakdowns"]] == xs


def test_a_failed_run_cancels_queued_evaluations_instead_of_waiting(*, tmp_path):
    """Finding 4: a crash must not block on the evaluation queue; what was cancelled is
    recorded rather than silently dropped."""
    output = tmp_path / "run"
    evaluations = FakeEvaluations(complete=False)
    with pytest.raises(RuntimeError, match="planner exploded"):
        StepPracticeRunner.run(
            environment=FakeStepProtocol,
            args=_step_args(output_dir=output, budget=50, interval=2),
            method=_scripted_method(script=[3, RuntimeError("planner exploded")]),
            problem=FakeProblem(),
            evaluations=evaluations,
        )
    assert evaluations.aborted and not evaluations.finished
    records = json.loads((output / "step_evaluations.json").read_text())
    assert len(records) == 2
    assert all(r["complete"] is False and r["cancelled"] is True for r in records)
    assert json.loads((output / "status.json").read_text())["phase"] == "failed"


class CountsLearning(ScriptedMethod):
    """Records each `end_cycle`, so a test can see which cycles were learned from."""

    def model_post_init(self, context, /) -> None:  # noqa: PLR0917
        super().model_post_init(context)
        self._learned = 0

    def end_cycle(self) -> None:
        self._learned += 1


@pytest.mark.parametrize(
    ("max_actions", "expected_cycles"),
    [
        (20, 1),  # the budget ends the first cycle after 5 of its 20 actions
        (5, 1),  # the first cycle's last action lands exactly on the budget
        (3, 2),  # one full cycle, then a partial one of 2 actions
    ],
)
def test_the_cycle_the_budget_ends_is_learned_from_before_the_final_measurement(
    *, tmp_path, max_actions: int, expected_cycles: int
):
    output = tmp_path / "run"
    method = CountsLearning(
        env=LightSwitchEnvironment(grid_size=4),
        skill_provider=LightSwitchSkillProvider(env=LightSwitchEnvironment(grid_size=4)),
        seed=0,
    )
    StepPracticeRunner.run(
        environment=FakeStepProtocol,
        args=_step_args(
            output_dir=output, budget=5, interval=100, max_steps_per_interaction=max_actions
        ),
        method=method,
        problem=FakeProblem(),
        evaluations=FakeEvaluations(),
    )
    assert method._learned == expected_cycles
    events = _events(output=output)
    names = [e["event"] for e in events]
    updates = [e for e in events if e["event"] == "learning_update"]
    assert len(updates) == expected_cycles
    assert updates[-1]["reason"] == "practice_step_budget"
    final = max(i for i, name in enumerate(names) if name == "measurement")
    assert names.index("learning_update") < final
    assert max(i for i, name in enumerate(names) if name == "learning_update") < final
    records = json.loads((output / "measurements.json").read_text())
    assert records[-1]["learning_updates"] == expected_cycles


def test_a_skill_interrupted_at_the_budget_is_logged_and_its_state_adopted(*, tmp_path):
    """Finding 9: the interrupted skill is reported as interrupted, with a reason, and
    the environment adopts the simulator's actual post-interruption state."""
    output = tmp_path / "run"
    problem = FakeProblem()
    StepPracticeRunner.run(
        environment=FakeStepProtocol,
        args=_step_args(output_dir=output, budget=3, interval=100),
        method=_scripted_method(script=[5]),
        problem=problem,
        evaluations=FakeEvaluations(),
    )
    robot = [e for e in _events(output=output) if e["event"] == "robot"]
    assert len(robot) == 1
    assert robot[0]["interrupted"] is True
    assert robot[0]["interruption"] == "practice_step_budget"
    assert robot[0]["skill_error"] is not None
    assert problem.env.adopted == [(5.0, robot[0]["skill_error"])]
    assert problem.state == ("adopted", 5.0)


def test_a_jump_over_several_thresholds_measures_each_one():
    """Finding 8: a human invocation longer than the interval crosses several measurement
    thresholds and several progress multiples; each is reported once, at its threshold."""
    measured = []
    progress = []
    clock = PracticeClock(
        budget=1000,
        interval=2,
        human_steps=250,
        on_measure=lambda: measured.append(clock.measurement_steps),
        on_progress=lambda: progress.append(clock.steps),
    )
    clock.before_human()
    clock.human_invoked()
    assert measured == list(range(2, 251, 2))
    assert len(progress) == 2
    assert clock.measurement_steps == 250


class RecordingMethodCli:
    calls: ClassVar[list] = []

    @staticmethod
    def run(*, args, env_cli) -> None:
        RecordingMethodCli.calls.append(args)


def test_existing_output_is_refused_before_anything_is_built_or_opened(*, tmp_path):
    """Finding 3: an existing run is refused before the problem is built or the state log
    is opened for writing, so its files are untouched."""
    RecordingMethodCli.calls.clear()
    output = tmp_path / "run"
    output.mkdir()
    (output / "step_events.jsonl").write_text("old events\n")
    (output / "tossing3d_state_log.jsonl").write_text("old states\n")
    with pytest.raises(FileExistsError):
        StepProtocolEntry.run(
            env_cli=RecordingMethodCli,
            environment=FakeStepProtocol,
            args=_step_args(output_dir=output, budget=5, interval=1),
            method_cli=RecordingMethodCli,
        )
    assert RecordingMethodCli.calls == []
    assert (output / "tossing3d_state_log.jsonl").read_text() == "old states\n"
    assert (output / "step_events.jsonl").read_text() == "old events\n"


@pytest.mark.parametrize(
    "override",
    [
        {"method": "skill-oracle"},
        {"defer_rendering": False},
        {"practice_reset_interval": 3},
    ],
)
def test_unsupported_configurations_fail_before_anything_is_built(*, tmp_path, override):
    RecordingMethodCli.calls.clear()
    with pytest.raises(ValueError, match="[Ss]tep protocol|defer-rendering"):
        StepProtocolEntry.run(
            env_cli=RecordingMethodCli,
            environment=FakeStepProtocol,
            args=_step_args(output_dir=tmp_path / "run", budget=5, interval=1, **override),
            method_cli=RecordingMethodCli,
        )
    assert RecordingMethodCli.calls == []
    assert not (tmp_path / "run").exists()


def test_episode_traces_are_refused_rather_than_silently_not_written(*, tmp_path):
    """The step protocol never runs `run_task_episode`, so it has no `EpisodeTrace` to
    write: it refuses `--record-episode-traces` instead of accepting and ignoring it."""
    RecordingMethodCli.calls.clear()
    with pytest.raises(ValueError, match="--record-episode-traces"):
        StepProtocolEntry.run(
            env_cli=RecordingMethodCli,
            environment=FakeStepProtocol,
            args=_step_args(
                output_dir=tmp_path / "run", budget=5, interval=1, record_episode_traces=True
            ),
            method_cli=RecordingMethodCli,
        )
    assert RecordingMethodCli.calls == []
    assert not (tmp_path / "run").exists()


def test_a_domain_without_a_step_protocol_is_refused_by_the_cli(*, tmp_path):
    from hitl_pmp.cli import Cli

    argv = ["--env", "lightswitch", "--method", "ees", "--practice-step-budget", "5"]
    with pytest.raises(ValueError, match="step protocol does not support --env lightswitch"):
        Cli.main(argv=[*argv, "--output-dir", str(tmp_path / "run")])
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    "flag",
    [
        "--practice-step-budget",
        "--measurement-interval-steps",
        "--human-skill-steps",
        "--evaluation-control-steps",
    ],
)
@pytest.mark.parametrize("value", ["0", "-3"])
def test_step_flags_reject_non_positive_values_at_parse_time(*, flag, value):
    """Finding 5."""
    from hitl_pmp.cli import Cli

    with pytest.raises(SystemExit):
        Cli.parse_args(argv=["--env", "lightswitch", "--method", "ees", flag, value])


def test_each_evaluation_runs_in_its_own_fresh_process():
    """Finding 7: no long-lived worker accumulates simulator memory across evaluations."""
    from hitl_pmp.step_protocol import FreshProcessEvaluations

    evaluations = FreshProcessEvaluations()
    futures = [evaluations.submit(fn=os.getpid) for _ in range(2)]
    evaluations.finish()
    pids = [future.result() for future in futures]
    assert len(set(pids)) == 2
    assert os.getpid() not in pids


def test_aborting_terminates_the_running_evaluation_and_cancels_the_queue():
    """Finding 4, against the real queue: abort returns promptly, the queued evaluation is
    cancelled, and the running one's process is terminated rather than awaited."""
    from hitl_pmp.step_protocol import FreshProcessEvaluations

    evaluations = FreshProcessEvaluations()
    running = evaluations.submit(fn=functools.partial(time.sleep, 120))
    queued = evaluations.submit(fn=functools.partial(time.sleep, 120))
    deadline = time.monotonic() + 60
    while not evaluations.has_running_process() and time.monotonic() < deadline:
        time.sleep(0.05)
    started = time.monotonic()
    evaluations.abort()
    assert time.monotonic() - started < 10
    assert queued.cancelled()
    with pytest.raises(BrokenProcessPool):
        running.result(timeout=30)


def test_early_stop_requires_three_consecutive_complete_distinct_perfect_sweeps():
    from hitl_pmp.step_protocol import EvaluationStopping

    def perfect(step):  # noqa: PLR0917 -- test record builder
        return dict(practice_steps=step, num_solved=10, num_total=10, complete=True)

    assert EvaluationStopping.reached(
        records=[perfect(1700), perfect(3400), perfect(5100)], num_tasks=10
    )
    assert not EvaluationStopping.reached(
        records=[perfect(1700), perfect(1700), perfect(3400)], num_tasks=10
    )
    assert not EvaluationStopping.reached(
        records=[
            perfect(1700),
            dict(practice_steps=3400, complete=False),
            perfect(5100),
            perfect(6800),
        ],
        num_tasks=10,
    )
    assert not EvaluationStopping.reached(
        records=[
            perfect(1700),
            dict(practice_steps=3400, num_solved=9, num_total=10),
            perfect(5100),
        ],
        num_tasks=10,
    )
    assert not EvaluationStopping.reached(
        records=[dict(practice_steps=s, num_solved=1, num_total=1) for s in range(3)],
        num_tasks=10,
    )


def test_perfect_means_every_configured_task_not_ten():
    from hitl_pmp.step_protocol import EvaluationStopping

    def sweep(step, solved, total):  # noqa: PLR0917 -- test record builder
        return dict(practice_steps=step, num_solved=solved, num_total=total, complete=True)

    three_of_three = [sweep(s, 3, 3) for s in (15000, 30000, 45000)]
    assert EvaluationStopping.reached(records=three_of_three, num_tasks=3)
    assert not EvaluationStopping.reached(records=three_of_three, num_tasks=10)
    ten_of_ten = [sweep(s, 10, 10) for s in (15000, 30000, 45000)]
    assert not EvaluationStopping.reached(records=ten_of_ten, num_tasks=3)


def test_early_stop_waits_for_preceding_pending_evaluations():
    from concurrent.futures import Future

    from hitl_pmp.step_protocol import EvaluationStopping

    pending = Future()
    later = Future()
    later.set_result(dict(num_solved=10, num_total=10, complete=True))
    futures = [(dict(practice_steps=1700), pending), (dict(practice_steps=3400), later)]
    assert EvaluationStopping.completed(futures=futures) == []
    pending.set_exception(RuntimeError("evaluation failed"))
    records = EvaluationStopping.completed(futures=futures)
    assert not records[0]["complete"]
    assert records[1]["num_solved"] == 10
    assert not EvaluationStopping.reached(records=records, num_tasks=10)


def test_an_empty_practice_session_ends_the_run_only_when_opted_in():
    from hitl_pmp.step_protocol import StepPracticeRunner

    assert StepPracticeRunner.session_endpoint(actions=0, stop_on_no_practice_action=False) is None
    assert (
        StepPracticeRunner.session_endpoint(actions=0, stop_on_no_practice_action=True)
        == "no_practice_action"
    )
    assert StepPracticeRunner.session_endpoint(actions=3, stop_on_no_practice_action=True) is None


def test_the_no_practice_action_stop_is_off_unless_requested():
    import argparse

    from hitl_pmp.cli import Cli

    parser = argparse.ArgumentParser()
    Cli.add_global_arguments(parser=parser)
    required = ["--env", "tossing3d", "--method", "pomdp"]
    assert parser.parse_args(required).stop_on_no_practice_action is False
    assert parser.parse_args([*required, "--stop-on-no-practice-action"]).stop_on_no_practice_action
    assert parser.parse_args(required).stop_after_perfect_evaluations == 0


def test_the_environment_supplies_the_cost_observation(*, monkeypatch, tmp_path):
    import argparse
    import json
    import sys
    import types

    from hitl_pmp.step_protocol import StepPracticeRunner

    module = types.ModuleType("cost_observation_from_env")
    module.price = lambda **kw: kw["context"]["observation"]["price"]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    config = tmp_path / "costs.json"
    config.write_text(json.dumps({"robot_step": {"function": "cost_observation_from_env:price"}}))
    configured = []

    class Method:
        def configure_practice_accounting(self, *, accounting):
            configured.append(accounting)

    class Env:
        price = 4

        def cost_observation(self):
            return {"price": self.price}

    env = Env()
    accounting = StepPracticeRunner.configure_accounting(
        args=argparse.Namespace(practice_cost_config=config, human_skill_steps=1),
        method=Method(),
        problem=types.SimpleNamespace(env=env),
        output=tmp_path,
    )
    assert configured == [accounting]
    assert accounting.robot_charge().cost == 4
    env.price = 6
    assert accounting.robot_charge().cost == 6
    assert (tmp_path / "practice_costs.json").exists()


def test_without_a_cost_file_there_is_no_accounting(*, tmp_path):
    import argparse

    from hitl_pmp.step_protocol import StepPracticeRunner

    args = argparse.Namespace(practice_cost_config=None, human_skill_steps=1)
    assert (
        StepPracticeRunner.configure_accounting(
            args=args, method=None, problem=None, output=tmp_path
        )
        is None
    )


def test_a_domain_without_a_cost_observation_supplies_none():
    assert LightSwitchEnvironment().cost_observation() is None


def test_a_cost_file_with_human_skill_steps_is_refused_before_anything_is_built(*, tmp_path):
    from hitl_pmp.step_protocol import StepProtocolValidation

    args = _step_args(
        output_dir=tmp_path / "run",
        budget=10,
        interval=5,
        human_steps=2,
        practice_cost_config=tmp_path / "costs.json",
    )
    with pytest.raises(ValueError, match="human durations in the cost file"):
        StepProtocolValidation.check(args=args)


class PerfectEvaluations(FakeEvaluations):
    def submit(self, *, fn, **kwargs) -> Future:
        future: Future = Future()
        future.set_result({"complete": True, "num_solved": 10, "num_total": 10})
        self.futures.append(future)
        return future


def _endpoint(*, output) -> str:
    (end,) = [e for e in _events(output=output) if e["event"] == "practice_end"]
    return end["reason"]


@pytest.mark.parametrize(
    ("opt_in", "endpoint"),
    [(False, "practice_step_budget"), (True, "no_practice_action")],
)
def test_an_empty_session_ends_the_run_only_with_the_opt_in_flag(*, tmp_path, opt_in, endpoint):
    from hitl_pmp.core.method.method import InteractionComplete

    output = tmp_path / "run"
    StepPracticeRunner.run(
        environment=FakeStepProtocol,
        args=_step_args(output_dir=output, budget=6, interval=3, stop_on_no_practice_action=opt_in),
        method=_scripted_method(script=[2, InteractionComplete(), InteractionComplete()]),
        problem=FakeProblem(),
        evaluations=FakeEvaluations(),
    )
    assert _endpoint(output=output) == endpoint
    protocol = json.loads((output / "step_protocol.json").read_text())
    assert protocol["stop_on_no_practice_action"] is opt_in


def test_perfect_evaluations_stop_practice_only_when_requested(*, tmp_path):
    for patience, endpoint in [(0, "practice_step_budget"), (3, "three_perfect_evaluations")]:
        output = tmp_path / f"run{patience}"
        StepPracticeRunner.run(
            environment=FakeStepProtocol,
            args=_step_args(
                output_dir=output, budget=40, interval=2, stop_after_perfect_evaluations=patience
            ),
            method=_scripted_method(script=[]),
            problem=FakeProblem(),
            evaluations=PerfectEvaluations(),
        )
        assert _endpoint(output=output) == endpoint
    # Measurements at 0, 2 and 4 are the three perfect ones; the session sees it before
    # its next action and stops well short of the budget.
    assert _measured(output=tmp_path / "run3")[:3] == [0, 2, 4]
    assert _measured(output=tmp_path / "run3")[-1] < 40


def _stop_events(*, output) -> list[dict]:
    """The `stop` events of a run, in order."""
    return [e for e in _events(output=output) if e["event"] == "stop"]


def test_a_stop_is_a_one_step_skill_that_always_succeeds(*, tmp_path):
    """STOP is a skill with competence fixed at 1: it costs one counted step, is recorded
    like any other action, and ends its session."""
    from hitl_pmp.core.method.method import InteractionComplete

    output = tmp_path / "run"
    StepPracticeRunner.run(
        environment=FakeStepProtocol,
        args=_step_args(output_dir=output, budget=6, interval=100),
        method=_scripted_method(script=[2, InteractionComplete(planner_stop=True)]),
        problem=FakeProblem(),
        evaluations=FakeEvaluations(),
        recorder=RunRecordWriter(
            output_path=output / "recording.jsonl", header=RecordingHeader(environment="fake")
        ),
    )
    (stop,) = _stop_events(output=output)[:1]
    assert (stop["start_step"], stop["end_step"]) == (2, 3)
    assert stop["success"] is True
    starts = [e["practice_steps"] for e in _events(output=output) if e["event"] == "session_start"]
    assert starts[:2] == [0, 3]
    kinds = [json.loads(line) for line in (output / "recording.jsonl").read_text().splitlines()]
    labels = [entry.get("label") for entry in kinds if entry.get("kind") == "action"]
    assert "STOP" in labels


def test_ten_stop_only_sessions_in_a_row_end_the_run(*, tmp_path):
    from hitl_pmp.core.method.method import InteractionComplete

    output = tmp_path / "run"
    StepPracticeRunner.run(
        environment=FakeStepProtocol,
        args=_step_args(output_dir=output, budget=1000, interval=100),
        method=_scripted_method(script=[1] + [InteractionComplete(planner_stop=True)] * 12),
        problem=FakeProblem(),
        evaluations=FakeEvaluations(),
    )
    assert _endpoint(output=output) == "repeated_stop"
    # One session with the action and its stop, then ten STOP-only sessions.
    assert len(_stop_events(output=output)) == 11
    assert _measured(output=output)[-1] == 1 + 11
    assert json.loads((output / "status.json").read_text())["phase"] == "complete"


def test_a_session_with_an_action_resets_the_stop_streak(*, tmp_path):
    from hitl_pmp.core.method.method import InteractionComplete

    stop = InteractionComplete(planner_stop=True)
    output = tmp_path / "run"
    StepPracticeRunner.run(
        environment=FakeStepProtocol,
        args=_step_args(output_dir=output, budget=1000, interval=100),
        method=_scripted_method(script=[stop] * 9 + [1, stop] + [stop] * 10),
        problem=FakeProblem(),
        evaluations=FakeEvaluations(),
    )
    assert _endpoint(output=output) == "repeated_stop"
    assert len(_stop_events(output=output)) == 20


def _session_starts(*, output) -> list[int]:
    return [e["practice_steps"] for e in _events(output=output) if e["event"] == "session_start"]


def test_sessions_of_refused_skills_end_the_run_as_repeated_zero_step(*, tmp_path):
    """A skill refused at planning executes 0 control steps; a session made only of
    them changes nothing, so K of them in a row end practice."""
    output = tmp_path / "run"
    StepPracticeRunner.run(
        environment=FakeStepProtocol,
        args=_step_args(
            output_dir=output,
            budget=1000,
            interval=100,
            max_steps_per_interaction=3,
            repeated_zero_step_limit=5,
        ),
        method=_scripted_method(script=[2, 0, 0] + [0] * 30),
        problem=FakeProblem(),
        evaluations=FakeEvaluations(),
    )
    end = [e for e in _events(output=output) if e["event"] == "practice_end"][0]
    assert end["reason"] == "repeated_zero_step"
    # Session 0 took 2 control steps; sessions 1-5 took none, and the fifth ends it.
    assert end["practice_steps"] == 2
    assert len(_session_starts(output=output)) == 6
    protocol = json.loads((output / "step_protocol.json").read_text())
    assert protocol["repeated_zero_step_limit"] == 5


def test_refusal_then_stop_sessions_count_as_zero_step_despite_the_stop_charge(*, tmp_path):
    """determinized-always-ask s3's livelock: sessions of refusals alternate with "one
    refusal, then STOP". The STOP's own 1-step charge is not a control step."""
    from hitl_pmp.core.method.method import InteractionComplete

    stop = InteractionComplete(planner_stop=True)
    output = tmp_path / "run"
    StepPracticeRunner.run(
        environment=FakeStepProtocol,
        args=_step_args(
            output_dir=output,
            budget=1000,
            interval=100,
            max_steps_per_interaction=3,
            repeated_zero_step_limit=4,
        ),
        method=_scripted_method(script=[0, 0, 0, 0, stop] * 2 + [0, 0, 0] + [0] * 30),
        problem=FakeProblem(),
        evaluations=FakeEvaluations(),
    )
    assert _endpoint(output=output) == "repeated_zero_step"
    assert len(_session_starts(output=output)) == 4
    assert len(_stop_events(output=output)) == 2
    assert _measured(output=output)[-1] == 2  # two STOP charges, no control steps


def test_a_session_with_a_control_step_resets_the_zero_step_streak(*, tmp_path):
    output = tmp_path / "run"
    StepPracticeRunner.run(
        environment=FakeStepProtocol,
        args=_step_args(
            output_dir=output,
            budget=1000,
            interval=100,
            max_steps_per_interaction=2,
            repeated_zero_step_limit=3,
        ),
        method=_scripted_method(script=[0, 0] * 2 + [0, 1] + [0, 0] * 10),
        problem=FakeProblem(),
        evaluations=FakeEvaluations(),
    )
    assert _endpoint(output=output) == "repeated_zero_step"
    # Two zero-step sessions, one with a step, then three more zero-step sessions.
    assert len(_session_starts(output=output)) == 6


def test_a_session_with_a_human_reset_resets_the_zero_step_streak(*, tmp_path):
    """A human reset changes the scene, so it counts as progress."""
    reset = HumanCubeBinResetRequested(cost=1.0, destination="robot_side")
    output = tmp_path / "run"
    StepPracticeRunner.run(
        environment=FakeStepProtocol,
        args=_step_args(
            output_dir=output,
            budget=1000,
            interval=100,
            max_steps_per_interaction=2,
            repeated_zero_step_limit=3,
        ),
        method=_scripted_method(script=[0, 0] * 2 + [0, reset] + [0, 0] * 10),
        problem=FakeProblem(),
        evaluations=FakeEvaluations(),
    )
    assert _endpoint(output=output) == "repeated_zero_step"
    assert len(_session_starts(output=output)) == 6


def test_stop_only_sessions_are_left_to_repeated_stop(*, tmp_path):
    """A STOP-only session neither counts toward the zero-step streak nor resets it: a
    planner that chooses to stop is `repeated_stop`'s to end, at its own limit."""
    from hitl_pmp.core.method.method import InteractionComplete

    stop = InteractionComplete(planner_stop=True)
    output = tmp_path / "run"
    StepPracticeRunner.run(
        environment=FakeStepProtocol,
        args=_step_args(
            output_dir=output,
            budget=1000,
            interval=100,
            max_steps_per_interaction=1,
            repeated_zero_step_limit=2,
        ),
        method=_scripted_method(script=[stop] * 12),
        problem=FakeProblem(),
        evaluations=FakeEvaluations(),
    )
    assert _endpoint(output=output) == "repeated_stop"
    assert len(_stop_events(output=output)) == 10

    # Nor does one reset the streak: zero-step, STOP-only, zero-step ends it at K=2.
    output = tmp_path / "interleaved"
    StepPracticeRunner.run(
        environment=FakeStepProtocol,
        args=_step_args(
            output_dir=output,
            budget=1000,
            interval=100,
            max_steps_per_interaction=1,
            repeated_zero_step_limit=2,
        ),
        method=_scripted_method(script=[0, stop, 0] + [1] * 5),
        problem=FakeProblem(),
        evaluations=FakeEvaluations(),
    )
    assert _endpoint(output=output) == "repeated_zero_step"
    assert len(_session_starts(output=output)) == 3


def test_the_zero_step_limit_defaults_to_five_on_the_cli():
    from hitl_pmp.cli import Cli

    args = Cli.parse_args(argv=["--env", "lightswitch", "--method", "ees"])
    assert args.repeated_zero_step_limit == 5


def test_a_stop_is_charged_one_robot_step_through_the_accounting():
    from hitl_pmp.core.practice_costs import PracticeAccounting, PracticeCosts

    accounting = PracticeAccounting(costs=PracticeCosts())
    clock = PracticeClock(budget=10, interval=5, accounting=accounting, on_measure=lambda: None)
    price = accounting.robot_charge().cost
    clock.stop_invoked()
    assert clock.steps == 1
    assert accounting.total_cost == pytest.approx(price)


_ALWAYS_ASK_PLANNERS = [
    {"method": "ees"},
    {"method": "pomdp", "pomdp_solver": "expectimax"},
    {"method": "pomdp", "pomdp_solver": "determinized_astar"},
]


@pytest.mark.parametrize("planner", _ALWAYS_ASK_PLANNERS)
def test_an_always_ask_run_whose_cost_file_cannot_price_its_reset_is_refused_up_front(
    *, tmp_path, planner
):
    """Its first full reset would otherwise raise `Unknown human skill` mid-run, after
    the first sweep -- as it does on the shelf's canonical cost file."""
    from hitl_pmp.step_protocol import StepProtocolValidation

    costs = tmp_path / "costs.json"
    costs.write_text(json.dumps({"human_skills": {"reset_cubes": {}, "reset_wiper": {}}}))
    args = _step_args(
        output_dir=tmp_path / "run",
        budget=10,
        interval=5,
        practice_cost_config=costs,
        always_ask_reset=True,
        **planner,
    )
    with pytest.raises(ValueError, match="reset_to_sampled_start"):
        StepProtocolValidation.check(args=args)
    # Without the wrapper nothing requests it, so the same file stays valid.
    StepProtocolValidation.check(
        args=argparse.Namespace(**{**vars(args), "always_ask_reset": False})
    )


@pytest.mark.parametrize("planner", _ALWAYS_ASK_PLANNERS)
@pytest.mark.parametrize(
    "name",
    [
        "normalized_practice_costs_sampled_start.json",
        "sweepoffshelf3d_practice_costs_sampled_start.json",
    ],
)
def test_an_always_ask_run_whose_cost_file_prices_its_reset_is_accepted(
    *, tmp_path, name: str, planner
):
    from hitl_pmp.step_protocol import StepProtocolValidation

    configs = Path(__file__).resolve().parents[1] / "configs"
    StepProtocolValidation.check(
        args=_step_args(
            output_dir=tmp_path / "run",
            budget=10,
            interval=5,
            practice_cost_config=configs / name,
            always_ask_reset=True,
            **planner,
        )
    )


def test_a_policy_snapshot_is_identified_by_its_files_bytes_under_the_new_key() -> None:
    raw = b"pickled policy"
    record = PolicyFileHash.record(raw=raw)
    assert record == {"policy_file_sha256": hashlib.sha256(raw).hexdigest()}
    assert "policy_sha256" not in record


def test_a_record_written_before_the_rename_still_reads() -> None:
    digest = hashlib.sha256(b"x").hexdigest()
    assert PolicyFileHash.read(record={"policy_sha256": digest}) == digest
    assert PolicyFileHash.read(record={"policy_file_sha256": digest}) == digest
    assert PolicyFileHash.read(record={}) is None


def test_a_simulated_step_zero_measurement_takes_no_checkpoint_and_no_restore(*, tmp_path):
    """Evaluations spawned beside practice never use its scene: the step-0 measurement
    is followed straight by practice, with no checkpoint and nothing put back."""
    output = tmp_path / "run"
    StepPracticeRunner.run(
        environment=FakeStepProtocol,
        args=_step_args(output_dir=output, budget=3, interval=100),
        method=_scripted_method(script=[1, 1, 1]),
        problem=FakeProblem(),
        evaluations=FakeEvaluations(),
    )
    kinds = [e["event"] for e in _events(output=output)]
    assert kinds[0] == "measurement" and kinds[1] == "session_start"
    assert "practice_scene_restored" not in kinds
    assert _measured(output=output)[0] == 0
    assert not (output / CHECKPOINT_FILENAME).exists()
