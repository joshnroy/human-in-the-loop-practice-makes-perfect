import argparse
import json
import pickle
from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ConfigDict, Field, PrivateAttr

from hitl_pmp import step_protocol
from hitl_pmp.core.control_steps import ControlStepLimitReached
from hitl_pmp.core.method.method import Method
from hitl_pmp.core.method.types import (
    EpisodeTrace,
    GroundSkill,
    LabeledAction,
    Policy,
    Rollout,
    SetupCommand,
)
from hitl_pmp.core.metrics.metrics import Metrics
from hitl_pmp.core.problem.environment.environment import Environment
from hitl_pmp.core.problem.environment.types import Action, Object, State, Type
from hitl_pmp.core.problem.human.human import HumanOracle
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.core.problem.tasks.tasks import Tasks
from hitl_pmp.core.problem.tasks.types import Goal, Task
from hitl_pmp.core.renderer.renderer import Renderer
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.skill_provider import LightSwitchSkillProvider
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod
from hitl_pmp.practice_loop import PracticeResetPolicy
from hitl_pmp.step_protocol import DeploymentSnapshot, PracticeClock, StepPracticeRunner


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
        on_measure=lambda: seen.append((clock.steps, ledger.total_cost)),
    )
    clock.robot_active = True
    clock.before_robot_step()
    clock.after_robot_step()
    clock.robot_active = False
    charge = clock.before_human(skill="near")
    clock.human_invoked(charge=charge)
    assert seen == [(4, 9)]
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


def test_early_stop_requires_three_consecutive_complete_distinct_perfect_sweeps():
    from hitl_pmp.step_protocol import EvaluationStopping

    def perfect(step):  # noqa: PLR0917 -- test record builder
        return dict(practice_steps=step, num_solved=10, num_total=10, complete=True)

    assert EvaluationStopping.reached(records=[perfect(1700), perfect(3400), perfect(5100)])
    assert not EvaluationStopping.reached(records=[perfect(1700), perfect(1700), perfect(3400)])
    assert not EvaluationStopping.reached(
        records=[
            perfect(1700),
            dict(practice_steps=3400, complete=False),
            perfect(5100),
            perfect(6800),
        ]
    )
    assert not EvaluationStopping.reached(
        records=[
            perfect(1700),
            dict(practice_steps=3400, num_solved=9, num_total=10),
            perfect(5100),
        ]
    )
    assert not EvaluationStopping.reached(
        records=[dict(practice_steps=s, num_solved=1, num_total=1) for s in range(3)]
    )


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
    assert not EvaluationStopping.reached(records=records)


# --------------------------------------------------------------- practice schedule
# A fake domain just rich enough for StepPracticeRunner.run: every robot action burns
# a fixed number of control steps through the clock's observers, no KINDER involved.

_SCHEDULE_TYPE = Type(name="thing", feature_names=("x",))
_SCHEDULE_OBJ = Object(name="thing", type=_SCHEDULE_TYPE)
_STEPS_PER_ACTION = 8_500


def _schedule_state(*, x: float) -> State:
    return State(data={_SCHEDULE_OBJ: np.array([x])})


class _FakeBackend:
    def __init__(self) -> None:
        self.before: Callable[[], None] | None = None
        self.after: Callable[[], None] | None = None

    def set_control_step_observers(
        self, *, before: Callable[[], None] | None, after: Callable[[], None] | None
    ) -> None:
        self.before, self.after = before, after

    @staticmethod
    def drain_damage_events() -> list[dict[str, Any]]:
        return []


class _ScheduleEnv(Environment):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _backend: _FakeBackend = PrivateAttr(default_factory=_FakeBackend)

    def backend(self) -> _FakeBackend:
        return self._backend

    def take_action(self, *, action: Action) -> State:
        del action
        for _ in range(_STEPS_PER_ACTION):
            assert self._backend.before is not None and self._backend.after is not None
            self._backend.before()
            self._backend.after()
        state = _schedule_state(x=float(self.get_current_state()[_SCHEDULE_OBJ][0]) + 1.0)
        self.set_state(state=state)
        return state

    def get_valid_actions(self) -> list[Action]:
        return []

    def noop_action(self) -> Action:
        return np.zeros(1)

    def hard_reset(self) -> None:
        self.set_state(state=_schedule_state(x=0.0))

    def reset_movables(self) -> bool:
        return True

    def _log_skill_ticks(self, *, action: Action) -> None:
        del action

    @staticmethod
    def last_skill_error() -> str | None:
        return None


class _ScheduleTasks(Tasks):
    def sample_train_task(self) -> Task:
        raise NotImplementedError

    def sample_train_task_in_place(self) -> Task:
        return Task(initial_state=self.env.get_current_state(), goal=Goal(atoms=frozenset()))

    def sample_test_task(self) -> Task:
        raise NotImplementedError


class _ScheduleProblem(Problem):
    env: _ScheduleEnv
    tasks: _ScheduleTasks

    def run_task_episode(
        self, *, task: Task, policy: Policy, renderer: type[Renderer] | None = None
    ) -> tuple[bool, list[np.ndarray], EpisodeTrace]:
        raise NotImplementedError


class _PlainMethod(Method):
    """Overrides nothing the schedule touches: the ABC's default must carry it."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    policy_calls: int = 0

    def reset_environment(self, *, start_state: State) -> bool:
        raise NotImplementedError

    def get_task_policy(self, *, task: Task) -> Policy:
        del task
        return lambda state: self._act(state=state)  # noqa: E731

    def _act(self, *, state: State) -> LabeledAction:
        del state
        self.policy_calls += 1
        return LabeledAction(action=np.zeros(1), label="fake")

    # The runner's method contract is wider than the ABC: this one is EesMethod's.
    def observe_execution_cost(self, *, cost: float, steps: int, complete: bool) -> None:
        del cost, steps, complete

    def generate_train_task(self, *, tbd_inputs: object) -> Task:
        raise NotImplementedError

    def execute_setup_command(self, *, setup_command: SetupCommand) -> None:
        raise NotImplementedError

    def execute_skill(self, *, skill: GroundSkill) -> Rollout:
        raise NotImplementedError

    def improve_skill_parameters(self, *, skill: GroundSkill, rollout: Rollout) -> None:
        raise NotImplementedError


class _RecordingMethod(_PlainMethod):
    schedules: list[dict[str, Any]] = Field(default_factory=list)
    # Number of schedule calls seen at the moment of each policy call.
    schedules_before_policy: list[int] = Field(default_factory=list)
    on_policy: Callable[..., None] = lambda **_: None

    def observe_practice_schedule(
        self,
        *,
        step_budget: int,
        measurement_interval_steps: int,
        steps_used: int,
        model_budget_remaining_usd: float | None,
    ) -> None:
        self.schedules.append({
            "step_budget": step_budget,
            "measurement_interval_steps": measurement_interval_steps,
            "steps_used": steps_used,
            "model_budget_remaining_usd": model_budget_remaining_usd,
        })

    def _act(self, *, state: State) -> LabeledAction:
        self.schedules_before_policy.append(len(self.schedules))
        action = super()._act(state=state)
        self.on_policy(policy_calls=self.policy_calls)
        return action


class _SynchronousPool:
    """Stands in for the spawn pool; evaluations are not what these tests measure."""

    def __init__(self, **kwargs: Any) -> None:
        del kwargs

    @staticmethod
    def submit(fn: Any, **kwargs: Any) -> Future[dict[str, Any]]:  # noqa: PLR0917
        del fn, kwargs
        future: Future[dict[str, Any]] = Future()
        future.set_result({
            "num_solved": 0,
            "num_total": 1,
            "tasks": [{"task": 0, "solved": False}],
            "complete": True,
        })
        return future

    @staticmethod
    def shutdown(*, wait: bool) -> None:
        del wait


def _schedule_args(*, output: Path) -> argparse.Namespace:
    return argparse.Namespace(
        env="tossing3d",
        method="ees",
        seed=0,
        output_dir=str(output),
        defer_rendering=True,
        practice_reset_policy=PracticeResetPolicy.NEVER,
        practice_reset_interval=None,
        practice_cost_config=None,
        human_skill_steps=1,
        practice_step_budget=85_000,
        measurement_interval_steps=1_700,
        evaluation_control_steps=1,
        max_steps_per_interaction=2,
        num_cycles=None,
        stop_after_perfect_evaluations=0,
    )


def _run_schedule_protocol(
    *, method: _PlainMethod, output: Path, monkeypatch: pytest.MonkeyPatch, **run_kwargs: Any
) -> Metrics:
    monkeypatch.setattr(step_protocol, "ProcessPoolExecutor", _SynchronousPool)
    assert isinstance(method.env, _ScheduleEnv)
    problem = _ScheduleProblem(
        env=method.env, tasks=_ScheduleTasks(env=method.env), human=HumanOracle
    )
    return StepPracticeRunner.run(
        args=_schedule_args(output=output),
        method=method,
        problem=problem,
        encode_snapshot=lambda **_: b"snapshot",
        **run_kwargs,
    )


def test_schedule_is_observed_once_before_every_practice_decision(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    method = _RecordingMethod(env=_ScheduleEnv())
    metrics = _run_schedule_protocol(method=method, output=tmp_path, monkeypatch=monkeypatch)
    expected_decisions = 85_000 // _STEPS_PER_ACTION
    assert method.policy_calls == expected_decisions == 10
    assert len(method.schedules) == expected_decisions
    # Every decision saw its schedule first, and only one schedule per decision.
    assert method.schedules_before_policy == list(range(1, expected_decisions + 1))
    assert {s["step_budget"] for s in method.schedules} == {85_000}
    assert {s["measurement_interval_steps"] for s in method.schedules} == {1_700}
    assert {s["model_budget_remaining_usd"] for s in method.schedules} == {None}
    used = [s["steps_used"] for s in method.schedules]
    assert used == [i * _STEPS_PER_ACTION for i in range(expected_decisions)]
    assert json.loads((tmp_path / "status.json").read_text())["phase"] == "complete"
    assert metrics.practice_session_ends[0].action_limit == 2


def test_schedule_carries_the_live_remaining_model_budget(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hitl_pmp.agentic_states.budget import SharedBudget

    ledger = SharedBudget(path=tmp_path / "model_budget.json", limit=20, session="coding")
    ledger.record(cumulative=7.5)

    def spend(*, policy_calls: int) -> None:
        # Spend mid-run (after the third decision) so the hook must read the ledger
        # at each decision rather than once at start.
        if policy_calls == 3:
            SharedBudget(path=ledger.path, limit=20, session="classifier").record(cumulative=2.5)

    method = _RecordingMethod(env=_ScheduleEnv(), on_policy=spend)
    _run_schedule_protocol(
        method=method, output=tmp_path, monkeypatch=monkeypatch, model_budget=ledger
    )
    remaining = [s["model_budget_remaining_usd"] for s in method.schedules]
    assert remaining == [12.5] * 3 + [10.0] * 7
    assert all(isinstance(value, float) for value in remaining)


def test_schedule_accepts_the_in_memory_session_ledger(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hitl_pmp.hybrid_skills.artifacts import SessionBudget

    ledger = SessionBudget(limit=20)
    ledger.record(cumulative=4)
    method = _RecordingMethod(env=_ScheduleEnv())
    _run_schedule_protocol(
        method=method, output=tmp_path, monkeypatch=monkeypatch, model_budget=ledger
    )
    assert {s["model_budget_remaining_usd"] for s in method.schedules} == {16.0}


def test_a_method_that_ignores_the_schedule_still_runs(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    method = _PlainMethod(env=_ScheduleEnv())
    assert (
        method.observe_practice_schedule(
            step_budget=1,
            measurement_interval_steps=1,
            steps_used=0,
            model_budget_remaining_usd=None,
        )
        is None
    )
    _run_schedule_protocol(method=method, output=tmp_path, monkeypatch=monkeypatch)
    assert method.policy_calls == 10
    assert json.loads((tmp_path / "status.json").read_text())["phase"] == "complete"
