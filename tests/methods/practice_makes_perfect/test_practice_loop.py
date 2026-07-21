from collections.abc import Iterator
from typing import ClassVar

import numpy as np
import pytest

from hitl_pmp.core.method.method import Method
from hitl_pmp.core.method.types import GroundSkill, LabeledAction, Policy, Rollout, SetupCommand
from hitl_pmp.core.metrics.metrics import Metrics
from hitl_pmp.core.problem.environment.environment import Environment
from hitl_pmp.core.problem.environment.types import Action, Object, State, Type
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.core.problem.tasks.tasks import Tasks
from hitl_pmp.core.problem.tasks.types import Goal, Task
from hitl_pmp.core.renderer.renderer import Renderer
from hitl_pmp.methods.practice_makes_perfect.practice_loop import PracticeLoop

_BLOCK = Type(name="block", feature_names=("x",))
_OBJ = Object(name="thing", type=_BLOCK)


def _state(*, x: float) -> State:
    return State(data={_OBJ: np.array([x])})


class _FakeEnv(Environment):
    hard_reset_count: ClassVar[int] = 0
    # x observed immediately before each take_action call, in call order -- lets
    # tests check what state an interaction period actually started from.
    pre_action_xs: ClassVar[list[float]] = []

    @staticmethod
    def take_action(*, action: Action) -> State:
        # Goes through the inherited get_current_state/set_state (not a direct
        # _FakeEnv.current_state assignment), since Environment.current_state is
        # meant to live on the base class itself regardless of which concrete
        # subclass is active -- matches LightSwitchEnvironment's own pattern.
        current_x = float(_FakeEnv.get_current_state()[_OBJ][0])
        _FakeEnv.pre_action_xs.append(current_x)
        next_state = _state(x=current_x + 1.0)
        _FakeEnv.set_state(state=next_state)
        return next_state

    @staticmethod
    def get_valid_actions() -> list[Action]:
        return []

    @staticmethod
    def hard_reset() -> None:
        _FakeEnv.hard_reset_count += 1
        _FakeEnv.set_state(state=_state(x=0.0))


class _FakeTasks(Tasks):
    train_task_count: ClassVar[int] = 0
    test_task_count: ClassVar[int] = 0

    @staticmethod
    def sample_train_task() -> Task:
        _FakeTasks.train_task_count += 1
        return Task(initial_state=_state(x=0.0), goal=Goal(atoms=frozenset()))

    @staticmethod
    def sample_test_task() -> Task:
        _FakeTasks.test_task_count += 1
        return Task(initial_state=_state(x=0.0), goal=Goal(atoms=frozenset()))


class _FakeRenderer(Renderer):
    @staticmethod
    def render_frame(*, state: State, label: str | None = None) -> np.ndarray:
        del state, label
        return np.zeros((1, 1, 3), dtype=np.uint8)


class _FakeProblem(Problem):
    run_task_episode_calls: ClassVar[int] = 0
    # renderer arguments this fake was called with, in call order -- lets tests
    # check exactly which run_task_episode calls actually rendered.
    renderer_calls: ClassVar[list[bool]] = []

    @staticmethod
    def run_task_episode(
        *, task: Task, policy: Policy, renderer: type[Renderer] | None = None
    ) -> tuple[bool, list[np.ndarray]]:
        _FakeProblem.run_task_episode_calls += 1
        _FakeProblem.renderer_calls.append(renderer is not None)
        # Mirrors the real per-domain override (e.g. LightSwitchProblem's own
        # run_task_episode): resets env state to the task's initial_state before
        # running -- this is exactly the behavior that makes PracticeLoop
        # NOT reset-free end to end, only within one interaction period.
        _FakeEnv.set_state(state=task.initial_state)
        policy(_FakeProblem.get_current_state())  # exercised, but the fake doesn't need its result
        frames = [renderer.render_frame(state=_FakeEnv.get_current_state())] if renderer else []
        return True, frames


class _FakeMethod(Method):
    policy_call_count: ClassVar[int] = 0

    @staticmethod
    def reset_environment(*, start_state: State) -> bool:
        raise NotImplementedError

    @staticmethod
    def get_task_policy(*, task: Task) -> Policy:
        del task
        # Policy is a positional Callable[[State], LabeledAction] per its interface
        # contract -- this lambda just adapts that into a call to _get_action's
        # keyword-only signature, same pattern as RandomSkillsMethod's own
        # get_task_policy.
        return lambda state: _FakeMethod._get_action(state=state)  # noqa: E731

    @staticmethod
    def _get_action(*, state: State) -> LabeledAction:
        del state
        _FakeMethod.policy_call_count += 1
        return LabeledAction(action=np.array([0.0]), label="fake")

    @staticmethod
    def generate_train_task(*, tbd_inputs: object) -> Task:
        raise NotImplementedError

    @staticmethod
    def execute_setup_command(*, setup_command: SetupCommand) -> None:
        raise NotImplementedError

    @staticmethod
    def execute_skill(*, skill: GroundSkill) -> Rollout:
        raise NotImplementedError

    @staticmethod
    def improve_skill_parameters(*, skill: GroundSkill, rollout: Rollout) -> None:
        raise NotImplementedError


@pytest.fixture(autouse=True)
def _wire_fakes() -> Iterator[None]:
    # Problem.env/Problem.tasks are shared, un-defaulted ClassVars on the base
    # Problem class -- Problem's own inherited facade methods (hard_reset,
    # sample_train_task, ...) hardcode references to Problem.env/Problem.tasks
    # specifically (never cls, matching this project's static-method-container
    # convention), so PracticeLoop genuinely needs them wired here, not
    # just on whichever concrete Problem subclass is passed in. Snapshot/restore
    # around the test so this doesn't leak into other test files.
    original_env = getattr(Problem, "env", None)
    original_tasks = getattr(Problem, "tasks", None)
    Problem.env = _FakeEnv
    Problem.tasks = _FakeTasks
    _FakeEnv.hard_reset_count = 0
    _FakeEnv.pre_action_xs = []
    _FakeTasks.train_task_count = 0
    _FakeTasks.test_task_count = 0
    _FakeProblem.run_task_episode_calls = 0
    _FakeProblem.renderer_calls = []
    _FakeMethod.policy_call_count = 0
    Metrics.reset()
    try:
        yield
    finally:
        if original_env is not None:
            Problem.env = original_env
        if original_tasks is not None:
            Problem.tasks = original_tasks


def test_run_hard_resets_exactly_once_before_the_first_evaluation() -> None:
    PracticeLoop.run(
        problem=_FakeProblem,
        method=_FakeMethod,
        metrics=Metrics,
        num_cycles=2,
        max_steps_per_interaction=3,
        num_test_tasks=1,
    )
    assert _FakeEnv.hard_reset_count == 1


def test_run_evaluates_once_before_any_cycle_and_once_after_each_cycle() -> None:
    PracticeLoop.run(
        problem=_FakeProblem,
        method=_FakeMethod,
        metrics=Metrics,
        num_cycles=3,
        max_steps_per_interaction=2,
        num_test_tasks=1,
    )
    # 1 initial evaluation + 1 per cycle = 4 total.
    assert len(Metrics.evaluations) == 4


def test_run_records_num_online_transitions_at_each_evaluation_checkpoint() -> None:
    PracticeLoop.run(
        problem=_FakeProblem,
        method=_FakeMethod,
        metrics=Metrics,
        num_cycles=3,
        max_steps_per_interaction=5,
        num_test_tasks=1,
    )
    transitions_recorded = [transitions for transitions, _, _ in Metrics.evaluations]
    assert transitions_recorded == [0, 5, 10, 15]


def test_run_evaluates_exactly_num_test_tasks_per_checkpoint() -> None:
    PracticeLoop.run(
        problem=_FakeProblem,
        method=_FakeMethod,
        metrics=Metrics,
        num_cycles=1,
        max_steps_per_interaction=1,
        num_test_tasks=4,
    )
    assert all(num_total == 4 for _, _, num_total in Metrics.evaluations)
    assert _FakeProblem.run_task_episode_calls == 4 * 2  # initial + 1 cycle's checkpoint


def test_run_calls_on_cycle_end_once_per_cycle() -> None:
    calls: list[None] = []
    PracticeLoop.run(
        problem=_FakeProblem,
        method=_FakeMethod,
        metrics=Metrics,
        num_cycles=3,
        max_steps_per_interaction=1,
        num_test_tasks=1,
        on_cycle_end=lambda: calls.append(None),
    )
    assert len(calls) == 3


def test_run_stays_reset_free_within_one_interaction_period() -> None:
    """Within a single interaction period (no evaluation sweep in between), state
    keeps accumulating across steps rather than snapping back -- this is the
    narrower claim PracticeLoop's docstring actually makes."""
    PracticeLoop.run(
        problem=_FakeProblem,
        method=_FakeMethod,
        metrics=Metrics,
        num_cycles=1,
        max_steps_per_interaction=3,
        num_test_tasks=1,
    )
    # Steps within the one interaction period run: x=0 -> 1 -> 2 -> 3, so
    # pre_action_xs (recorded before each increment) for that period is [0, 1, 2].
    assert _FakeEnv.pre_action_xs[:3] == [0.0, 1.0, 2.0]


def test_run_interaction_period_resumes_from_the_prior_evaluations_reset_state() -> None:
    """NOT reset-free end to end: an evaluation sweep resets env state via
    run_task_episode's own env.set_state(state=task.initial_state) call, so
    cycle N+1's interaction period does not resume from wherever cycle N's own
    training steps left off -- it resumes from wherever the intervening
    evaluation sweep's last episode reset state to (here, always
    task.initial_state's x=0.0, since _FakeTasks always returns x=0.0)."""
    PracticeLoop.run(
        problem=_FakeProblem,
        method=_FakeMethod,
        metrics=Metrics,
        num_cycles=2,
        max_steps_per_interaction=3,
        num_test_tasks=1,
    )
    # If training resumed from where the previous period's own steps left off
    # (genuinely reset-free end to end), cycle 2's period would start at x=3.0.
    # Instead it restarts at x=0.0, because the evaluation sweep in between reset
    # it via run_task_episode.
    first_step_of_each_period = _FakeEnv.pre_action_xs[0::3]
    assert first_step_of_each_period == [0.0, 0.0]


def test_run_without_a_num_cycles_zero_still_runs_the_initial_evaluation() -> None:
    PracticeLoop.run(
        problem=_FakeProblem,
        method=_FakeMethod,
        metrics=Metrics,
        num_cycles=0,
        max_steps_per_interaction=5,
        num_test_tasks=2,
    )
    assert Metrics.evaluations == [(0, 2, 2)]


def test_run_without_a_renderer_returns_no_frames() -> None:
    frames = PracticeLoop.run(
        problem=_FakeProblem,
        method=_FakeMethod,
        metrics=Metrics,
        num_cycles=2,
        max_steps_per_interaction=1,
        num_test_tasks=2,
    )
    assert frames == []
    assert _FakeProblem.renderer_calls == [False] * (2 * 3)  # 3 evaluations x 2 test tasks


def test_run_with_a_renderer_and_zero_cycles_renders_the_initial_evaluation() -> None:
    frames = PracticeLoop.run(
        problem=_FakeProblem,
        method=_FakeMethod,
        metrics=Metrics,
        num_cycles=0,
        max_steps_per_interaction=1,
        num_test_tasks=3,
        renderer=_FakeRenderer,
    )
    assert len(frames) == 1
    # Only the first test task of the (sole) evaluation sweep renders.
    assert _FakeProblem.renderer_calls == [True, False, False]


def test_run_with_a_renderer_renders_only_the_last_evaluation_sweeps_first_task() -> None:
    PracticeLoop.run(
        problem=_FakeProblem,
        method=_FakeMethod,
        metrics=Metrics,
        num_cycles=2,
        max_steps_per_interaction=1,
        num_test_tasks=2,
        renderer=_FakeRenderer,
    )
    # 3 evaluation sweeps of 2 test tasks each: initial, after cycle 1, after
    # cycle 2. Only the very first task of the very last sweep renders.
    assert _FakeProblem.renderer_calls == [
        False,
        False,  # initial evaluation
        False,
        False,  # after cycle 1
        True,
        False,  # after cycle 2 (the last sweep)
    ]
