import numpy as np
from pydantic import ConfigDict, Field

from hitl_pmp.core.method.method import InteractionComplete, Method
from hitl_pmp.core.method.types import GroundSkill, LabeledAction, Policy, Rollout, SetupCommand
from hitl_pmp.core.metrics.metrics import Metrics
from hitl_pmp.core.problem.environment.environment import Environment
from hitl_pmp.core.problem.environment.types import Action, Object, State, Type
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.core.problem.tasks.tasks import Tasks
from hitl_pmp.core.problem.tasks.types import Goal, Task
from hitl_pmp.core.renderer.renderer import Renderer
from hitl_pmp.practice_loop import PracticeLoop

_BLOCK = Type(name="block", feature_names=("x",))
_OBJ = Object(name="thing", type=_BLOCK)


def _state(*, x: float) -> State:
    return State(data={_OBJ: np.array([x])})


class _EventLog:
    """A plain, non-pydantic container so _FakeProblem and _FakeMethod genuinely
    share ONE object. A `list[str]` field would not work: pydantic validates
    list-typed fields by *copying* them, so each model would silently end up with
    its own list and the interleaving assertion would be vacuous."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def record(self, *, event: str) -> None:
        self.events.append(event)


class _FakeEnv(Environment):
    hard_reset_count: int = 0
    # x observed immediately before each take_action call, in call order -- lets
    # tests check what state an interaction period actually started from.
    pre_action_xs: list[float] = Field(default_factory=list)

    def take_action(self, *, action: Action) -> State:
        del action
        # Goes through the inherited get_current_state/set_state (not a direct
        # self.current_state assignment), since Environment.current_state is
        # meant to live on the base class itself regardless of which concrete
        # subclass is active -- matches LightSwitchEnvironment's own pattern.
        current_x = float(self.get_current_state()[_OBJ][0])
        self.pre_action_xs.append(current_x)
        next_state = _state(x=current_x + 1.0)
        self.set_state(state=next_state)
        return next_state

    def get_valid_actions(self) -> list[Action]:
        return []

    def hard_reset(self) -> None:
        self.hard_reset_count += 1
        self.set_state(state=_state(x=0.0))


class _FakeTasks(Tasks):
    train_task_count: int = 0
    test_task_count: int = 0

    def sample_train_task(self) -> Task:
        self.train_task_count += 1
        # Deliberately distinguishable from sample_test_task's x=0.0: if both
        # returned the same value, a practice period that (incorrectly) resumed
        # from the preceding evaluation sweep's state would be indistinguishable
        # from one that correctly starts at its own train task's initial state.
        return Task(initial_state=_state(x=100.0), goal=Goal(atoms=frozenset()))

    def sample_test_task(self) -> Task:
        self.test_task_count += 1
        return Task(initial_state=_state(x=0.0), goal=Goal(atoms=frozenset()))


class _FakeRenderer(Renderer):
    @staticmethod
    def render_frame(*, state: State, env: Environment, label: str | None = None) -> np.ndarray:
        del state, env, label
        return np.zeros((1, 1, 3), dtype=np.uint8)


class _FakeProblem(Problem):
    # Narrowed to _FakeEnv (matching LightSwitchProblem's own env: LightSwitchEnvironment
    # pattern), so problem.env.hard_reset_count/.pre_action_xs type-check below --
    # Problem.env's base type is just Environment, which doesn't declare either.
    env: _FakeEnv
    tasks: _FakeTasks
    run_task_episode_calls: int = 0
    # renderer arguments this fake was called with, in call order -- lets tests
    # check exactly which run_task_episode calls actually rendered.
    renderer_calls: list[bool] = Field(default_factory=list)
    # Shared with _FakeMethod (same object, wired in _build) so a test can assert
    # the *interleaving* of evaluation sweeps and end_cycle calls, not just counts.
    event_log: _EventLog = Field(default_factory=_EventLog)

    def run_task_episode(
        self, *, task: Task, policy: Policy, renderer: type[Renderer] | None = None
    ) -> tuple[bool, list[np.ndarray]]:
        self.run_task_episode_calls += 1
        self.event_log.record(event="evaluate")
        self.renderer_calls.append(renderer is not None)
        # Mirrors the real per-domain override (e.g. LightSwitchProblem's own
        # run_task_episode): resets env state to the task's initial_state before
        # running -- this is exactly the behavior that makes PracticeLoop
        # NOT reset-free end to end, only within one interaction period.
        self.env.set_state(state=task.initial_state)
        policy(self.get_current_state())  # exercised, but the fake doesn't need its result
        frames = (
            [renderer.render_frame(state=self.env.get_current_state(), env=self.env)]
            if renderer
            else []
        )
        return True, frames


class _FakeMethod(Method):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    policy_call_count: int = 0
    task_policy_calls: int = 0
    practice_policy_calls: int = 0
    end_cycle_calls: int = 0
    # Shared with _FakeProblem -- see its own event_log comment.
    event_log: _EventLog = Field(default_factory=_EventLog)

    def reset_environment(self, *, start_state: State) -> bool:
        raise NotImplementedError

    def get_task_policy(self, *, task: Task) -> Policy:
        del task
        self.task_policy_calls += 1
        # Policy is a positional Callable[[State], LabeledAction] per its interface
        # contract -- this lambda just adapts that into a call to _get_action's
        # keyword-only signature, same pattern as RandomSkillsMethod's own
        # get_task_policy.
        return lambda state: self._get_action(state=state)  # noqa: E731

    def get_practice_policy(self, *, task: Task) -> Policy:
        del task
        self.practice_policy_calls += 1
        return lambda state: self._get_action(state=state)  # noqa: E731

    def end_cycle(self) -> None:
        self.end_cycle_calls += 1
        self.event_log.record(event="end_cycle")

    def _get_action(self, *, state: State) -> LabeledAction:
        del state
        self.policy_call_count += 1
        return LabeledAction(action=np.array([0.0]), label="fake")

    def generate_train_task(self, *, tbd_inputs: object) -> Task:
        raise NotImplementedError

    def execute_setup_command(self, *, setup_command: SetupCommand) -> None:
        raise NotImplementedError

    def execute_skill(self, *, skill: GroundSkill) -> Rollout:
        raise NotImplementedError

    def improve_skill_parameters(self, *, skill: GroundSkill, rollout: Rollout) -> None:
        raise NotImplementedError


def _build() -> tuple[_FakeProblem, _FakeMethod, Metrics]:
    """Fresh, independently-wired fake instances per test -- no shared ClassVar
    state to reset and no global Problem.env-style wiring left to snapshot/restore,
    since each of these instances carries its own state."""
    env = _FakeEnv()
    # One list object shared by both fakes, so a test can assert the interleaving
    # of evaluation sweeps and end_cycle calls.
    event_log = _EventLog()
    problem = _FakeProblem(env=env, tasks=_FakeTasks(env=env), event_log=event_log)
    method = _FakeMethod(env=env, event_log=event_log)
    return problem, method, Metrics()


def test_run_hard_resets_exactly_once_before_the_first_evaluation() -> None:
    problem, method, metrics = _build()
    PracticeLoop.run(
        problem=problem,
        method=method,
        metrics=metrics,
        num_cycles=2,
        max_steps_per_interaction=3,
        num_test_tasks=1,
    )
    assert problem.env.hard_reset_count == 1


def test_run_evaluates_once_before_any_cycle_and_once_after_each_cycle() -> None:
    problem, method, metrics = _build()
    PracticeLoop.run(
        problem=problem,
        method=method,
        metrics=metrics,
        num_cycles=3,
        max_steps_per_interaction=2,
        num_test_tasks=1,
    )
    # 1 initial evaluation + 1 per cycle = 4 total.
    assert len(metrics.evaluations) == 4


def test_run_records_num_online_transitions_at_each_evaluation_checkpoint() -> None:
    problem, method, metrics = _build()
    PracticeLoop.run(
        problem=problem,
        method=method,
        metrics=metrics,
        num_cycles=3,
        max_steps_per_interaction=5,
        num_test_tasks=1,
    )
    transitions_recorded = [transitions for transitions, _, _ in metrics.evaluations]
    assert transitions_recorded == [0, 5, 10, 15]


def test_run_evaluates_exactly_num_test_tasks_per_checkpoint() -> None:
    problem, method, metrics = _build()
    PracticeLoop.run(
        problem=problem,
        method=method,
        metrics=metrics,
        num_cycles=1,
        max_steps_per_interaction=1,
        num_test_tasks=4,
    )
    assert all(num_total == 4 for _, _, num_total in metrics.evaluations)
    assert problem.run_task_episode_calls == 4 * 2  # initial + 1 cycle's checkpoint


def test_run_calls_on_cycle_end_once_per_cycle() -> None:
    problem, method, metrics = _build()
    calls: list[None] = []
    PracticeLoop.run(
        problem=problem,
        method=method,
        metrics=metrics,
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
    problem, method, metrics = _build()
    PracticeLoop.run(
        problem=problem,
        method=method,
        metrics=metrics,
        num_cycles=1,
        max_steps_per_interaction=3,
        num_test_tasks=1,
    )
    # The period starts at its train task's x=100, then accumulates: 100 -> 101 ->
    # 102 -> 103, so pre_action_xs (recorded before each increment) is [100, 101, 102].
    assert problem.env.pre_action_xs[:3] == [100.0, 101.0, 102.0]


def test_run_starts_each_interaction_period_from_its_own_train_tasks_initial_state() -> None:
    """Every free period begins at the train task the loop just sampled -- not at
    whatever state the preceding evaluation sweep happened to leave behind, and not
    where the previous period's own steps left off.

    This matches predicators, which resets per interaction request (main.py:301-302:
    `env_task = env.get_train_tasks()[request.train_task_idx]` then
    `cogman.reset(env_task)`), and getting it wrong is not cosmetic: an evaluation
    episode ends in a *solved* state, so on Light Switch a resuming period would
    start with the robot already at the light, skip the traversal entirely, and
    spend its whole budget practicing the toggle."""
    problem, method, metrics = _build()
    PracticeLoop.run(
        problem=problem,
        method=method,
        metrics=metrics,
        num_cycles=2,
        max_steps_per_interaction=3,
        num_test_tasks=1,
    )
    # Train tasks start at x=100, test tasks at x=0. Resuming from the evaluation
    # sweep would show 0.0; resuming from the previous period's own last step would
    # show 103.0 for the second period.
    first_step_of_each_period = problem.env.pre_action_xs[0::3]
    assert first_step_of_each_period == [100.0, 100.0]


def test_run_without_a_num_cycles_zero_still_runs_the_initial_evaluation() -> None:
    problem, method, metrics = _build()
    PracticeLoop.run(
        problem=problem,
        method=method,
        metrics=metrics,
        num_cycles=0,
        max_steps_per_interaction=5,
        num_test_tasks=2,
    )
    assert metrics.evaluations == [(0, 2, 2)]


def test_run_without_a_renderer_returns_no_frames() -> None:
    problem, method, metrics = _build()
    frames = PracticeLoop.run(
        problem=problem,
        method=method,
        metrics=metrics,
        num_cycles=2,
        max_steps_per_interaction=1,
        num_test_tasks=2,
    )
    assert frames == {}
    assert problem.renderer_calls == [False] * (2 * 3)  # 3 evaluations x 2 test tasks


def test_run_with_a_renderer_and_zero_cycles_renders_the_initial_evaluation() -> None:
    problem, method, metrics = _build()
    frames = PracticeLoop.run(
        problem=problem,
        method=method,
        metrics=metrics,
        num_cycles=0,
        max_steps_per_interaction=1,
        num_test_tasks=3,
        renderer=_FakeRenderer,
    )
    assert len(frames) == 1
    # Only the first test task of the (sole) evaluation sweep renders.
    assert problem.renderer_calls == [True, False, False]


def test_run_with_a_renderer_renders_only_the_last_evaluation_sweeps_first_task() -> None:
    problem, method, metrics = _build()
    PracticeLoop.run(
        problem=problem,
        method=method,
        metrics=metrics,
        num_cycles=2,
        max_steps_per_interaction=1,
        num_test_tasks=2,
        renderer=_FakeRenderer,
    )
    # 3 evaluation sweeps of 2 test tasks each: initial, after cycle 1, after
    # cycle 2. Only the very first task of the very last sweep renders.
    assert problem.renderer_calls == [
        False,
        False,  # initial evaluation
        False,
        False,  # after cycle 1
        True,
        False,  # after cycle 2 (the last sweep)
    ]


def test_render_sweep_indices_defaults_to_the_final_sweep_only() -> None:
    """Backwards-compatible default: one clip of the finished policy."""
    assert PracticeLoop.render_sweep_indices(num_cycles=10, num_render_checkpoints=1) == frozenset({
        10
    })


def test_render_sweep_indices_spreads_evenly_across_training() -> None:
    """Five clips over ten cycles, spanning untrained (sweep 0, before any
    practice) through fully trained (sweep 10) -- so a set of them reads as a
    progression rather than five samples of the same finished policy."""
    assert PracticeLoop.render_sweep_indices(num_cycles=10, num_render_checkpoints=5) == frozenset({
        0,
        2,
        5,
        8,
        10,
    })


def test_render_sweep_indices_never_exceeds_the_number_of_sweeps() -> None:
    """Asking for more checkpoints than there are sweeps yields every sweep, not
    duplicates or an index past the end."""
    assert PracticeLoop.render_sweep_indices(num_cycles=2, num_render_checkpoints=9) == frozenset({
        0,
        1,
        2,
    })


def test_run_returns_frames_keyed_by_transitions_for_each_checkpoint() -> None:
    problem, method, metrics = _build()
    frames_by_transitions = PracticeLoop.run(
        problem=problem,
        method=method,
        metrics=metrics,
        num_cycles=4,
        max_steps_per_interaction=2,
        num_test_tasks=1,
        renderer=_FakeRenderer,
        num_render_checkpoints=3,
    )
    # Sweeps 0, 2, 4 -> 0, 4, 8 transitions at 2 steps per cycle.
    assert sorted(frames_by_transitions) == [0, 4, 8]
    assert all(frames for frames in frames_by_transitions.values())


def test_run_without_a_renderer_returns_nothing_even_with_checkpoints_requested() -> None:
    problem, method, metrics = _build()
    assert (
        PracticeLoop.run(
            problem=problem,
            method=method,
            metrics=metrics,
            num_cycles=4,
            max_steps_per_interaction=1,
            num_test_tasks=1,
            num_render_checkpoints=5,
        )
        == {}
    )


def test_run_uses_practice_policy_for_interaction_and_task_policy_for_evaluation() -> None:
    """The practice/evaluate split a learning Method (EES) depends on: exploration
    happens only in the interaction period, never during the held-out evaluation
    sweep (which would be training on the test set)."""
    problem, method, metrics = _build()
    PracticeLoop.run(
        problem=problem,
        method=method,
        metrics=metrics,
        num_cycles=2,
        max_steps_per_interaction=1,
        num_test_tasks=3,
    )
    # One practice policy per cycle.
    assert method.practice_policy_calls == 2
    # One task policy per test task per evaluation sweep (initial + one per cycle).
    assert method.task_policy_calls == 3 * 3


def test_run_calls_end_cycle_once_per_cycle() -> None:
    problem, method, metrics = _build()
    PracticeLoop.run(
        problem=problem,
        method=method,
        metrics=metrics,
        num_cycles=3,
        max_steps_per_interaction=1,
        num_test_tasks=1,
    )
    assert method.end_cycle_calls == 3


def test_run_calls_end_cycle_before_that_cycles_evaluation() -> None:
    """Retraining has to land before the sweep that's meant to measure it --
    otherwise every reported evaluation lags a cycle behind the learning."""
    problem, method, metrics = _build()
    PracticeLoop.run(
        problem=problem,
        method=method,
        metrics=metrics,
        num_cycles=2,
        max_steps_per_interaction=1,
        num_test_tasks=1,
    )
    # Initial sweep before any cycle, then each cycle ends (retrains) before the
    # sweep that measures it.
    assert problem.event_log.events == [
        "evaluate",
        "end_cycle",
        "evaluate",
        "end_cycle",
        "evaluate",
    ]


def test_run_with_zero_cycles_never_calls_end_cycle_or_practice_policy() -> None:
    problem, method, metrics = _build()
    PracticeLoop.run(
        problem=problem,
        method=method,
        metrics=metrics,
        num_cycles=0,
        max_steps_per_interaction=5,
        num_test_tasks=2,
    )
    assert method.end_cycle_calls == 0
    assert method.practice_policy_calls == 0


class _EarlyStoppingMethod(_FakeMethod):
    """Signals it has nothing left to practice after a fixed number of steps --
    predicators' explorers do the equivalent by raising out of
    run_episode_and_get_observations, which then returns a short trajectory."""

    steps_before_stopping: int = 2
    steps_taken: int = 0

    def get_practice_policy(self, *, task: Task) -> Policy:
        del task
        self.practice_policy_calls += 1
        # Lambda adapter, per this project's convention for interfaces that
        # demand a positional callable (Policy is Callable[[State], ...]).
        return lambda state: self._early_action(state=state)

    def _early_action(self, *, state: State) -> LabeledAction:
        del state
        if self.steps_taken >= self.steps_before_stopping:
            raise InteractionComplete
        self.steps_taken += 1
        return LabeledAction(action=np.array([0.0]), label="early")


def test_run_counts_only_the_transitions_actually_taken() -> None:
    """Data-driven, matching predicators' `num_online_transitions += sum(
    len(result.actions) for result in interaction_results)`: an interaction period
    that ends early contributes only the steps it really took, not its budget."""
    env = _FakeEnv()
    event_log = _EventLog()
    problem = _FakeProblem(env=env, tasks=_FakeTasks(env=env), event_log=event_log)
    method = _EarlyStoppingMethod(env=env, event_log=event_log, steps_before_stopping=2)
    metrics = Metrics()

    PracticeLoop.run(
        problem=problem,
        method=method,
        metrics=metrics,
        num_cycles=1,
        max_steps_per_interaction=100,
        num_test_tasks=1,
    )

    transitions_recorded = [transitions for transitions, _, _ in metrics.evaluations]
    # 0 before any practice, then only the 2 steps actually taken -- not 100.
    assert transitions_recorded == [0, 2]


def test_early_stopping_still_ends_the_cycle_and_evaluates() -> None:
    """Stopping early is a normal end to the period, not an error: the cycle must
    still retrain and be measured."""
    env = _FakeEnv()
    event_log = _EventLog()
    problem = _FakeProblem(env=env, tasks=_FakeTasks(env=env), event_log=event_log)
    method = _EarlyStoppingMethod(env=env, event_log=event_log, steps_before_stopping=1)
    metrics = Metrics()

    PracticeLoop.run(
        problem=problem,
        method=method,
        metrics=metrics,
        num_cycles=2,
        max_steps_per_interaction=50,
        num_test_tasks=1,
    )

    assert method.end_cycle_calls == 2
    assert problem.event_log.events == [
        "evaluate",
        "end_cycle",
        "evaluate",
        "end_cycle",
        "evaluate",
    ]
