import inspect
from pathlib import Path

import numpy as np
import pytest
from pydantic import ConfigDict, Field

from hitl_pmp.core.method.method import HumanHelpRequested, InteractionComplete, Method
from hitl_pmp.core.method.types import GroundSkill, LabeledAction, Policy, Rollout, SetupCommand
from hitl_pmp.core.metrics.metrics import Metrics
from hitl_pmp.core.problem.environment.environment import Environment
from hitl_pmp.core.problem.environment.types import Action, Object, State, Type
from hitl_pmp.core.problem.human.human import HumanOracle
from hitl_pmp.core.problem.human.types import (
    CommandGoalDescription,
    CommandStartStateDescription,
    Cost,
)
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.core.problem.tasks.tasks import Tasks
from hitl_pmp.core.problem.tasks.types import Goal, Task
from hitl_pmp.core.renderer.renderer import Renderer, VideoStream
from hitl_pmp.human_intervention import HumanResetTarget
from hitl_pmp.practice_loop import PracticeLoop, PracticeResetPolicy
from hitl_pmp.recording.loop_recorder import LoopRecorder

_BLOCK = Type(name="block", feature_names=("x",))
_OBJ = Object(name="thing", type=_BLOCK)


def _state(*, x: float) -> State:
    return State(data={_OBJ: np.array([x])})


class _CollectingSink:
    """A CheckpointFramesSink that keeps what it is handed, so a test can assert on
    frames without PracticeLoop retaining any itself.

    Production code streams frames straight to disk; a test that wants them in hand
    collects them here instead. That is deliberately the *same* code path production
    uses -- there is no parallel retaining mode inside PracticeLoop to exercise
    instead, and reintroducing one is the bug this arrangement exists to prevent."""

    def __init__(self) -> None:
        self.frames_by_transitions: dict[int, list[np.ndarray]] = {}

    def __call__(self, *, transitions: int, frames: list[np.ndarray]) -> None:
        self.frames_by_transitions[transitions] = frames


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

    def noop_action(self) -> Action:
        return np.zeros(1)

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


class _SamplingMovesTheWorldTasks(_FakeTasks):
    """Tossing3D's shape: the only way to obtain an initial `State` is to build it in
    the world, so *sampling* a task is a simulator operation with a side effect.

    Not a hypothetical. `Tossing3DTasks.build_task` calls `env.reset_to_seed`, which
    rebuilds the live MuJoCo scene, and that is what made `practice_reset_policy=never`
    a no-op there: the loop declined to *install* the sampled task's initial state and
    recorded zero resets while the scene had already been rebuilt underneath it. This
    fake reproduces that side effect and nothing else."""

    def sample_train_task(self) -> Task:
        task = super().sample_train_task()
        self.env.set_state(state=_state(x=float(task.initial_state[_OBJ][0])))
        return task


class _SamplingMovesTheWorldInPlaceTasks(_SamplingMovesTheWorldTasks):
    """The migrated form: same side effect on `sample_train_task`, plus the in-place
    path a reset-free arm actually takes. Mirrors `Tossing3DTasks` after the fix --
    the task is addressed to the world the environment is already in, so no simulator
    operation happens at all."""

    in_place_task_count: int = 0

    def sample_train_task_in_place(self) -> Task:
        self.in_place_task_count += 1
        return Task(initial_state=self.env.get_current_state(), goal=Goal(atoms=frozenset()))


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
    # x observed at each observe_environment_reset call, in call order -- lets a
    # test check the Method is handed the state the harness is about to discard
    # rather than the one it is about to reset to.
    reset_observation_xs: list[float] = Field(default_factory=list)
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

    def observe_environment_reset(self, *, state: State) -> None:
        self.reset_observation_xs.append(float(state[_OBJ][0]))
        self.event_log.record(event="observe_environment_reset")

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


def _build_split() -> tuple[_FakeProblem, _FakeProblem, _FakeMethod, Metrics]:
    """A practice triple and a wholly separate evaluation triple, as a domain's
    composition root now builds them. Two independent _FakeEnv instances, so a write
    to one is observable as an absence on the other -- which is the entire point."""
    practice_env = _FakeEnv()
    evaluation_env = _FakeEnv()
    event_log = _EventLog()
    practice = _FakeProblem(
        env=practice_env, tasks=_FakeTasks(env=practice_env), event_log=event_log
    )
    evaluation = _FakeProblem(
        env=evaluation_env, tasks=_FakeTasks(env=evaluation_env), event_log=event_log
    )
    method = _FakeMethod(env=practice_env, event_log=event_log)
    return practice, evaluation, method, Metrics()


def _build_split_with(
    *, tasks_type: type[_FakeTasks]
) -> tuple[_FakeProblem, _FakeProblem, _FakeMethod, Metrics]:
    """`_build_split`, but with the practice side's `Tasks` swapped for a domain whose
    task sampling moves the environment. Evaluation keeps the plain fake: the defect
    being pinned lives entirely on the practice side."""
    practice_env = _FakeEnv()
    evaluation_env = _FakeEnv()
    event_log = _EventLog()
    practice = _FakeProblem(
        env=practice_env, tasks=tasks_type(env=practice_env), event_log=event_log
    )
    evaluation = _FakeProblem(
        env=evaluation_env, tasks=_FakeTasks(env=evaluation_env), event_log=event_log
    )
    method = _FakeMethod(env=practice_env, event_log=event_log)
    return practice, evaluation, method, Metrics()


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


def test_run_without_a_renderer_renders_nothing() -> None:
    problem, method, metrics = _build()
    assert (
        PracticeLoop.run(
            problem=problem,
            method=method,
            metrics=metrics,
            num_cycles=2,
            max_steps_per_interaction=1,
            num_test_tasks=2,
        )
        is None
    )
    assert problem.renderer_calls == [False] * (2 * 3)  # 3 evaluations x 2 test tasks


def test_run_with_a_renderer_and_zero_cycles_renders_the_initial_evaluation() -> None:
    problem, method, metrics = _build()
    sink = _CollectingSink()
    PracticeLoop.run(
        problem=problem,
        method=method,
        metrics=metrics,
        num_cycles=0,
        max_steps_per_interaction=1,
        num_test_tasks=3,
        renderer=_FakeRenderer,
        on_checkpoint_frames=sink,
    )
    assert len(sink.frames_by_transitions) == 1
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
        on_checkpoint_frames=_CollectingSink(),
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


def test_each_checkpoints_frames_reach_the_sink_keyed_by_transitions() -> None:
    problem, method, metrics = _build()
    sink = _CollectingSink()
    PracticeLoop.run(
        problem=problem,
        method=method,
        metrics=metrics,
        num_cycles=4,
        max_steps_per_interaction=2,
        num_test_tasks=1,
        renderer=_FakeRenderer,
        num_render_checkpoints=3,
        on_checkpoint_frames=sink,
    )
    # Sweeps 0, 2, 4 -> 0, 4, 8 transitions at 2 steps per cycle.
    assert sorted(sink.frames_by_transitions) == [0, 4, 8]
    assert all(frames for frames in sink.frames_by_transitions.values())


def test_without_a_renderer_the_sink_is_never_called_even_with_checkpoints_requested() -> None:
    """Checkpoints select which *sweeps* would record; with no renderer none of
    them produce a frame, so the sink must stay untouched rather than being handed
    empty lists."""
    problem, method, metrics = _build()
    sink = _CollectingSink()
    PracticeLoop.run(
        problem=problem,
        method=method,
        metrics=metrics,
        num_cycles=4,
        max_steps_per_interaction=1,
        num_test_tasks=1,
        num_render_checkpoints=5,
        on_checkpoint_frames=sink,
    )
    assert sink.frames_by_transitions == {}


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


def test_the_test_set_is_drawn_once_and_reused_by_every_sweep() -> None:
    """The evaluation set must be fixed for the whole run, matching predicators'
    cached `BaseEnv.get_test_tasks`. Re-sampling it per sweep makes consecutive
    points on a learning curve measure different task sets, so the curve carries
    task-sampling variance on top of the policy change it is meant to isolate --
    which on Ball-Ring produced ~3x the reference implementation's per-seed
    variance and apparent within-one-cycle collapses."""
    problem, method, metrics = _build()
    PracticeLoop.run(
        problem=problem,
        method=method,
        metrics=metrics,
        num_cycles=4,
        max_steps_per_interaction=2,
        num_test_tasks=3,
    )
    # 5 sweeps happen (1 initial + 4 cycles), but only 3 test tasks are ever drawn.
    assert len(metrics.evaluations) == 5
    assert problem.tasks.test_task_count == 3


def test_evaluation_runs_on_the_evaluation_problem_when_one_is_given() -> None:
    """The defect this fixes: `_evaluate` used to take the SAME Problem the practice
    loop drives, so every evaluation episode's opening `reset_to_task` was a
    privileged state-write on the environment practice inherits. A 30-task sweep is
    30 such writes, which silently resets a "never reset" practice arm 30 times per
    sweep and makes the manipulation unmeasurable."""
    practice, evaluation, method, metrics = _build_split()
    PracticeLoop.run(
        problem=practice,
        evaluation_problem=evaluation,
        method=method,
        metrics=metrics,
        num_cycles=2,
        max_steps_per_interaction=2,
        num_test_tasks=3,
    )
    # 3 sweeps (1 initial + 2 cycles) x 3 test tasks, all on the evaluation problem.
    assert evaluation.run_task_episode_calls == 9
    assert practice.run_task_episode_calls == 0


def test_the_test_set_is_drawn_from_the_evaluation_problem() -> None:
    """Test tasks come off the evaluation Tasks' own stream, not the practice one's.
    Both are constructed identically, so the tasks are the same -- but drawing them
    from the practice instance would leave the evaluation environment un-exercised
    and keep the two halves entangled through a shared Tasks."""
    practice, evaluation, method, metrics = _build_split()
    PracticeLoop.run(
        problem=practice,
        evaluation_problem=evaluation,
        method=method,
        metrics=metrics,
        num_cycles=1,
        max_steps_per_interaction=2,
        num_test_tasks=4,
    )
    assert evaluation.tasks.test_task_count == 4
    assert practice.tasks.test_task_count == 0


def test_evaluation_leaves_the_practice_environments_state_untouched() -> None:
    """The property the whole split exists for, stated as behaviour rather than as
    call counts: after a run, the practice environment holds what *practice* left
    there. Before the split, the final evaluation sweep's `reset_to_task` overwrote
    it with a test task's initial state (x=0.0 here)."""
    practice, evaluation, method, metrics = _build_split()
    PracticeLoop.run(
        problem=practice,
        evaluation_problem=evaluation,
        method=method,
        metrics=metrics,
        num_cycles=1,
        max_steps_per_interaction=3,
        num_test_tasks=2,
    )
    # The period starts at the train task's x=100.0 and takes 3 steps of +1.0 each.
    assert float(practice.env.get_current_state()[_OBJ][0]) == 103.0
    # ...and the evaluation environment holds a *test* task's state, never 100-ish.
    assert float(evaluation.env.get_current_state()[_OBJ][0]) == 0.0


def test_both_environments_are_hard_reset_exactly_once() -> None:
    """Each environment needs its own one-time hard_reset before the run; skipping
    the evaluation one would leave it with no current_state at all."""
    practice, evaluation, method, metrics = _build_split()
    PracticeLoop.run(
        problem=practice,
        evaluation_problem=evaluation,
        method=method,
        metrics=metrics,
        num_cycles=2,
        max_steps_per_interaction=2,
        num_test_tasks=1,
    )
    assert practice.env.hard_reset_count == 1
    assert evaluation.env.hard_reset_count == 1


def test_omitting_the_evaluation_problem_evaluates_on_the_practice_problem() -> None:
    """Back-compatibility, and what keeps every un-migrated domain byte-identical:
    with no evaluation_problem the loop behaves exactly as it did before, evaluating
    on the one Problem it was given and hard-resetting it exactly once."""
    problem, method, metrics = _build()
    PracticeLoop.run(
        problem=problem,
        method=method,
        metrics=metrics,
        num_cycles=1,
        max_steps_per_interaction=2,
        num_test_tasks=3,
    )
    assert problem.run_task_episode_calls == 6
    assert problem.tasks.test_task_count == 3
    assert problem.env.hard_reset_count == 1


def test_streaming_hands_over_every_rendered_checkpoint_and_returns_nothing() -> None:
    """The memory property streaming exists for: peak retention is one sweep's
    frames, not every checkpoint's. Retaining all of them is an unbounded buffer
    (checkpoints x episode length x frame bytes), and a runaway one already
    OOM-killed a whole session on this project -- fatally, because a tmux pane's
    systemd scope defaults to OOMPolicy=stop, so the kill took the session down
    with it rather than just the offending process."""
    problem, method, metrics = _build()
    received: list[int] = []

    def sink(*, transitions: int, frames: list[np.ndarray]) -> None:
        assert frames, "a rendered sweep must hand over a non-empty frame list"
        received.append(transitions)

    returned = PracticeLoop.run(
        problem=problem,
        method=method,
        metrics=metrics,
        num_cycles=4,
        max_steps_per_interaction=2,
        num_test_tasks=2,
        renderer=_FakeRenderer,
        num_render_checkpoints=5,
        on_checkpoint_frames=sink,
    )
    # Every rendered checkpoint reached the sink...
    assert len(received) == 5
    assert received == sorted(received)
    # ...and run() hands back nothing, because there is no longer anywhere for
    # frames to accumulate. This fails if someone reintroduces a retaining path.
    assert returned is None


def test_a_renderer_without_a_sink_is_rejected_before_the_run_starts() -> None:
    """Streaming is the only path, so a renderer with nowhere to send its frames is
    a caller error rather than a silent discard -- and it is caught up front, before
    hard_reset(), so the environment is untouched when it raises."""
    problem, method, metrics = _build()
    with pytest.raises(ValueError, match="on_checkpoint_frames is required"):
        PracticeLoop.run(
            problem=problem,
            method=method,
            metrics=metrics,
            num_cycles=2,
            max_steps_per_interaction=2,
            num_test_tasks=2,
            renderer=_FakeRenderer,
            num_render_checkpoints=3,
        )
    assert problem.env.hard_reset_count == 0
    assert metrics.evaluations == []


def test_streaming_hands_over_frames_as_each_sweep_ends_not_at_the_end() -> None:
    """Ordering is the whole point: if the sink were called after run() finished,
    the frames would have been held for the entire run and nothing would be
    bounded. Interleaving with the evaluate events proves they are handed over
    while the run is still going."""
    problem, method, metrics = _build()

    def sink(*, transitions: int, frames: list[np.ndarray]) -> None:
        problem.event_log.events.append(f"frames:{transitions}")

    PracticeLoop.run(
        problem=problem,
        method=method,
        metrics=metrics,
        num_cycles=2,
        max_steps_per_interaction=2,
        num_test_tasks=1,
        renderer=_FakeRenderer,
        num_render_checkpoints=3,
        on_checkpoint_frames=sink,
    )
    # Each handover sits immediately after its own sweep, interleaved with
    # end_cycle -- not batched at the tail.
    assert problem.event_log.events == [
        "evaluate",
        "frames:0",
        "end_cycle",
        "evaluate",
        "frames:2",
        "end_cycle",
        "evaluate",
        "frames:4",
    ]


def test_practice_reset_interval_defaults_to_the_cycle_boundary_only() -> None:
    """The default (None) is exactly the behaviour that shipped before the knob
    existed: one reset per cycle, at the top of the period, and nothing inside it.

    Asserted on the recorded count rather than only on the state trace, because
    the count is what an experiment varying this knob reads back out of
    stats.json to check the manipulation happened."""
    problem, method, metrics = _build()
    PracticeLoop.run(
        problem=problem,
        method=method,
        metrics=metrics,
        num_cycles=3,
        max_steps_per_interaction=4,
        num_test_tasks=1,
    )
    assert metrics.num_practice_resets == 3
    assert method.reset_observation_xs == []


def test_practice_reset_interval_restarts_the_period_partway_through() -> None:
    problem, method, metrics = _build()
    PracticeLoop.run(
        problem=problem,
        method=method,
        metrics=metrics,
        num_cycles=1,
        max_steps_per_interaction=6,
        num_test_tasks=1,
        practice_reset_interval=2,
    )
    # Train tasks start at x=100 and every step adds 1, so an interval of 2 shows
    # up as 100 -> 101, reset, 100 -> 101, reset, 100 -> 101 (the last reset, after
    # step 6, is suppressed as the period's final step).
    assert problem.env.pre_action_xs == [100.0, 101.0, 100.0, 101.0, 100.0, 101.0]


def test_practice_reset_interval_returns_to_the_same_task_not_a_fresh_one() -> None:
    """Sampling a new train task at each within-period reset would change the
    train-task distribution along with the reset rate, reintroducing exactly the
    kind of confound this knob exists to remove."""
    problem, method, metrics = _build()
    PracticeLoop.run(
        problem=problem,
        method=method,
        metrics=metrics,
        num_cycles=2,
        max_steps_per_interaction=6,
        num_test_tasks=1,
        practice_reset_interval=2,
    )
    assert problem.tasks.train_task_count == 2


def test_practice_reset_interval_counts_every_reset_it_performs() -> None:
    problem, method, metrics = _build()
    PracticeLoop.run(
        problem=problem,
        method=method,
        metrics=metrics,
        num_cycles=3,
        max_steps_per_interaction=10,
        num_test_tasks=1,
        practice_reset_interval=5,
    )
    # 10 // 5 = 2 resets per period (the period-opening one plus one mid-period),
    # times 3 cycles.
    assert metrics.num_practice_resets == 6


def test_practice_reset_interval_equal_to_the_period_reproduces_the_default() -> None:
    """The arm that is meant to be "today's behaviour, stated explicitly" has to
    actually be today's behaviour -- otherwise a sweep's control arm differs from
    its baseline for a reason nobody wrote down."""
    problem_default, method_default, metrics_default = _build()
    PracticeLoop.run(
        problem=problem_default,
        method=method_default,
        metrics=metrics_default,
        num_cycles=2,
        max_steps_per_interaction=4,
        num_test_tasks=1,
    )
    problem_explicit, method_explicit, metrics_explicit = _build()
    PracticeLoop.run(
        problem=problem_explicit,
        method=method_explicit,
        metrics=metrics_explicit,
        num_cycles=2,
        max_steps_per_interaction=4,
        num_test_tasks=1,
        practice_reset_interval=4,
    )
    assert problem_explicit.env.pre_action_xs == problem_default.env.pre_action_xs
    assert metrics_explicit.num_practice_resets == metrics_default.num_practice_resets
    assert method_explicit.reset_observation_xs == []


def test_practice_reset_interval_does_not_end_the_cycle_or_retrain() -> None:
    """The whole point: reset frequency and refit frequency come apart. A
    within-period reset must not fire end_cycle(), must not ask for a fresh
    practice policy, and must not trigger an evaluation sweep."""
    problem, method, metrics = _build()
    PracticeLoop.run(
        problem=problem,
        method=method,
        metrics=metrics,
        num_cycles=2,
        max_steps_per_interaction=8,
        num_test_tasks=1,
        practice_reset_interval=2,
    )
    assert method.end_cycle_calls == 2
    assert method.practice_policy_calls == 2
    assert len(metrics.evaluations) == 3  # initial + one per cycle


def test_practice_reset_interval_does_not_charge_resets_as_transitions() -> None:
    problem, method, metrics = _build()
    PracticeLoop.run(
        problem=problem,
        method=method,
        metrics=metrics,
        num_cycles=2,
        max_steps_per_interaction=8,
        num_test_tasks=1,
        practice_reset_interval=2,
    )
    assert [transitions for transitions, _, _ in metrics.evaluations] == [0, 8, 16]


def test_practice_reset_interval_tells_the_method_before_each_within_period_reset() -> None:
    """The Method is handed the state the environment is about to leave, before it
    leaves it. Without this, a Method that scores a skill by checking its effects
    on the next state it sees would score every pre-reset skill against a freshly
    reset environment -- one mislabelled outcome per reset, scaling with exactly
    the quantity a reset-interval sweep varies."""
    problem, method, metrics = _build()
    PracticeLoop.run(
        problem=problem,
        method=method,
        metrics=metrics,
        num_cycles=1,
        max_steps_per_interaction=6,
        num_test_tasks=1,
        practice_reset_interval=2,
    )
    # Two mid-period resets, each observed at x=102 (the train task's 100 plus the
    # two steps taken since the last reset) -- not at the post-reset 100.
    assert method.reset_observation_xs == [102.0, 102.0]


def test_practice_reset_interval_leaves_the_period_boundary_unannounced() -> None:
    """observe_environment_reset fires only inside a period, never at its end.
    Firing at the boundary too would change long-standing behaviour, and it is
    what keeps every arm of a reset-interval sweep dropping exactly one
    observation per period instead of a number that varies with the interval."""
    problem, method, metrics = _build()
    PracticeLoop.run(
        problem=problem,
        method=method,
        metrics=metrics,
        num_cycles=3,
        max_steps_per_interaction=4,
        num_test_tasks=1,
        practice_reset_interval=2,
    )
    # One mid-period reset per period (after step 2; the one after step 4 is the
    # boundary and is suppressed), so three announcements for three cycles.
    assert len(method.reset_observation_xs) == 3
    assert method.event_log.events.count("observe_environment_reset") == 3


def test_reset_is_due_never_fires_on_the_periods_last_step() -> None:
    assert not PracticeLoop._reset_is_due(
        step=9, max_steps_per_interaction=10, practice_reset_interval=10
    )
    assert not PracticeLoop._reset_is_due(
        step=9, max_steps_per_interaction=10, practice_reset_interval=5
    )


def test_reset_is_due_is_never_due_without_an_interval() -> None:
    assert not any(
        PracticeLoop._reset_is_due(
            step=step, max_steps_per_interaction=10, practice_reset_interval=None
        )
        for step in range(10)
    )


def test_reset_is_due_yields_the_designed_resets_per_period() -> None:
    """The exact per-arm reset counts a reset-frequency sweep is designed around:
    at a 100-step period, intervals of 10/25/50/100 give 10/4/2/1 resets per
    period once the period-opening reset is counted."""
    for interval, expected_per_period in ((10, 10), (25, 4), (50, 2), (100, 1)):
        within_period = sum(
            PracticeLoop._reset_is_due(
                step=step, max_steps_per_interaction=100, practice_reset_interval=interval
            )
            for step in range(100)
        )
        assert within_period + 1 == expected_per_period


class _SpyRecorder(LoopRecorder):
    """Logs which recording hook PracticeLoop called, in order, without rendering
    or writing anything -- these tests are about the *loop's* coverage of its own
    phases and resets, not about what a frame looks like (that is
    tests/recording/'s job)."""

    calls: list[str] = Field(default_factory=list)

    def record_hard_reset(self, *, state: State) -> None:
        del state
        self.calls.append("hard_reset")

    def begin_evaluation(self, *, sweep_index: int, transitions: int) -> None:
        del transitions
        self.calls.append(f"begin_evaluation:{sweep_index}")

    def watch_policy(self, *, policy: Policy) -> Policy:
        self.calls.append("watch_policy")
        return policy

    def record_evaluation_episode(
        self,
        *,
        task_index: int,
        num_tasks: int,
        task: str,
        frames: list[np.ndarray],
        solved: bool,
    ) -> None:
        del num_tasks, task, frames, solved
        self.calls.append(f"evaluation_episode:{task_index}")

    def begin_practice(self, *, cycle_index: int, transitions: int, task: str) -> None:
        del transitions, task
        self.calls.append(f"begin_practice:{cycle_index}")

    def record_period_reset(self, *, state: State) -> None:
        del state
        self.calls.append("period_reset")

    def record_practice_step(
        self, *, state: State, skill: str, step_index: int, transitions: int
    ) -> None:
        del state, skill, transitions
        self.calls.append(f"practice_step:{step_index}")

    def record_interval_reset(self, *, state: State, step_index: int, transitions: int) -> None:
        del state, transitions
        self.calls.append(f"interval_reset:{step_index}")

    def record_human_reset(self, *, state: State, step_index: int, transitions: int) -> None:
        del state, transitions
        self.calls.append(f"human_reset:{step_index}")

    def record_interaction_complete(
        self, *, state: State, step_index: int, transitions: int
    ) -> None:
        del state, transitions
        self.calls.append(f"interaction_complete:{step_index}")


def _spy_recorder(*, problem: _FakeProblem, num_cycles: int, max_steps: int) -> _SpyRecorder:
    return _SpyRecorder(
        renderer=_FakeRenderer,
        env=problem.env,
        video=VideoStream(output_path=Path("unused.mp4"), fps=4),
        num_cycles=num_cycles,
        max_steps_per_interaction=max_steps,
    )


def test_run_hands_the_recorder_every_phase_and_every_reset_of_the_loop() -> None:
    problem, method, metrics = _build()
    recorder = _spy_recorder(problem=problem, num_cycles=2, max_steps=3)
    PracticeLoop.run(
        problem=problem,
        method=method,
        metrics=metrics,
        num_cycles=2,
        max_steps_per_interaction=3,
        num_test_tasks=1,
        practice_reset_interval=2,
        recorder=recorder,
    )
    assert recorder.calls == [
        "hard_reset",
        "begin_evaluation:0",
        "watch_policy",
        "evaluation_episode:0",
        "begin_practice:0",
        "period_reset",
        "practice_step:0",
        "practice_step:1",
        "interval_reset:1",
        "practice_step:2",
        "begin_evaluation:1",
        "watch_policy",
        "evaluation_episode:0",
        "begin_practice:1",
        "period_reset",
        "practice_step:0",
        "practice_step:1",
        "interval_reset:1",
        "practice_step:2",
        "begin_evaluation:2",
        "watch_policy",
        "evaluation_episode:0",
    ]


def test_run_records_every_test_task_of_a_sweep_not_only_the_first() -> None:
    """The rendered-checkpoint path deliberately records only test task 0; a
    full-loop recording is a record of the whole loop, so every episode of the
    sweep has to appear in it."""
    problem, method, metrics = _build()
    recorder = _spy_recorder(problem=problem, num_cycles=0, max_steps=1)
    PracticeLoop.run(
        problem=problem,
        method=method,
        metrics=metrics,
        num_cycles=0,
        max_steps_per_interaction=1,
        num_test_tasks=3,
        recorder=recorder,
    )
    assert [call for call in recorder.calls if call.startswith("evaluation_episode")] == [
        "evaluation_episode:0",
        "evaluation_episode:1",
        "evaluation_episode:2",
    ]


def test_run_records_an_interaction_complete_when_a_period_ends_early() -> None:
    env = _FakeEnv()
    event_log = _EventLog()
    problem = _FakeProblem(env=env, tasks=_FakeTasks(env=env), event_log=event_log)
    method = _EarlyStoppingMethod(env=env, event_log=event_log, steps_before_stopping=2)
    recorder = _spy_recorder(problem=problem, num_cycles=1, max_steps=100)
    PracticeLoop.run(
        problem=problem,
        method=method,
        metrics=Metrics(),
        num_cycles=1,
        max_steps_per_interaction=100,
        num_test_tasks=1,
        recorder=recorder,
    )
    assert "interaction_complete:2" in recorder.calls


def test_run_defaults_to_not_recording() -> None:
    assert inspect.signature(PracticeLoop.run).parameters["recorder"].default is None


def _run_for_comparison(
    *, recorder: _SpyRecorder | None
) -> tuple[Metrics, _FakeProblem, _FakeMethod]:
    problem, method, metrics = _build()
    PracticeLoop.run(
        problem=problem,
        method=method,
        metrics=metrics,
        num_cycles=3,
        max_steps_per_interaction=4,
        num_test_tasks=2,
        practice_reset_interval=2,
        recorder=recorder,
    )
    return metrics, problem, method


def test_recording_does_not_change_what_the_run_does() -> None:
    """The whole flag is only usable if it is an observer: same actions, same
    step counts, same serialized Metrics down to the byte."""
    plain_metrics, plain_problem, plain_method = _run_for_comparison(recorder=None)
    problem, _, _ = _build()
    recorded_metrics, recorded_problem, recorded_method = _run_for_comparison(
        recorder=_spy_recorder(problem=problem, num_cycles=3, max_steps=4)
    )

    assert recorded_metrics.model_dump_json() == plain_metrics.model_dump_json()
    assert recorded_problem.env.pre_action_xs == plain_problem.env.pre_action_xs
    assert recorded_method.policy_call_count == plain_method.policy_call_count
    assert recorded_method.reset_observation_xs == plain_method.reset_observation_xs


def test_practice_reset_policy_defaults_to_scheduled() -> None:
    """The default must be the behaviour that shipped before the knob existed:
    one reset at the top of every period. Every existing run depends on it."""
    problem, method, metrics = _build()
    PracticeLoop.run(
        problem=problem,
        method=method,
        metrics=metrics,
        num_cycles=3,
        max_steps_per_interaction=2,
        num_test_tasks=1,
    )
    assert metrics.num_practice_resets == 3
    # Every period restarts at its own train task's x=100.0.
    assert problem.env.pre_action_xs == [100.0, 101.0, 100.0, 101.0, 100.0, 101.0]


def test_practice_reset_policy_never_skips_the_per_period_reset() -> None:
    """The manipulation this flag exists to express: practice state runs
    continuously across period boundaries instead of being put back to each
    freshly-sampled train task's initial state."""
    practice, evaluation, method, metrics = _build_split()
    PracticeLoop.run(
        problem=practice,
        evaluation_problem=evaluation,
        method=method,
        metrics=metrics,
        num_cycles=3,
        max_steps_per_interaction=2,
        num_test_tasks=1,
        practice_reset_policy=PracticeResetPolicy.NEVER,
    )
    # hard_reset leaves x=0.0, and nothing ever puts it back: the trace is one
    # unbroken +1.0 ramp across all three periods, never returning to 100.0.
    assert practice.env.pre_action_xs == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def test_practice_reset_policy_never_records_no_practice_resets() -> None:
    """num_practice_resets is what an experiment reads back out of stats.json to
    confirm the manipulation actually happened, rather than trusting the flag."""
    practice, evaluation, method, metrics = _build_split()
    PracticeLoop.run(
        problem=practice,
        evaluation_problem=evaluation,
        method=method,
        metrics=metrics,
        num_cycles=4,
        max_steps_per_interaction=2,
        num_test_tasks=1,
        practice_reset_policy=PracticeResetPolicy.NEVER,
    )
    assert metrics.num_practice_resets == 0


def test_practice_reset_policy_never_still_samples_a_train_task_per_cycle() -> None:
    """Only the reset is suppressed, not the cycle structure: each period still
    draws its own train task and still hands it to get_practice_policy, so the
    train-task distribution is identical between the two arms and the only
    difference is whether the environment is put back."""
    practice, evaluation, method, metrics = _build_split()
    PracticeLoop.run(
        problem=practice,
        evaluation_problem=evaluation,
        method=method,
        metrics=metrics,
        num_cycles=3,
        max_steps_per_interaction=2,
        num_test_tasks=1,
        practice_reset_policy=PracticeResetPolicy.NEVER,
    )
    assert practice.tasks.train_task_count == 3
    assert method.practice_policy_calls == 3
    assert method.end_cycle_calls == 3


def test_practice_reset_policy_never_rejects_a_reset_interval() -> None:
    """A within-period reset interval under a 'never reset' policy is a
    contradiction, and a silently-ignored flag would make an arm look reset-free
    while it was being reset every k steps. Rejected up front, before the run
    mutates anything."""
    practice, evaluation, method, metrics = _build_split()
    with pytest.raises(ValueError, match="practice_reset_policy"):
        PracticeLoop.run(
            problem=practice,
            evaluation_problem=evaluation,
            method=method,
            metrics=metrics,
            num_cycles=1,
            max_steps_per_interaction=4,
            num_test_tasks=1,
            practice_reset_interval=2,
            practice_reset_policy=PracticeResetPolicy.NEVER,
        )
    assert practice.env.hard_reset_count == 0


def test_practice_reset_policy_never_requires_a_separate_evaluation_problem() -> None:
    """The larger of the two confounds, and the one that was unguarded. Without its
    own evaluation environment, every sweep's per-episode `reset_to_task` resets the
    practice environment `num_test_tasks` times -- so a `never` arm would be reset
    hundreds of times while `Metrics.num_practice_resets` still reported 0, which is
    the very field an experiment reads to certify the manipulation happened. Rejected
    up front, exactly like the `practice_reset_interval` combination."""
    problem, method, metrics = _build()
    with pytest.raises(ValueError, match="evaluation_problem"):
        PracticeLoop.run(
            problem=problem,
            method=method,
            metrics=metrics,
            num_cycles=1,
            max_steps_per_interaction=2,
            num_test_tasks=3,
            practice_reset_policy=PracticeResetPolicy.NEVER,
        )
    assert problem.env.hard_reset_count == 0


def test_practice_reset_policy_never_rejects_an_evaluation_problem_that_is_the_same_object() -> (
    None
):
    """Passing the practice Problem as its own evaluation Problem satisfies "not
    None" while providing none of the isolation, so it must be rejected too."""
    problem, method, metrics = _build()
    with pytest.raises(ValueError, match="evaluation_problem"):
        PracticeLoop.run(
            problem=problem,
            evaluation_problem=problem,
            method=method,
            metrics=metrics,
            num_cycles=1,
            max_steps_per_interaction=2,
            num_test_tasks=3,
            practice_reset_policy=PracticeResetPolicy.NEVER,
        )


def test_practice_reset_policy_never_tells_the_recorder_of_no_period_reset() -> None:
    """The full-loop recording must not draw a reset marker for a reset that did
    not happen -- the markers are how a viewer reads the arm off the video."""
    practice, evaluation, method, metrics = _build_split()
    recorder = _spy_recorder(problem=practice, num_cycles=2, max_steps=2)
    PracticeLoop.run(
        problem=practice,
        evaluation_problem=evaluation,
        method=method,
        metrics=metrics,
        num_cycles=2,
        max_steps_per_interaction=2,
        num_test_tasks=1,
        practice_reset_policy=PracticeResetPolicy.NEVER,
        recorder=recorder,
    )
    assert "period_reset" not in recorder.calls


def test_the_two_reset_policies_must_not_run_identically_when_sampling_moves_the_world() -> None:
    """The test whose absence let a no-op arm ship.

    On Tossing3D the `never` and `scheduled` arms came back byte-identical on 10/10
    seeds in every `stats.json` field except `num_practice_resets` (100 against 0),
    because `sample_train_task` rebuilt the simulator before the loop got as far as
    deciding not to reset. Nothing asserted the manipulation had an effect, so twenty
    committed runs measured one condition twice.

    In the fake, `pre_action_xs` -- the state each practice period actually starts
    from -- is what `stats.json` is downstream of. Identical traces here are the
    identical stats.json there, so this fails if the arms are the same run twice."""
    scheduled, evaluation, method, metrics = _build_split_with(
        tasks_type=_SamplingMovesTheWorldInPlaceTasks
    )
    PracticeLoop.run(
        problem=scheduled,
        evaluation_problem=evaluation,
        method=method,
        metrics=metrics,
        num_cycles=3,
        max_steps_per_interaction=2,
        num_test_tasks=1,
        practice_reset_policy=PracticeResetPolicy.SCHEDULED,
    )
    never, evaluation, method, metrics = _build_split_with(
        tasks_type=_SamplingMovesTheWorldInPlaceTasks
    )
    PracticeLoop.run(
        problem=never,
        evaluation_problem=evaluation,
        method=method,
        metrics=metrics,
        num_cycles=3,
        max_steps_per_interaction=2,
        num_test_tasks=1,
        practice_reset_policy=PracticeResetPolicy.NEVER,
    )
    assert never.env.pre_action_xs != scheduled.env.pre_action_xs, (
        "the two policies practiced from exactly the same states, so `never` is a "
        "label rather than a condition"
    )
    # Spelled out rather than left to the inequality, so a future change that breaks
    # both arms the same way cannot satisfy this test by accident.
    assert scheduled.env.pre_action_xs == [100.0, 101.0, 100.0, 101.0, 100.0, 101.0]
    assert never.env.pre_action_xs == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def test_practice_reset_policy_never_samples_the_train_task_in_place() -> None:
    """Which of the two sampling paths the loop takes is the whole fix, so it is
    pinned directly rather than inferred from the trace above."""
    practice, evaluation, method, metrics = _build_split_with(
        tasks_type=_SamplingMovesTheWorldInPlaceTasks
    )
    PracticeLoop.run(
        problem=practice,
        evaluation_problem=evaluation,
        method=method,
        metrics=metrics,
        num_cycles=3,
        max_steps_per_interaction=2,
        num_test_tasks=1,
        practice_reset_policy=PracticeResetPolicy.NEVER,
    )
    assert isinstance(practice.tasks, _SamplingMovesTheWorldInPlaceTasks)
    assert practice.tasks.in_place_task_count == 3
    assert practice.tasks.train_task_count == 0


def test_practice_reset_policy_scheduled_samples_the_train_task_the_ordinary_way() -> None:
    """The incumbent arm must be untouched by the fix: it is about to be reset into
    the task anyway, so any side effect sampling has is overwritten."""
    practice, evaluation, method, metrics = _build_split_with(
        tasks_type=_SamplingMovesTheWorldInPlaceTasks
    )
    PracticeLoop.run(
        problem=practice,
        evaluation_problem=evaluation,
        method=method,
        metrics=metrics,
        num_cycles=3,
        max_steps_per_interaction=2,
        num_test_tasks=1,
        practice_reset_policy=PracticeResetPolicy.SCHEDULED,
    )
    assert isinstance(practice.tasks, _SamplingMovesTheWorldInPlaceTasks)
    assert practice.tasks.train_task_count == 3
    assert practice.tasks.in_place_task_count == 0


def test_practice_reset_policy_never_refuses_a_domain_whose_in_place_sampling_still_resets() -> (
    None
):
    """The guard that turns this class of defect from silent into loud.

    `Tasks.sample_train_task_in_place` defaults to `sample_train_task`, which is right
    for every domain that builds a task arithmetically and wrong for every domain that
    builds one in the world. A domain of the second kind that has not overridden it
    would otherwise get exactly the original defect back -- a reset-free arm reporting
    zero resets while the scene is rebuilt every cycle -- so the loop checks rather
    than trusts, and refuses mid-run instead of producing a run to publish."""
    practice, evaluation, method, metrics = _build_split_with(
        tasks_type=_SamplingMovesTheWorldTasks
    )
    with pytest.raises(RuntimeError, match="sample_train_task_in_place"):
        PracticeLoop.run(
            problem=practice,
            evaluation_problem=evaluation,
            method=method,
            metrics=metrics,
            num_cycles=1,
            max_steps_per_interaction=2,
            num_test_tasks=1,
            practice_reset_policy=PracticeResetPolicy.NEVER,
        )


class _StuckEnv(_FakeEnv):
    """An environment that never reaches anywhere new: every action leaves x alone.

    The absorbing region a human rescue exists to break, reduced to its essentials --
    _FakeEnv increments x on every step, so with it every state is novel forever and a
    novelty-based help-seeking Method would never fire."""

    def take_action(self, *, action: Action) -> State:
        del action
        current = self.get_current_state()
        self.pre_action_xs.append(float(current[_OBJ][0]))
        return current


class _RecordingHuman(HumanOracle):
    """Costs 2.0 per rescue and writes the commanded target state.

    A distinct, non-1.0 cost so a test can tell a recorded cost apart from a recorded
    count; the real v0 oracle's flat 1.0 would make the two indistinguishable."""

    commands: list[CommandGoalDescription] = []

    @staticmethod
    def calculate_cost_for_human_command(
        *,
        command_start_state_description: CommandStartStateDescription,
        command_goal_description: CommandGoalDescription,
    ) -> Cost:
        del command_start_state_description, command_goal_description
        return 2.0

    @staticmethod
    def execute_human_command(
        *,
        command_start_state_description: CommandStartStateDescription,
        command_goal_description: CommandGoalDescription,
        env: Environment,
    ) -> None:
        del command_start_state_description
        _RecordingHuman.commands.append(command_goal_description)
        target = command_goal_description.target_state
        assert target is not None
        env.set_state(state=target.model_copy(deep=True))


class _AskingMethod(_FakeMethod):
    """A Method that owns the decision to ask, which is the whole point of the split:
    the harness learns of it only by catching the exception this raises.

    Asks once `steps_before_asking` calls have gone by, and stops asking the moment it
    is told help was granted -- a minimal stand-in for what HelpSeekingPolicy does with
    a StuckDetector, without importing it. Deliberately not imported: practice_loop.py
    sits *below* methods/ in the layering, so testing it through a real Method would
    test the pair rather than the mechanism."""

    steps_before_asking: int = 2
    steps_taken: int = 0
    may_ask: bool = True
    # x observed at each observe_help_granted call, in call order.
    help_granted_xs: list[float] = Field(default_factory=list)

    def may_request_human_help(self) -> bool:
        return self.may_ask

    def observe_help_granted(self, *, state: State) -> None:
        self.help_granted_xs.append(float(state[_OBJ][0]))
        self.event_log.record(event="observe_help_granted")
        self.steps_taken = 0

    def get_practice_policy(self, *, task: Task) -> Policy:
        del task
        self.practice_policy_calls += 1
        return lambda state: self._asking_action(state=state)

    def _asking_action(self, *, state: State) -> LabeledAction:
        del state
        if self.steps_taken >= self.steps_before_asking:
            raise HumanHelpRequested
        self.steps_taken += 1
        return LabeledAction(action=np.array([0.0]), label="asking")


def _build_asking(*, steps_before_asking: int = 2):
    """A practice/evaluation split whose Method asks for help, with a human wired in."""
    _RecordingHuman.commands = []
    practice_env = _FakeEnv()
    evaluation_env = _FakeEnv()
    event_log = _EventLog()
    practice = _FakeProblem(
        env=practice_env,
        tasks=_FakeTasks(env=practice_env),
        event_log=event_log,
        human=_RecordingHuman,
    )
    evaluation = _FakeProblem(env=evaluation_env, tasks=_FakeTasks(env=evaluation_env))
    method = _AskingMethod(
        env=practice_env, event_log=event_log, steps_before_asking=steps_before_asking
    )
    return practice, evaluation, method, Metrics()


def test_a_method_that_never_asks_is_never_rescued() -> None:
    """The incumbent path is untouched: a plain Method on a maximally absorbing
    environment behaves exactly as it did before any of this existed. The harness has
    no way to notice on its own, which is the design."""
    _RecordingHuman.commands = []
    practice_env = _StuckEnv()
    practice = _FakeProblem(
        env=practice_env, tasks=_FakeTasks(env=practice_env), human=_RecordingHuman
    )
    evaluation_env = _FakeEnv()
    evaluation = _FakeProblem(env=evaluation_env, tasks=_FakeTasks(env=evaluation_env))
    metrics = Metrics()
    PracticeLoop.run(
        problem=practice,
        evaluation_problem=evaluation,
        method=_FakeMethod(env=practice_env),
        metrics=metrics,
        num_cycles=2,
        max_steps_per_interaction=10,
        num_test_tasks=1,
    )
    assert _RecordingHuman.commands == []
    assert metrics.num_human_interventions() == (0.0, 0)


def test_a_method_that_asks_is_rescued_and_its_cost_recorded() -> None:
    practice, evaluation, method, metrics = _build_asking(steps_before_asking=2)
    PracticeLoop.run(
        problem=practice,
        evaluation_problem=evaluation,
        method=method,
        metrics=metrics,
        num_cycles=1,
        max_steps_per_interaction=9,
        num_test_tasks=1,
    )
    summed_cost, count = metrics.num_human_interventions()
    assert count == len(_RecordingHuman.commands)
    assert count > 0
    # The recording human charges 2.0, so the cost recorded is the one it quoted rather
    # than an assumed unit.
    assert summed_cost == 2.0 * count


def test_the_period_continues_after_a_rescue_rather_than_ending() -> None:
    """The difference from InteractionComplete, measured rather than asserted in prose:
    a period whose Method asks after 2 steps keeps going and is rescued repeatedly,
    where one that raised InteractionComplete would have stopped at 2.

    Each rescue costs the loop iteration it was raised on, so the period spends 3
    iterations per "act, act, ask" cycle: 9 steps is exactly 3 of them."""
    practice, evaluation, method, metrics = _build_asking(steps_before_asking=2)
    PracticeLoop.run(
        problem=practice,
        evaluation_problem=evaluation,
        method=method,
        metrics=metrics,
        num_cycles=1,
        max_steps_per_interaction=9,
        num_test_tasks=1,
    )
    assert metrics.num_human_interventions()[1] == 3
    assert [transitions for transitions, _s, _t in metrics.evaluations] == [0, 6]


def test_a_rescue_consumes_its_loop_iteration_so_an_always_asking_method_cannot_spin() -> None:
    """Bounded by the step budget like everything else. A Method that asks on its very
    first call gets exactly max_steps_per_interaction rescues and no more."""
    practice, evaluation, method, metrics = _build_asking(steps_before_asking=0)
    PracticeLoop.run(
        problem=practice,
        evaluation_problem=evaluation,
        method=method,
        metrics=metrics,
        num_cycles=1,
        max_steps_per_interaction=7,
        num_test_tasks=1,
    )
    assert metrics.num_human_interventions()[1] == 7
    assert [transitions for transitions, _s, _t in metrics.evaluations] == [0, 0]


def test_the_method_is_told_which_state_the_human_left_behind() -> None:
    """observe_help_granted is what lets a Method restart its detector. It gets the
    state *read back* from the environment, not the one that was commanded -- a v1+
    human that only partially succeeds would leave it somewhere else."""
    practice, evaluation, method, metrics = _build_asking(steps_before_asking=2)
    PracticeLoop.run(
        problem=practice,
        evaluation_problem=evaluation,
        method=method,
        metrics=metrics,
        num_cycles=1,
        max_steps_per_interaction=9,
        num_test_tasks=1,
    )
    # _FakeTasks' train task starts at x=100.0 and TASK_INITIAL is the default target.
    assert method.help_granted_xs == [100.0, 100.0, 100.0]


def test_both_method_hooks_fire_on_a_rescue_in_order() -> None:
    """observe_environment_reset against the state about to be lost, then
    observe_help_granted against the state that replaced it. The first is the same
    contract a mid-period interval reset has; the second is new and answers a different
    question."""
    practice, evaluation, method, metrics = _build_asking(steps_before_asking=2)
    PracticeLoop.run(
        problem=practice,
        evaluation_problem=evaluation,
        method=method,
        metrics=metrics,
        num_cycles=1,
        max_steps_per_interaction=3,
        num_test_tasks=1,
    )
    assert [event for event in method.event_log.events if "observe" in event] == [
        "observe_environment_reset",
        "observe_help_granted",
    ]
    # Two actions were taken from x=100, so the state about to be lost is 102.
    assert method.reset_observation_xs == [102.0]


def test_a_rescue_is_not_counted_as_a_free_practice_reset() -> None:
    """num_practice_resets certifies the reset-free manipulation, so a charged human
    rescue must not land in it -- otherwise a `never` arm that asks would look like an
    arm that was quietly reset for free."""
    practice, evaluation, method, metrics = _build_asking(steps_before_asking=2)
    PracticeLoop.run(
        problem=practice,
        evaluation_problem=evaluation,
        method=method,
        metrics=metrics,
        num_cycles=1,
        max_steps_per_interaction=9,
        num_test_tasks=1,
        practice_reset_policy=PracticeResetPolicy.NEVER,
    )
    assert metrics.num_practice_resets == 0
    assert metrics.num_human_interventions()[1] > 0


def test_a_rescue_does_not_end_the_cycle_or_retrain() -> None:
    practice, evaluation, method, metrics = _build_asking(steps_before_asking=2)
    PracticeLoop.run(
        problem=practice,
        evaluation_problem=evaluation,
        method=method,
        metrics=metrics,
        num_cycles=2,
        max_steps_per_interaction=9,
        num_test_tasks=1,
    )
    assert metrics.num_human_interventions()[1] > 2
    assert method.end_cycle_calls == 2


def test_the_task_initial_target_asks_for_this_periods_own_task() -> None:
    practice, evaluation, method, metrics = _build_asking(steps_before_asking=2)
    PracticeLoop.run(
        problem=practice,
        evaluation_problem=evaluation,
        method=method,
        metrics=metrics,
        num_cycles=1,
        max_steps_per_interaction=9,
        num_test_tasks=1,
        human_reset_target=HumanResetTarget.TASK_INITIAL,
    )
    # One train task drawn for the period and none for any rescue.
    assert practice.tasks.train_task_count == 1
    assert all(
        float(command.target_state[_OBJ][0]) == 100.0 for command in _RecordingHuman.commands
    )


def test_the_random_target_draws_a_fresh_train_task_per_rescue() -> None:
    practice, evaluation, method, metrics = _build_asking(steps_before_asking=2)
    PracticeLoop.run(
        problem=practice,
        evaluation_problem=evaluation,
        method=method,
        metrics=metrics,
        num_cycles=1,
        max_steps_per_interaction=9,
        num_test_tasks=1,
        human_reset_target=HumanResetTarget.RANDOM,
    )
    rescues = len(_RecordingHuman.commands)
    assert rescues > 0
    # One for the period itself, plus one per rescue.
    assert practice.tasks.train_task_count == 1 + rescues


def test_no_evaluation_episode_is_ever_rescued() -> None:
    """Measurement runs on the evaluation Problem, which is never handed a practice
    policy -- so a rescued arm's score is still the robot's own. Pinned by giving the
    evaluation Problem no human at all: if the loop ever tried, it would raise."""
    practice, evaluation, method, metrics = _build_asking(steps_before_asking=0)
    assert evaluation.human is None
    PracticeLoop.run(
        problem=practice,
        evaluation_problem=evaluation,
        method=method,
        metrics=metrics,
        num_cycles=2,
        max_steps_per_interaction=4,
        num_test_tasks=3,
    )
    assert metrics.num_human_interventions()[1] == 8


def test_a_method_that_may_ask_requires_a_human_on_the_problem() -> None:
    """Up front, before hard_reset, rather than at the first rescue: a run that would
    crash three cycles in should not first spend three cycles. Read from the Method
    exactly once -- the loop never polls it per step."""
    practice_env = _FakeEnv()
    practice = _FakeProblem(env=practice_env, tasks=_FakeTasks(env=practice_env))
    evaluation_env = _FakeEnv()
    evaluation = _FakeProblem(env=evaluation_env, tasks=_FakeTasks(env=evaluation_env))
    with pytest.raises(ValueError, match="HumanOracle"):
        PracticeLoop.run(
            problem=practice,
            evaluation_problem=evaluation,
            method=_AskingMethod(env=practice_env),
            metrics=Metrics(),
            num_cycles=1,
            max_steps_per_interaction=10,
            num_test_tasks=1,
        )
    assert practice.env.hard_reset_count == 0


def test_a_method_that_cannot_ask_needs_no_human_at_all() -> None:
    """The converse: the validation keys off the Method's own declaration, not off the
    environment or any flag, so a run whose Method never asks needs nothing wired."""
    practice_env = _StuckEnv()
    practice = _FakeProblem(env=practice_env, tasks=_FakeTasks(env=practice_env))
    evaluation_env = _FakeEnv()
    evaluation = _FakeProblem(env=evaluation_env, tasks=_FakeTasks(env=evaluation_env))
    PracticeLoop.run(
        problem=practice,
        evaluation_problem=evaluation,
        method=_AskingMethod(env=practice_env, may_ask=False),
        metrics=Metrics(),
        num_cycles=0,
        max_steps_per_interaction=10,
        num_test_tasks=1,
    )
    assert practice.env.hard_reset_count == 1


def test_interaction_complete_still_ends_the_period_for_a_method_with_a_human() -> None:
    """The two signals stay distinct at the harness too: a Method that raises
    InteractionComplete ends its period exactly as it always did, and is not rescued --
    there is no longer any path from that exception to the human."""
    _RecordingHuman.commands = []
    practice_env = _FakeEnv()
    practice = _FakeProblem(
        env=practice_env, tasks=_FakeTasks(env=practice_env), human=_RecordingHuman
    )
    evaluation_env = _FakeEnv()
    evaluation = _FakeProblem(env=evaluation_env, tasks=_FakeTasks(env=evaluation_env))
    metrics = Metrics()
    PracticeLoop.run(
        problem=practice,
        evaluation_problem=evaluation,
        method=_EarlyStoppingMethod(env=practice_env, steps_before_stopping=2),
        metrics=metrics,
        num_cycles=1,
        max_steps_per_interaction=10,
        num_test_tasks=1,
    )
    assert metrics.num_human_interventions() == (0.0, 0)
    assert _RecordingHuman.commands == []
    assert [transitions for transitions, _s, _t in metrics.evaluations] == [0, 2]


def test_a_rescue_is_handed_to_the_recorder_as_its_own_reset_kind() -> None:
    practice, evaluation, method, metrics = _build_asking(steps_before_asking=2)
    recorder = _spy_recorder(problem=practice, num_cycles=1, max_steps=9)
    PracticeLoop.run(
        problem=practice,
        evaluation_problem=evaluation,
        method=method,
        metrics=metrics,
        num_cycles=1,
        max_steps_per_interaction=9,
        num_test_tasks=1,
        recorder=recorder,
    )
    assert [call for call in recorder.calls if call.startswith("human_reset")] == [
        "human_reset:2",
        "human_reset:5",
        "human_reset:8",
    ]
