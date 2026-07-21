from collections.abc import Callable

import numpy as np

from hitl_pmp.core.method.method import Method
from hitl_pmp.core.metrics.metrics import Metrics
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.core.renderer.renderer import Renderer


class PracticeLoop:
    """Drives PMP-style online learning: one initial evaluation, then num_cycles
    rounds of (one interaction period, an optional per-cycle retraining hook,
    an evaluation sweep over sampled test tasks) -- mirrors predicators'
    main.py:_run_pipeline. hard_reset() is called exactly once, before the very
    first evaluation, and never again -- but this does NOT mean state is reset-free
    end to end: run_task_episode (called once per test task inside _evaluate) resets
    the environment itself via env.set_state(state=task.initial_state), since it has
    to start each evaluation episode from that task's own initial state. So state is
    only continuous *within* one interaction period (a Method's own get_task_policy
    is expected to decide internally, e.g. by checking task.goal.is_satisfied
    against the current state on every call, when to keep pursuing the sampled
    task's goal versus switch to self-directed practice for the rest of the period);
    each new interaction period actually resumes from wherever the immediately
    preceding evaluation sweep's last episode left off, not from where the
    interaction period itself ended.

    Domain- and Method-agnostic (any core.Problem/core.Method/core.Metrics
    triple) -- lives at the top level, alongside cli.py, since it's the one
    execution harness every core.Method runs through, not something specific
    to the "Practice Makes Perfect" paper reproduction. A non-learning Method
    (e.g. a privileged oracle) is just num_cycles=0 through this same loop --
    mirrors predicators' own main.py:_run_pipeline, whose only structural
    fork for a non-learning approach is skipping the online-learning cycles
    entirely and running one evaluation sweep; there's no separate pipeline
    anywhere in predicators either, and no is_learning_based-style branch
    needed here -- a Method that doesn't learn just gets called with
    num_cycles=0 by its caller, or has no-op on_cycle_end/
    improve_skill_parameters if it does use cycles for some other reason
    (e.g. re-evaluating checkpoints without retraining).

    problem/method/metrics are real instances now (constructed by the caller's
    own composition root, e.g. environments/lightswitch/cli.py's
    LightSwitchCli.run_method), not classes with shared ClassVar state to wire
    beforehand -- there is no separate "remember to set Problem.env/
    Problem.tasks first" step anymore; whatever problem instance is passed in
    already has everything it needs.

    If renderer is given, the *first* test task of each rendered evaluation sweep
    is recorded, and run() returns {num_online_transitions: frames}. Which sweeps
    those are is set by num_render_checkpoints: 1 (the default) records only the
    final sweep, i.e. a single post-hoc demo clip of the finished policy. A larger
    value spreads recordings evenly from sweep 0 (before any practice) through the
    last, which turns the output into a visible *progression* -- the same task
    attempted by a policy at increasing levels of competence, which is the thing
    worth looking at for a Method that actually learns. Unrendered sweeps cost
    nothing, so this stays opt-in rather than always-on."""

    @staticmethod
    def run(
        *,
        problem: Problem,
        method: Method,
        metrics: Metrics,
        num_cycles: int,
        max_steps_per_interaction: int,
        num_test_tasks: int,
        on_cycle_end: Callable[[], None] | None = None,
        renderer: type[Renderer] | None = None,
        num_render_checkpoints: int = 1,
    ) -> dict[int, list[np.ndarray]]:
        rendered_sweeps = PracticeLoop.render_sweep_indices(
            num_cycles=num_cycles, num_render_checkpoints=num_render_checkpoints
        )
        frames_by_transitions: dict[int, list[np.ndarray]] = {}

        problem.hard_reset()
        num_online_transitions = 0
        frames = PracticeLoop._evaluate(
            problem=problem,
            method=method,
            metrics=metrics,
            num_test_tasks=num_test_tasks,
            num_online_transitions=num_online_transitions,
            renderer=renderer if 0 in rendered_sweeps else None,
        )
        if frames:
            frames_by_transitions[num_online_transitions] = frames
        for cycle in range(num_cycles):
            task = problem.sample_train_task()
            policy = method.get_task_policy(task=task)
            state = problem.get_current_state()
            for _ in range(max_steps_per_interaction):
                labeled_action = policy(state)
                state = problem.take_action(action=labeled_action.action)
                num_online_transitions += 1
            if on_cycle_end is not None:
                on_cycle_end()
            frames = PracticeLoop._evaluate(
                problem=problem,
                method=method,
                metrics=metrics,
                num_test_tasks=num_test_tasks,
                num_online_transitions=num_online_transitions,
                renderer=renderer if (cycle + 1) in rendered_sweeps else None,
            )
            if frames:
                frames_by_transitions[num_online_transitions] = frames
        return frames_by_transitions

    @staticmethod
    def render_sweep_indices(*, num_cycles: int, num_render_checkpoints: int) -> frozenset[int]:
        """Which evaluation sweeps to record, as indices into the num_cycles + 1
        sweeps (0 = the initial one, before any practice). Evenly spaced and
        always inclusive of both ends, so recordings span untrained through fully
        trained rather than clustering at one end."""
        num_sweeps = num_cycles + 1
        checkpoints = max(1, min(num_render_checkpoints, num_sweeps))
        if checkpoints == 1:
            return frozenset({num_cycles})
        step = num_cycles / (checkpoints - 1)
        return frozenset(round(index * step) for index in range(checkpoints))

    @staticmethod
    def _evaluate(
        *,
        problem: Problem,
        method: Method,
        metrics: Metrics,
        num_test_tasks: int,
        num_online_transitions: int,
        renderer: type[Renderer] | None = None,
    ) -> list[np.ndarray]:
        num_solved = 0
        frames: list[np.ndarray] = []
        for i in range(num_test_tasks):
            task = problem.sample_test_task()
            solved, task_frames = problem.run_task_episode(
                task=task,
                policy=method.get_task_policy(task=task),
                renderer=renderer if i == 0 else None,
            )
            if i == 0 and renderer is not None:
                frames = task_frames
            num_solved += int(solved)
        metrics.record_evaluation(
            num_online_transitions=num_online_transitions,
            num_solved=num_solved,
            num_total=num_test_tasks,
        )
        return frames
