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

    If renderer is given, the *first* test task of the *last* evaluation sweep
    (the one after the final cycle, or the sole initial evaluation if
    num_cycles=0) is recorded and returned as a list of frames -- an empty list
    otherwise. Only the last sweep is ever rendered (not every one) since this is
    meant for a single post-hoc demo clip, not per-checkpoint video; earlier
    sweeps render nothing so run() doesn't pay rendering cost it can't use."""

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
    ) -> list[np.ndarray]:
        problem.hard_reset()
        num_online_transitions = 0
        frames = PracticeLoop._evaluate(
            problem=problem,
            method=method,
            metrics=metrics,
            num_test_tasks=num_test_tasks,
            num_online_transitions=num_online_transitions,
            renderer=renderer if num_cycles == 0 else None,
        )
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
            is_last_cycle = cycle == num_cycles - 1
            frames = PracticeLoop._evaluate(
                problem=problem,
                method=method,
                metrics=metrics,
                num_test_tasks=num_test_tasks,
                num_online_transitions=num_online_transitions,
                renderer=renderer if is_last_cycle else None,
            )
        return frames

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
