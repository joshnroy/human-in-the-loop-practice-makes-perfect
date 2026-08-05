from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np

from hitl_pmp.core.method.method import InteractionComplete, Method
from hitl_pmp.core.metrics.metrics import Metrics
from hitl_pmp.core.metrics.types import TaskOutcome
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.core.renderer.renderer import Renderer
from hitl_pmp.recording.loop_recorder import LoopRecorder


class PracticeLoop:
    """Drives PMP-style online learning: one initial evaluation, then num_cycles
    rounds of (one interaction period, an optional per-cycle retraining hook,
    an evaluation sweep over sampled test tasks) -- mirrors predicators'
    main.py:_run_pipeline. hard_reset() is called exactly once, before the very
    first evaluation, and never again -- but this does NOT mean state is reset-free
    end to end. Every episode-like unit starts from a task's own initial state:
    run_task_episode (once per test task inside _evaluate) and each interaction
    period (via problem.reset_to_task on the train task just sampled) both do so,
    matching predicators, which resets per interaction request (main.py:301-302).
    State is therefore continuous only *within* one interaction period -- a Method's
    own get_task_policy is expected to decide internally, e.g. by checking
    task.goal.is_satisfied against the current state on every call, when to keep
    pursuing the sampled task's goal versus switch to self-directed practice for
    the rest of the period.

    That per-period reset is load-bearing rather than tidiness. An evaluation
    episode ends with the environment in a *solved* state, so a period that resumed
    from it would begin having already achieved the goal -- on Light Switch, with
    the robot standing at the light, skipping the whole traversal and spending
    every step of its budget practicing the toggle. That inflates practice
    throughput per period and makes grid_size stop affecting results at all.

    **Reset frequency, decoupled from refit frequency.** `practice_reset_interval`
    (None by default, i.e. exactly the behaviour described above) additionally puts
    the environment back to the *current* practice task's initial state every k
    steps *within* a period, without ending the cycle and therefore without firing
    `end_cycle()`. The two were previously welded to the same loop boundary, so
    `num_cycles` set how often the robot is rescued and how often the samplers
    refit with a single number -- and an experiment varying one necessarily varied
    the other (PR #39's Tossing Room reset-frequency experiment, whose arms ended
    ~40 competence points apart on identical experience). It
    resets to the same task rather than sampling a new one, so the train-task
    distribution is untouched; resets are not charged as transitions, matching how
    the per-period reset is already uncharged; and `Metrics.num_practice_resets`
    counts what actually happened, so the manipulation is measurable from a run's
    own output rather than inferred from the flag. Immediately before each such
    reset the Method is told (`observe_environment_reset`) -- see that method for
    why skipping it silently mislabels one skill outcome per reset.

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

    num_online_transitions -- the x-axis of every learning curve this produces --
    counts environment steps taken during *interaction periods only*; the
    evaluation sweeps cost real compute but are deliberately not charged, since
    the metric measures how much experience the agent needed, not how much was
    spent measuring it. The count is data-driven rather than budget-driven: a
    period that ends early (the Method raises InteractionComplete) contributes
    only the steps it actually took. That matches predicators, which accumulates
    `sum(len(result.actions) for result in interaction_results)` over the
    trajectories its explorers really produced (main.py:244) rather than assuming
    each request ran to max_num_steps_interaction_request.

    **Fixed test set.** The evaluation set is drawn once, before the first sweep, and
    reused for the whole run -- matching predicators, whose `BaseEnv.get_test_tasks`
    generates once and caches (`envs/base_env.py:180-193`).

    This is load-bearing, not tidiness. Re-sampling it per sweep
    (which this loop used to do) means every point on a learning curve is measured on
    a different set, so the curve carries task-sampling variance on top of the policy
    change it is supposed to isolate -- and a Method whose competence is uneven across
    the task distribution can then swing between a favourable and an unfavourable draw
    from one cycle to the next. On Ball-Ring, where tasks differ precisely in the
    quantity the sampler is learning, that produced ~3x the per-seed variance of the
    reference implementation and apparent 100%-to-0% collapses within a single cycle.
    A fixed set makes consecutive points comparable, which is the whole point of a
    learning curve. Measured on Light Switch EES at 10 seeds, it cuts pooled
    across-seed sd during the climb from 16.6 to 10.4 and downward steps from 5 to 2
    (of 100 adjacent pairs). It also means `num_test_tasks` is the *denominator*
    everywhere, so `_evaluate` reads it off the list rather than being told twice.

    The *train* side has a separate, unrelated fidelity gap -- predicators draws each
    period's task with replacement from a cached pool of `num_train_tasks = 50`
    (`approaches/online_nsrt_learning_approach.py:66-67`) where this loop draws from an
    unbounded never-repeating stream. That is deliberately NOT fixed here: implementing
    it alongside this change measurably cancelled this change's benefit on Light Switch
    (sd back to 15.7, 6 regressions, final mean 100% -> 97%), so it needs its own PR
    and its own evidence rather than riding along with a protocol fix.

    If renderer is given, the *first* test task of each rendered evaluation sweep
    is recorded. Which sweeps those are is set by num_render_checkpoints: 1 (the
    default) records only the final sweep, i.e. a single post-hoc demo clip of the
    finished policy. A larger value spreads recordings evenly from sweep 0 (before
    any practice) through the last, which turns the output into a visible
    *progression* -- the same task attempted by a policy at increasing levels of
    competence, which is the thing worth looking at for a Method that actually
    learns. Unrendered sweeps cost nothing, so this stays opt-in rather than
    always-on.

    **Rendered frames are always streamed, never accumulated.** Each sweep's
    frames are handed to on_checkpoint_frames the moment that sweep finishes and
    then dropped; run() returns nothing, and nothing in this class accumulates
    frames across sweeps. Peak retention is therefore one sweep's worth,
    regardless of how long a run is.

    That is not a default, it is the only behaviour, and deliberately so.
    Retaining every rendered checkpoint until the run ends is an unbounded buffer
    whose size is (checkpoints x episode length x frame bytes), none of which this
    class controls. It is fine for a small 2D domain and not fine in general: a 3D
    domain rendering 1920x1080 RGB reaches gigabytes within a few checkpoints, and
    on this project a runaway ~48 GB process already triggered the kernel OOM
    killer, which -- because a tmux pane's systemd scope defaults to
    OOMPolicy=stop -- tore down the entire session rather than just the offending
    process. Retaining used to be available by simply *omitting* a sink, which
    made reintroducing that buffer a matter of forgetting an argument; the
    accumulation code is gone rather than merely discouraged.

    **The one rule for on_checkpoint_frames**: it is required whenever renderer is
    given, and ignored (so it may be omitted) when renderer is None, since a run
    with no renderer produces no frames for anyone to receive. Passing a renderer
    without a sink is a ValueError raised up front, before the run starts, rather
    than a silent discard of every clip.

    **recorder is a separate, orthogonal thing from renderer.** renderer records
    *evaluation episodes* at checkpoints, into one clip each. recorder (a
    recording.LoopRecorder, off by default) records the whole outer loop -- every
    practice period, every evaluation episode of every sweep, and every reset --
    into a single continuous video. The hooks below are the only difference a
    recorder makes: it is handed state this loop already has, and never gets to
    decide anything, so a recorded run takes exactly the actions an unrecorded one
    does. See LoopRecorder's own docstring."""

    @staticmethod
    def run(
        *,
        problem: Problem,
        method: Method,
        metrics: Metrics,
        num_cycles: int,
        max_steps_per_interaction: int,
        num_test_tasks: int,
        practice_reset_interval: int | None = None,
        on_cycle_end: Callable[[], None] | None = None,
        renderer: type[Renderer] | None = None,
        num_render_checkpoints: int = 1,
        on_checkpoint_frames: CheckpointFramesSink | None = None,
        recorder: LoopRecorder | None = None,
    ) -> None:
        # Up front, before hard_reset(), so a caller that forgot the sink finds out
        # before the run mutates the environment rather than one sweep in.
        if renderer is not None and on_checkpoint_frames is None:
            raise ValueError(
                "on_checkpoint_frames is required when renderer is given: rendered "
                "frames are streamed to the sink and never retained, so without one "
                "every recorded clip would be silently discarded."
            )
        rendered_sweeps = PracticeLoop.render_sweep_indices(
            num_cycles=num_cycles, num_render_checkpoints=num_render_checkpoints
        )

        def hand_over(*, transitions: int, sweep_frames: list[np.ndarray]) -> None:
            """Give this sweep's frames to the sink and keep no reference to them.

            The empty-list guard is load-bearing, not an optimization: it is what
            lets a renderer-free run legally have no sink at all, since every sweep
            of such a run produces an empty list. The `is not None` check is for
            mypy's benefit -- narrowing from the guard above does not reach into
            this closure -- and is unreachable in practice for a non-empty list."""
            if not sweep_frames or on_checkpoint_frames is None:
                return
            on_checkpoint_frames(transitions=transitions, frames=sweep_frames)

        problem.hard_reset()
        if recorder is not None:
            recorder.record_hard_reset(state=problem.get_current_state())
        # Drawn ONCE, up front -- see the class docstring's "fixed test set"
        # section for why re-sampling it per sweep corrupts the learning curve.
        test_tasks = [problem.sample_test_task() for _ in range(num_test_tasks)]
        num_online_transitions = 0
        frames = PracticeLoop._evaluate(
            problem=problem,
            method=method,
            metrics=metrics,
            test_tasks=test_tasks,
            num_online_transitions=num_online_transitions,
            renderer=renderer if 0 in rendered_sweeps else None,
            recorder=recorder,
            sweep_index=0,
        )
        hand_over(transitions=num_online_transitions, sweep_frames=frames)
        for cycle in range(num_cycles):
            task = problem.sample_train_task()
            # get_practice_policy, not get_task_policy: a learning Method explores
            # (and records training data) during the interaction period, but must
            # not do either during the evaluation sweep below, which runs on
            # held-out test tasks. Non-learning Methods inherit the default, which
            # just forwards to get_task_policy -- see Method's own docstrings.
            policy = method.get_practice_policy(task=task)
            if recorder is not None:
                recorder.begin_practice(
                    cycle_index=cycle,
                    transitions=num_online_transitions,
                    task=task.goal.describe(),
                )
            # Start the period at the task just sampled, rather than resuming from
            # whatever the preceding evaluation sweep left behind. predicators does
            # the same (main.py:301-302, `cogman.reset(env_task)` per interaction
            # request), and it is load-bearing: an evaluation episode ends with the
            # environment in a *solved* state, so resuming from it would hand every
            # free period a head start it never earned.
            state = problem.reset_to_task(task=task)
            metrics.record_practice_reset()
            if recorder is not None:
                recorder.record_period_reset(state=state)
            for step in range(max_steps_per_interaction):
                try:
                    labeled_action = policy(state)
                except InteractionComplete:
                    # The Method has nothing further worth practicing. Ending
                    # early is normal, and the steps not taken are not charged --
                    # see InteractionComplete's own docstring for why the count is
                    # data-driven rather than budget-driven.
                    if recorder is not None:
                        recorder.record_interaction_complete(
                            state=state, step_index=step, transitions=num_online_transitions
                        )
                    break
                state = problem.take_action(action=labeled_action.action)
                num_online_transitions += 1
                if recorder is not None:
                    recorder.record_practice_step(
                        state=state,
                        skill=labeled_action.label,
                        step_index=step,
                        transitions=num_online_transitions,
                    )
                if PracticeLoop._reset_is_due(
                    step=step,
                    max_steps_per_interaction=max_steps_per_interaction,
                    practice_reset_interval=practice_reset_interval,
                ):
                    # Hand back what the environment is about to lose, so a Method
                    # scoring an in-flight skill scores it against what really
                    # happened rather than against the initial state it is about to
                    # be teleported to -- see Method.observe_environment_reset.
                    method.observe_environment_reset(state=state)
                    # The SAME task, deliberately: sampling a fresh one here would
                    # change the train-task distribution along with the reset rate,
                    # which is the confound this knob exists to avoid.
                    state = problem.reset_to_task(task=task)
                    metrics.record_practice_reset()
                    if recorder is not None:
                        recorder.record_interval_reset(
                            state=state, step_index=step, transitions=num_online_transitions
                        )
            # Before this cycle's evaluation, so the sweep actually measures what
            # the Method just learned rather than lagging a cycle behind.
            method.end_cycle()
            if on_cycle_end is not None:
                on_cycle_end()
            frames = PracticeLoop._evaluate(
                problem=problem,
                method=method,
                metrics=metrics,
                test_tasks=test_tasks,
                num_online_transitions=num_online_transitions,
                renderer=renderer if (cycle + 1) in rendered_sweeps else None,
                recorder=recorder,
                sweep_index=cycle + 1,
            )
            hand_over(transitions=num_online_transitions, sweep_frames=frames)

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
    def _reset_is_due(
        *, step: int, max_steps_per_interaction: int, practice_reset_interval: int | None
    ) -> bool:
        """Whether the interaction period should be put back to its task's initial
        state after the (0-indexed) step just taken.

        Never on the period's last step: the next period opens with its own reset,
        so firing here would double-count a reset that changes nothing. With
        practice_reset_interval == max_steps_per_interaction that leaves exactly
        one reset per period -- today's behaviour, reached by an explicit interval
        rather than by the default -- and an interval k that divides the period
        length n leaves exactly n // k resets per period."""
        if practice_reset_interval is None:
            return False
        if step + 1 >= max_steps_per_interaction:
            return False
        return (step + 1) % practice_reset_interval == 0

    @staticmethod
    def _evaluate(
        *,
        problem: Problem,
        method: Method,
        metrics: Metrics,
        test_tasks: list[Task],
        num_online_transitions: int,
        renderer: type[Renderer] | None = None,
        recorder: LoopRecorder | None = None,
        sweep_index: int = 0,
    ) -> list[np.ndarray]:
        num_solved = 0
        frames: list[np.ndarray] = []
        outcomes: list[TaskOutcome] = []
        if recorder is not None:
            recorder.begin_evaluation(sweep_index=sweep_index, transitions=num_online_transitions)
        for i, task in enumerate(test_tasks):
            policy = method.get_task_policy(task=task)
            # The checkpoint clip is of test task 0 only; a full-loop recording is
            # of the whole loop, so it renders every episode of the sweep. Its
            # renderer wins where both are on -- they are the same renderer in
            # practice, and rendering the episode twice would only cost time.
            episode_renderer = renderer if i == 0 else None
            if recorder is not None:
                policy = recorder.watch_policy(policy=policy)
                episode_renderer = recorder.renderer
            solved, task_frames = problem.run_task_episode(
                task=task, policy=policy, renderer=episode_renderer
            )
            if i == 0 and renderer is not None:
                frames = task_frames
            if recorder is not None:
                recorder.record_evaluation_episode(
                    task_index=i,
                    num_tasks=len(test_tasks),
                    task=task.goal.describe(),
                    frames=task_frames,
                    solved=solved,
                )
            num_solved += int(solved)
            # test_tasks is drawn once for the whole run, so i identifies the
            # same Task at every checkpoint -- a task can be followed across the
            # curve, not just counted within one sweep.
            outcomes.append(TaskOutcome(task_index=i, goal=task.goal.describe(), solved=solved))
        metrics.record_evaluation(
            num_online_transitions=num_online_transitions,
            num_solved=num_solved,
            num_total=len(test_tasks),
            outcomes=tuple(outcomes),
        )
        return frames


class CheckpointFramesSink(Protocol):
    """Receives one evaluation sweep's rendered frames as soon as that sweep ends.

    A Protocol rather than a bare Callable because every parameter in this
    codebase is keyword-only (ruff PLR0917, max-positional-args = 0), which
    Callable cannot express. Implementations are expected to consume the frames
    immediately -- writing them to disk -- and not retain them: PracticeLoop has
    already dropped its own reference by the time this returns, so a sink that
    accumulates reintroduces exactly the unbounded buffer that streaming exists to
    remove. This is the only route rendered frames take out of a run; there is no
    return value to read them from instead."""

    def __call__(self, *, transitions: int, frames: list[np.ndarray]) -> None: ...
