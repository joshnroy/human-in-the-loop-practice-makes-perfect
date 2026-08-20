from __future__ import annotations

import enum
from collections.abc import Callable
from typing import Protocol

import numpy as np

from hitl_pmp.core.method.method import (
    HumanHelpRequested,
    HumanRandomTaskResetRequested,
    InteractionComplete,
    Method,
)
from hitl_pmp.core.metrics.metrics import Metrics
from hitl_pmp.core.metrics.types import TaskOutcome
from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.core.renderer.renderer import Renderer
from hitl_pmp.episode_traces import EpisodeTraceRecorder
from hitl_pmp.recording.loop_recorder import LoopRecorder


class PracticeResetPolicy(str, enum.Enum):
    """Whether an interaction period is put back to its train task's initial state
    before it begins.

    Defined above `PracticeLoop` rather than below it (which the file's top-down
    convention would otherwise want) for one mechanical reason: it is the *default
    value* of a `PracticeLoop.run` parameter, so it has to exist by the time that
    `def` is evaluated.

    A `(str, Enum)` -- matching this project's other domain enums, e.g.
    `environments/tossingroom/tasks.py`'s
    `TossingRoomGoalType` -- so
    argparse can offer the members directly as `choices`, so a member compares equal
    to its own wire string, and so the chosen value lands in `config_snapshot.json`
    as a readable word rather than an integer nobody can interpret later."""

    # Today's behaviour, and the default: reset at the top of every period. Named
    # "scheduled" rather than "always" because --practice-reset-interval may add
    # further resets *within* the period on a schedule of its own.
    SCHEDULED = "scheduled"
    # No practice reset, ever. The environment carries whatever state the previous
    # period left it in, across the period boundary and across the train task
    # changing underneath it. This is only truthful because evaluation now runs on
    # its own Environment -- otherwise each sweep's per-episode reset_to_task would
    # silently reset this arm num_test_tasks times per sweep.
    NEVER = "never"

    def __str__(self) -> str:
        """So argparse prints `scheduled`/`never` in --help and in its error message
        for a bad value, rather than `PracticeResetPolicy.SCHEDULED`."""
        return self.value


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

    **The per-period reset is optional.** `practice_reset_policy` (`SCHEDULED` by
    default, i.e. exactly the behaviour described above) turns that reset off
    entirely at `NEVER`: the environment carries whatever the previous period left
    behind, across the period boundary and across the train task changing underneath
    it. A train task is still produced per period and still handed to
    `get_practice_policy`, so the cycle structure is identical between the two
    policies.

    **The two policies draw that task differently, and must.** `NEVER` goes through
    `Tasks.sample_train_task_in_place` (`_sample_practice_task` below), because on a
    domain that builds its tasks *in the world* -- Tossing3D, whose only way to obtain
    an initial `State` is to rebuild the MuJoCo scene -- an ordinary `sample_train_task`
    is itself a reset, and one this loop then reported as zero. So on such a domain the
    *sequence of `Task` objects* is NOT identical between the arms: the reset-free arm
    practices in the one scene `hard_reset` left it, because there handing the robot a
    new scene and resetting it are the same physical act. On every arithmetic domain
    the default keeps the sequences identical, as before.

    **But the two policies are NOT "the same run minus one reset", and an experiment
    comparing them must say so.** A `Task`'s continuous parameters live in its
    `initial_state`, and `reset_to_task` is the only thing that installs one. Under
    `NEVER` the environment therefore keeps whatever `hard_reset()` put there for the
    whole run, so any state feature no action writes is **frozen at its canonical
    value**. On the retired frozen-weight Tossing Room forks those features were a bin's
    `throw_distance` and an item's `weight` -- exactly the learned sampler's input row
    -- so a `NEVER` run practiced every throw at a single point of the task
    distribution while a `SCHEDULED` run saw a fresh draw per period. Measured on the
    2026-08-06 A/B: 194 greedy throws at **1** distinct required-force target under
    `NEVER`, against 86 distinct targets over 440 throws under `SCHEDULED`.

    `tossingroom` exists to break that entanglement: it draws the
    weight at pickup, on an action the robot takes, so `NEVER` no longer collapses the
    sampler's training distribution. The general hazard below still applies to any
    domain whose sampler inputs no action writes.

    That is a second, entangled difference, not a side note: it collapses the
    sampler's training distribution at the same time as it removes the rescue. The
    two cannot be separated by this flag alone, and any result comparing the policies
    is a result about both mechanisms together. See
    `docs/experiment-logs/2026-08-06-reset-free-practice-ab.md`.

    That is the reset-free condition the paragraph above argues *against* on Light
    Switch, and stating it as a flag is the point: the argument is an empirical claim
    about how much a free reset is worth, and it was never measured. `NEVER` is what
    makes it measurable, and `Metrics.num_practice_resets` (0 for the whole run) is
    how a reader confirms from a run's own output that the manipulation happened.

    It is only truthful in combination with `evaluation_problem` above. Without a
    separate evaluation environment, every sweep's per-episode `reset_to_task` resets
    this arm `num_test_tasks` times anyway, and `NEVER` would be a label rather than
    a condition. Combining `NEVER` with `practice_reset_interval` is rejected up
    front for the same reason -- it would be a reset-free arm that is reset every k
    steps.

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

    **Separate evaluation environment.** `evaluation_problem` (None by default) is a
    second, independently-constructed Problem -- its own Environment and its own Tasks
    -- that every evaluation sweep runs on. The practice Problem is then never touched
    by measurement at all.

    This is not tidiness either. Every domain's `run_task_episode` opens with
    `reset_to_task`, a privileged state-write, so a sweep over an n-task test set
    performs n such writes **on the environment practice inherits**. At the default 30
    test tasks that is 30 resets per sweep handed to the practice environment for free.
    Under the default per-period reset the next period overwrites that state anyway, so
    it is invisible -- but it makes a reset-free practice arm impossible to express:
    "never reset" would still be reset 30 times per sweep, and the manipulation being
    measured would not exist. Splitting the environments is what makes
    `practice_reset_policy` mean anything.

    Passing one is safe only where the split is genuinely byte-identical -- the
    environment must consume no randomness of its own, or a second instance changes
    which draws practice sees. Two domains are wired today: `tossingroom` (no
    environment RNG, pure `take_action`), and `tossing3d`, which holds no RNG field at
    all -- its only randomness lives in `Tossing3DTasks`' train/test streams, and the
    simulator is re-seeded from the scene seed on every reset, so no history carries
    across instances. `tossing3d` pays a cost the others do not, a **second live MuJoCo
    scene** for the length of the run, which is why it was deferred rather than
    excluded outright; a sweep's memory cap has to be sized for it.

    `ballring` remains *excluded on the merits* and should not simply be wired by the
    next reader: its `_noise_rng` is consumed by evaluation today and therefore shifts
    the practice stream, so a split needs a re-baseline rather than an identity check.
    `lightswitch` is simply **not migrated yet**, not assessed and rejected; it is
    RNG-free in the same way and could be wired when something needs it. Every
    unmigrated domain omits the argument and keeps exactly its old behaviour, one
    Problem for both roles.

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
        evaluation_problem: Problem | None = None,
        practice_reset_policy: PracticeResetPolicy = PracticeResetPolicy.SCHEDULED,
        practice_reset_interval: int | None = None,
        on_cycle_end: Callable[[], None] | None = None,
        # Fired after every evaluation sweep, including the one before any practice --
        # so a caller sees num_cycles + 1 calls. Distinct from on_cycle_end, which
        # fires *before* the sweep that measures the cycle it ends: a caller reporting
        # progress needs the sweep's result, which does not exist yet at that point.
        # Purely an observer, like recorder: it is handed nothing to decide and its
        # return value is ignored, so a run with one takes the same actions.
        on_sweep_end: Callable[[], None] | None = None,
        renderer: type[Renderer] | None = None,
        num_render_checkpoints: int = 1,
        on_checkpoint_frames: CheckpointFramesSink | None = None,
        recorder: LoopRecorder | None = None,
        trace_recorder: EpisodeTraceRecorder | None = None,
    ) -> None:
        # Up front, before hard_reset(), so a caller that forgot the sink finds out
        # before the run mutates the environment rather than one sweep in.
        if renderer is not None and on_checkpoint_frames is None:
            raise ValueError(
                "on_checkpoint_frames is required when renderer is given: rendered "
                "frames are streamed to the sink and never retained, so without one "
                "every recorded clip would be silently discarded."
            )
        if practice_reset_policy is PracticeResetPolicy.NEVER and (
            evaluation_problem is None or evaluation_problem is problem
        ):
            # The larger of the two ways a "never reset" arm silently gets reset
            # anyway, and the one worth refusing rather than documenting: every
            # evaluation episode opens with reset_to_task, so a shared Problem hands
            # this arm num_test_tasks resets per sweep while num_practice_resets --
            # the field an experiment reads to certify the manipulation -- still
            # reports 0. `is problem` is checked as well as None because passing the
            # practice Problem as its own evaluation Problem provides no isolation.
            raise ValueError(
                "practice_reset_policy=never requires its own evaluation_problem: "
                "without one, every evaluation episode's reset_to_task writes into "
                "the practice environment, so the arm is reset num_test_tasks times "
                "per sweep while num_practice_resets still reports 0. Build a second, "
                "independent Environment + Tasks + Problem for evaluation."
            )
        if (
            practice_reset_policy is PracticeResetPolicy.NEVER
            and practice_reset_interval is not None
        ):
            # Rejected rather than resolved by precedence: whichever way it went,
            # one of the two flags would be silently ignored, and an arm that looked
            # reset-free while being reset every k steps is exactly the kind of
            # invisible confound this whole change exists to remove.
            raise ValueError(
                "practice_reset_policy=never is incompatible with a "
                f"practice_reset_interval ({practice_reset_interval}): the interval "
                "resets the environment within a period, so the arm would not be "
                "reset-free. Drop one of the two."
            )
        if method.may_request_human_help() and problem.human is None:
            # Up front, before hard_reset(), for the same reason as the sink check: a
            # run that would fail at its first rescue three cycles in should not first
            # spend three cycles. The evaluation Problem deliberately needs no human --
            # nobody is rescued during measurement.
            #
            # This is the ONLY place the loop asks the Method whether it can ask. It is
            # never polled per step: whether a rescue happens on a given step is the
            # Method's own business, and it says so by raising HumanHelpRequested.
            raise ValueError(
                "this Method may request human help (Method.may_request_human_help is "
                "True) but the practice Problem has no HumanOracle: Problem.human is "
                "None, so there is nobody to ask. Wire one in the domain's composition "
                "root (e.g. humans/oracle.py's UnconditionalHumanOracle)."
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

        # Falls back to the practice Problem, which is exactly the old behaviour --
        # see the class docstring's "separate evaluation environment" section for what
        # a caller buys by passing a distinct one, and why every domain that has not
        # been migrated stays byte-identical by omitting it.
        eval_problem = evaluation_problem if evaluation_problem is not None else problem
        problem.hard_reset()
        if eval_problem is not problem:
            # Its own one-time reset: a fresh Environment has no current_state at all
            # until something installs one. Guarded on identity rather than called
            # unconditionally so the un-migrated path still hard-resets exactly once.
            eval_problem.hard_reset()
        if recorder is not None:
            recorder.record_hard_reset(state=problem.get_current_state())
        # Drawn ONCE, up front -- see the class docstring's "fixed test set"
        # section for why re-sampling it per sweep corrupts the learning curve. From
        # the *evaluation* Problem: its Tasks is constructed identically to practice's,
        # so the tasks are the same ones, but they now come off a stream nothing in
        # the practice loop can advance.
        test_tasks = [eval_problem.sample_test_task() for _ in range(num_test_tasks)]
        num_online_transitions = 0
        frames = PracticeLoop._evaluate(
            problem=eval_problem,
            method=method,
            metrics=metrics,
            test_tasks=test_tasks,
            num_online_transitions=num_online_transitions,
            renderer=renderer if 0 in rendered_sweeps else None,
            recorder=recorder,
            sweep_index=0,
            trace_recorder=trace_recorder,
        )
        hand_over(transitions=num_online_transitions, sweep_frames=frames)
        if on_sweep_end is not None:
            on_sweep_end()
        for cycle in range(num_cycles):
            task = PracticeLoop._sample_practice_task(
                problem=problem, practice_reset_policy=practice_reset_policy
            )
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
            if practice_reset_policy is PracticeResetPolicy.NEVER:
                # Pick up wherever the last period left off. The sampled `task` is
                # still what get_practice_policy was given, but its initial_state is
                # NOT installed -- so every state feature no action writes stays at
                # its hard_reset value for the whole run. See the class docstring:
                # that freezes the sampler's own input features, which is a second
                # difference from the scheduled arm, not merely a missing reset.
                state = problem.get_current_state()
            else:
                # Tested against NEVER rather than for SCHEDULED so that anything
                # unexpected -- a hand-built argparse.Namespace carrying the string
                # "never", say -- degrades to the INCUMBENT behaviour rather than
                # silently running the experimental arm.
                state = problem.reset_to_task(task=task)
                metrics.record_practice_reset()
                if recorder is not None:
                    recorder.record_period_reset(state=state)
            for step in range(max_steps_per_interaction):
                try:
                    labeled_action = policy(state)
                except HumanHelpRequested as request:
                    # The robot asked, so a human answers and the period CONTINUES --
                    # the difference from InteractionComplete below, which ends it. The
                    # harness's whole role here is mechanism: it does not decide that a
                    # rescue is warranted, it only performs one that was requested.
                    #
                    # `continue` consumes this loop iteration, and that is deliberate
                    # rather than incidental: it is what guarantees a Method that asks
                    # on every call cannot spin, since it gets at most one rescue per
                    # remaining step. The cost is that an asking arm takes about one
                    # fewer online transition per rescue than the old harness-triggered
                    # arm did -- roughly 36 out of ~15000 per seed, since the rescue is
                    # charged a step the old design took for free.
                    state = PracticeLoop._grant_human_help(
                        problem=problem,
                        method=method,
                        metrics=metrics,
                        task=task,
                        target_state=task.initial_state,
                        cost=request.cost,
                        state=state,
                    )
                    # After the write, with the state the human actually left behind, so
                    # the Method can restart whatever made it ask. Without this a
                    # rescued robot is re-rescued forever -- see
                    # Method.observe_help_granted.
                    method.observe_help_granted(state=state)
                    if recorder is not None:
                        recorder.record_human_reset(
                            state=state, step_index=step, transitions=num_online_transitions
                        )
                    continue
                except HumanRandomTaskResetRequested as request:
                    # Modeled like InteractionComplete: the period ends here, no goal
                    # necessarily achieved. Unlike InteractionComplete, ending is not
                    # free -- a human resets the environment onto a FRESHLY SAMPLED
                    # train task (advancing the train-task stream, same as
                    # `--human-reset-target random` used to), priced and banked exactly
                    # like a HumanHelpRequested rescue. See
                    # HumanRandomTaskResetRequested's own docstring for why this cannot
                    # be a mid-plan step the way a HumanHelpRequested rescue is.
                    fresh_task = problem.sample_train_task()
                    state = PracticeLoop._grant_human_help(
                        problem=problem,
                        method=method,
                        metrics=metrics,
                        task=task,
                        target_state=fresh_task.initial_state,
                        cost=request.cost,
                        state=state,
                    )
                    method.observe_help_granted(state=state)
                    if recorder is not None:
                        recorder.record_human_reset(
                            state=state, step_index=step, transitions=num_online_transitions
                        )
                    break
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
                problem=eval_problem,
                method=method,
                metrics=metrics,
                test_tasks=test_tasks,
                num_online_transitions=num_online_transitions,
                renderer=renderer if (cycle + 1) in rendered_sweeps else None,
                recorder=recorder,
                sweep_index=cycle + 1,
                trace_recorder=trace_recorder,
            )
            hand_over(transitions=num_online_transitions, sweep_frames=frames)
            if on_sweep_end is not None:
                on_sweep_end()

    @staticmethod
    def _grant_human_help(
        *,
        problem: Problem,
        method: Method,
        metrics: Metrics,
        task: Task,
        target_state: State,
        cost: float | None,
        state: State,
    ) -> State:
        """Perform the rescue a `Method` asked for, charge it, and return the state that
        results.

        **Mechanism, not policy.** Nothing here decides that a rescue is warranted --
        that decision was the Method's, and it arrived as a `HumanHelpRequested` or a
        `HumanRandomTaskResetRequested`. This function only carries it out; which state
        to restore is the caller's decision (this period's own task-initial state for
        the former, a freshly sampled train task's for the latter -- see each
        exception's own docstring for why that axis now lives in which exception is
        raised rather than in a shared `--human-reset-target` flag).

        **Practice only.** No evaluation episode is ever rescued: measurement runs on
        `evaluation_problem`, which is never handed a practice policy, so a rescued
        arm's score is still the robot's own. A human who could be called during a sweep
        would be measuring the human.

        **Three things a rescue is deliberately not.** It is not a
        `record_practice_reset` -- that counter certifies the reset-free manipulation,
        and a charged rescue landing in it would make a `never` arm look like one that
        was quietly reset for free. It is not charged as an online transition, matching
        every other reset here, or an arm rescued more often would advance along every
        learning curve's x-axis for free. And it does not end the period or fire
        `end_cycle` on its own -- ending the period is the caller's decision (the
        `HumanRandomTaskResetRequested` handler `break`s after this returns; the
        `HumanHelpRequested` handler does not), exactly like a `practice_reset_interval`
        reset never ending one either.

        **`cost=None` means "price this the harness's own way"**, by querying
        `Problem.calculate_cost_for_human_command` -- the incumbent behaviour, and what
        every caller that raises a bare exception (no `cost=`) still gets. A `Method`
        that priced the request itself (EES, from its own planner's
        `ground_skill_costs`) passes that number through instead, bypassing the query
        entirely so the harness banks the price the plan was actually built against.

        **Order matters in two places.** The `Method` is told
        (`observe_environment_reset`) *before* the environment is written, so a Method
        scoring an in-flight skill scores it against what really happened rather than
        against the state it is about to be teleported to -- the identical contract a
        mid-period interval reset has. And a harness-priced cost is *queried before* the
        command is executed, so what is banked is the price the oracle actually quoted
        for the command that then ran, not a recomputed one."""
        # The goal handed over is the one the robot was *pursuing*, not the target
        # state's: a rescue is a request to be repositioned, and what the robot is
        # trying to achieve has not changed. A capability-aware human (v1+) prices
        # exactly that pairing -- "put it somewhere it can still finish this".
        priced_cost = (
            cost
            if cost is not None
            else problem.calculate_cost_for_human_command(goal=task.goal, target_state=target_state)
        )
        method.observe_environment_reset(state=state)
        problem.execute_human_command(goal=task.goal, target_state=target_state)
        metrics.record_human_intervention(cost=priced_cost)
        # Read back rather than assumed: the HumanOracle owns what actually happened to
        # the environment, and a v1+ human that only partially succeeds would leave it
        # somewhere other than the state that was asked for.
        return problem.get_current_state()

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
    def _sample_practice_task(
        *, problem: Problem, practice_reset_policy: PracticeResetPolicy
    ) -> Task:
        """The train task this period practices, drawn the way the policy allows.

        Under SCHEDULED the period is reset into the task immediately afterwards, so
        whatever sampling did to the environment is overwritten and the ordinary draw
        is the right one. This is the incumbent path and is deliberately untouched.

        Under NEVER there is no such reset, so sampling itself must not move the world
        -- see `Tasks.sample_train_task_in_place`. The check below is not defensive
        tidiness: on a domain that builds its tasks in the world, the failure mode is a
        *silent* one. The arm reports `num_practice_resets == 0`, because that counter
        records the branch this method's caller takes and that branch really was not
        taken, while the simulator has been rebuilt every cycle regardless. Tossing3D
        shipped 20 committed runs that way, two arms that were one condition measured
        twice. Identity rather than equality because `State` wraps numpy arrays and has
        no usable `__eq__`; every `Environment` here installs a new `State` object when
        it moves, so a replaced object is the signal that something happened.
        """
        if practice_reset_policy is not PracticeResetPolicy.NEVER:
            return problem.sample_train_task()
        before = problem.get_current_state()
        task = problem.sample_train_task_in_place()
        if problem.get_current_state() is not before:
            raise RuntimeError(
                "practice_reset_policy=never sampled a train task and the environment "
                f"moved: {type(problem.tasks).__name__}.sample_train_task_in_place "
                "replaced the current state. On this policy nothing may reset the "
                "practice environment, and a domain that builds its tasks in the world "
                "(as Tossing3D does, via env.reset_to_seed) must override "
                "Tasks.sample_train_task_in_place to address the world it is already "
                "in. Left unchecked this is invisible: num_practice_resets would still "
                "report 0 while the scene was rebuilt every cycle."
            )
        return task

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
        trace_recorder: EpisodeTraceRecorder | None = None,
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
            solved, task_frames, trace = problem.run_task_episode(
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
            if trace_recorder is not None:
                trace_recorder.record_episode(
                    checkpoint=sweep_index,
                    num_online_transitions=num_online_transitions,
                    task_index=i,
                    goal=task.goal.describe(),
                    solved=solved,
                    trace=trace,
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
