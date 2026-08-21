import abc
from typing import Any

from pydantic import BaseModel

from hitl_pmp.core.problem.environment.environment import Environment
from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.core.problem.tasks.types import Task

from .types import (
    GroundSkill,
    Policy,
    PracticeTargetTally,
    Rollout,
    SetupCommand,
    SkillPracticeTally,
)


class InteractionComplete(Exception):  # noqa: N818
    """Raised by a practice policy that has nothing further worth doing, ending
    the current interaction period early.

    This is what makes the online-transition count *data-driven* rather than
    budget-driven: practice_loop.py charges only the steps actually taken, the
    way predicators sums `len(result.actions)` over the trajectories its
    explorers actually produced (main.py:244) rather than assuming every request
    ran to `max_num_steps_interaction_request`. predicators' explorers signal the
    same condition by raising out of `run_episode_and_get_observations`, which
    then returns a correspondingly short trajectory.

    Not an error: ending early is a normal end to a period. The cycle still
    retrains (`end_cycle`) and is still evaluated. Named without the `Error`
    suffix (hence the ruff N818 waiver) precisely because it is control flow, not
    a failure.

    **Distinct from `HumanHelpRequested` below**, and deliberately never widened to
    cover it -- see that exception's own docstring for the two differences."""


class HumanHelpRequested(Exception):  # noqa: N818
    """Raised by a practice policy that cannot get anywhere from where it is and is
    asking a human to reposition it. Control flow, not an error, exactly like
    `InteractionComplete` above (hence the same ruff N818 waiver).

    **This is the robot asking, not a monitor noticing.** ("Robot", not "agent", per
    CLAUDE.md's naming rule -- this is our own `Method` acting, and nothing here
    involves an LLM. The rule exists partly because the arm this exception replaces was
    called `agent-signal`.) Deciding when to ask is
    *policy*, and policy belongs to the `Method`: the harness's whole job here is
    mechanism -- invoke the `HumanOracle`, price the command, bank the cost, and carry
    on. An external watcher of the state stream that summoned a human on the `Method`'s
    behalf would be measuring the watcher.

    **Two differences from `InteractionComplete`, both load-bearing.**

      * *What it claims.* `InteractionComplete` means "no ground skill is applicable at
        all", which on Tossing Room essentially never happens because `MoveRoom` always
        is. This means "I am still perfectly able to act, and acting is getting me
        nowhere" -- the absorbing region behind the one-way ledge, where a robot paces
        between rooms forever.
      * *What happens next.* `InteractionComplete` **ends** the interaction period.
        This does not: the human answers and the period **continues**, bounded by the
        step budget like everything else. The rescue does consume the loop iteration it
        was raised on, so a `Method` that asks every step cannot spin.

    Kept a separate exception rather than a widened `InteractionComplete` because that
    exception's meaning is EES-wide and already-merged results depend on it. Reusing it
    would silently give every arm that catches one the behaviour of the other, and the
    two could then not be compared.

    **Always this period's own task-initial state.** `--human-reset-target` (which also
    offered "a freshly sampled train task") is gone: that second target names a
    physically different action (resample a task and advance the train-task stream,
    rather than restore this one), so it stays a separate exception,
    `HumanRandomTaskResetRequested` below, rather than a variant of this one -- even
    though, on a domain whose task family is shape-invariant, both are now dispatched
    the same way (see that exception's own docstring).

    **Carries one optional field, `cost`**, exactly the extension this docstring used to
    invite ("a `Method` that later needs to ask for something specific adds a field here
    rather than a second exception"). `None` (the default) means "price this the
    harness's own way" -- `practice_loop.py` then queries
    `Problem.calculate_cost_for_human_command`, the incumbent behaviour, still exercised
    by anything that raises the bare exception. A `Method` that has already priced the
    request as part of its own planning (EES, whose planner chose this ground skill
    against a configured `--ask-for-reset-task-initial-cost`) sets it instead, so the
    harness banks the number the plan was actually built against rather than re-deriving
    an unrelated one from the `HumanOracle`'s own pricing -- see
    `EesMethod.ask_for_reset_task_initial_cost` for why that pricing is deliberately
    independent of the human's own (`UnconditionalHumanOracle.intervention_cost` is a
    flat structural constant, not a per-skill dial)."""

    def __init__(self, *, cost: float | None = None) -> None:
        super().__init__(cost)
        self.cost = cost


class HumanRandomTaskResetRequested(Exception):  # noqa: N818
    """Raised by a practice policy that is asking a human to reposition it onto a
    freshly sampled train task. Control flow, not an error, exactly like the other two
    (hence the same ruff N818 waiver).

    **Modeled like `HumanHelpRequested`, not like `InteractionComplete` -- the period
    CONTINUES.** This used to end the period outright, on the reasoning that resetting
    onto a *freshly sampled* train task can change the goal atoms the episode is
    pursuing, not merely the state, and a classical plan built for one fixed goal
    cannot have a goal-changing step as a mid-plan transition. That reasoning is still
    correct in general, but it does not apply on a domain whose task family is
    *shape-invariant*: if every task the domain ever samples shares the same goal and
    the same abstract initial-state shape (Tossing3D's case -- single goal family over
    `ClassVar` objects, verified empirically that a freshly sampled task's abstracted
    initial atoms always match `task_initial_atoms`), then resetting onto a fresh task
    is, symbolically, indistinguishable from resetting onto this period's own task --
    see `HumanResetSkillBuilder` for how `ask_for_reset_random_task` reuses exactly
    `ask_for_reset_task_initial`'s operator construction on that basis. So `EesMethod`
    now offers this as a genuine mid-plan ground skill too (`plan_to`), and selecting
    it no longer ends the interaction period: `practice_loop.py`'s handler `continue`s,
    exactly like `HumanHelpRequested`'s.

    **This is a domain-dependent relaxation, not a general framework fact.** A domain
    whose sampled tasks vary in goal or initial-state shape must NOT rely on this: the
    ground skill's declared effect (task_initial_atoms) would then be only an estimate
    of what a real reset produces, and continuing to pursue a stale goal built for the
    OLD task after installing a NEW one would be a real bug, not merely a suboptimal
    plan. `_EesEpisode.step`'s closed-loop divergence check (discarding a queued plan
    whose next precondition the real post-reset state does not satisfy) prevents an
    inapplicable ACTION from ever being taken on such a mismatch, but it cannot rescue
    a stale GOAL -- `_EesEpisode._goal` is fixed at construction and is not rebuilt
    against whatever task a mid-cycle reset actually lands on. No domain in this repo
    other than Tossing3D configures `ask_for_reset_random_task_cost` today.

    **Still what makes it a distinct event from `HumanHelpRequested` at all.** Where
    that exception restores THIS period's own task-initial state, this one resolves
    through a real human reset onto a *freshly sampled* train task, which
    **advances the train-task stream** -- the same real, intended difference
    `--human-reset-target random` used to describe, and the reason `practice_loop.py`
    still dispatches it through its own handler (`PracticeLoop._grant_human_help` with
    `target_state=fresh_task.initial_state`) rather than folding it into
    `HumanHelpRequested`. Priced the same way (`cost`, `None`-means-harness-priced).

    **NOT an unconditional substitute for `InteractionComplete` at any configured
    cost.** An earlier version of the mid-plan offering above also raised this
    unconditionally from a `step()`-level "nothing left to practice" fallback whenever
    a cost was configured at all -- a bare presence check, not a comparison against
    the cost's own VALUE. Removed: a classical cost-minimizing search cannot weigh "no
    plan" as a competing finite-cost alternative, so once a reset is offered at all it
    is accepted at any magnitude -- which silently made "sweep the reset cost" mean
    "always reset regardless of cost", and made a genuinely stuck robot structurally
    unable to just stay stuck the way the `never` baseline with no reset skill
    configured can. `EesMethod.plan_to` now enforces a real, cost-derived ceiling
    (`max` of the ordinary ground-skill costs already in scope for that call) before
    accepting a reset it finds -- see that method's own docstring -- and the
    `step()`-level fallback that used to bypass it is gone outright.

    Kept a third, separate exception rather than a payload on either existing one for
    the same reason `HumanHelpRequested` stays separate from `InteractionComplete`: the
    three describe three different things a caller needs to tell apart by `except`, and
    conflating any two would silently give every existing catcher of one the behaviour
    of another."""

    def __init__(self, *, cost: float | None = None) -> None:
        super().__init__(cost)
        self.cost = cost


class HumanCubeBinResetRequested(Exception):  # noqa: N818
    """Raised by a practice policy that is choosing to ask a human for a *partial*
    reset: reposition whichever of this domain's own non-robot objects a human could
    casually tidy up, while leaving the robot's own configuration exactly as it is.
    Control flow, not an error, exactly like the other three (hence the same ruff
    N818 waiver).

    **A third, structurally different kind of reset from either existing one.**
    `HumanHelpRequested` and `HumanRandomTaskResetRequested` both restore a *complete*
    symbolic state (this period's own task-initial atoms, or a freshly sampled task's)
    via `Problem.execute_human_command`'s `target_state: State`. This one cannot go
    through that path at all: `target_state` describes every object at once, and on a
    simulator-backed domain like Tossing3D the only way to install one is a whole-scene
    rebuild (`Tossing3DEnvironment.set_state` -- see its own docstring) which
    necessarily also relocates the robot. A partial reset needs a state-mutation
    primitive `HumanOracle`/`Environment` did not previously have -- see
    `Environment.reset_movables` and `HumanOracle.execute_movables_reset`, and
    `Problem.execute_movables_reset`, the facade `PracticeLoop` calls instead of
    `execute_human_command` for this exception.

    **Modeled like `HumanHelpRequested`, not like `HumanRandomTaskResetRequested`.**
    The robot is not relocated and the goal is not changed, so there is no structural
    reason this has to end the interaction period -- exactly the reasoning that keeps
    `HumanHelpRequested` a mid-plan step rather than a period-ending one. Whatever plan
    was in flight is simply invalidated (the touched objects' atoms may have changed)
    and `_EesEpisode.step`'s closed-loop replanning handles that the same way any other
    stochastic divergence does.

    **Which domain objects "reposition" touches is entirely the domain's business.**
    This exception carries no target description at all -- unlike the other two, whose
    caller supplies `target_state`, this one's caller (a `Method`) does not know which
    objects a given domain considers "not the robot"; only the domain's own
    `Environment`/`SkillProvider` do. See `SkillProvider.human_cube_bin_reset_skill`
    for how a domain opts in, and `Environment.reset_movables` for how it is carried
    out; every domain that does not override either stays exactly as it was, since both
    default to declining (`None`/`False`).

    **Carries one required `cost` field**, unlike the other two's optional one. Both of
    those support `cost=None` ("price this the harness's own way", queried from
    `Problem.calculate_cost_for_human_command`) because that query is well-defined for
    a goal-directed or full-state-restore command. No such query exists for "reposition
    whatever a domain calls its movables" -- there is no `Goal`/`target_state` pair to
    price against -- so a `Method` raising this must always have priced it itself
    (`EesMethod` does, from its own `--ask-for-reset-cube-bin-cost`), and the field is
    required rather than optional to make that contract explicit rather than silently
    accepting a `None` no caller can currently supply a price for."""

    def __init__(self, *, cost: float) -> None:
        super().__init__(cost)
        self.cost = cost


class Method(BaseModel, abc.ABC):
    """The agent side: decides what to practice, executes skills, improves them.

    A real, constructor-injected instance now (not a static-method container):
    env is the one piece of context every Method concretely needs to act at all
    (e.g. SkillOracleMethod's oracle logic has to know which domain it's cheating
    in) -- there is no global Problem.env to read anymore, so whatever a Method
    needs from its environment must be handed to it explicitly, and the natural
    place for that is construction time, mirroring Environment/Tasks. Methods that
    turn out not to need env at all are still free to ignore self.env entirely;
    nothing about this field forces every concrete Method to use it.
    """

    env: Environment

    @abc.abstractmethod
    def reset_environment(self, *, start_state: State) -> bool:
        """The agent's own attempt to self-navigate to start_state, without human help.

        True means the agent genuinely got there by acting. It is NOT satisfiable by
        calling `env.set_state` -- that is the privileged external override reserved
        for `HumanOracle`, and using it here would report a
        recovery no agent performed. Every implementation in this repo today returns
        False, because none of them can navigate anywhere on purpose; nothing calls
        this method yet, so False is the honest answer rather than a regression."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_task_policy(self, *, task: Task) -> Policy:
        """The policy used to *evaluate* on a task: pursue the goal, exploiting
        whatever has been learned so far. Never explores, and must never record
        training data -- practice_loop.py calls this once per held-out test task,
        so learning from it would be training on the test set."""
        raise NotImplementedError

    def get_practice_policy(self, *, task: Task) -> Policy:
        """The policy used during an interaction/practice period, where a Method
        is free to explore and to record whatever training data it wants.

        Concrete (not abstract) and defaults to get_task_policy, because a Method
        that doesn't learn -- every baseline built so far (SkillOracleMethod,
        RandomSkillsMethod) -- behaves identically in both phases and shouldn't
        need boilerplate to say so. A learning Method (EES) overrides this to
        explore, keeping exploration strictly out of get_task_policy: predicators
        splits the same way, with the approach's own _solve() used for evaluation
        and a separate explorer used during interaction (see
        predicators/approaches/active_sampler_learning_approach.py, whose
        _create_explorer is only ever consulted for interaction requests)."""
        return self.get_task_policy(task=task)

    def observe_environment_reset(self, *, state: State) -> None:
        """Called by practice_loop.py immediately *before* it resets the
        environment part-way through an interaction period
        (`practice_reset_interval`), handing over the state the environment is
        about to leave -- the last chance to score whatever the policy has in
        flight against what actually happened.

        Without this, a mid-period reset silently corrupts the very data it is
        supposed to leave alone. A Method that judges a skill by checking its
        effects on the *next* state it sees (EesMethod does) would check them
        against the post-reset initial state instead, which almost never
        satisfies them -- so every skill executed just before a reset gets
        recorded as a failure. That mislabelling scales with how often the
        harness resets, which is exactly the quantity an experiment varying
        `practice_reset_interval` is trying to isolate.

        Concrete no-op by default, for the same reason as end_cycle: a Method
        with nothing in flight has nothing to settle. NOT called at the
        interaction period's own boundary -- that would change long-standing
        behaviour (the last skill of a period has always gone unobserved), and
        keeping it uncalled there is what makes every arm of a reset-interval
        sweep drop exactly one observation per period rather than a number that
        varies with the interval."""

    def may_request_human_help(self) -> bool:
        """Whether this Method's *practice* policy can raise `HumanHelpRequested`.

        Its one and only consumer is practice_loop.py's up-front validation: a Method
        that can ask, paired with a `Problem` that has no `HumanOracle`, is a
        misconfiguration worth refusing before `hard_reset()` rather than three cycles
        in. The loop must never poll this per step -- whether a rescue happens on a
        given step is the Method's business, and it says so by raising.

        Concrete `False` by default, for the same reason as `end_cycle`: no baseline
        built so far asks for anything, and none of them should need boilerplate to say
        so. `EesMethod` overrides it, True exactly when one of its three ground-skill
        cost flags (`ask_for_reset_task_initial_cost`/`ask_for_reset_random_task_cost`/
        `ask_for_reset_cube_bin_cost`) is configured -- see that class."""
        return False

    def observe_help_granted(self, *, state: State) -> None:
        """Called by practice_loop.py immediately *after* a rescue has been executed,
        handing over the state the human actually left behind.

        This is what lets a Method restart whatever detector made it ask -- a hook that
        exists for any future novelty-based trigger a Method might carry, though
        `EesMethod` today needs no such restart: it decides to ask fresh at every
        planning call (a real ground skill priced like any other), not from an
        accumulated per-period detector, so it inherits this as a no-op. Without a
        restart, a Method that DID carry such state would be instantly re-rescued
        forever: a human by construction puts it back somewhere it has already been, so
        under any novelty-based rule every state is non-novel the moment it arrives.

        The *readback* state, not the state that was commanded -- a capability-aware
        human (v1+) that only partially succeeds leaves the environment somewhere other
        than what was asked for, and a Method restarting on the commanded state would
        be restarting on a place the robot is not.

        Distinct from `observe_environment_reset`, which fires *before* the write and
        exists to settle an in-flight skill against what really happened. Both fire on
        a rescue, in that order, and they answer different questions: one is "score
        this against the state you are about to lose", the other is "here is where you
        now are". Concrete no-op by default, like `end_cycle`."""

    def planning_outcomes(self) -> tuple[int, int]:
        """(failures, attempts) for this Method's own planning so far, cumulative over
        the whole run. `(0, 0)` for a Method that does not plan.

        Read by method_runner.py once per cycle and differenced into
        `Metrics.record_planning_outcomes`, so what lands in stats.json is per-window
        while all a Method maintains is two monotonic counters -- the cheaper half of
        the job, and the half that cannot get out of step with the loop's cadence.

        A *pair*, never a bare failure count: EES asks its planner speculatively, so a
        failure there is often routine and the number is uninterpretable without what
        it is out of. Returning both together is what makes it impossible to report one
        without the other -- see Metrics.record_planning_outcomes.

        Concrete default, for the same reason as end_cycle: every baseline built so far
        plans nothing, and none of them should need boilerplate to say so. A planning
        Method overrides it (EesMethod does).

        On Method rather than Metrics because only the Method knows -- a
        `PlanningFailure` is caught deep inside its own policy, where the harness
        cannot see it. Pulling a counter at a known boundary is what keeps this from
        needing a Metrics reference threaded down into every Method."""
        return (0, 0)

    def practice_outcomes(self) -> dict[str, SkillPracticeTally]:
        """{lifted skill name: what practicing it has done so far}, cumulative over the
        whole run. `{}` for a Method that never scores its own skill executions.

        The sibling of `planning_outcomes`, threaded the same way and for the same
        reason: read by method_runner.py once per cycle and differenced into
        `Metrics.record_practice_outcomes`, so what lands in stats.json is per-window
        while all a Method maintains is one monotonic tally per skill.

        **What this makes answerable.** Until it existed, a run's stats.json recorded
        tasks solved and nothing about practice, so a null result could not be told
        apart from a starved one -- PR #108's author said so explicitly, that "the
        discriminating quantity between 'too few labels' and 'cannot use them' does not
        exist in any run to date". `SkillPracticeTally` is where that quantity now
        lives; see it for how to read the three pools.

        **Keyed by the LIFTED skill**, not the grounding, because that is the unit the
        learning happens in: one `LearnedSkillSampler` is fitted per skill *name*
        (predicators' `active_sampler_learning_object_specific_samplers = False`), so
        "was this sampler starved?" is a question about the lifted name. Competence
        models are per *ground* skill and are deliberately not surfaced here -- a
        different quantity, aggregated differently, and one a caller can reach through
        the Method it already holds.

        **`{}` and a zero-attempt entry are different answers.** `{}` means this Method
        does not measure practice at all -- `RandomSkillsMethod` checks no add effects
        and `SkillOracleMethod` never practices, so neither has an outcome to report,
        and neither should need boilerplate to say so (concrete default, same reason as
        `end_cycle`). A *present* entry reading 0/0 means the opposite: this Method does
        measure, and this skill was not practiced. Only the second is evidence, which is
        why a Method must never pad its mapping with skills it has not seen.

        On Method rather than Metrics because only the Method knows: the add-effect
        check that decides a success, and the `SamplerChoice` flags that classify the
        draw, both happen deep inside its own policy where the harness cannot see
        them."""
        return {}

    def practice_target_outcomes(self) -> dict[str, PracticeTargetTally]:
        """{lifted skill name: how often practicing it was *chosen*}, cumulative over
        the whole run. `{}` for a Method that does not select practice targets.

        The sibling of `practice_outcomes`, threaded and differenced identically, and
        deliberately *not* folded into it: the two count different events. That one
        counts executions, this one counts decisions, and the gap between them is
        precisely where a skill EES declines to practice hides -- it goes on being
        executed as a prefix step toward some other candidate, so its execution tally
        looks healthy while its selection tally is zero. See `PracticeTargetTally`.

        `{}` and a zero entry differ here for the same reason as on `practice_outcomes`:
        `{}` is "this Method does not choose practice targets at all", an absent skill
        is "never a candidate", and a present entry reading zero selections is "was a
        candidate and was passed over". Only the last two are evidence about the skill,
        and they mean opposite things."""
        return {}

    def current_competences(self) -> dict[GroundSkill, float]:
        """{ground skill: this Method's own currently tracked competence estimate for
        it}, read at every evaluation checkpoint by method_runner.py's
        --record-skill-competence sidecar (see hitl_pmp/competence_log.py). `{}` for a
        Method that tracks no such thing.

        Unlike practice_outcomes/planning_outcomes, this is NOT a cumulative counter
        differenced into a per-window Metrics field -- it is a live read of whatever
        value the Method's own planning currently uses, taken as-is at each checkpoint.
        There is nothing to difference: competence is a belief, not a tally, and
        recording it at every checkpoint is exactly what makes its trajectory
        plottable, the same way SamplerDrawRecorder makes a sampler's chosen
        parameters plottable rather than only inferrable from success-rate tallies.

        **Keyed by ground skill, not lifted skill name** -- unlike practice_outcomes,
        which is keyed by the lifted name because one sampler is fitted per lifted
        skill. Competence is estimated per *grounding* (predicators' `_ground_op_hist`
        keying, matched by EesMethod.competence_model), so two groundings of the same
        lifted skill can have different competence and must stay distinguishable.

        Concrete default, for the same reason as practice_outcomes: a Method that
        tracks no competence model (every non-learning baseline) has nothing to report
        and should need no boilerplate to say so."""
        return {}

    def end_cycle(self) -> None:
        """Called by practice_loop.py once after each interaction period, before
        that cycle's evaluation sweep -- the hook where a learning Method
        retrains on everything it just collected (predicators does exactly this
        between cycles: _update_sampler_data, then _learn_wrapped_samplers, then
        advance_cycle on every competence model).

        Concrete no-op by default, for the same reason as get_practice_policy: a
        non-learning Method has nothing to do here. Distinct from
        improve_skill_parameters, which is per-skill-execution rather than
        per-cycle."""

    @abc.abstractmethod
    def generate_train_task(self, *, tbd_inputs: Any) -> Task:
        """Decides what to practice next; exact inputs still TBD per the design doc."""
        raise NotImplementedError

    @abc.abstractmethod
    def execute_setup_command(self, *, setup_command: SetupCommand) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def execute_skill(self, *, skill: GroundSkill) -> Rollout:
        raise NotImplementedError

    @abc.abstractmethod
    def improve_skill_parameters(self, *, skill: GroundSkill, rollout: Rollout) -> None:
        raise NotImplementedError
