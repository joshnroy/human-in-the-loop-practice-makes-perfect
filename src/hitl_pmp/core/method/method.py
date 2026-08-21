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
    """Raised by a practice policy with nothing further worth doing, ending the
    period early so the online-transition count stays data-driven, not
    budget-driven. Not an error, hence no `Error` suffix (ruff N818 waiver).

    Distinct from `HumanHelpRequested` below -- see that docstring."""


class HumanHelpRequested(Exception):  # noqa: N818
    """Raised by a practice policy asking a human to reposition it -- the robot
    asking, not a monitor noticing (see CLAUDE.md's robot-vs-agent rule). Control
    flow, not an error, same N818 waiver as `InteractionComplete`.

    Two differences from `InteractionComplete`: it means "I can still act but
    acting is getting me nowhere" rather than "nothing is applicable", and the
    period *continues* after the rescue instead of ending.

    Carries an optional `cost`: `None` means "price it the harness's own way"
    (`Problem.calculate_cost_for_human_command`); a `Method` that already priced
    the request itself (EES, via its planner) sets it directly instead, so the
    harness banks the number the plan was built against."""

    def __init__(self, *, cost: float | None = None) -> None:
        super().__init__(cost)
        self.cost = cost


class HumanRandomTaskResetRequested(Exception):  # noqa: N818
    """Raised by a practice policy asking a human to reset it onto a freshly
    sampled train task (advancing the train-task stream), rather than this
    period's own task-initial state. Control flow, not an error (N818 waiver).

    Modeled like `HumanHelpRequested` -- the period continues, not like
    `InteractionComplete`. Safe here because Tossing3D's task family is
    shape-invariant (every sampled task shares the same goal and initial-state
    shape), so a fresh task is symbolically indistinguishable from this period's
    own -- see `HumanResetSkillBuilder`. That's a domain-dependent fact, not a
    general one: a domain whose tasks vary in goal/shape must not reuse this.

    Priced like `HumanHelpRequested` (`cost`, `None` = harness-priced), and
    gated by the same cost ceiling in `EesMethod.plan_to` -- so a genuinely
    stuck robot can still stay stuck rather than always resetting."""

    def __init__(self, *, cost: float | None = None) -> None:
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
        so. `EesMethod` overrides it, True exactly when one of its two ground-skill
        cost flags (`ask_for_reset_task_initial_cost`/`ask_for_reset_random_task_cost`)
        is configured -- see that class."""
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
