import abc
from typing import Any

from pydantic import BaseModel

from hitl_pmp.core.problem.environment.environment import Environment
from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.core.problem.tasks.types import Task

from .types import GroundSkill, Policy, Rollout, SetupCommand


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
    a failure."""


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
        """The agent's own attempt to self-navigate to start_state, without human help."""
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
