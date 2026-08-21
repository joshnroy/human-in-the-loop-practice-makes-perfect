import math
from typing import ClassVar

from hitl_pmp.core.problem.environment.environment import Environment
from hitl_pmp.core.problem.human.human import HumanOracle
from hitl_pmp.core.problem.human.types import (
    CommandGoalDescription,
    CommandStartStateDescription,
    Cost,
)


class UnconditionalHumanOracle(HumanOracle):
    """**v0** of the design doc's v0-v3 human axis, and the first concrete `HumanOracle`
    in this project: a human who always complies, immediately, at a flat cost.

    Ask it to put the world into a configuration and the world is in that configuration.
    There is no feasibility check, no capability model, no dependence on how far the
    robot has drifted, and no notion of the human refusing -- all of which are exactly
    what v1 (`cost_model.py`) onwards exist to add. It is the trivial baseline against
    which a real cost model is later measured, and its value is that it makes human
    intervention *representable*: before it, `Metrics.num_human_interventions()` returned
    a hardcoded zero because no `HumanOracle` existed for a `Problem` to hold.

    **Domain-agnostic, and testably so.** Nothing here knows a domain's dynamics, state
    layout or action space. It reads exactly one field off the command
    (`target_state`), calls exactly one method on the `Environment` (`set_state`, the
    privileged external override this class of caller is the documented owner of), and
    never evaluates the goal -- `tests/humans/test_oracle.py` passes a `Goal` whose
    classifier raises, so "unconditional" is asserted rather than merely claimed. It is
    therefore usable on `tossingroom` and `tossing3d` alike, though see the caveat below
    for what `set_state` means on a simulator-backed domain.

    **Why the command has to carry a target state.** `HumanOracle`'s interface hands over
    a `CommandGoalDescription`, which originally wrapped only a symbolic `Goal`. A `Goal`
    is a frozenset of `GroundAtom`s whose truth is an opaque `holds` callable, so no
    domain-agnostic implementation can synthesise a `State` satisfying one -- v0 as
    originally sketched ("just calls `env.set_state(...)` to satisfy the
    goal") was not implementable as written. `CommandGoalDescription.target_state` closes
    that, and see its own comment for why a reset is genuinely a different command rather
    than a weakening of the goal-directed one.

    **A static-method container, never instantiated**, like every `HumanOracle`: there is
    nothing to carry between calls. `intervention_cost` is a `ClassVar` because a flat
    cost really is a structural constant of *this* version of the human -- a human whose
    price varies is v1, a different class, not a differently-configured v0. That keeps
    `summed_human_cost` degenerate with the intervention count here, which is honest: at
    v0 they are the same measurement, and `Metrics` reports them separately so that they
    can come apart later without the metric changing shape.

    **One caveat for simulator-backed domains.** `set_state` is only as capable as the
    domain makes it. `tossing3d`'s can restore an **episode-initial** state and raises
    otherwise, because a flat `State` cannot carry MuJoCo's `qpos`/`qvel`. So on that
    domain this oracle can rescue to a task's initial state and not to an arbitrary one.
    That is a property of the environment, not of this class, and it fails loudly."""

    # What one intervention costs. 1.0 rather than 0.0 so that a run with a human is
    # distinguishable from one without in `summed_human_cost` alone, and rather than some
    # domain-derived number because v0 has no domain to derive it from.
    intervention_cost: ClassVar[Cost] = 1.0

    @staticmethod
    def calculate_cost_for_human_command(
        *,
        command_start_state_description: CommandStartStateDescription,
        command_goal_description: CommandGoalDescription,
    ) -> Cost:
        """`intervention_cost` for a reset, `inf` for anything else.

        The `inf` branch is the interface's own documented "cost is inf if infeasible":
        a v0 human can put the world where it is told and cannot reason toward a symbolic
        goal, so a goal-only command is genuinely infeasible for it rather than merely
        unimplemented. Side-effect free, so it is safe to call repeatedly for
        planning/ROI -- it takes no `Environment` at all, which is the interface making
        that guarantee structural."""
        del command_start_state_description
        if command_goal_description.target_state is None:
            return math.inf
        return UnconditionalHumanOracle.intervention_cost

    @staticmethod
    def execute_human_command(
        *,
        command_start_state_description: CommandStartStateDescription,
        command_goal_description: CommandGoalDescription,
        env: Environment,
    ) -> None:
        """Put `env` into the commanded configuration.

        Raises rather than no-opping when there is no target state. By the time this
        runs the caller has already priced the command and is about to record the cost,
        so silently doing nothing would bill an intervention that never happened -- and
        `calculate_cost_for_human_command` already returned `inf` for this case, so a
        caller that checked it cannot get here.

        **Deep-copied**, so the environment never shares an array with the caller's
        target. That target is typically a `Task.initial_state`, and the evaluation set
        is drawn once and replayed at every checkpoint, so aliasing would let one
        intervention rewrite a task for the remainder of the run."""
        del command_start_state_description
        target_state = command_goal_description.target_state
        if target_state is None:
            raise ValueError(
                "UnconditionalHumanOracle was asked to execute a command with no "
                "target_state. A v0 human can only restore a configuration it is given, "
                "not bring about a symbolic goal -- calculate_cost_for_human_command "
                "reports inf for exactly this command, so check it first."
            )
        env.set_state(state=target_state.model_copy(deep=True))

    @staticmethod
    def execute_movables_reset(*, env: Environment) -> None:
        """`env.reset_movables()`, and nothing else: a v0 human complies immediately
        and unconditionally, so there is no capability model to consult and no
        `target_state` to copy -- unlike `execute_human_command`, the domain itself
        decides what changes.

        Raises if the domain declined (`reset_movables()` returned False), the same
        shape as `execute_human_command`'s missing-target_state check: by the time
        this runs, the caller has already priced and is about to record an
        intervention, so silently doing nothing would bill one that never happened."""
        if not env.reset_movables():
            raise ValueError(
                "UnconditionalHumanOracle was asked to execute a movables reset, but "
                f"{type(env).__name__}.reset_movables() returned False -- this domain "
                "has no notion of 'the robot' distinct from 'everything else' (or has "
                "not implemented one). A Method should only raise "
                "HumanCubeBinResetRequested against a domain whose SkillProvider."
                "human_cube_bin_reset_skill opted in, which is what makes this call "
                "reachable in the first place."
            )
