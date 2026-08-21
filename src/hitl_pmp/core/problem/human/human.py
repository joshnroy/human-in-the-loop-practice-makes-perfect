import abc

from hitl_pmp.core.problem.environment.environment import Environment

from .types import CommandGoalDescription, CommandStartStateDescription, Cost


class HumanOracle(abc.ABC):
    """The v0-v3 human-cost-model axis from the design doc; a static-method container,
    never instantiated, swappable independent of Environment. Unlike
    Environment/Problem/Tasks/Method, this one stays static rather than becoming a
    constructor-injected instance: it has no state of its own to hold between calls
    -- execute_human_command already receives the one Environment *instance* it
    needs to mutate as an explicit per-call argument, the same pattern the rest of
    this refactor moved everything else toward, HumanOracle just already had it."""

    @staticmethod
    @abc.abstractmethod
    def calculate_cost_for_human_command(
        *,
        command_start_state_description: CommandStartStateDescription,
        command_goal_description: CommandGoalDescription,
    ) -> Cost:
        """Estimate the cost of asking the human for this, without actually asking;
        cost is inf if infeasible. Safe to call repeatedly for planning/ROI purposes."""
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def execute_human_command(
        *,
        command_start_state_description: CommandStartStateDescription,
        command_goal_description: CommandGoalDescription,
        env: Environment,
    ) -> None:
        """Actually ask the human to satisfy command_goal_description, starting from
        command_start_state_description. No return value — the cost was already
        known from calculate_cost_for_human_command. Each concrete HumanOracle
        implements its own policy for how the human actually goes about this and
        how env ends up reflecting it (e.g. via env.set_state) — this is deliberately
        hand-waved at this level so different versions can model humans of different
        capability/efficiency without changing the interface."""
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def execute_movables_reset(*, env: Environment) -> None:
        """Ask the human for a *partial* reset: `env.reset_movables()`, whatever that
        means for this domain, and nothing else. No `CommandStartStateDescription`/
        `CommandGoalDescription` pair, unlike `execute_human_command`, because there is
        no goal being pursued and no target state to describe -- the caller
        (`Problem.execute_movables_reset`, reached only from `HumanCubeBinResetRequested`)
        does not know which objects a domain considers "not the robot"; only the
        domain's own `Environment` does, via `reset_movables`.

        A concrete oracle should raise if `env.reset_movables()` returns False (this
        domain declined -- there is nothing to execute), the same way
        `execute_human_command` raises rather than no-opping when there is no
        target_state to restore. Unlike that method, there is no cost-inf branch on
        `calculate_cost_for_human_command` to have checked first: a Method that may
        raise `HumanCubeBinResetRequested` at all is only ever built against a domain
        that supports it (see `SkillProvider.human_cube_bin_reset_skill`, which is
        what makes the ground skill reachable in the first place), so reaching this
        call with a declining Environment would mean that contract was broken."""
        raise NotImplementedError
