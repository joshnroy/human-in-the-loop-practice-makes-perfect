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
