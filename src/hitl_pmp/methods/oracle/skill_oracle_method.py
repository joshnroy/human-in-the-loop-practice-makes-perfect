from typing import Any

from hitl_pmp.core.method.method import Method
from hitl_pmp.core.method.skill_provider import OraclePolicyProvider
from hitl_pmp.core.method.types import GroundSkill, Policy, Rollout, SetupCommand
from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.core.problem.tasks.types import Task


class SkillOracleMethod(Method):
    """Wraps a domain's privileged, hand-authored solver (its OraclePolicyProvider)
    as a core.Method, so the upper-bound baseline runs through the same
    practice_loop.py:PracticeLoop harness as every other Method.

    Fully domain-agnostic now: the domain-specific oracle logic lives in the injected
    `oracle` (e.g. `environments/lightswitch/skill_provider.py`'s LightSwitchOracle,
    `environments/ballring/skill_provider.py`'s BallRingOracle,
    `environments/tossingroomsplitpickupweight/skill_provider.py`'s
    TossingRoomSplitPickupWeightOracle), rather than in
    an `isinstance(self.env, ...)` branch here. Adding a domain means adding its
    OraclePolicyProvider, not editing this class.

    The oracle is handed the task's goal as well as the state: most domains ignore it
    (Light Switch, Ball-Ring drive toward their single fixed objective from privileged
    state), but a goal-dependent oracle -- Tossing Room, whose state alone can't tell
    throw-recycling from throw-trash -- reads it to pick which item/bin/room to head
    for."""

    oracle: OraclePolicyProvider

    def reset_environment(self, *, start_state: State) -> bool:
        """Always False: this oracle has no self-navigation to offer, so it declines
        rather than reporting a success it did not achieve (matches RandomSkillsMethod).

        This used to `self.env.set_state(state=start_state); return True` -- a
        privileged external state write dressed up as the agent recovering under its
        own power. Nothing calls this method today, so the lie cost nothing; the moment
        a reset-free loop branches on the return value it would silently treat every
        stranded robot as rescued."""
        del start_state  # nothing to navigate towards -- see above
        return False

    def get_task_policy(self, *, task: Task) -> Policy:
        # Close over the goal so a goal-dependent oracle (Tossing Room) can read it;
        # goal-agnostic oracles (Light Switch, Ball-Ring) ignore it.
        goal = task.goal
        return lambda state: self.oracle.get_labeled_action(state=state, goal=goal)

    def generate_train_task(self, *, tbd_inputs: Any) -> Task:
        raise NotImplementedError(
            "SkillOracleMethod.generate_train_task is unreachable: this oracle never practices."
        )

    def execute_setup_command(self, *, setup_command: SetupCommand) -> None:
        raise NotImplementedError(
            "SkillOracleMethod.execute_setup_command is unreachable: "
            "no HumanOracle is ever used in this reproduction."
        )

    def execute_skill(self, *, skill: GroundSkill) -> Rollout:
        raise NotImplementedError(
            "SkillOracleMethod.execute_skill is unreachable: this oracle "
            "computes its own ground skill choice directly, it never practices one."
        )

    def improve_skill_parameters(self, *, skill: GroundSkill, rollout: Rollout) -> None:
        raise NotImplementedError(
            "SkillOracleMethod.improve_skill_parameters is unreachable: this oracle never learns."
        )
