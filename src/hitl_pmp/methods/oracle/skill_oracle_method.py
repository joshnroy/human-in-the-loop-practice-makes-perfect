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
    `environments/tossingroom/skill_provider.py`'s TossingRoomOracle), rather than in
    an `isinstance(self.env, ...)` branch here. Adding a domain means adding its
    OraclePolicyProvider, not editing this class.

    The oracle is handed the task's goal as well as the state: most domains ignore it
    (Light Switch, Ball-Ring drive toward their single fixed objective from privileged
    state), but a goal-dependent oracle -- Tossing Room, whose state alone can't tell
    throw-recycling from throw-trash -- reads it to pick which item/bin/room to head
    for."""

    oracle: OraclePolicyProvider

    def reset_environment(self, *, start_state: State) -> bool:
        """No irreversible actions matter to this oracle and the base PMP paper has no
        human-in-the-loop layer -- a direct environment set stands in for a real
        "self-navigate without help" recovery (matches RandomSkillsMethod)."""
        self.env.set_state(state=start_state)
        return True

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
