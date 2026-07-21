from typing import Any

from hitl_pmp.core.method.method import Method
from hitl_pmp.core.method.types import GroundSkill, LabeledAction, Policy, Rollout, SetupCommand
from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.skill_oracle_policy import SkillOraclePolicy


class SkillOracleMethod(Method):
    """Wraps SkillOraclePolicy (environments/lightswitch/skill_oracle_policy.py)
    as a core.Method, so it runs through the same practice_loop.py:PracticeLoop
    harness as every other Method. Domain dispatch (get_labeled_action, below)
    lives here rather than on SkillOraclePolicy itself: SkillOraclePolicy only
    knows its own domain's oracle logic (matching ActionOraclePolicy's
    precedent), while this class -- living under methods/, not
    environments/lightswitch/ -- is the layer allowed to reason about which
    environment self.env actually is (inherited from Method; see
    core/README.md's dependency-direction section). Only Light Switch exists so
    far, so there's only one branch -- a second domain would add its own
    environments/<domain>/skill_oracle_policy.py plus one more branch in
    get_labeled_action, not a second SkillOracleMethod-like class."""

    def get_labeled_action(self, *, state: State) -> LabeledAction:
        if isinstance(self.env, LightSwitchEnvironment):
            return SkillOraclePolicy.get_labeled_action(state=state, env=self.env)
        raise NotImplementedError(
            f"SkillOracleMethod has no oracle logic for env={self.env!r} yet."
        )

    def reset_environment(self, *, start_state: State) -> bool:
        """No irreversible actions exist in Light Switch and the base PMP paper
        has no human-in-the-loop layer at all -- this reproduction never needs a
        real "self-navigate without help" recovery, so a direct environment set
        stands in for it (matches RandomSkillsMethod's own reasoning)."""
        self.env.set_state(state=start_state)
        return True

    def get_task_policy(self, *, task: Task) -> Policy:
        del task  # never consulted -- this oracle always drives toward the
        # light using privileged state, regardless of which task it's handed
        return lambda state: self.get_labeled_action(state=state)

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
