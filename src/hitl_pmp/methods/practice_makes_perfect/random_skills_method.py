from typing import Any

import numpy as np
from pydantic import PrivateAttr

from hitl_pmp.core.method.method import Method
from hitl_pmp.core.method.types import GroundSkill, LabeledAction, Policy, Rollout, SetupCommand
from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.random_skills_policy import RandomSkillsPolicy


class RandomSkillsMethod(Method):
    """Random Skills: at each step, uniformly sample among the currently-applicable
    ground skills and execute one -- no planning, no competence model, no sampler
    learning. Matches predicators' own RandomOptionsApproach. Wraps
    RandomSkillsPolicy (environments/lightswitch/random_skills_policy.py) as a
    core.Method, so it runs through the same practice_loop.py:PracticeLoop harness
    as every other Method. Domain dispatch (get_labeled_action, below) lives here
    rather than on RandomSkillsPolicy itself: RandomSkillsPolicy only knows its own
    domain's uniform-sampling logic (matching SkillOraclePolicy's precedent), while
    this class -- living under methods/, not environments/lightswitch/ -- is the
    layer allowed to reason about which environment self.env actually is (inherited
    from Method; see core/README.md's dependency-direction section). Only Light
    Switch exists so far, so there's only one branch -- a second domain would add
    its own environments/<domain>/random_skills_policy.py plus one more branch in
    get_labeled_action, not a second RandomSkillsMethod-like class.

    seed carries this Method's own RNG stream (no existing Method previously
    needed one) -- same private-RNG-derived-from-seed pattern as
    LightSwitchTasks: a public seed field, a PrivateAttr populated in
    model_post_init, never reassigned directly."""

    seed: int = 0

    _rng: np.random.Generator = PrivateAttr()

    def model_post_init(self, __context: object) -> None:
        self._rng = np.random.default_rng(self.seed)

    def get_labeled_action(self, *, state: State) -> LabeledAction:
        if isinstance(self.env, LightSwitchEnvironment):
            return RandomSkillsPolicy.get_labeled_action(state=state, env=self.env, rng=self._rng)
        raise NotImplementedError(
            f"RandomSkillsMethod has no policy logic for env={self.env!r} yet."
        )

    def reset_environment(self, *, start_state: State) -> bool:
        """No irreversible actions exist in Light Switch and the base PMP paper
        has no human-in-the-loop layer at all -- this reproduction never needs a
        real "self-navigate without help" recovery, so a direct environment set
        stands in for it (matches SkillOracleMethod's own reasoning)."""
        self.env.set_state(state=start_state)
        return True

    def get_task_policy(self, *, task: Task) -> Policy:
        del task  # never consulted -- this baseline always samples uniformly among
        # applicable ground skills, regardless of which task it's handed
        return lambda state: self.get_labeled_action(state=state)

    def generate_train_task(self, *, tbd_inputs: Any) -> Task:
        raise NotImplementedError(
            "RandomSkillsMethod.generate_train_task is unreachable: this baseline never practices."
        )

    def execute_setup_command(self, *, setup_command: SetupCommand) -> None:
        raise NotImplementedError(
            "RandomSkillsMethod.execute_setup_command is unreachable: "
            "no HumanOracle is ever used in this reproduction."
        )

    def execute_skill(self, *, skill: GroundSkill) -> Rollout:
        raise NotImplementedError(
            "RandomSkillsMethod.execute_skill is unreachable: this baseline "
            "computes its own ground skill choice directly, it never practices one."
        )

    def improve_skill_parameters(self, *, skill: GroundSkill, rollout: Rollout) -> None:
        raise NotImplementedError(
            "RandomSkillsMethod.improve_skill_parameters is unreachable: "
            "this baseline never learns."
        )
