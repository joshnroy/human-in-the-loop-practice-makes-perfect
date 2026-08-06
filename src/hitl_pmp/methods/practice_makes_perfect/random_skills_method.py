from typing import Any

import numpy as np
from pydantic import PrivateAttr

from hitl_pmp.core.method.method import Method
from hitl_pmp.core.method.skill_provider import SkillProvider
from hitl_pmp.core.method.types import GroundSkill, LabeledAction, Policy, Rollout, SetupCommand
from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.planning.grounding import SkillGrounder


class RandomSkillsMethod(Method):
    """Random Skills: at each step, uniformly sample among the currently-applicable
    ground skills and execute one -- no planning, no competence model, no sampler
    learning. Matches predicators' own RandomOptionsApproach.

    Fully domain-agnostic: everything domain-specific (the lifted skills, the
    predicates/objects to ground over, and how a chosen ground skill becomes a raw
    Action) comes from the injected `SkillProvider`, so this one class runs on any
    `--env` -- there is no per-domain `RandomSkillsPolicy` and no
    `isinstance(self.env, ...)` dispatch anymore.

    seed carries this Method's own RNG stream (same private-RNG-derived-from-seed
    pattern as LightSwitchTasks: a public seed field, a PrivateAttr populated in
    model_post_init, never reassigned directly)."""

    skill_provider: SkillProvider
    seed: int = 0

    _rng: np.random.Generator = PrivateAttr()

    def model_post_init(self, __context: object) -> None:
        self._rng = np.random.default_rng(self.seed)

    def get_labeled_action(self, *, state: State) -> LabeledAction:
        provider = self.skill_provider
        objects = provider.objects()
        true_atoms = SkillGrounder.abstract_state(
            state=state, objects=objects, predicates=provider.predicates()
        )
        ground_skills = SkillGrounder.applicable_ground_skills(
            skills=provider.skills(), objects=objects, true_atoms=true_atoms
        )
        assert ground_skills, f"No applicable ground skills for state={state!r}"
        ground_skill = ground_skills[int(self._rng.integers(len(ground_skills)))]

        params = provider.sample_params(ground_skill=ground_skill, rng=self._rng)
        action = provider.compute_action(ground_skill=ground_skill, params=params, state=state)
        objects_desc = ", ".join(obj.name for obj in ground_skill.objects)
        label = f"{ground_skill.skill.name}({objects_desc})"
        if params.size > 0:
            label += f", params={[round(float(p), 2) for p in params]}"
        return LabeledAction(action=action, label=label)

    def reset_environment(self, *, start_state: State) -> bool:
        """Always False: this baseline has no self-navigation to offer, so it declines
        rather than reporting a success it did not achieve (matches SkillOracleMethod).

        This used to `self.env.set_state(state=start_state); return True` -- a
        privileged external state write dressed up as the agent recovering under its
        own power. Nothing calls this method today, so the lie cost nothing; the moment
        a reset-free loop branches on the return value it would silently treat every
        stranded robot as rescued."""
        del start_state  # nothing to navigate towards -- see above
        return False

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
