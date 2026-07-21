import numpy as np

from hitl_pmp.core.method.types import LabeledAction
from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.planning.grounding import SkillGrounder

from .environment import LightSwitchEnvironment
from .predicates import ADJACENT, LIGHT_IN_CELL, LIGHT_OFF, LIGHT_ON, ROBOT_IN_CELL
from .skills import LightSwitchSkills


class RandomSkillsPolicy:
    """Uniformly samples among the currently-applicable ground skills (via
    planning.grounding.SkillGrounder) and executes one -- no planning, no
    competence model, no sampler improvement, matching predicators' own
    RandomOptionsApproach (random_options_approach.py). Light-Switch-only, same
    as ActionOraclePolicy/SkillOraclePolicy -- the cross-environment dispatch
    lives one layer up, in
    methods/practice_makes_perfect/random_skills_method.py's
    RandomSkillsMethod. A static-method container, never instantiated, same as
    every other business-logic class in this project."""

    @staticmethod
    def get_labeled_action(
        *, state: State, env: LightSwitchEnvironment, rng: np.random.Generator
    ) -> LabeledAction:
        objects = (env.robot, env.light, *env.get_cells())
        true_atoms = SkillGrounder.abstract_state(
            state=state,
            objects=objects,
            predicates=(LIGHT_ON, LIGHT_OFF, ROBOT_IN_CELL, LIGHT_IN_CELL, ADJACENT),
        )
        ground_skills = SkillGrounder.applicable_ground_skills(
            skills=(
                LightSwitchSkills.MOVE_ROBOT,
                LightSwitchSkills.TURN_ON_LIGHT,
                LightSwitchSkills.TURN_OFF_LIGHT,
                LightSwitchSkills.JUMP_TO_LIGHT,
            ),
            objects=objects,
            true_atoms=true_atoms,
        )
        assert ground_skills, f"No applicable ground skills for state={state!r}"
        ground_skill = ground_skills[int(rng.integers(len(ground_skills)))]

        params = LightSwitchSkills.sample_params(ground_skill=ground_skill, rng=rng)
        action = LightSwitchSkills.compute_action(
            ground_skill=ground_skill, params=params, state=state
        )
        objects_desc = ", ".join(obj.name for obj in ground_skill.objects)
        label = f"{ground_skill.skill.name}({objects_desc})"
        if params.size > 0:
            # Show every entry in params, not just params[0] -- same reasoning as
            # SkillOraclePolicy's identical label-building: params has no name of
            # its own at the GroundSkill/compute_action level, so this must stay
            # generic over param_dim.
            rounded_params = [round(float(p), 2) for p in params]
            label += f", params={rounded_params}"
        return LabeledAction(action=action, label=label)
