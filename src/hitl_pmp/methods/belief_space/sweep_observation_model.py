"""Sweep uses the existing Model B inference and per-attempt training clocks."""

from hitl_pmp.core.method.types import GroundSkill

from .competence_inference import InferenceConfig, create_bayesian_prior
from .tossing3d_observation_model import PracticeExampleSource, SkillBeliefModel
from .types.belief_state import SamplerTrainingState, Tossing3DBeliefState


class SweepBeliefs:
    @staticmethod
    def models(
        *, ground_skills: tuple[GroundSkill, ...], trainable_skill_names: tuple[str, ...]
    ) -> tuple[dict[GroundSkill, SkillBeliefModel], dict[str, SkillBeliefModel]]:
        by_ground = {
            ground: SkillBeliefModel(
                skill=ground.skill,
                example_source=PracticeExampleSource.SAMPLER
                if ground.skill.name in trainable_skill_names
                else PracticeExampleSource.OUTCOME,
            )
            for ground in ground_skills
        }
        return by_ground, {ground.skill.name: model for ground, model in by_ground.items()}

    @staticmethod
    def prior(
        *,
        skill_names: tuple[str, ...],
        trainable_skill_names: tuple[str, ...],
        seed: int,
        num_particles: int,
        config: InferenceConfig,
    ) -> Tossing3DBeliefState:
        return Tossing3DBeliefState(
            skill_beliefs={
                name: create_bayesian_prior(
                    model="local_trend",
                    engine="particle",
                    seed=seed + index,
                    num_particles=num_particles,
                    config=config,
                )
                for index, name in enumerate(sorted(set(skill_names)))
            },
            sampler_training={name: SamplerTrainingState() for name in trainable_skill_names},
        )
