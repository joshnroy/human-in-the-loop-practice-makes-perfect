"""Exact expectation of a fixed symbolic deployment policy."""

from collections.abc import Mapping
from functools import cache

from pydantic import BaseModel, ConfigDict, Field

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.problem.tasks.types import GroundAtom

from .failure_effect_model import EmpiricalFailureEffects
from .tossing3d_transition_model import apply_success_effects
from .types.failure_effects import FailureEffectCount
from .types.skill_belief import SkillBelief


class SweepDeploymentExpectation(BaseModel):
    """Select the first applicable stock skill, with no human deployment action.

    Outcome counts preserve the same latent competence across retries. This is
    exact integration of this fixed policy, not optimization for sampled theta.
    """

    model_config = ConfigDict(frozen=True)
    ordered_skills: tuple[GroundSkill, ...]
    initial_atoms: frozenset[GroundAtom]
    goal_atoms: frozenset[GroundAtom]
    horizon: int = Field(ge=0)
    failure_effect_counts: tuple[FailureEffectCount, ...] = ()

    def evaluate(self, *, beliefs: Mapping[str, SkillBelief]) -> float:
        names = tuple(sorted({skill.skill.name for skill in self.ordered_skills}))
        indexes = {name: index for index, name in enumerate(names)}
        effects: dict[
            GroundSkill, tuple[frozenset[GroundAtom], frozenset[GroundAtom], frozenset[object]]
        ] = {
            skill: (skill.add_effects, skill.delete_effects, skill.ignore_effects)
            for skill in self.ordered_skills
        }

        @cache
        def moment(*, name: str, successes: int, failures: int) -> float:
            return beliefs[name].competence_outcome_probability(
                successes=successes, failures=failures
            )

        @cache
        def visit(
            *, atoms: frozenset[GroundAtom], remaining: int, counts: tuple[tuple[int, int], ...]
        ) -> float:
            if self.goal_atoms <= atoms:
                return 1.0
            if remaining == 0:
                return 0.0
            action = next(
                (skill for skill in self.ordered_skills if skill.preconditions <= atoms), None
            )
            if action is None:
                return 0.0
            index = indexes[action.skill.name]
            successes, failures = counts[index]
            denominator = moment(name=action.skill.name, successes=successes, failures=failures)
            if denominator == 0:
                return 0.0
            value = 0.0
            for success in (True, False):
                updated = (successes + int(success), failures + int(not success))
                probability = (
                    moment(name=action.skill.name, successes=updated[0], failures=updated[1])
                    / denominator
                )
                if probability <= 0:
                    continue
                successors = (
                    (
                        (
                            1.0,
                            apply_success_effects(
                                true_atoms=atoms, ground_skill=action, effects=effects
                            ),
                        ),
                    )
                    if success
                    else EmpiricalFailureEffects.outcomes(
                        counts=self.failure_effect_counts,
                        ground_skill=action,
                        true_atoms=atoms,
                        was_random_exploration=False,
                    )
                )
                next_counts = counts[:index] + (updated,) + counts[index + 1 :]
                value += probability * sum(
                    weight * visit(atoms=next_atoms, remaining=remaining - 1, counts=next_counts)
                    for weight, next_atoms in successors
                )
            return value

        return visit(
            atoms=self.initial_atoms, remaining=self.horizon, counts=tuple((0, 0) for _ in names)
        )
