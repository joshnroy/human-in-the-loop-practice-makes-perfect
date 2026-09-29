"""Exact expectation of a fixed symbolic deployment policy."""

from collections.abc import Mapping
from functools import cache, lru_cache

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
        names, monomials = self._compile(
            ordered_skills=self.ordered_skills,
            initial_atoms=self.initial_atoms,
            goal_atoms=self.goal_atoms,
            horizon=self.horizon,
            failure_effect_counts=self.failure_effect_counts,
        )

        @cache
        def moment(*, name: str, successes: int, failures: int) -> float:
            return beliefs[name].competence_outcome_probability(
                successes=successes, failures=failures
            )

        value = 0.0
        for counts, coefficient in monomials:
            term = coefficient
            for name, (successes, failures) in zip(names, counts, strict=True):
                if successes == failures == 0:
                    continue
                denominator = moment(name=name, successes=0, failures=0)
                if denominator == 0:
                    term = 0.0
                    break
                term *= moment(name=name, successes=successes, failures=failures) / denominator
            value += term
        return value

    @staticmethod
    @lru_cache(maxsize=128)
    def _compile(
        *,
        ordered_skills: tuple[GroundSkill, ...],
        initial_atoms: frozenset[GroundAtom],
        goal_atoms: frozenset[GroundAtom],
        horizon: int,
        failure_effect_counts: tuple[FailureEffectCount, ...],
    ) -> tuple[tuple[str, ...], tuple[tuple[tuple[tuple[int, int], ...], float], ...]]:
        names = tuple(sorted({skill.skill.name for skill in ordered_skills}))
        indexes = {name: index for index, name in enumerate(names)}
        zero = tuple((0, 0) for _ in names)
        effects: dict[
            GroundSkill, tuple[frozenset[GroundAtom], frozenset[GroundAtom], frozenset[object]]
        ] = {
            skill: (skill.add_effects, skill.delete_effects, skill.ignore_effects)
            for skill in ordered_skills
        }

        @cache
        def visit(
            *, atoms: frozenset[GroundAtom], remaining: int
        ) -> tuple[tuple[tuple[tuple[int, int], ...], float], ...]:
            if goal_atoms <= atoms:
                return ((zero, 1.0),)
            if remaining == 0:
                return ()
            action = next((skill for skill in ordered_skills if skill.preconditions <= atoms), None)
            if action is None:
                return ()
            index = indexes[action.skill.name]
            polynomial: dict[tuple[tuple[int, int], ...], float] = {}
            for success in (True, False):
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
                        counts=failure_effect_counts,
                        ground_skill=action,
                        true_atoms=atoms,
                        was_random_exploration=False,
                    )
                )
                for weight, next_atoms in successors:
                    for counts, coefficient in visit(atoms=next_atoms, remaining=remaining - 1):
                        successes, failures = counts[index]
                        updated = (
                            counts[:index]
                            + ((successes + int(success), failures + int(not success)),)
                            + counts[index + 1 :]
                        )
                        polynomial[updated] = polynomial.get(updated, 0.0) + weight * coefficient
            return tuple(polynomial.items())

        return names, visit(atoms=initial_atoms, remaining=horizon)
