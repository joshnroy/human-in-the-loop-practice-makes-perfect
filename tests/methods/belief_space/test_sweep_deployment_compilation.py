"""Exact expectation of a fixed symbolic deployment policy."""

from collections.abc import Mapping
from functools import cache

import pytest
from pydantic import BaseModel, ConfigDict, Field

from hitl_pmp.core.method.types import GroundSkill, LiftedAtom, Skill
from hitl_pmp.core.problem.tasks.types import GroundAtom, Predicate
from hitl_pmp.methods.belief_space.failure_effect_model import EmpiricalFailureEffects
from hitl_pmp.methods.belief_space.sweep_deployment_model import SweepDeploymentExpectation
from hitl_pmp.methods.belief_space.tossing3d_transition_model import apply_success_effects
from hitl_pmp.methods.belief_space.types.failure_effects import FailureEffectCount
from hitl_pmp.methods.belief_space.types.skill_belief import (
    SkillBelief,
    SkillHypothesis,
    WeightedHypothesis,
)
from hitl_pmp.methods.belief_space.types.weighted_hypothesis_belief import WeightedHypothesisBelief


class _RecursiveReference(BaseModel):
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


@pytest.mark.parametrize("horizon", [0, 2, 3, 5, 7])
@pytest.mark.parametrize("failure_shortcut", [False, True])
def test_compiled_policy_matches_original_recursive_evaluator(*, horizon, failure_shortcut):
    predicates = tuple(
        Predicate(name=f"Stage{i}", types=(), holds=lambda s, o: False) for i in range(4)
    )
    atoms = tuple(GroundAtom(predicate=p, objects=()) for p in predicates)
    skills = tuple(
        GroundSkill(
            skill=Skill(
                name=f"Stock{i}",
                parameters=(),
                preconditions=frozenset({LiftedAtom(predicate=predicates[i], variables=())}),
                add_effects=frozenset({LiftedAtom(predicate=predicates[i + 1], variables=())}),
                delete_effects=frozenset({LiftedAtom(predicate=predicates[i], variables=())}),
                param_dim=1,
                practice_cost=1,
            ),
            objects=(),
        )
        for i in range(3)
    )
    failures = (
        (
            FailureEffectCount(
                ground_skill=skills[0],
                before_atoms=frozenset({atoms[0]}),
                was_random_exploration=False,
                add_effects=frozenset({atoms[2]}),
                delete_effects=frozenset({atoms[0]}),
                count=2,
            ),
            FailureEffectCount(
                ground_skill=skills[0],
                before_atoms=frozenset({atoms[0]}),
                was_random_exploration=False,
                add_effects=frozenset(),
                delete_effects=frozenset(),
                count=1,
            ),
        )
        if failure_shortcut
        else ()
    )
    settings = dict(
        ordered_skills=skills,
        initial_atoms=frozenset({atoms[0]}),
        goal_atoms=frozenset({atoms[3]}),
        horizon=horizon,
        failure_effect_counts=failures,
    )
    for values in ((0.0, 1.0), (0.2, 0.8), (0.0, 0.0), (1.0, 1.0)):
        beliefs = {
            skill.skill.name: WeightedHypothesisBelief(
                hypotheses=tuple(
                    WeightedHypothesis(
                        hypothesis=SkillHypothesis(competence=value, learning_rate=0),
                        probability=0.5,
                    )
                    for value in values
                )
            )
            for skill in skills
        }
        expected = _RecursiveReference(**settings).evaluate(beliefs=beliefs)
        actual = SweepDeploymentExpectation(**settings).evaluate(beliefs=beliefs)
        assert actual == pytest.approx(expected, rel=0, abs=2e-14)
