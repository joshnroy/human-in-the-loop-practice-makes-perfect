"""Expectimax-facing composition of the Tossing3D practice model."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.problem.tasks.types import GroundAtom

from .tossing3d_constants import OPEN_GRIPPER_SKILL, PICK_SKILL, PRACTICE_BUDGET, TOSS_SKILL
from .tossing3d_deployment_model import evaluate_deployment_policies, evaluate_deployment_policy
from .tossing3d_observation_model import (
    SkillBeliefModel,
    make_skill_belief_models,
    refit_belief_state,
)
from .tossing3d_transition_model import make_tossing3d_search_state, transition_outcomes
from .types.belief_state import Tossing3DBeliefState
from .types.search_state import Tossing3DSearchState
from .types.skill_belief import SkillBelief, SkillHypothesis
from .types.theta import Tossing3DTheta


class Tossing3DPracticeModel(BaseModel):
    """Connect Tossing3D dynamics and beliefs to the generic expectimax protocol."""

    model_config = ConfigDict(frozen=True)

    seed: int = 0
    ground_skills: tuple[GroundSkill, ...] = Field(default=(), exclude=True)
    random_toss_competence: float = Field(default=0.25, ge=0.0, le=1.0)
    exploration_epsilon: float = Field(default=0.5, ge=0.0, le=1.0)
    deployment_horizon: int = Field(default=4, ge=0)

    _rng: np.random.Generator = PrivateAttr()
    _atom_indexes: dict[GroundAtom, int] = PrivateAttr(default_factory=dict)
    _precondition_masks: tuple[int, ...] = PrivateAttr(default=())
    _effects: dict[
        GroundSkill,
        tuple[frozenset[GroundAtom], frozenset[GroundAtom], frozenset[object]],
    ] = PrivateAttr(default_factory=dict)
    _belief_ids: dict[object, int] = PrivateAttr(default_factory=dict)
    _skill_belief_models: dict[GroundSkill, SkillBeliefModel] = PrivateAttr(default_factory=dict)
    _skill_belief_models_by_name: dict[str, SkillBeliefModel] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: object) -> None:
        self._rng = np.random.default_rng(self.seed)
        relevant_atoms = sorted(
            {
                atom
                for skill in self.ground_skills
                for atom in (*skill.preconditions, *skill.add_effects, *skill.delete_effects)
            },
            key=str,
        )
        self._atom_indexes = {atom: index for index, atom in enumerate(relevant_atoms)}
        self._precondition_masks = tuple(
            self._atoms_mask(atoms=skill.preconditions) for skill in self.ground_skills
        )
        self._effects = {
            skill: (skill.add_effects, skill.delete_effects, skill.ignore_effects)
            for skill in self.ground_skills
        }
        self._skill_belief_models, self._skill_belief_models_by_name = make_skill_belief_models(
            ground_skills=self.ground_skills
        )

    def sample_theta_from_belief(self, *, belief_state: Tossing3DBeliefState) -> Tossing3DTheta:
        return self.sample_thetas_from_belief(belief_state=belief_state, num_samples=1)[0]

    def sample_thetas_from_belief(
        self, *, belief_state: Tossing3DBeliefState, num_samples: int
    ) -> list[Tossing3DTheta]:
        projected = refit_belief_state(state=belief_state)
        pick = self.sample_skills(belief=projected.skill_beliefs[PICK_SKILL], count=num_samples)
        toss = self.sample_skills(belief=projected.skill_beliefs[TOSS_SKILL], count=num_samples)
        opened = self.sample_skills(
            belief=projected.skill_beliefs[OPEN_GRIPPER_SKILL], count=num_samples
        )
        return [
            Tossing3DTheta.model_construct(
                pick=pick[index], toss=toss[index], open_gripper=opened[index]
            )
            for index in range(num_samples)
        ]

    def sample_skill(self, *, belief: SkillBelief) -> SkillHypothesis:
        return self.sample_skills(belief=belief, count=1)[0]

    def sample_skills(self, *, belief: SkillBelief, count: int) -> list[SkillHypothesis]:
        parameters = belief.sample(rng=self._rng, count=count)
        return [
            SkillHypothesis.model_construct(competence=float(row[0]), learning_rate=float(row[1]))
            for row in parameters
        ]

    def evaluate_policy(self, *, sampled_theta: Tossing3DTheta) -> float:
        return evaluate_deployment_policy(
            toss_competence=sampled_theta.toss.competence,
            pick_competence=sampled_theta.pick.competence,
            open_competence=sampled_theta.open_gripper.competence,
            horizon=self.deployment_horizon,
        )

    def sample_policy_values_from_belief(
        self, *, belief_state: Tossing3DBeliefState, num_samples: int
    ) -> np.ndarray:
        projected = refit_belief_state(state=belief_state)
        competences = []
        for skill_name in (PICK_SKILL, TOSS_SKILL, OPEN_GRIPPER_SKILL):
            competences.append(
                projected.skill_beliefs[skill_name].sample(rng=self._rng, count=num_samples)[:, 0]
            )
        return evaluate_deployment_policies(
            toss_competences=competences[1],
            pick_competences=competences[0],
            open_competences=competences[2],
            horizon=self.deployment_horizon,
        )

    def G(self, *, policy_value: float, summed_cost: float) -> float:
        """Return deployment value for feasible practice, otherwise negative infinity."""
        return policy_value if summed_cost <= PRACTICE_BUDGET else -np.inf

    def observe_outcome(
        self,
        *,
        state: Tossing3DBeliefState,
        ground_skill: GroundSkill,
        success: bool,
        was_random_exploration: bool,
    ) -> Tossing3DBeliefState:
        """Apply any belief observation associated with a practiced skill."""
        return self._skill_belief_models[ground_skill].observe_outcome(
            state=state,
            success=success,
            was_random_exploration=was_random_exploration,
        )

    def observe_training_example(
        self, *, state: Tossing3DBeliefState, skill_name: str
    ) -> Tossing3DBeliefState:
        """Apply any learning-curve update associated with a sampler example."""
        return self._skill_belief_models_by_name[skill_name].observe_training_example(state=state)

    def get_valid_actions(self, *, environment_state: Tossing3DSearchState) -> list[GroundSkill]:
        state_mask = self._atoms_mask(atoms=environment_state.true_atoms)
        return [
            ground_skill
            for index, ground_skill in enumerate(self.ground_skills)
            if self._precondition_masks[index] & state_mask == self._precondition_masks[index]
        ]

    def _atoms_mask(self, *, atoms: Iterable[GroundAtom]) -> int:
        mask = 0
        for atom in atoms:
            index = self._atom_indexes.get(atom)
            if index is not None:
                mask |= 1 << index
        return mask

    @staticmethod
    def _belief_signature(*, belief: SkillBelief) -> tuple[object, ...]:
        return (type(belief).__name__, *belief.signature())

    def _belief_id(self, *, belief: SkillBelief) -> int:
        signature = self._belief_signature(belief=belief)
        identifier = self._belief_ids.get(signature)
        if identifier is None:
            identifier = len(self._belief_ids)
            self._belief_ids[signature] = identifier
        return identifier

    def search_cache_key(
        self,
        *,
        environment_state: Tossing3DSearchState,
        summed_cost: float,
        belief_state: Tossing3DBeliefState,
        horizon: int,
    ) -> object:
        assert summed_cost == belief_state.accumulated_cost
        return (
            self._atoms_mask(atoms=environment_state.true_atoms),
            tuple(
                (skill_name, self._belief_id(belief=belief))
                for skill_name, belief in sorted(belief_state.skill_beliefs.items())
            ),
            tuple(sorted(belief_state.pending_examples.items())),
            belief_state.accumulated_cost,
            horizon,
        )

    def outcomes(
        self,
        *,
        environment_state: Tossing3DSearchState,
        state: Tossing3DBeliefState,
        action: GroundSkill,
    ) -> tuple[tuple[float, Tossing3DBeliefState, frozenset[GroundAtom]], ...]:
        return transition_outcomes(
            environment_state=environment_state,
            state=state,
            action=action,
            ground_skills=self.ground_skills,
            effects=self._effects,
            exploration_epsilon=self.exploration_epsilon,
            random_toss_competence=self.random_toss_competence,
        )

    def sample_next_states(
        self,
        *,
        environment_state: Tossing3DSearchState,
        practice_action: GroundSkill,
        belief_state: Tossing3DBeliefState,
    ) -> list[tuple[Tossing3DSearchState, float]]:
        successors: dict[object, tuple[Tossing3DSearchState, float]] = {}
        for _probability, next_state, next_true_atoms in self.outcomes(
            environment_state=environment_state,
            state=belief_state,
            action=practice_action,
        ):
            next_environment = make_tossing3d_search_state(
                state=next_state, true_atoms=next_true_atoms
            )
            cost = next_state.accumulated_cost - belief_state.accumulated_cost
            successors[self.transition_key(environment_state=next_environment, cost=cost)] = (
                next_environment,
                cost,
            )
        return list(successors.values())

    def transition_outcomes(
        self,
        *,
        environment_state: Tossing3DSearchState,
        practice_action: GroundSkill,
        belief_state: Tossing3DBeliefState,
    ) -> list[tuple[Tossing3DSearchState, float, float]]:
        merged: dict[object, tuple[Tossing3DSearchState, float, float]] = {}
        for probability, next_state, next_true_atoms in self.outcomes(
            environment_state=environment_state,
            state=belief_state,
            action=practice_action,
        ):
            next_environment = make_tossing3d_search_state(
                state=next_state, true_atoms=next_true_atoms
            )
            cost = next_state.accumulated_cost - belief_state.accumulated_cost
            key = self.transition_key(environment_state=next_environment, cost=cost)
            previous_probability = merged.get(key, (next_environment, cost, 0.0))[2]
            merged[key] = (next_environment, cost, previous_probability + probability)
        return list(merged.values())

    @staticmethod
    def transition_key(*, environment_state: Tossing3DSearchState, cost: float) -> object:
        state = environment_state.state
        return (
            environment_state.atoms,
            tuple(sorted(state.skill_beliefs.items())),
            tuple(sorted(state.pending_examples.items())),
            state.accumulated_cost,
            cost,
        )

    def update_belief_state(
        self,
        *,
        belief_state: Tossing3DBeliefState,
        environment_state: Tossing3DSearchState,
        potential_next_environment_state: Tossing3DSearchState,
        practice_action: GroundSkill,
    ) -> Tossing3DBeliefState:
        return potential_next_environment_state.state

    def transition_probability(
        self,
        *,
        potential_next_environment_state: Tossing3DSearchState,
        sampled_cost: float,
        environment_state: Tossing3DSearchState,
        practice_action: GroundSkill,
        belief_state: Tossing3DBeliefState,
    ) -> float:
        total_probability = 0.0
        for probability, next_state, next_true_atoms in self.outcomes(
            environment_state=environment_state,
            state=belief_state,
            action=practice_action,
        ):
            if (
                next_state == potential_next_environment_state.state
                and next_true_atoms == potential_next_environment_state.true_atoms
                and next_state.accumulated_cost - belief_state.accumulated_cost == sampled_cost
            ):
                total_probability += probability
        return total_probability
