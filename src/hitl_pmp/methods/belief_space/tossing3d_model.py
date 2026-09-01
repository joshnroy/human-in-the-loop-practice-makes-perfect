"""Expectimax-facing composition of the Tossing3D practice model."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.problem.tasks.types import GroundAtom

from .tossing3d_constants import OPEN_GRIPPER_SKILL, PICK_SKILL, RESET_SKILL, TOSS_SKILL
from .tossing3d_deployment_model import evaluate_deployment_policy
from .tossing3d_observation_model import refit_belief_state
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
    practice_cost: float = Field(default=0.01, ge=0.0, allow_inf_nan=False)
    random_toss_competence: float = Field(default=0.25, ge=0.0, le=1.0)
    exploration_epsilon: float = Field(default=0.5, ge=0.0, le=1.0)
    pick_cost: float = Field(default=1.0, ge=0.0, allow_inf_nan=False)
    toss_cost: float = Field(default=1.0, ge=0.0, allow_inf_nan=False)
    open_gripper_cost: float = Field(default=1.0, ge=0.0, allow_inf_nan=False)
    reset_cost: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    hard_budget: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    deployment_horizon: int = Field(default=4, ge=0)

    _rng: np.random.Generator = PrivateAttr()
    _atom_indexes: dict[GroundAtom, int] = PrivateAttr(default_factory=dict)
    _precondition_masks: tuple[int, ...] = PrivateAttr(default=())
    _effects: dict[
        GroundSkill,
        tuple[frozenset[GroundAtom], frozenset[GroundAtom], frozenset[object]],
    ] = PrivateAttr(default_factory=dict)
    _belief_ids: dict[object, int] = PrivateAttr(default_factory=dict)

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

    def sample_theta_from_belief(self, *, belief_state: Tossing3DBeliefState) -> Tossing3DTheta:
        return self.sample_thetas_from_belief(belief_state=belief_state, num_samples=1)[0]

    def sample_thetas_from_belief(
        self, *, belief_state: Tossing3DBeliefState, num_samples: int
    ) -> list[Tossing3DTheta]:
        projected = refit_belief_state(state=belief_state)
        pick = self.sample_skills(belief=projected.pick_belief, count=num_samples)
        toss = self.sample_skills(belief=projected.toss_belief, count=num_samples)
        opened = self.sample_skills(belief=projected.open_gripper_belief, count=num_samples)
        return [
            Tossing3DTheta(pick=pick[index], toss=toss[index], open_gripper=opened[index])
            for index in range(num_samples)
        ]

    def sample_skill(self, *, belief: SkillBelief) -> SkillHypothesis:
        return self.sample_skills(belief=belief, count=1)[0]

    def sample_skills(self, *, belief: SkillBelief, count: int) -> list[SkillHypothesis]:
        indexes = np.atleast_1d(
            self._rng.choice(
                len(belief.hypotheses),
                size=count,
                p=np.fromiter((item.probability for item in belief.hypotheses), dtype=np.float64),
            )
        )
        return [belief.hypotheses[int(index)].hypothesis for index in indexes]

    def evaluate_policy(self, *, sampled_theta: Tossing3DTheta) -> float:
        return evaluate_deployment_policy(
            toss_competence=sampled_theta.toss.competence,
            pick_competence=sampled_theta.pick.competence,
            open_competence=sampled_theta.open_gripper.competence,
            horizon=self.deployment_horizon,
        )

    def score_pomdp_value_from_policy_value_and_cost(
        self, *, policy_value: float, summed_cost: float
    ) -> float:
        if self.hard_budget is not None:
            assert summed_cost <= self.hard_budget
            return policy_value
        return policy_value - self.practice_cost * summed_cost

    def get_valid_actions(self, *, environment_state: Tossing3DSearchState) -> list[GroundSkill]:
        costs = {
            PICK_SKILL: self.pick_cost,
            TOSS_SKILL: self.toss_cost,
            OPEN_GRIPPER_SKILL: self.open_gripper_cost,
            RESET_SKILL: self.reset_cost,
        }
        state_mask = self._atoms_mask(atoms=environment_state.true_atoms)
        return [
            ground_skill
            for index, ground_skill in enumerate(self.ground_skills)
            if self._precondition_masks[index] & state_mask == self._precondition_masks[index]
            for cost in [costs[ground_skill.skill.name]]
            if self.hard_budget is None
            or (
                cost is not None
                and environment_state.state.accumulated_cost + cost <= self.hard_budget
            )
        ]

    def _atoms_mask(self, *, atoms: Iterable[GroundAtom]) -> int:
        mask = 0
        for atom in atoms:
            index = self._atom_indexes.get(atom)
            if index is not None:
                mask |= 1 << index
        return mask

    @staticmethod
    def _belief_signature(*, belief: SkillBelief) -> tuple[tuple[float, float, float], ...]:
        return tuple(
            (item.hypothesis.competence, item.hypothesis.learning_rate, item.probability)
            for item in belief.hypotheses
        )

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
            self._belief_id(belief=belief_state.pick_belief),
            self._belief_id(belief=belief_state.toss_belief),
            self._belief_id(belief=belief_state.open_gripper_belief),
            belief_state.pending_pick_examples,
            belief_state.pending_training_examples,
            belief_state.pending_open_gripper_examples,
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
            pick_cost=self.pick_cost,
            toss_cost=self.toss_cost,
            open_gripper_cost=self.open_gripper_cost,
            reset_cost=self.reset_cost,
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
        return list(
            dict.fromkeys(
                (
                    make_tossing3d_search_state(state=next_state, true_atoms=next_true_atoms),
                    next_state.accumulated_cost - belief_state.accumulated_cost,
                )
                for _probability, next_state, next_true_atoms in self.outcomes(
                    environment_state=environment_state,
                    state=belief_state,
                    action=practice_action,
                )
            )
        )

    def transition_outcomes(
        self,
        *,
        environment_state: Tossing3DSearchState,
        practice_action: GroundSkill,
        belief_state: Tossing3DBeliefState,
    ) -> list[tuple[Tossing3DSearchState, float, float]]:
        merged: dict[tuple[Tossing3DSearchState, float], float] = {}
        for probability, next_state, next_true_atoms in self.outcomes(
            environment_state=environment_state,
            state=belief_state,
            action=practice_action,
        ):
            next_environment = make_tossing3d_search_state(
                state=next_state, true_atoms=next_true_atoms
            )
            cost = next_state.accumulated_cost - belief_state.accumulated_cost
            key = (next_environment, cost)
            merged[key] = merged.get(key, 0.0) + probability
        return [(*key, probability) for key, probability in merged.items()]

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
