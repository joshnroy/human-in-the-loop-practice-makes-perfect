"""Expectimax-facing composition of the Tossing3D practice model."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from hitl_pmp.core.method.types import GroundSkill, SamplerConsultation
from hitl_pmp.core.problem.tasks.types import GroundAtom

from .competence_inference import BayesianSkillBelief
from .object_symmetry import InterchangeableObjects
from .practice_domain import TOSSING3D_DOMAIN, PracticeDomain
from .tossing3d_constants import PRACTICE_BUDGET
from .tossing3d_observation_model import (
    SkillBeliefModel,
    make_skill_belief_models,
    refit_belief_state,
)
from .tossing3d_transition_model import (
    estimated_action_cost,
    make_tossing3d_search_state,
    transition_outcomes,
)
from .types.belief_state import Tossing3DBeliefState
from .types.competence_evidence import CompetenceEvidence
from .types.failure_effects import FailureEffectCount
from .types.search_state import Tossing3DSearchState
from .types.skill_belief import SkillBelief, SkillHypothesis


class Tossing3DPracticeModel(BaseModel):
    """Connect Tossing3D dynamics and beliefs to the generic expectimax protocol."""

    model_config = ConfigDict(frozen=True)

    seed: int = 0
    domain: PracticeDomain = Field(default=TOSSING3D_DOMAIN, exclude=True)
    ground_skills: tuple[GroundSkill, ...] = Field(default=(), exclude=True)
    random_toss_competence: float = Field(default=0.25, ge=0.0, le=1.0)
    exploration_epsilon: float = Field(default=0.5, ge=0.0, le=1.0)
    competence_evidence: CompetenceEvidence = CompetenceEvidence.NON_EPSILON
    # None: the domain's own deployment horizon.
    deployment_horizon: int | None = Field(default=None, ge=0)
    linear_cost_lambda: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    # STOP is a skill with competence 1 and this price (one robot step under step
    # accounting, 0 otherwise): every stopping value is taken after paying it.
    stop_cost: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    failure_effect_counts: tuple[FailureEffectCount, ...] = Field(default=(), exclude=True)
    # (symbolic state, ground action) pairs whose parameter pool starved this
    # session (see EesMethod.record_starved_parameter_pool). Masked in
    # get_valid_actions so a re-run of the search *chooses again* rather than
    # deterministically re-picking an action that cannot be dispatched. Excluded
    # from dumps: session-scoped scratch, not part of the learned model.
    starved_pools: tuple[tuple[frozenset[GroundAtom], GroundSkill], ...] = Field(
        default=(), exclude=True
    )
    # Exact search reductions (see object_symmetry.py and search_cache_key); the
    # method turns both on for expectimax on a domain with `reduce_expectimax_search`.
    object_symmetry: bool = False
    merge_beliefs_by_counts: bool = False

    _rng: np.random.Generator = PrivateAttr()
    _dynamic_predicates: frozenset[str] = PrivateAttr(default=frozenset())
    # (costs, failure records, starved pools, static atoms, symmetry) of the last
    # detection; the first three compared by identity, as they are fixed in a search.
    _symmetry_cache: (
        tuple[object, object, object, frozenset[GroundAtom], InterchangeableObjects] | None
    ) = PrivateAttr(default=None)
    _atom_indexes: dict[GroundAtom, int] = PrivateAttr(default_factory=dict)
    _precondition_masks: tuple[int, ...] = PrivateAttr(default=())
    _effects: dict[
        GroundSkill,
        tuple[frozenset[GroundAtom], frozenset[GroundAtom], frozenset[object]],
    ] = PrivateAttr(default_factory=dict)
    _skill_belief_models: dict[GroundSkill, SkillBeliefModel] = PrivateAttr(default_factory=dict)
    _skill_belief_models_by_name: dict[str, SkillBeliefModel] = PrivateAttr(default_factory=dict)
    _domain: PracticeDomain = PrivateAttr()

    def model_post_init(self, __context: object) -> None:
        self._rng = np.random.default_rng(self.seed)
        relevant_atoms = sorted(
            {
                atom
                for skill in self.ground_skills
                for atom in (
                    *skill.preconditions,
                    *skill.add_effects,
                    *skill.delete_effects,
                    *(
                        atom
                        for effect in skill.conditional_add_effects
                        for atom in (*effect.conditions, *effect.add_effects)
                    ),
                )
            },
            key=str,
        )
        self._atom_indexes = {atom: index for index, atom in enumerate(relevant_atoms)}
        self._dynamic_predicates = frozenset(
            atom.predicate.name
            for skill in self.ground_skills
            for atom in (
                *skill.add_effects,
                *skill.delete_effects,
                *(atom for effect in skill.conditional_add_effects for atom in effect.add_effects),
            )
        ) | frozenset(
            predicate.name for skill in self.ground_skills for predicate in skill.ignore_effects
        )
        self._precondition_masks = tuple(
            self._atoms_mask(atoms=skill.preconditions) for skill in self.ground_skills
        )
        self._effects = {
            skill: (skill.add_effects, skill.delete_effects, skill.ignore_effects)
            for skill in self.ground_skills
        }
        self._skill_belief_models, self._skill_belief_models_by_name = make_skill_belief_models(
            ground_skills=self.ground_skills, domain=self.domain
        )
        self._domain = (
            self.domain
            if self.deployment_horizon is None
            else self.domain.model_copy(update={"deployment_horizon": self.deployment_horizon})
        )

    def sample_theta_from_belief(self, *, belief_state: Tossing3DBeliefState) -> Any:
        return self.sample_thetas_from_belief(belief_state=belief_state, num_samples=1)[0]

    def sample_thetas_from_belief(
        self, *, belief_state: Tossing3DBeliefState, num_samples: int
    ) -> list[Any]:
        projected = refit_belief_state(state=belief_state)
        return self._domain.sample_thetas(
            beliefs=projected.skill_beliefs,
            sample_skills=lambda belief, count: self.sample_skills(belief=belief, count=count),
            count=num_samples,
        )

    def sample_skill(self, *, belief: SkillBelief) -> SkillHypothesis:
        return self.sample_skills(belief=belief, count=1)[0]

    def sample_skills(self, *, belief: SkillBelief, count: int) -> list[SkillHypothesis]:
        parameters = belief.sample(rng=self._rng, count=count)
        return [
            SkillHypothesis.model_construct(
                competence=float(row[0]), learning_rate=float(row[1]), cost=float(row[2])
            )
            for row in parameters
        ]

    def evaluate_policy(self, *, sampled_theta: Any) -> float:
        return self._domain.evaluate_theta(theta=sampled_theta)

    def sample_policy_values_from_belief(
        self, *, belief_state: Tossing3DBeliefState, num_samples: int
    ) -> np.ndarray:
        projected = refit_belief_state(state=belief_state)
        return self._domain.sample_policy_values(
            beliefs=projected.skill_beliefs, rng=self._rng, count=num_samples
        )

    def G(self, *, policy_value: float, summed_cost: float) -> float:
        """Apply either the PDF's hard-budget or linear-cost objective."""
        summed_cost += self.stop_cost
        if self.linear_cost_lambda is not None:
            return policy_value - self.linear_cost_lambda * summed_cost
        return policy_value if summed_cost <= PRACTICE_BUDGET else -np.inf

    def J(
        self, *, belief_state: Tossing3DBeliefState, summed_cost: float, num_samples: int
    ) -> float:
        """Integrate J(C, b) exactly over the represented competence distributions.

        G is affine in deployment value for either supported cost objective.
        No fresh Monte Carlo draw is needed to compare stopping values, which
        otherwise introduces noise much larger than one reset's cost penalty.
        Particle/grid approximation and the existing training forecasts remain.
        """
        assert num_samples >= 1
        projected = refit_belief_state(state=belief_state)
        policy_value = self._domain.expected_policy_value(beliefs=projected.skill_beliefs)
        return self.G(policy_value=policy_value, summed_cost=summed_cost)

    def observe_outcome(
        self,
        *,
        state: Tossing3DBeliefState,
        ground_skill: GroundSkill,
        success: bool,
        was_random_exploration: bool,
        observed_cost: float | None = None,
        consultation: SamplerConsultation | None = None,
    ) -> Tossing3DBeliefState:
        """Apply any belief observation associated with a practiced skill.

        With a `consultation`, `competence_evidence` decides whether the outcome
        conditions competence; without one (a reset) the attempt is not
        epsilon-random and is admitted."""
        return self._skill_belief_models[ground_skill].observe_outcome(
            state=state,
            success=success,
            was_random_exploration=was_random_exploration,
            observed_cost=observed_cost,
            condition_competence=(
                None
                if consultation is None
                else self.competence_evidence.admits(consultation=consultation)
            ),
        )

    def observe_training_example(
        self, *, state: Tossing3DBeliefState, skill_name: str, success: bool
    ) -> Tossing3DBeliefState:
        """Apply any learning-curve update associated with a sampler example."""
        return self._skill_belief_models_by_name[skill_name].observe_training_example(
            state=state, success=success
        )

    def get_valid_actions(self, *, environment_state: Tossing3DSearchState) -> list[GroundSkill]:
        """Return every applicable action not masked by an observed starved pool.

        A reset whose symbolic effect changes no atom is still offered: it
        re-randomizes the geometry, which the abstract state does not represent,
        so whether that is worth its cost is the planner's choice.
        """
        state_mask = self._atoms_mask(atoms=environment_state.true_atoms)
        actions = [
            ground_skill
            for index, ground_skill in enumerate(self.ground_skills)
            if self._precondition_masks[index] & state_mask == self._precondition_masks[index]
            and (environment_state.true_atoms, ground_skill) not in self.starved_pools
        ]
        if not self.object_symmetry:
            return actions
        return self.object_symmetry_for(environment_state=environment_state).representatives(
            atoms=environment_state.true_atoms, actions=actions
        )

    def object_symmetry_for(
        self, *, environment_state: Tossing3DSearchState
    ) -> InterchangeableObjects:
        """The interchangeable objects under this model and the state's cost forecasts.

        Derived from the model's *current* failure records and starved pools on every
        call rather than at construction, because the method adds them with
        `model_copy`, which skips `model_post_init`. Within one search none of the
        inputs changes (beliefs are shallow-copied, so the forecast dict is the same
        object at every node), so one detection serves the whole search."""
        if not self.object_symmetry:
            return InterchangeableObjects()
        belief = environment_state.state
        static = frozenset(
            atom
            for atom in environment_state.true_atoms
            if atom.predicate.name not in self._dynamic_predicates
        )
        cached = self._symmetry_cache
        if (
            cached is not None
            and cached[0] is belief.expected_execution_costs
            and cached[1] is self.failure_effect_counts
            and cached[2] is self.starved_pools
            and cached[3] == static
        ):
            return cached[4]
        symmetry = InterchangeableObjects.detect(
            ground_skills=self.ground_skills,
            dynamic_predicates=self._dynamic_predicates,
            static_atoms=static,
            cost_of=lambda ground_skill: estimated_action_cost(state=belief, action=ground_skill),
            failure_effect_counts=self.failure_effect_counts,
            starved_pools=self.starved_pools,
        )
        self._symmetry_cache = (
            belief.expected_execution_costs,
            self.failure_effect_counts,
            self.starved_pools,
            static,
            symmetry,
        )
        return symmetry

    def _atoms_mask(self, *, atoms: Iterable[GroundAtom]) -> int:
        mask = 0
        for atom in atoms:
            index = self._atom_indexes.get(atom)
            if index is not None:
                mask |= 1 << index
        return mask

    @staticmethod
    def _belief_signature(*, belief: SkillBelief) -> bytes:
        """Identify a posterior without retaining its potentially large buffers.

        Search memo tables live for one solve. A persistent signature-to-integer
        table previously kept every hypothetical posterior alive across solves,
        including hundreds of KB per fixed-grid posterior. A SHA-256 identifier
        is constant-size, requires no interning table, and includes all the same
        signature information. Type tags and lengths prevent ambiguous joins.
        """
        digest = hashlib.sha256()

        def update(*, value: object) -> None:
            if isinstance(value, tuple):
                digest.update(b"tuple" + len(value).to_bytes(8, "big"))
                for item in value:
                    update(value=item)
                return
            if isinstance(value, BaseModel):
                digest.update(b"model")
                update(
                    value=(
                        "pydantic",
                        type(value).__module__,
                        type(value).__qualname__,
                        value.model_dump_json(),
                    )
                )
                return
            if isinstance(value, bytes):
                tag, payload = b"bytes", value
            elif isinstance(value, str):
                tag, payload = b"string", value.encode("utf-8")
            elif isinstance(value, bool):
                tag, payload = b"bool", bytes([value])
            elif isinstance(value, int):
                tag, payload = b"int", str(value).encode("ascii")
            elif isinstance(value, float):
                # Python's old tuple keys treat positive and negative zero equally.
                tag, payload = b"float", (0.0 if value == 0 else value).hex().encode("ascii")
            elif value is None:
                tag, payload = b"none", b""
            else:
                raise TypeError(f"unsupported belief signature component: {type(value).__name__}")
            digest.update(tag + len(payload).to_bytes(8, "big"))
            digest.update(payload)

        update(value=(type(belief).__module__, type(belief).__qualname__, belief.signature()))
        return digest.digest()

    def search_cache_key(
        self,
        *,
        environment_state: Tossing3DSearchState,
        summed_cost: float,
        belief_state: Tossing3DBeliefState,
        horizon: int | None,
    ) -> object:
        assert summed_cost == belief_state.accumulated_cost
        atoms = environment_state.true_atoms
        if self.object_symmetry:
            atoms = self.object_symmetry_for(environment_state=environment_state).canonical_atoms(
                atoms=atoms
            )
        if not self.merge_beliefs_by_counts:
            return (
                self._atoms_mask(atoms=atoms),
                tuple(
                    (skill_name, self._belief_signature(belief=belief))
                    for skill_name, belief in sorted(belief_state.skill_beliefs.items())
                ),
                tuple(sorted(belief_state.pending_examples.items())),
                tuple(sorted(belief_state.sampler_training.items())),
                belief_state.accumulated_cost,
                horizon,
            )
        # Within one search every belief descends from the root's by likelihood
        # products alone -- no refit, no cycle advance, no cost observation -- so equal
        # S/F counts are the same posterior up to floating-point order, and the
        # accumulated cost of one multiset of actions is the same sum up to order.
        # Exact to rounding, not bitwise; valid only for a key that lives one search.
        return (
            self._atoms_mask(atoms=atoms),
            tuple(
                (skill_name, belief.cycle_successes, belief.cycle_failures)
                if isinstance(belief, BayesianSkillBelief)
                else (skill_name, self._belief_signature(belief=belief))
                for skill_name, belief in sorted(belief_state.skill_beliefs.items())
            ),
            tuple(sorted(belief_state.pending_examples.items())),
            tuple(sorted(belief_state.sampler_training.items())),
            float(f"{belief_state.accumulated_cost:.12g}"),
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
            failure_effect_counts=self.failure_effect_counts,
            competence_evidence=self.competence_evidence,
            domain=self._domain,
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
            tuple(sorted(state.sampler_training.items())),
            state.accumulated_cost,
            cost,
        )

    def compute_next_belief_state(
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
