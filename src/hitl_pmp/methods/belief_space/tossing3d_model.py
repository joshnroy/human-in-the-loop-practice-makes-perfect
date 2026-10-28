"""Expectimax-facing composition of the Tossing3D practice model."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from hitl_pmp.core.method.types import GroundSkill, SamplerConsultation
from hitl_pmp.core.problem.tasks.types import GroundAtom

from .belief_digest import BeliefDigest
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
from .types.forced_start_reset import ForcedStartReset
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
    # The always-ask wrapper's reset, known to the planner (Josh, 2026-10-19): every
    # skill with continuous parameters is followed by it. None for every other arm,
    # whose model is then exactly the one without this field. Excluded from dumps so
    # those arms' decision logs are unchanged; the method logs it on its own.
    forced_start_reset: ForcedStartReset | None = Field(default=None, exclude=True)

    _rng: np.random.Generator = PrivateAttr()
    _dynamic_predicates: frozenset[str] = PrivateAttr(default=frozenset())
    # (costs, failure records, starved pools, forced reset, static atoms, symmetry) of
    # the last detection; the first four compared by identity, as they are fixed in a
    # search.
    _symmetry_cache: (
        tuple[object, object, object, object, frozenset[GroundAtom], InterchangeableObjects] | None
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
        self,
        *,
        state: Tossing3DBeliefState,
        skill_name: str,
        success: bool,
        ground_skill: GroundSkill | None = None,
    ) -> Tossing3DBeliefState:
        """Apply any learning-curve update associated with a sampler example."""
        if ground_skill is None:
            contracts = {
                m.belief_name
                for g, m in self._skill_belief_models.items()
                if g.skill.name == skill_name
            }
            if len(contracts) > 1:
                raise ValueError("Count-conditioned training requires the selected ground skill")
            model = self._skill_belief_models_by_name[skill_name]
        else:
            if ground_skill.skill.name != skill_name:
                raise ValueError("Sampler skill differs from selected grounding")
            model = self._skill_belief_models[ground_skill]
        return model.observe_training_example(state=state, success=success)

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
            and cached[3] is self.forced_start_reset
            and cached[4] == static
        ):
            return cached[5]
        symmetry = InterchangeableObjects.detect(
            ground_skills=self.ground_skills,
            dynamic_predicates=self._dynamic_predicates,
            static_atoms=static,
            cost_of=lambda ground_skill: estimated_action_cost(state=belief, action=ground_skill),
            failure_effect_counts=self.failure_effect_counts,
            starved_pools=self.starved_pools,
            # The reset's destination is one fixed state, so a swap is a symmetry of
            # the search only if it maps that state onto itself.
            invariant_atoms=(
                frozenset()
                if self.forced_start_reset is None
                else self.forced_start_reset.start_atoms
            ),
        )
        self._symmetry_cache = (
            belief.expected_execution_costs,
            self.failure_effect_counts,
            self.starved_pools,
            self.forced_start_reset,
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
        return BeliefDigest.of(belief=belief)

    @staticmethod
    def cost_key(*, cost: float) -> float:
        """The accumulated cost as a merge key, independent of summation order.

        One multiset of actions sums to costs that differ in the last bit by order
        whenever a cost is inexact in binary (0.61 + 1.04 + ... at a 0.01 step price):
        an error near n * 2**-52 relative. Nine significant digits leave six orders of
        magnitude above that, so such sums share a key except in the rare case that
        straddles a rounding boundary, while genuinely different costs -- at least
        one control step's price apart, or a fraction of one in a mean of step counts
        -- stay apart up to costs of order 1e5 steps. Decimal digits make the key
        scale-free under the power-of-ten cost rescale (721 and 7.21 round alike)."""
        return float(f"{cost:.9g}")

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
                self.cost_key(cost=belief_state.accumulated_cost),
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
            self.cost_key(cost=belief_state.accumulated_cost),
            horizon,
        )

    def belief_cache_key(
        self,
        *,
        environment_state: Tossing3DSearchState,
        belief_state: Tossing3DBeliefState,
        horizon: int | None,
    ) -> object:
        """`search_cache_key` less the accumulated cost, which J reads only through G's
        separate cost term. Every other field moves J itself and stays: the training
        clocks and sampler state feed the refit J forecasts from, and the horizon bounds
        the continuation."""
        key = self.search_cache_key(
            environment_state=environment_state,
            summed_cost=belief_state.accumulated_cost,
            belief_state=belief_state,
            horizon=horizon,
        )
        assert isinstance(key, tuple)
        return (*key[:-2], key[-1])

    def outcome_draw_key(
        self,
        *,
        environment_state: Tossing3DSearchState,
        belief_state: Tossing3DBeliefState,
        action: GroundSkill,
        scope: str,
    ) -> object:
        """What `transition_outcomes`' probabilities read, and nothing else.

        The success probability reads only the acting skill's own belief and, for the
        sampler skill, its training state (whether epsilon applies); the failure branches
        also read the atoms (failure records match on them); failure records, costs and
        epsilon are fixed within a search. The accumulated cost, every training clock
        (`pending_examples`), the horizon and the other skills' beliefs are left out:
        they move J but not this outcome, and they are what a reset changes, so keying
        on them is what let a reset re-roll a pick.

        "state" keeps the atoms and the ground action, so a reset that changes the atoms
        still buys a fresh draw. "skill" drops them: the same belief is the same trial
        wherever it is attempted, so success and failure agree across states (a shared
        uniform falls on the same side of the same success probability) while the
        failure branch's own effects follow the state."""
        name = action.skill.name
        belief_name = self.domain.ground_belief_name(ground_skill=action) or name
        belief = belief_state.skill_beliefs.get(belief_name)
        read = (
            belief_name,
            None if belief is None else self._belief_signature(belief=belief),
            belief_state.sampler_training.get(name) if name == self.domain.sampler_skill else None,
        )
        if scope == "skill":
            return read
        assert scope == "state", f"unknown draw scope {scope!r}"
        # The whole state, not `_atoms_mask`: failure records match on every atom.
        return (environment_state.true_atoms, action, *read)

    def outcomes(
        self,
        *,
        environment_state: Tossing3DSearchState,
        state: Tossing3DBeliefState,
        action: GroundSkill,
    ) -> tuple[tuple[float, Tossing3DBeliefState, frozenset[GroundAtom]], ...]:
        """The skill's own outcome branches; under the always-ask wrapper a skill with
        continuous parameters is then followed, in every branch, by the human's reset:
        the branch keeps its belief update and lands in the start state, and the
        reset's price joins the plan cost. A toss is then worth "learn, then be reset
        and practise again" rather than a dead end."""
        branches = transition_outcomes(
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
        reset = self.forced_start_reset
        if reset is None or action.skill.param_dim == 0:
            return branches
        # One step cost for skill and reset together, added to the parent's cost once:
        # the searches carry `summed_cost + step cost` and assert it equals the
        # branch's own accumulated cost, which two separate float additions can miss.
        return tuple(
            (
                probability,
                next_state.model_copy(
                    update={
                        "accumulated_cost": state.accumulated_cost
                        + (next_state.accumulated_cost + reset.cost - state.accumulated_cost)
                    }
                ),
                reset.start_atoms,
            )
            for probability, next_state, _atoms in branches
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
