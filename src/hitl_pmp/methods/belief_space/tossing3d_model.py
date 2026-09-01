"""Finite situated practice POMDP for the canonical Tossing3D task."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, computed_field, model_validator

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.problem.tasks.types import GroundAtom

from .types import BeliefState, EnvironmentState, POMDPAction, Theta

PICK_SKILL = "PickCube"
TOSS_SKILL = "MoveToTossLocationAndToss"
OPEN_GRIPPER_SKILL = "OpenGripper"
RESET_SKILL = "ask_for_reset_cube_bin_only"


class SkillHypothesis(Theta):
    model_config = ConfigDict(frozen=True)

    competence: float = Field(ge=0.0, le=1.0)
    learning_rate: float = Field(ge=0.0, le=1.0)

    def after_training_examples(self, *, count: int) -> SkillHypothesis:
        if count < 0:
            raise ValueError("training-example count must be nonnegative")
        competence = 1.0 - (1.0 - self.competence) * (1.0 - self.learning_rate) ** count
        return SkillHypothesis(competence=competence, learning_rate=self.learning_rate)


class WeightedHypothesis(BaseModel):
    model_config = ConfigDict(frozen=True)

    hypothesis: SkillHypothesis
    probability: float = Field(gt=0.0, le=1.0)


class SkillBelief(BaseModel):
    """Posterior over a skill's competence and improvement rate."""

    model_config = ConfigDict(frozen=True)

    hypotheses: tuple[WeightedHypothesis, ...]

    @staticmethod
    def prior() -> SkillBelief:
        return SkillBelief(
            hypotheses=tuple(
                WeightedHypothesis(
                    hypothesis=SkillHypothesis(competence=p, learning_rate=rate),
                    probability=1.0 / 10,
                )
                for p in (0.0, 0.25, 0.5, 0.75, 1.0)
                for rate in (0.0, 0.1)
            )
        )

    @model_validator(mode="after")
    def _valid_distribution(self) -> SkillBelief:
        if not self.hypotheses:
            raise ValueError("a skill belief needs at least one hypothesis")
        total = sum(item.probability for item in self.hypotheses)
        if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"hypothesis probabilities sum to {total}, not 1")
        return self

    @property
    def mean_competence(self) -> float:
        return sum(item.probability * item.hypothesis.competence for item in self.hypotheses)

    @property
    def mean_learning_rate(self) -> float:
        return sum(item.probability * item.hypothesis.learning_rate for item in self.hypotheses)

    def condition(self, *, success: bool) -> SkillBelief:
        """Condition on a greedy-policy outcome without pretending a refit occurred."""
        weighted: list[tuple[SkillHypothesis, float]] = []
        for item in self.hypotheses:
            likelihood = item.hypothesis.competence if success else 1.0 - item.hypothesis.competence
            mass = item.probability * likelihood
            if mass > 0.0:
                weighted.append((item.hypothesis, mass))
        normalizer = sum(mass for _hypothesis, mass in weighted)
        if normalizer <= 0.0:
            raise ValueError(f"observation success={success} has zero probability")
        return SkillBelief(
            hypotheses=tuple(
                WeightedHypothesis(hypothesis=hypothesis, probability=mass / normalizer)
                for hypothesis, mass in weighted
            )
        )

    def after_refit(self, *, training_examples: int) -> SkillBelief:
        return SkillBelief(
            hypotheses=tuple(
                WeightedHypothesis(
                    hypothesis=item.hypothesis.after_training_examples(count=training_examples),
                    probability=item.probability,
                )
                for item in self.hypotheses
            )
        )


class Tossing3DBeliefState(BeliefState):
    """Latent-controller posterior, pending examples, and paid practice cost."""

    model_config = ConfigDict(frozen=True)

    toss_belief: SkillBelief
    pick_belief: SkillBelief = Field(default_factory=SkillBelief.prior)
    open_gripper_belief: SkillBelief = Field(default_factory=SkillBelief.prior)
    pending_pick_examples: int = Field(default=0, ge=0)
    pending_open_gripper_examples: int = Field(default=0, ge=0)
    pending_training_examples: int = Field(default=0, ge=0)
    accumulated_cost: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)

    def transition(
        self,
        *,
        added_cost: float,
        toss_belief: SkillBelief | None = None,
        added_training_examples: int = 0,
    ) -> Tossing3DBeliefState:
        return self.model_copy(
            update={
                "toss_belief": self.toss_belief if toss_belief is None else toss_belief,
                "pending_training_examples": self.pending_training_examples
                + added_training_examples,
                "accumulated_cost": self.accumulated_cost + added_cost,
            }
        )

    def after_refit(self) -> Tossing3DBeliefState:
        return self.model_copy(
            update={
                "toss_belief": self.toss_belief.after_refit(
                    training_examples=self.pending_training_examples
                ),
                "pending_training_examples": 0,
                "pick_belief": self.pick_belief.after_refit(
                    training_examples=self.pending_pick_examples
                ),
                "open_gripper_belief": self.open_gripper_belief.after_refit(
                    training_examples=self.pending_open_gripper_examples
                ),
                "pending_pick_examples": 0,
                "pending_open_gripper_examples": 0,
            }
        )


class Tossing3DSearchState(EnvironmentState):
    """EES symbolic state paired with the posterior state used by the search."""

    state: Tossing3DBeliefState
    true_atoms: frozenset[GroundAtom] = Field(exclude=True)

    @computed_field
    def atoms(self) -> tuple[str, ...]:
        """Serializable identity for traces; true_atoms remains the executable form."""
        return tuple(sorted(str(atom) for atom in self.true_atoms))


class Tossing3DAction(POMDPAction):
    name: str
    ground_skill: GroundSkill = Field(exclude=True)


class Tossing3DOutcome(BaseModel):
    probability: float
    next_state: Tossing3DBeliefState
    next_true_atoms: frozenset[GroundAtom]


class Tossing3DTheta(Theta):
    pick: SkillHypothesis
    toss: SkillHypothesis
    open_gripper: SkillHypothesis


class Tossing3DPracticeModel(BaseModel):
    """Applicable actions, situated transitions, costs, and deployment value."""

    model_config = ConfigDict(frozen=True)

    seed: int = 0
    ground_skills: tuple[GroundSkill, ...] = Field(default=(), exclude=True)
    _rng: np.random.Generator = PrivateAttr()
    _atom_indexes: dict[GroundAtom, int] = PrivateAttr(default_factory=dict)
    _precondition_masks: tuple[int, ...] = PrivateAttr(default=())
    _actions: tuple[Tossing3DAction, ...] = PrivateAttr(default=())
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
        self._actions = tuple(
            Tossing3DAction(name=skill.skill.name, ground_skill=skill)
            for skill in self.ground_skills
        )
        self._effects = {
            skill: (skill.add_effects, skill.delete_effects, skill.ignore_effects)
            for skill in self.ground_skills
        }

    def sample_theta_from_belief(self, *, belief_state: BeliefState) -> Theta:
        return self.sample_thetas_from_belief(belief_state=belief_state, num_samples=1)[0]

    def sample_thetas_from_belief(
        self, *, belief_state: BeliefState, num_samples: int
    ) -> list[Theta]:
        assert isinstance(belief_state, Tossing3DBeliefState)
        projected = belief_state.after_refit()
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

    def evaluate_policy(self, *, sampled_theta: Theta) -> float:
        assert isinstance(sampled_theta, Tossing3DTheta)
        return self._evaluate_deployment_policy(
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

    def get_valid_actions(self, *, environment_state: EnvironmentState) -> list[POMDPAction]:
        assert isinstance(environment_state, Tossing3DSearchState)
        state = environment_state.state
        costs = {
            PICK_SKILL: self.pick_cost,
            TOSS_SKILL: self.toss_cost,
            OPEN_GRIPPER_SKILL: self.open_gripper_cost,
            RESET_SKILL: self.reset_cost,
        }
        state_mask = self._atoms_mask(atoms=environment_state.true_atoms)
        return [
            action
            for index, (ground_skill, action) in enumerate(
                zip(self.ground_skills, self._actions, strict=True)
            )
            if self._precondition_masks[index] & state_mask == self._precondition_masks[index]
            for cost in [costs[ground_skill.skill.name]]
            if self.hard_budget is None
            or (cost is not None and state.accumulated_cost + cost <= self.hard_budget)
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
            (
                item.hypothesis.competence,
                item.hypothesis.learning_rate,
                item.probability,
            )
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
        environment_state: EnvironmentState,
        summed_cost: float,
        belief_state: BeliefState,
        horizon: int,
    ) -> object:
        assert isinstance(environment_state, Tossing3DSearchState)
        assert isinstance(belief_state, Tossing3DBeliefState)
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

    def sample_next_states(
        self,
        *,
        environment_state: EnvironmentState,
        practice_action: POMDPAction,
        belief_state: BeliefState,
    ) -> list[tuple[EnvironmentState, float]]:
        assert isinstance(environment_state, Tossing3DSearchState)
        assert isinstance(practice_action, Tossing3DAction)
        assert isinstance(belief_state, Tossing3DBeliefState)
        # Enumerate the finite model, merging observationally identical branches.
        return list(
            dict.fromkeys(
                (
                    Tossing3DSearchState(
                        state=outcome.next_state, true_atoms=outcome.next_true_atoms
                    ),
                    outcome.next_state.accumulated_cost - belief_state.accumulated_cost,
                )
                for outcome in self.outcomes(
                    environment_state=environment_state,
                    state=belief_state,
                    action=practice_action,
                )
            )
        )

    def transition_outcomes(
        self,
        *,
        environment_state: EnvironmentState,
        practice_action: POMDPAction,
        belief_state: BeliefState,
    ) -> list[tuple[EnvironmentState, float, float]]:
        assert isinstance(environment_state, Tossing3DSearchState)
        assert isinstance(practice_action, Tossing3DAction)
        assert isinstance(belief_state, Tossing3DBeliefState)
        merged: dict[tuple[Tossing3DSearchState, float], float] = {}
        for outcome in self.outcomes(
            environment_state=environment_state,
            state=belief_state,
            action=practice_action,
        ):
            next_environment = Tossing3DSearchState(
                state=outcome.next_state, true_atoms=outcome.next_true_atoms
            )
            cost = outcome.next_state.accumulated_cost - belief_state.accumulated_cost
            key = (next_environment, cost)
            merged[key] = merged.get(key, 0.0) + outcome.probability
        return [(*key, probability) for key, probability in merged.items()]

    def update_belief_state(
        self,
        *,
        belief_state: BeliefState,
        environment_state: EnvironmentState,
        potential_next_environment_state: EnvironmentState,
        practice_action: POMDPAction,
    ) -> BeliefState:
        assert isinstance(potential_next_environment_state, Tossing3DSearchState)
        return potential_next_environment_state.state

    def transition_probability(
        self,
        *,
        potential_next_environment_state: EnvironmentState,
        sampled_cost: float,
        environment_state: EnvironmentState,
        practice_action: POMDPAction,
        belief_state: BeliefState,
    ) -> float:
        assert isinstance(potential_next_environment_state, Tossing3DSearchState)
        assert isinstance(environment_state, Tossing3DSearchState)
        assert isinstance(practice_action, Tossing3DAction)
        assert isinstance(belief_state, Tossing3DBeliefState)
        total_probability = 0.0
        for outcome in self.outcomes(
            environment_state=environment_state,
            state=belief_state,
            action=practice_action,
        ):
            if (
                outcome.next_state == potential_next_environment_state.state
                and outcome.next_true_atoms == potential_next_environment_state.true_atoms
                and outcome.next_state.accumulated_cost - belief_state.accumulated_cost
                == sampled_cost
            ):
                total_probability += outcome.probability
        return total_probability

    practice_cost: float = Field(default=0.01, ge=0.0, allow_inf_nan=False)
    random_toss_competence: float = Field(default=0.25, ge=0.0, le=1.0)
    exploration_epsilon: float = Field(default=0.5, ge=0.0, le=1.0)
    pick_cost: float = Field(default=1.0, ge=0.0, allow_inf_nan=False)
    toss_cost: float = Field(default=1.0, ge=0.0, allow_inf_nan=False)
    open_gripper_cost: float = Field(default=1.0, ge=0.0, allow_inf_nan=False)
    reset_cost: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    hard_budget: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    deployment_horizon: int = Field(default=4, ge=0)

    def _evaluate_deployment_policy(
        self,
        *,
        toss_competence: float,
        pick_competence: float,
        open_competence: float,
        horizon: int,
    ) -> float:
        """Solve the canonical deployment MDP; human reset is unavailable at test."""
        ready_value = holding_value = closed_gripper_value = 0.0
        for _ in range(horizon):
            previous_ready = ready_value
            previous_holding = holding_value
            previous_closed_gripper = closed_gripper_value
            holding_value = toss_competence
            ready_value = (
                pick_competence * previous_holding
                + (1.0 - pick_competence) * previous_closed_gripper
            )
            closed_gripper_value = (
                open_competence * previous_ready + (1.0 - open_competence) * previous_closed_gripper
            )
        return ready_value

    def outcomes(
        self,
        *,
        environment_state: Tossing3DSearchState,
        state: Tossing3DBeliefState,
        action: Tossing3DAction,
    ) -> tuple[Tossing3DOutcome, ...]:
        ground_skill = action.ground_skill
        assert ground_skill in self.ground_skills
        assert ground_skill.preconditions <= environment_state.true_atoms
        if action.name == PICK_SKILL:
            return self._binary_outcomes(
                state=state,
                true_atoms=environment_state.true_atoms,
                ground_skill=ground_skill,
                probability=state.pick_belief.mean_competence,
                skill_name=PICK_SKILL,
                cost=self.pick_cost,
            )
        if action.name == OPEN_GRIPPER_SKILL:
            return self._binary_outcomes(
                state=state,
                true_atoms=environment_state.true_atoms,
                ground_skill=ground_skill,
                probability=state.open_gripper_belief.mean_competence,
                skill_name=OPEN_GRIPPER_SKILL,
                cost=self.open_gripper_cost,
            )
        if action.name == RESET_SKILL:
            assert self.reset_cost is not None
            return self._deterministic(
                state=state,
                true_atoms=environment_state.true_atoms,
                ground_skill=ground_skill,
                cost=self.reset_cost,
            )
        assert action.name == TOSS_SKILL
        return self._toss_outcomes(
            state=state,
            true_atoms=environment_state.true_atoms,
            ground_skill=ground_skill,
        )

    def _deterministic(
        self,
        *,
        state: Tossing3DBeliefState,
        true_atoms: frozenset[GroundAtom],
        ground_skill: GroundSkill,
        cost: float,
    ) -> tuple[Tossing3DOutcome, ...]:
        next_true_atoms = self.apply_success_effects(
            true_atoms=true_atoms, ground_skill=ground_skill
        )
        return (
            Tossing3DOutcome(
                probability=1.0,
                next_state=state.transition(
                    added_cost=cost,
                ),
                next_true_atoms=next_true_atoms,
            ),
        )

    def _binary_outcomes(
        self,
        *,
        state: Tossing3DBeliefState,
        true_atoms: frozenset[GroundAtom],
        ground_skill: GroundSkill,
        probability: float,
        skill_name: str,
        cost: float,
    ) -> tuple[Tossing3DOutcome, ...]:
        outcomes = []
        for success, branch_probability in ((True, probability), (False, 1.0 - probability)):
            if branch_probability <= 0.0:
                continue
            next_true_atoms = (
                self.apply_success_effects(true_atoms=true_atoms, ground_skill=ground_skill)
                if success
                else true_atoms
            )
            outcomes.append(
                Tossing3DOutcome(
                    probability=branch_probability,
                    next_state=Tossing3DPracticeModel.observe_robot_skill(
                        state=state.transition(
                            added_cost=cost,
                        ),
                        skill_name=skill_name,
                        success=success,
                    ),
                    next_true_atoms=next_true_atoms,
                )
            )
        return tuple(outcomes)

    @staticmethod
    def observe_robot_skill(
        *, state: Tossing3DBeliefState, skill_name: str, success: bool
    ) -> Tossing3DBeliefState:
        belief_field, count_field = {
            PICK_SKILL: ("pick_belief", "pending_pick_examples"),
            OPEN_GRIPPER_SKILL: ("open_gripper_belief", "pending_open_gripper_examples"),
        }[skill_name]
        belief = getattr(state, belief_field)
        return state.model_copy(
            update={
                belief_field: belief.condition(success=success),
                count_field: getattr(state, count_field) + 1,
            }
        )

    def _toss_outcomes(
        self,
        *,
        state: Tossing3DBeliefState,
        true_atoms: frozenset[GroundAtom],
        ground_skill: GroundSkill,
    ) -> tuple[Tossing3DOutcome, ...]:
        branches: list[Tossing3DOutcome] = []
        for is_random, choice_probability, success_probability in (
            (False, 1.0 - self.exploration_epsilon, state.toss_belief.mean_competence),
            (True, self.exploration_epsilon, self.random_toss_competence),
        ):
            for success, observation_probability in (
                (True, success_probability),
                (False, 1.0 - success_probability),
            ):
                probability = choice_probability * observation_probability
                if probability <= 0.0:
                    continue
                belief = (
                    state.toss_belief if is_random else state.toss_belief.condition(success=success)
                )
                next_true_atoms = (
                    self.apply_success_effects(true_atoms=true_atoms, ground_skill=ground_skill)
                    if success
                    else true_atoms
                )
                branches.append(
                    Tossing3DOutcome(
                        probability=probability,
                        next_state=state.transition(
                            added_cost=self.toss_cost,
                            toss_belief=belief,
                            added_training_examples=1,
                        ),
                        next_true_atoms=next_true_atoms,
                    )
                )
        return tuple(branches)

    def apply_success_effects(
        self, *, true_atoms: frozenset[GroundAtom], ground_skill: GroundSkill
    ) -> frozenset[GroundAtom]:
        add_effects, delete_effects, ignore_effects = self._effects[ground_skill]
        kept = {
            atom
            for atom in true_atoms
            if atom.predicate not in ignore_effects and atom not in delete_effects
        }
        return frozenset(kept | set(add_effects))

    def observe_toss(
        self,
        *,
        state: Tossing3DBeliefState,
        success: bool,
        was_random_exploration: bool,
    ) -> Tossing3DBeliefState:
        belief = state.toss_belief
        if not was_random_exploration:
            belief = belief.condition(success=success)
        return state.model_copy(update={"toss_belief": belief})

    def record_training_example(self, *, state: Tossing3DBeliefState) -> Tossing3DBeliefState:
        return state.model_copy(
            update={"pending_training_examples": state.pending_training_examples + 1}
        )


def make_default_tossing3d_belief() -> Tossing3DBeliefState:
    """Independent broad priors for all robot skills; human reset is known."""
    return Tossing3DBeliefState(
        toss_belief=SkillBelief.prior(),
    )
