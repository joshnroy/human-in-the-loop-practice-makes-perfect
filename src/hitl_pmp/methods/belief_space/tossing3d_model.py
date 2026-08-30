"""Finite situated practice POMDP for the canonical Tossing3D task."""

from __future__ import annotations

import math
from collections.abc import Iterable
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .expectimax import ChanceOutcome

PICK_SKILL = "PickCube"
TOSS_SKILL = "MoveToTossLocationAndToss"
OPEN_GRIPPER_SKILL = "OpenGripper"
RESET_SKILL = "ask_for_reset_cube_bin_only"


class Tossing3DEnvironmentState(Enum):
    """Canonical symbolic phases that determine applicable environment skills."""

    READY = "ready"
    HOLDING = "holding"
    UNREACHABLE_HOLDING = "unreachable_holding"
    GRIPPER_CLOSED = "gripper_closed"
    CLOSED_STRANDED = "closed_stranded"
    STRANDED = "stranded"
    SOLVED = "solved"
    CLOSED_SOLVED = "closed_solved"


class SkillHypothesis(BaseModel):
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
    """Normalized posterior over the learnable toss controller."""

    model_config = ConfigDict(frozen=True)

    hypotheses: tuple[WeightedHypothesis, ...]

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


class Tossing3DBeliefState(BaseModel):
    """Physical state, latent-controller posterior, and paid practice cost."""

    model_config = ConfigDict(frozen=True)

    environment_state: Tossing3DEnvironmentState
    toss_belief: SkillBelief
    pending_training_examples: int = Field(default=0, ge=0)
    accumulated_cost: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)

    def transition(
        self,
        *,
        environment_state: Tossing3DEnvironmentState,
        added_cost: float,
        toss_belief: SkillBelief | None = None,
        added_training_examples: int = 0,
    ) -> Tossing3DBeliefState:
        return Tossing3DBeliefState(
            environment_state=environment_state,
            toss_belief=self.toss_belief if toss_belief is None else toss_belief,
            pending_training_examples=self.pending_training_examples + added_training_examples,
            accumulated_cost=self.accumulated_cost + added_cost,
        )

    def after_refit(self) -> Tossing3DBeliefState:
        return self.model_copy(
            update={
                "toss_belief": self.toss_belief.after_refit(
                    training_examples=self.pending_training_examples
                ),
                "pending_training_examples": 0,
            }
        )


class Tossing3DPracticeModel(BaseModel):
    """Applicable actions, situated transitions, costs, and deployment value."""

    model_config = ConfigDict(frozen=True)

    practice_cost: float = Field(default=0.01, ge=0.0, allow_inf_nan=False)
    pick_competence: float = Field(default=0.5, ge=0.0, le=1.0)
    random_toss_competence: float = Field(default=0.25, ge=0.0, le=1.0)
    exploration_epsilon: float = Field(default=0.5, ge=0.0, le=1.0)
    pick_cost: float = Field(default=1.0, ge=0.0, allow_inf_nan=False)
    toss_cost: float = Field(default=1.0, ge=0.0, allow_inf_nan=False)
    open_gripper_cost: float = Field(default=1.0, ge=0.0, allow_inf_nan=False)
    reset_cost: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    deployment_horizon: int = Field(default=4, ge=0)

    @property
    def required_skills(self) -> tuple[str, ...]:
        return (PICK_SKILL, TOSS_SKILL, OPEN_GRIPPER_SKILL)

    def actions(self, state: Tossing3DBeliefState) -> Iterable[str]:  # noqa: PLR0917
        by_state = {
            Tossing3DEnvironmentState.READY: (PICK_SKILL,),
            Tossing3DEnvironmentState.HOLDING: (TOSS_SKILL,),
            Tossing3DEnvironmentState.UNREACHABLE_HOLDING: (OPEN_GRIPPER_SKILL,),
            Tossing3DEnvironmentState.GRIPPER_CLOSED: (OPEN_GRIPPER_SKILL,),
            Tossing3DEnvironmentState.CLOSED_STRANDED: (OPEN_GRIPPER_SKILL,),
            Tossing3DEnvironmentState.STRANDED: (),
            Tossing3DEnvironmentState.SOLVED: (),
            Tossing3DEnvironmentState.CLOSED_SOLVED: (OPEN_GRIPPER_SKILL,),
        }
        actions = by_state[state.environment_state]
        if self.reset_cost is not None and state.environment_state in {
            Tossing3DEnvironmentState.CLOSED_STRANDED,
            Tossing3DEnvironmentState.STRANDED,
            Tossing3DEnvironmentState.SOLVED,
            Tossing3DEnvironmentState.CLOSED_SOLVED,
        }:
            return (*actions, RESET_SKILL)
        return actions

    def stop_value(self, state: Tossing3DBeliefState) -> float:  # noqa: PLR0917
        projected = state.toss_belief.after_refit(training_examples=state.pending_training_examples)
        deployment_value = self._evaluate_deployment_policy(
            environment_state=Tossing3DEnvironmentState.READY,
            toss_competence=projected.mean_competence,
            horizon=self.deployment_horizon,
        )
        return deployment_value - self.practice_cost * state.accumulated_cost

    def _evaluate_deployment_policy(
        self,
        *,
        environment_state: Tossing3DEnvironmentState,
        toss_competence: float,
        horizon: int,
    ) -> float:
        """Solve the canonical deployment MDP; human reset is unavailable at test."""
        if environment_state is Tossing3DEnvironmentState.SOLVED:
            return 1.0
        if horizon == 0:
            return 0.0
        if environment_state is Tossing3DEnvironmentState.READY:
            success_value = self._evaluate_deployment_policy(
                environment_state=Tossing3DEnvironmentState.HOLDING,
                toss_competence=toss_competence,
                horizon=horizon - 1,
            )
            failure_value = self._evaluate_deployment_policy(
                environment_state=Tossing3DEnvironmentState.GRIPPER_CLOSED,
                toss_competence=toss_competence,
                horizon=horizon - 1,
            )
            return (
                self.pick_competence * success_value + (1.0 - self.pick_competence) * failure_value
            )
        if environment_state is Tossing3DEnvironmentState.GRIPPER_CLOSED:
            return self._evaluate_deployment_policy(
                environment_state=Tossing3DEnvironmentState.READY,
                toss_competence=toss_competence,
                horizon=horizon - 1,
            )
        if environment_state is Tossing3DEnvironmentState.HOLDING:
            return toss_competence * self._evaluate_deployment_policy(
                environment_state=Tossing3DEnvironmentState.SOLVED,
                toss_competence=toss_competence,
                horizon=horizon - 1,
            )
        return 0.0

    def outcomes(  # noqa: PLR0917
        self, state: Tossing3DBeliefState, action: str
    ) -> tuple[ChanceOutcome[Tossing3DBeliefState], ...]:
        if action not in set(self.actions(state)):
            raise ValueError(f"{action!r} is not applicable in {state.environment_state.value}")
        if action == PICK_SKILL:
            return self._binary_outcomes(
                state=state,
                probability=self.pick_competence,
                success_state=Tossing3DEnvironmentState.HOLDING,
                failure_state=Tossing3DEnvironmentState.GRIPPER_CLOSED,
                cost=self.pick_cost,
            )
        if action == OPEN_GRIPPER_SKILL:
            opened = {
                Tossing3DEnvironmentState.GRIPPER_CLOSED: Tossing3DEnvironmentState.READY,
                Tossing3DEnvironmentState.CLOSED_STRANDED: Tossing3DEnvironmentState.STRANDED,
                Tossing3DEnvironmentState.UNREACHABLE_HOLDING: Tossing3DEnvironmentState.STRANDED,
                Tossing3DEnvironmentState.CLOSED_SOLVED: Tossing3DEnvironmentState.SOLVED,
            }[state.environment_state]
            return self._deterministic(
                state=state,
                environment_state=opened,
                cost=self.open_gripper_cost,
            )
        if action == RESET_SKILL:
            assert self.reset_cost is not None
            reset_state = (
                Tossing3DEnvironmentState.GRIPPER_CLOSED
                if state.environment_state
                in {
                    Tossing3DEnvironmentState.CLOSED_STRANDED,
                    Tossing3DEnvironmentState.CLOSED_SOLVED,
                }
                else Tossing3DEnvironmentState.READY
            )
            return self._deterministic(
                state=state,
                environment_state=reset_state,
                cost=self.reset_cost,
            )
        assert action == TOSS_SKILL
        return self._toss_outcomes(state=state)

    @staticmethod
    def _deterministic(
        *, state: Tossing3DBeliefState, environment_state: Tossing3DEnvironmentState, cost: float
    ) -> tuple[ChanceOutcome[Tossing3DBeliefState], ...]:
        return (
            ChanceOutcome(
                probability=1.0,
                next_state=state.transition(environment_state=environment_state, added_cost=cost),
            ),
        )

    @staticmethod
    def _binary_outcomes(
        *,
        state: Tossing3DBeliefState,
        probability: float,
        success_state: Tossing3DEnvironmentState,
        failure_state: Tossing3DEnvironmentState,
        cost: float,
    ) -> tuple[ChanceOutcome[Tossing3DBeliefState], ...]:
        return tuple(
            ChanceOutcome(
                probability=branch_probability,
                next_state=state.transition(
                    environment_state=next_environment_state, added_cost=cost
                ),
            )
            for branch_probability, next_environment_state in (
                (probability, success_state),
                (1.0 - probability, failure_state),
            )
            if branch_probability > 0.0
        )

    def _toss_outcomes(
        self, *, state: Tossing3DBeliefState
    ) -> tuple[ChanceOutcome[Tossing3DBeliefState], ...]:
        branches: list[ChanceOutcome[Tossing3DBeliefState]] = []
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
                branches.append(
                    ChanceOutcome(
                        probability=probability,
                        next_state=state.transition(
                            environment_state=(
                                Tossing3DEnvironmentState.SOLVED
                                if success
                                else Tossing3DEnvironmentState.STRANDED
                            ),
                            added_cost=self.toss_cost,
                            toss_belief=belief,
                            added_training_examples=1,
                        ),
                    )
                )
        return tuple(branches)

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


def make_default_tossing3d_belief(
    *, environment_state: Tossing3DEnvironmentState = Tossing3DEnvironmentState.READY
) -> Tossing3DBeliefState:
    """Broad prior over the only parameterized and learnable Tossing3D skill."""
    hypotheses = tuple(
        SkillHypothesis(competence=competence, learning_rate=rate)
        for competence in (0.25, 0.5, 0.75)
        for rate in (0.0, 0.1)
    )
    probability = 1.0 / len(hypotheses)
    return Tossing3DBeliefState(
        environment_state=environment_state,
        toss_belief=SkillBelief(
            hypotheses=tuple(
                WeightedHypothesis(hypothesis=hypothesis, probability=probability)
                for hypothesis in hypotheses
            )
        ),
    )
