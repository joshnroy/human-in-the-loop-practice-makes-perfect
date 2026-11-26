"""Serving's skill provider: the chained toy's (`TwoRoutesSkillProvider` under
`chained_route_b`) over the 3D scene. The same skills, predicates, failure effects and
rewards, renamed; the continuous parameters are the split controllers' own."""

from typing import ClassVar

import numpy as np

from hitl_pmp.core.method.skill_provider import OraclePolicyProvider, SkillProvider
from hitl_pmp.core.method.types import GroundSkill, LabeledAction, Skill
from hitl_pmp.core.problem.environment.types import Action, Object, State, Type
from hitl_pmp.core.problem.tasks.types import Goal, Predicate

from .objects import RackServing3DObjects as Objects
from .skills import SWEEP_BOUNDS
from .toy_environment import RackServingToyEnvironment
from .toy_interface import (
    GATHER,
    HUMAN_RESET,
    SWEEP3,
    SWEEP5,
    TOY_SKILL_IDS,
    ServingToyPredicates,
    ServingToySkills,
)


class RackServingSkillProvider(SkillProvider):
    env: RackServingToyEnvironment
    offer_human_reset: bool = True

    def skills(self) -> tuple[Skill, ...]:
        return ServingToySkills.chained()

    def human_skill(self) -> Skill:
        return ServingToySkills.human_reset()

    def predicates(self) -> tuple[Predicate, ...]:
        return ServingToyPredicates.all()

    def types(self) -> tuple[Type, ...]:
        return (Objects.robot_type,)

    def objects(self) -> tuple[Object, ...]:
        return (Objects.robot,)

    def ground(self, *, name: str) -> GroundSkill:
        skill = next(s for s in (*self.skills(), self.human_skill()) if s.name == name)
        return GroundSkill(skill=skill, objects=(Objects.robot,))

    def bounds(self, *, name: str) -> tuple[tuple[float, float], ...]:
        gather_and_sweep = self.env.config.gather_and_sweep_bounds()
        return {
            SWEEP3: SWEEP_BOUNDS,
            GATHER: gather_and_sweep[:2],
            SWEEP5: gather_and_sweep[2:],
        }.get(name, ())

    def sample_params(self, *, ground_skill: GroundSkill, rng: np.random.Generator) -> np.ndarray:
        bounds = self.bounds(name=ground_skill.skill.name)
        return np.array([rng.uniform(lower, upper) for lower, upper in bounds])

    def compute_action(
        self, *, ground_skill: GroundSkill, params: np.ndarray, state: State
    ) -> Action:
        del state
        return RackServingActions.encode(
            name=ground_skill.skill.name,
            params=params,
            bounds=self.bounds(name=ground_skill.skill.name),
        )

    def hand_selected_feature_transform(
        self, *, ground_skill: GroundSkill, state: State, params: np.ndarray
    ) -> list[float] | None:
        """The learned sampler reads the parameters alone, as the launch line's
        `--sampler-features parameters` has it: the toy's state is one robot."""
        del state
        if ground_skill.skill.param_dim == 0:
            return None
        return [1.0, *(float(value) for value in params)]

    def failure_kept_predicates(self) -> dict[str, frozenset[str]]:
        return {
            SWEEP3: frozenset({"NeedsReset", "AfterSweep3"}),
            GATHER: frozenset({"NeedsReset", "AfterGather"}),
            SWEEP5: frozenset({"AfterSweep5"}),
        }

    def goal_rewards(self) -> dict[str, float]:
        return {"ServedThree": self.env.sweep3_reward, "ServedFive": self.env.sweep5_reward}

    def human_cube_bin_reset_skill(self) -> GroundSkill | None:
        if not self.offer_human_reset or self.env.config.mode == "deployment":
            return None
        return self.ground(name=HUMAN_RESET)

    def movables_reset_destination(self, *, ground_skill: GroundSkill) -> str | None:
        return ground_skill.skill.name


class RackServingActions:
    """[toy skill ID, 0, s, delta, d]: the rack's action layout, so its recording,
    replay and interruption paths read it unchanged."""

    @staticmethod
    def encode(*, name: str, params: np.ndarray, bounds: tuple[tuple[float, float], ...]) -> Action:
        if name not in TOY_SKILL_IDS:
            raise ValueError(f"not a robot skill of the toy: {name}")
        params = np.asarray(params, dtype=float)
        if params.shape != (len(bounds),) or not np.isfinite(params).all():
            raise ValueError(f"{name} requires {len(bounds)} finite parameters")
        if any(not low <= v <= high for v, (low, high) in zip(params, bounds, strict=True)):
            raise ValueError(f"{name} parameters are outside the declared bounds")
        continuous = np.zeros(3)
        if name == GATHER:
            continuous[0:2] = params
        elif name in (SWEEP3, SWEEP5):
            continuous[2] = params[0]
        return np.array([TOY_SKILL_IDS[name], 0.0, *continuous], dtype=float)


class RackServingOracle(OraclePolicyProvider):
    """sweep3 at its nominal d, once: an episode is one attempt (the toy's oracle
    takes one route too)."""

    env: RackServingToyEnvironment
    # The driven-blade grid's best single setting (2026-11-03): gather's (s, delta)
    # and either sweep's d. A starting policy, not a calibrated success claim.
    NOMINAL_GATHER: ClassVar[tuple[float, float]] = (0.07, 0.030)
    NOMINAL_D: ClassVar[float] = 0.10

    def get_labeled_action(self, *, state: State, goal: Goal) -> LabeledAction:
        del goal
        if ("FullRack", (Objects.robot.name,)) not in getattr(state, "abstract_atoms", ()):
            return LabeledAction(action=self.env.noop_action(), label="no-op")
        return LabeledAction(
            action=RackServingActions.encode(
                name=SWEEP3, params=np.array([self.NOMINAL_D]), bounds=SWEEP_BOUNDS
            ),
            label=SWEEP3,
        )
