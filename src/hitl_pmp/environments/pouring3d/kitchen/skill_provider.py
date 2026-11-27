"""Ordinary skills and samplers; no kitchen-specific learning algorithm."""

from functools import lru_cache

import numpy as np
from pydantic import model_validator

from hitl_pmp.core.method.types import GroundSkill, LabeledAction, Skill
from hitl_pmp.core.problem.environment.types import Action, State, Type
from hitl_pmp.core.problem.tasks.types import Goal, Predicate
from hitl_pmp.environments.pouring3d.practice.skill_provider import (
    FrozenPourOracle,
    FrozenPourSkillProvider,
)
from hitl_pmp.environments.pouring3d.practice.skills import SKILL_NAMES

from .environment import HUMAN_RESET, KitchenEnvironment
from .predicates import KitchenPredicates as P
from .skills import KitchenSkills
from .types import ClosedLoopPourParameters
from .types import KitchenObjects as O


class KitchenSkillProvider(FrozenPourSkillProvider):
    env: KitchenEnvironment
    # The closed-loop Pour learns only its tilt ramp rate (Josh, 2026-11-04); its approach,
    # ceiling, wiggle and return are fixed on `KitchenController`.
    ramp_rate_bounds: tuple[float, float] = ClosedLoopPourParameters.LIMITS

    @model_validator(mode="after")
    def validate_ramp_rate_bounds(self) -> "KitchenSkillProvider":
        stale = {"tilt_bounds", "amplitude_bounds", "duration_bounds"} & self.model_fields_set
        if stale:
            raise ValueError(
                f"The kitchen Pour learns only its ramp rate; {sorted(stale)} do not apply"
            )
        low, high = ClosedLoopPourParameters.LIMITS
        bounds = self.ramp_rate_bounds
        if not np.isfinite(bounds).all() or not low <= bounds[0] <= bounds[1] <= high:
            raise ValueError(f"Ramp rate bounds must be ordered and inside [{low}, {high}] deg/s")
        return self

    @staticmethod
    @lru_cache(maxsize=1)
    def domain(*, count: int) -> tuple[Skill, ...]:
        """The kitchen's own table (`skills.py`), the human pour-back last."""
        if count != 3:
            raise ValueError("The kitchen has three dumplings")
        return KitchenSkills.build()

    def ground(self, *, name: str, request: int | None = None, shrimp: int = 0) -> GroundSkill:
        """Bind a kitchen skill. A Pour's count is its name; no request object binds."""
        o = self.env.objects()
        if name.startswith("Pour") and request not in (None, int(name.removeprefix("Pour"))):
            raise ValueError(f"{name} has fixed count {name.removeprefix('Pour')}")
        if not 0 <= shrimp < o.count:
            raise ValueError("Invalid discrete skill argument")
        skill = next(s for s in self.domain(count=o.count) if s.name == name)
        bindings = {
            "r": o.robot,
            "p": o.pitcher,
            "t": o.tray,
            "s": o.shrimp()[shrimp],
            **{f"s{i}": obj for i, obj in enumerate(o.shrimp())},
        }
        return GroundSkill(skill=skill, objects=tuple(bindings[v.name] for v in skill.parameters))

    def encode(self, *, ground_skill: GroundSkill, params: np.ndarray) -> Action:
        name = ground_skill.skill.name
        if not name.startswith("Pour"):
            return super().encode(ground_skill=ground_skill, params=params)
        if params.shape != (ground_skill.skill.param_dim,) or not np.isfinite(params).all():
            raise ValueError("Parameter dimension or values invalid")
        return np.r_[
            SKILL_NAMES.index("Pour"),
            0,
            int(name.removeprefix("Pour")),
            self.native_pour_parameters(params=params),
        ]

    def compute_action(
        self, *, ground_skill: GroundSkill, params: np.ndarray, state: State
    ) -> Action:
        # A Pour dispatches its own fixed count; every other skill carries the observed
        # count as metadata only, as the legacy benchmark's non-count skills do.
        result = super().compute_action(ground_skill=ground_skill, params=params, state=state)
        if ground_skill.skill.name.startswith("Pour"):
            result[2] = int(ground_skill.skill.name.removeprefix("Pour"))
        return result

    def sample_params(self, *, ground_skill: GroundSkill, rng: np.random.Generator) -> np.ndarray:
        if not ground_skill.skill.name.startswith("Pour"):
            return np.zeros(0)
        return np.array([rng.uniform(*self.ramp_rate_bounds)])

    def native_pour_parameters(self, *, params: np.ndarray) -> np.ndarray:
        # Slots 4-5 carried the open-loop wiggle and duration; the kitchen fixes both.
        return np.r_[params, 0.0, 0.0]

    def predicates(self) -> tuple[Predicate, ...]:
        return P.all()

    def types(self) -> tuple[Type, ...]:
        return O.types()

    def human_cube_bin_reset_skill(self) -> GroundSkill | None:
        if self.env.deployment or not self.offer_human_reset:
            return None
        return self.ground(name=HUMAN_RESET)


class KitchenPourOracle(FrozenPourOracle):
    """The fixed-recipe reference with the closed-loop Pour's canonical ramp rate."""

    env: KitchenEnvironment
    # 1 deg/s delivered Pour1 and Pour2 exactly on seeds 0-3 in the 2026-11-04 prototype.
    parameters: tuple[float, ...] = (1.0,)

    def provider(self) -> FrozenPourSkillProvider:
        return KitchenSkillProvider(env=self.env)

    def get_labeled_action(self, *, state: State, goal: Goal) -> LabeledAction:
        """The legacy recipe on the kitchen's table: with `ReadyToPour` gone, the jug is
        picked when the hand is empty, the jug is home and every dumpling is in it and
        settled, which is what `ready()` measured once the skillet is welded."""
        del goal
        atoms: frozenset[tuple[str, tuple[str, ...]]] = getattr(
            state, "abstract_atoms", frozenset()
        )
        shrimp = self.env.objects().shrimp()
        all_in = all(("InPitcher", (o.name, "pitcher")) in atoms for o in shrimp)
        if ("HoldingPitcher", ("robot", "pitcher")) in atoms:
            request = int(state.get(obj=O.scene, feature_name="requested_count"))
            name = f"Pour{request}" if all_in else "PlacePitcherAtHome"
        elif (
            all_in
            and {("HandEmpty", ("robot",)), ("PitcherHome", ("pitcher",))} <= atoms
            and all(("Settled", (o.name,)) in atoms for o in shrimp)
        ):
            name = "PickPitcher"
        else:
            return LabeledAction(action=self.env.noop_action(), label="Wait")
        provider = self.provider()
        return LabeledAction(
            action=provider.compute_action(
                ground_skill=provider.ground(name=name),
                params=np.array(self.parameters) if name.startswith("Pour") else np.zeros(0),
                state=state,
            ),
            label=name,
        )
