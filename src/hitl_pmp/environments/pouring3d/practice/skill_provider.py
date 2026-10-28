"""Typed parameter sampling and robot/human actions for the generic learner."""

from functools import lru_cache

import numpy as np
from pydantic import Field, model_validator

from hitl_pmp.core.method.skill_provider import OraclePolicyProvider, SkillProvider
from hitl_pmp.core.method.types import GroundSkill, LabeledAction, Skill
from hitl_pmp.core.problem.environment.types import Action, Object, State, Type
from hitl_pmp.core.problem.tasks.types import Goal, Predicate

from .environment import FrozenPourEnvironment
from .predicates import PourPredicates
from .skills import RESET_NAME, SKILL_NAMES, PourSkills
from .types import PourObjects


class FrozenPourSkillProvider(SkillProvider):
    env: FrozenPourEnvironment
    human_reset_practice_cost: float = Field(default=1, ge=0, allow_inf_nan=False)
    offer_human_reset: bool = True
    # Sampling includes deliberately failing low/high-flow regions. These do not
    # depend on the task seed or hidden outcomes, and are frozen per experiment.
    tilt_bounds: tuple[float, float] = (87.0, 95.0)
    amplitude_bounds: tuple[float, float] = (0.0, 0.025)
    duration_bounds: tuple[float, float] = (0.0, 30.0)

    @model_validator(mode="after")
    def validate_bounds(self) -> "FrozenPourSkillProvider":
        for bounds, limits in zip(
            (self.tilt_bounds, self.amplitude_bounds, self.duration_bounds),
            ((60, 130), (0, 0.025), (0, 60)),
            strict=True,
        ):
            if (
                not np.isfinite(bounds).all()
                or not limits[0] <= bounds[0] <= bounds[1] <= limits[1]
            ):
                raise ValueError(
                    "Sampling bounds must be ordered, finite, and inside controller limits"
                )
        return self

    @staticmethod
    @lru_cache(maxsize=2)
    def domain(*, count: int) -> tuple[Skill, ...]:
        return PourSkills.build(count=count)

    def skills(self) -> tuple[Skill, ...]:
        return self.domain(count=self.env.payload_count)[:-1]

    def predicates(self) -> tuple[Predicate, ...]:
        return PourPredicates.all()

    def types(self) -> tuple[Type, ...]:
        return PourObjects.types()

    def objects(self) -> tuple[Object, ...]:
        return self.env.objects().all()

    def sample_params(self, *, ground_skill: GroundSkill, rng: np.random.Generator) -> np.ndarray:
        if not ground_skill.skill.name.startswith("Pour"):
            return np.zeros(0)
        return np.array([
            rng.uniform(*bound)
            for bound in (self.tilt_bounds, self.amplitude_bounds, self.duration_bounds)
        ])

    def ground(self, *, name: str, request: int | None = None, shrimp: int = 0) -> GroundSkill:
        o = self.env.objects()
        if name.startswith("Pour"):
            fixed_count = int(name.removeprefix("Pour"))
            if request is not None and request != fixed_count:
                raise ValueError(f"{name} has fixed count {fixed_count}")
            request = fixed_count
        request = self.env.requested_count if request is None else request
        if not 1 <= request <= o.count or not 0 <= shrimp < o.count:
            raise ValueError("Invalid discrete skill argument")
        skill = next(s for s in self.domain(count=o.count) if s.name == name)
        bindings = {
            "r": o.robot,
            "p": o.pitcher,
            "t": o.tray,
            "k": o.requests()[request - 1],
            "s": o.shrimp()[shrimp],
            **{f"s{i}": obj for i, obj in enumerate(o.shrimp())},
        }
        return GroundSkill(skill=skill, objects=tuple(bindings[v.name] for v in skill.parameters))

    def encode(self, *, ground_skill: GroundSkill, params: np.ndarray) -> Action:
        if params.shape != (ground_skill.skill.param_dim,) or not np.isfinite(params).all():
            raise ValueError("Parameter dimension or values invalid")
        name = ground_skill.skill.name
        native_name = "Pour" if name.startswith("Pour") else name
        if native_name not in SKILL_NAMES:
            raise ValueError("Human reset cannot be encoded as a robot action")
        bindings = dict(
            zip((v.name for v in ground_skill.skill.parameters), ground_skill.objects, strict=True)
        )
        shrimp = self.env.objects().shrimp().index(bindings["s"]) if "s" in bindings else 0
        requested = (
            int(bindings["k"].name.split("_")[-1]) if "k" in bindings else self.env.requested_count
        )
        if native_name == "Pour" and requested != int(name.removeprefix("Pour")):
            raise ValueError(f"{name} requires its fixed count binding")
        return np.r_[
            SKILL_NAMES.index(native_name),
            shrimp,
            requested,
            params if len(params) else np.zeros(3),
        ]

    def compute_action(
        self, *, ground_skill: GroundSkill, params: np.ndarray, state: State
    ) -> Action:
        result = self.encode(ground_skill=ground_skill, params=params)
        # Count-bearing skills use the selected grounding, including in practice.
        # Non-count actions carry observation metadata only (also for evaluation).
        if not any(v.name == "k" for v in ground_skill.skill.parameters):
            result[2] = int(state.get(obj=PourObjects.scene, feature_name="requested_count"))
        return result

    def human_cube_bin_reset_skill(self) -> GroundSkill | None:
        if not self.offer_human_reset:
            return None
        ground = self.ground(name=RESET_NAME)
        return ground.model_copy(
            update={
                "skill": ground.skill.model_copy(
                    update={"practice_cost": self.human_reset_practice_cost}
                )
            }
        )

    def movables_reset_destination(self, *, ground_skill: GroundSkill) -> str | None:
        return ground_skill.skill.name


class FrozenPourOracle(OraclePolicyProvider):
    """Fixed recipe reference; no per-seed search and no guaranteed success claim."""

    env: FrozenPourEnvironment
    parameters: tuple[float, float, float] = (90.5, 0.008, 30.0)

    def get_labeled_action(self, *, state: State, goal: Goal) -> LabeledAction:
        del goal
        provider = FrozenPourSkillProvider(env=self.env)
        atoms: frozenset[tuple[str, tuple[str, ...]]] = getattr(
            state, "abstract_atoms", frozenset()
        )
        held = ("HoldingPitcher", ("robot", "pitcher")) in atoms
        all_in = all(
            ("InPitcher", (o.name, "pitcher")) in atoms for o in self.env.objects().shrimp()
        )
        if held:
            request = int(state.get(obj=PourObjects.scene, feature_name="requested_count"))
            name = f"Pour{request}" if all_in else "PlacePitcherAtHome"
        elif ("ReadyToPour", ()) in atoms:
            name = "PickPitcher"
        else:
            return LabeledAction(action=self.env.noop_action(), label="Wait")
        request = int(state.get(obj=PourObjects.scene, feature_name="requested_count"))
        ground = provider.ground(name=name, request=request)
        return LabeledAction(
            action=provider.compute_action(
                ground_skill=ground,
                params=np.array(self.parameters) if name.startswith("Pour") else np.zeros(0),
                state=state,
            ),
            label=name,
        )
