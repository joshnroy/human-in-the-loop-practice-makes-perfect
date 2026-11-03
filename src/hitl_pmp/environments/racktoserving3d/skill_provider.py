"""Robot action encoding and explicitly separate, training-only human offerings."""

from typing import ClassVar

import numpy as np

from hitl_pmp.core.method.skill_provider import OraclePolicyProvider, SkillProvider
from hitl_pmp.core.method.types import GroundSkill, LabeledAction, Skill
from hitl_pmp.core.problem.environment.types import Action, Object, State, Type
from hitl_pmp.core.problem.tasks.types import Goal, Predicate

from .environment import RackServingEnvironment
from .objects import RackServing3DObjects as Objects
from .predicates import RackServing3DAtoms, RackServing3DPredicates
from .semantics import HUMAN_BATCH_RETURN, HUMAN_TOOL_RECOVERY
from .skills import GATHER_AND_SWEEP_BOUNDS, SKILL_IDS
from .skills import RackServing3DSkills as Skills


class RackServingSkillProvider(SkillProvider):
    env: RackServingEnvironment
    offer_human_reset: bool = True

    def skills(self) -> tuple[Skill, ...]:
        return Skills.robot_skills()

    def predicates(self) -> tuple[Predicate, ...]:
        return RackServing3DPredicates.all()

    def types(self) -> tuple[Type, ...]:
        return Objects.all_types()

    def objects(self) -> tuple[Object, ...]:
        return Objects.all_objects()

    def sample_params(self, *, ground_skill: GroundSkill, rng: np.random.Generator) -> np.ndarray:
        if ground_skill.skill != Skills.GATHER_AND_SWEEP:
            return np.zeros(0)
        return np.array([rng.uniform(lower, upper) for lower, upper in GATHER_AND_SWEEP_BOUNDS])

    def compute_action(
        self, *, ground_skill: GroundSkill, params: np.ndarray, state: State
    ) -> Action:
        del state
        return RackServingActions.encode(ground_skill=ground_skill, params=params)

    def human_cube_bin_reset_skills(self) -> tuple[GroundSkill, ...]:
        if not self.offer_human_reset or self.env.config.mode == "deployment":
            return ()
        batch, tool = Skills.human_skills(
            mode=self.env.config.mode,
            human_interruption_cost=self.env.config.human_interruption_cost,
        )
        return (
            GroundSkill(
                skill=batch, objects=(Objects.robot, Objects.rack, Objects.tray, *Objects.food)
            ),
            GroundSkill(skill=tool, objects=(Objects.robot, Objects.scraper)),
        )

    def human_cube_bin_reset_skill(self) -> GroundSkill | None:
        resets = self.human_cube_bin_reset_skills()
        return resets[0] if resets else None

    def movables_reset_destination(self, *, ground_skill: GroundSkill) -> str | None:
        name = ground_skill.skill.name
        if name not in (HUMAN_BATCH_RETURN, HUMAN_TOOL_RECOVERY):
            return None
        if self.env.config.mode == "deployment":
            raise PermissionError("human interventions are training-only")
        if not self.offer_human_reset:
            raise PermissionError("human interventions are disabled")
        return name


class RackServingActions:
    """Encode [robot skill ID, food index, squeeze s m, gather delta m, sweep depth d m]."""

    @staticmethod
    def encode(*, ground_skill: GroundSkill, params: np.ndarray) -> Action:
        skill = ground_skill.skill
        if skill.name in (HUMAN_BATCH_RETURN, HUMAN_TOOL_RECOVERY):
            raise PermissionError("human interventions are not robot actions")
        if skill not in Skills.robot_skills():
            raise ValueError(f"unknown rack-serving robot skill: {skill.name}")
        params = np.asarray(params, dtype=float)
        if params.shape != (skill.param_dim,) or not np.isfinite(params).all():
            raise ValueError(f"{skill.name} requires {skill.param_dim} finite parameters")
        continuous = np.zeros(3)
        if skill == Skills.GATHER_AND_SWEEP:
            if any(
                not lower <= value <= upper
                for value, (lower, upper) in zip(params, GATHER_AND_SWEEP_BOUNDS, strict=True)
            ):
                raise ValueError("GatherAndSweep s/delta/d are outside the declared bounds")
            continuous = params
        food_index = 0
        if skill in (Skills.PICK_FOOD, Skills.PLACE_FOOD_ON_TRAY, Skills.RETURN_FOOD_TO_RACK):
            food_index = Objects.food.index(ground_skill.objects[1])
        return np.array([SKILL_IDS[skill.name], food_index, *continuous], dtype=float)


class RackServingOracle(OraclePolicyProvider):
    """Nominal sweep-first baseline, with autonomous pick/place cleanup.

    The sweep parameters are a starting policy, not a calibrated success claim.
    No human operation or simulator write is hidden in this oracle.
    """

    env: RackServingEnvironment
    # (s, delta, d): the driven-blade grid's best single setting (5/20 layouts fully
    # served, 2026-11-03), not a calibrated success claim.
    NOMINAL_GATHER_AND_SWEEP: ClassVar[np.ndarray] = np.array([0.07, 0.015, 0.10])

    def get_labeled_action(self, *, state: State, goal: Goal) -> LabeledAction:
        def holds(*, name: str, objects: tuple[Object, ...] = ()) -> bool:
            return RackServing3DAtoms.holds(state=state, name=name, objects=objects)

        if goal.is_satisfied(state=state):
            return LabeledAction(action=self.env.noop_action(), label="no-op")
        params = np.zeros(0)
        held_food = next(
            (
                food
                for food in Objects.food
                if holds(name="HoldingFood", objects=(Objects.robot, food))
            ),
            None,
        )
        on_rack = [
            food for food in Objects.food if holds(name="OnRack", objects=(food, Objects.rack))
        ]
        holding_scraper = holds(name="HoldingScraper", objects=(Objects.robot, Objects.scraper))
        if holds(name="Waiting"):
            ground = GroundSkill(skill=Skills.WAIT, objects=())
        elif held_food is not None:
            ground = GroundSkill(
                skill=Skills.PLACE_FOOD_ON_TRAY, objects=(Objects.robot, held_food, Objects.tray)
            )
        elif on_rack and holding_scraper:
            ground = GroundSkill(
                skill=Skills.GATHER_AND_SWEEP,
                objects=(Objects.robot, Objects.scraper, Objects.rack, Objects.tray, *Objects.food),
            )
            params = RackServingOracle.NOMINAL_GATHER_AND_SWEEP
        elif holding_scraper:
            ground = GroundSkill(
                skill=Skills.RETURN_SCRAPER, objects=(Objects.robot, Objects.scraper)
            )
        elif holds(name="HandEmpty", objects=(Objects.robot,)):
            if on_rack and holds(name="ScraperHome", objects=(Objects.scraper,)):
                ground = GroundSkill(
                    skill=Skills.PICK_SCRAPER, objects=(Objects.robot, Objects.scraper)
                )
            else:
                food = next(
                    (
                        food
                        for food in Objects.food
                        if holds(name="FoodAvailable", objects=(food,))
                        and not holds(name="OnTray", objects=(food, Objects.tray))
                    ),
                    None,
                )
                if food is None:
                    return LabeledAction(action=self.env.noop_action(), label="no-op")
                ground = GroundSkill(
                    skill=Skills.PICK_FOOD,
                    objects=(Objects.robot, food, Objects.rack, Objects.tray),
                )
        else:
            return LabeledAction(action=self.env.noop_action(), label="no-op")
        return LabeledAction(
            action=RackServingActions.encode(ground_skill=ground, params=params),
            label=ground.skill.name,
        )
