"""SweepOffShelf3D's `SkillProvider` and privileged oracle."""

import numpy as np
from pydantic import Field

from hitl_pmp.core.method.skill_provider import OraclePolicyProvider, SkillProvider
from hitl_pmp.core.method.types import GroundSkill, LabeledAction, Skill
from hitl_pmp.core.problem.environment.types import Action, Object, State, Type
from hitl_pmp.core.problem.tasks.types import Goal, Predicate

from .environment import SKILL_IDS, SweepOffShelf3DEnvironment
from .objects import SweepOffShelf3DObjects as O
from .predicates import SweepOffShelf3DAtoms, SweepOffShelf3DPredicates
from .skills import SweepOffShelf3DSkills as S

# The oracle's sweep: the head skimming the board at a moderate pull.
ORACLE_SWEEP_PARAMS = (0.004, 0.1)


class SweepOffShelf3DSkillProvider(SkillProvider):
    env: SweepOffShelf3DEnvironment
    human_reset_practice_cost: float = Field(default=1.0, ge=0.0, allow_inf_nan=False)
    # Off only for a baseline without the human: the skills are then absent, not priced.
    offer_human_reset: bool = True

    def skills(self) -> tuple[Skill, ...]:
        return S.robot_skills()

    def predicates(self) -> tuple[Predicate, ...]:
        return SweepOffShelf3DPredicates.all()

    def types(self) -> tuple[Type, ...]:
        return O.all_types()

    def objects(self) -> tuple[Object, ...]:
        return O.all_objects()

    def sample_params(self, *, ground_skill: GroundSkill, rng: np.random.Generator) -> np.ndarray:
        if ground_skill.skill != S.SWEEP_SHELF:
            return np.zeros(0)
        (h_low, h_high), (v_low, v_high) = S.SWEEP_BOUNDS
        return np.array([rng.uniform(h_low, h_high), rng.uniform(v_low, v_high)])

    def compute_action(
        self, *, ground_skill: GroundSkill, params: np.ndarray, state: State
    ) -> Action:
        del state
        return SweepOffShelf3DActions.encode(ground_skill=ground_skill, params=params)

    def human_cube_bin_reset_skills(self) -> tuple[GroundSkill, ...]:
        """Both human skills, grounded once each, at the configured cost."""
        if not self.offer_human_reset:
            return ()
        priced = (
            skill.model_copy(update={"practice_cost": self.human_reset_practice_cost})
            for skill in S.human_skills()
        )
        cubes, wiper = priced
        return (
            GroundSkill(skill=cubes, objects=(O.robot, O.box, *O.cubes)),
            GroundSkill(skill=wiper, objects=(O.robot, O.wiper)),
        )

    def human_cube_bin_reset_skill(self) -> GroundSkill | None:
        resets = self.human_cube_bin_reset_skills()
        return resets[0] if resets else None

    def movables_reset_destination(self, *, ground_skill: GroundSkill) -> str | None:
        """The environment reads the human skill's name to know which one to perform."""
        return ground_skill.skill.name


class SweepOffShelf3DActions:
    """`[skill_id, cube_index, height, speed]`. A static-method container."""

    @staticmethod
    def encode(*, ground_skill: GroundSkill, params: np.ndarray) -> Action:
        single_cube = (S.PICK_CUBE_FROM_SHELF, S.PICK_CUBE_FROM_FLOOR, S.PLACE_CUBE_IN_BOX)
        cube_index = (
            float(O.cubes.index(ground_skill.objects[1]))
            if ground_skill.skill in single_cube
            else 0.0
        )
        height, speed = (float(v) for v in params) if len(params) == 2 else (0.0, 0.0)
        return np.array([float(SKILL_IDS[ground_skill.skill.name]), cube_index, height, speed])


class SweepOffShelf3DOracle(OraclePolicyProvider):
    """A hand-written policy: sweep with the wiper, then clean up by hand."""

    env: SweepOffShelf3DEnvironment

    def get_labeled_action(self, *, state: State, goal: Goal) -> LabeledAction:
        del goal

        def holds(*, name: str, objects: tuple[Object, ...]) -> bool:
            return SweepOffShelf3DAtoms.holds(state=state, name=name, objects=objects)

        hand_empty = holds(name="HandEmpty", objects=(O.robot,))
        holding_wiper = holds(
            name="HoldingWiper",
            objects=(
                O.robot,
                O.wiper,
            ),
        )
        on_shelf = [cube for cube in O.cubes if holds(name="OnShelf", objects=(cube,))]
        on_floor = [cube for cube in O.cubes if holds(name="OnFloor", objects=(cube,))]
        held = [
            cube
            for cube in O.cubes
            if holds(
                name="HoldingCube",
                objects=(
                    O.robot,
                    cube,
                ),
            )
        ]
        if held:
            ground = GroundSkill(skill=S.PLACE_CUBE_IN_BOX, objects=(O.robot, held[0], O.box))
        elif on_shelf and holding_wiper:
            ground = GroundSkill(skill=S.SWEEP_SHELF, objects=(O.robot, O.wiper, O.box, *O.cubes))
            return LabeledAction(
                action=SweepOffShelf3DActions.encode(
                    ground_skill=ground, params=np.array(ORACLE_SWEEP_PARAMS)
                ),
                label="SweepShelf",
            )
        elif on_shelf and hand_empty and holds(name="WiperHome", objects=(O.wiper,)):
            ground = GroundSkill(skill=S.PICK_WIPER, objects=(O.robot, O.wiper))
        elif holding_wiper:
            ground = GroundSkill(skill=S.RETURN_WIPER, objects=(O.robot, O.wiper))
        elif on_floor and hand_empty:
            ground = GroundSkill(skill=S.PICK_CUBE_FROM_FLOOR, objects=(O.robot, on_floor[0]))
        elif on_shelf and hand_empty:
            ground = GroundSkill(skill=S.PICK_CUBE_FROM_SHELF, objects=(O.robot, on_shelf[0]))
        else:
            return LabeledAction(action=self.env.noop_action(), label="no-op")
        return LabeledAction(
            action=SweepOffShelf3DActions.encode(ground_skill=ground, params=np.zeros(0)),
            label=ground.skill.name,
        )
