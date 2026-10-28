"""SweepTable3D's provider, action encoding and geometry-guided oracle."""

from typing import ClassVar

import numpy as np
from pydantic import Field

from hitl_pmp.core.method.skill_provider import OraclePolicyProvider, SkillProvider
from hitl_pmp.core.method.types import GroundSkill, LabeledAction, Skill
from hitl_pmp.core.problem.environment.environment import Environment
from hitl_pmp.core.problem.environment.types import Action, Object, State, Type
from hitl_pmp.core.problem.tasks.types import Goal, Predicate

from .objects import NUM_CUBES
from .objects import SweepTable3DObjects as O
from .predicates import SweepTable3DAtoms, SweepTable3DPredicates
from .skills import SKILL_IDS
from .skills import SweepTable3DSkills as S


class SweepTable3DSkillProvider(SkillProvider):
    env: Environment
    human_reset_practice_cost: float = Field(default=1.0, ge=0.0, allow_inf_nan=False)
    offer_human_reset: bool = True

    def skills(self) -> tuple[Skill, ...]:
        return S.robot_skills()

    def predicates(self) -> tuple[Predicate, ...]:
        return SweepTable3DPredicates.all()

    def types(self) -> tuple[Type, ...]:
        return O.all_types()

    def objects(self) -> tuple[Object, ...]:
        return O.all_objects()

    def sample_params(self, *, ground_skill: GroundSkill, rng: np.random.Generator) -> np.ndarray:
        if ground_skill.skill != S.SWEEP_TABLE:
            return np.zeros(0)
        return np.array([rng.uniform(low, high) for low, high in S.SWEEP_BOUNDS])

    def compute_action(
        self, *, ground_skill: GroundSkill, params: np.ndarray, state: State
    ) -> Action:
        del state
        return SweepTable3DActions.encode(ground_skill=ground_skill, params=params)

    def human_cube_bin_reset_skills(self) -> tuple[GroundSkill, ...]:
        """Both human reset operators grounded once at the configured practice cost."""
        if not self.offer_human_reset:
            return ()
        cubes, wiper = (
            skill.model_copy(update={"practice_cost": self.human_reset_practice_cost})
            for skill in S.human_skills()
        )
        return (
            GroundSkill(skill=cubes, objects=(O.robot, O.box, O.wiper, *O.cubes)),
            GroundSkill(skill=wiper, objects=(O.robot, O.wiper)),
        )

    def human_cube_bin_reset_skill(self) -> GroundSkill | None:
        resets = self.human_cube_bin_reset_skills()
        return resets[0] if resets else None

    def movables_reset_destination(self, *, ground_skill: GroundSkill) -> str | None:
        return ground_skill.skill.name


class SweepTable3DActions:
    """The runtime vector is [skill_id, ball_index, angle_deg, speed_m_s]."""

    @staticmethod
    def encode(*, ground_skill: GroundSkill, params: np.ndarray) -> Action:
        target = next(
            (
                obj
                for parameter, obj in zip(
                    ground_skill.skill.parameters, ground_skill.objects, strict=True
                )
                if parameter.name == "cube" and parameter.type == O.ball_type
            ),
            None,
        )
        cube_index = float(O.balls.index(target)) if target is not None else 0.0
        values = np.asarray(params, dtype=float)
        if values.shape != (ground_skill.skill.param_dim,) or not np.isfinite(values).all():
            raise ValueError("invalid skill parameter shape or nonfinite value")
        if ground_skill.skill == S.SWEEP_TABLE and not all(
            low <= value <= high for value, (low, high) in zip(values, S.SWEEP_BOUNDS, strict=True)
        ):
            raise ValueError("SweepTable parameters outside [angle_deg, speed_m_s] bounds")
        angle, speed = (float(value) for value in values) if len(values) == 2 else (0.0, 0.0)
        return np.array([float(SKILL_IDS[ground_skill.skill.name]), cube_index, angle, speed])


class SweepTable3DOracle(OraclePolicyProvider):
    """Aim one bounded sweep; never substitute pick/place delivery.

    The fixed table front is x=1.373; state geometry can override table_edge_x.
    Aiming uses the observed ball centroid and the live bin's interior
    midpoint along the table edge; clamping produces a bounded heuristic, not a
    guaranteed successful stroke.
    """

    env: Environment
    TABLE_EDGE_X: ClassVar[float] = 1.373

    def get_labeled_action(self, *, state: State, goal: Goal) -> LabeledAction:
        del goal

        def holds(*, name: str, objects: tuple[Object, ...]) -> bool:
            return SweepTable3DAtoms.holds(state=state, name=name, objects=objects)

        if not holds(name="TaskActive", objects=()):
            return LabeledAction(action=self.env.noop_action(), label="no-op")
        hand_empty = holds(name="HandEmpty", objects=(O.robot,))
        holding_wiper = holds(name="HoldingWiper", objects=(O.robot, O.wiper))
        on_table = [cube for cube in O.cubes if holds(name="OnTable", objects=(cube,))]
        held = [cube for cube in O.cubes if holds(name="HoldingCube", objects=(O.robot, cube))]
        if held:
            ground = GroundSkill(skill=S.PLACE_CUBE_IN_BOX, objects=(O.robot, held[0], O.box))
        elif len(on_table) == NUM_CUBES and holding_wiper:
            ground = GroundSkill(skill=S.SWEEP_TABLE, objects=(O.robot, O.wiper, O.box, *O.cubes))
            return LabeledAction(
                action=SweepTable3DActions.encode(
                    ground_skill=ground, params=self.sweep_params(state=state)
                ),
                label=ground.skill.name,
            )
        elif (
            len(on_table) == NUM_CUBES
            and hand_empty
            and holds(name="WiperHome", objects=(O.wiper,))
        ):
            ground = GroundSkill(skill=S.PICK_WIPER, objects=(O.robot, O.wiper))
        else:
            return LabeledAction(action=self.env.noop_action(), label="no-op")
        return LabeledAction(
            action=SweepTable3DActions.encode(ground_skill=ground, params=np.zeros(0)),
            label=ground.skill.name,
        )

    def sweep_params(self, *, state: State) -> np.ndarray:
        """Aim toward the tray at a fixed nominal speed, without moving the start."""
        centroid_x = float(np.mean([state.get(obj=cube, feature_name="x") for cube in O.cubes]))
        centroid_y = float(np.mean([state.get(obj=cube, feature_name="y") for cube in O.cubes]))
        bin_y = (
            state.get(obj=O.box, feature_name="y_min") + state.get(obj=O.box, feature_name="y_max")
        ) / 2.0
        edge_x = float(getattr(state, "table_edge_x", self.TABLE_EDGE_X))
        angle_bounds, _ = S.SWEEP_BOUNDS
        depth = max(centroid_x - edge_x, 1e-6)
        angle = float(np.clip(np.rad2deg(np.arctan2(bin_y - centroid_y, depth)), *angle_bounds))
        return np.array([angle, 0.15])
