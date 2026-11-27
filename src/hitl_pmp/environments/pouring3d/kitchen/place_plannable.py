"""`PlacePlannable`: false exactly where the dispatched jug placement already refused to plan.

Curtis seed 8 of pouring-cost-10seed-20261112 (human-reset cost 10) dispatched
PlacePitcherAtHome 117 times in a row from one state, every one a 0-step "No
collision-free base path to placement", until practice ended `repeated_zero_step`. Each
was scored as a practice attempt that cost nothing, which drove the skill's forecast
practice cost from 3.78 to 0.51, so practising it kept outscoring the human pour-back.

The placement plans everything (`PlacePitcherController.reset`) before the robot moves,
and deterministically, so one refusal is exact evidence about that state: the same
dispatch from it refuses again. As `PickPlannable` (`pick_plannable.py`) does for the
dumpling picks, the atom is history-dependent: false iff this environment instance has
seen the dispatched placement fail having run no control step, the jug still in the
hand, and the state is still the one it failed from: the base and arm within
`BASE_TOLERANCE_M` / `JOINT_TOLERANCE_RAD`, the jug and every dumpling within
`pick_plannable`'s pose tolerances. Any motion forgets it, and a failure after motion is
an ordinary failed attempt. Every state never tried, and every imagined planner state,
carries the atom; the human pour-back, which takes the jug home, re-asserts it.
"""

from typing import Any

import numpy as np

from hitl_pmp.environments.pouring3d.types import AbstractAtom

from .pick_plannable import ANGLE_TOLERANCE_DEG, POSITION_TOLERANCE_M, KitchenPickPlannable
from .types import PlaceRefusal

PLACE_PLANNABLE = "PlacePlannable"
BASE_TOLERANCE_M = POSITION_TOLERANCE_M
BASE_YAW_TOLERANCE_RAD = float(np.deg2rad(ANGLE_TOLERANCE_DEG))
JOINT_TOLERANCE_RAD = 0.01


class KitchenPlacePlannable:
    """A static-method container, never instantiated."""

    @staticmethod
    def snapshot(*, controller: Any) -> PlaceRefusal:
        c = controller
        return PlaceRefusal(
            base=tuple(float(v) for v in c.data.qpos[c.base_q]),
            arm=tuple(float(v) for v in c.data.qpos[c.arm_q]),
            bodies={
                name: KitchenPickPlannable.pose(controller=c, name=name)
                for name in ("pitcher", *c.scene.cube_names)
            },
        )

    @staticmethod
    def same_state(*, a: PlaceRefusal, b: PlaceRefusal) -> bool:
        base_a, base_b = np.asarray(a.base), np.asarray(b.base)
        if np.linalg.norm(base_a[:2] - base_b[:2]) > BASE_TOLERANCE_M:
            return False
        if abs(float(np.angle(np.exp(1j * (base_a[2] - base_b[2]))))) > BASE_YAW_TOLERANCE_RAD:
            return False
        if np.max(np.abs(np.subtract(a.arm, b.arm))) > JOINT_TOLERANCE_RAD:
            return False
        return a.bodies.keys() == b.bodies.keys() and all(
            KitchenPickPlannable.same_pose(a=a.bodies[n], b=b.bodies[n]) for n in a.bodies
        )

    @staticmethod
    def refusal(
        *, controller: Any, before: PlaceRefusal, error: str | None, steps: int
    ) -> PlaceRefusal | None:
        """The refusal to record after a dispatched placement, or None: it failed, ran
        no control step, the jug is still in the hand, and nothing moved."""
        if error is None or steps != 0 or controller.held != "pitcher":
            return None
        now = KitchenPlacePlannable.snapshot(controller=controller)
        if not KitchenPlacePlannable.same_state(a=now, b=before):
            return None
        return before.model_copy(update={"error": error})

    @staticmethod
    def atoms(*, controller: Any, refusals: dict[str, PlaceRefusal]) -> set[AbstractAtom]:
        """Forget the refusal once anything moved; while it stands the atom is false."""
        refused = refusals.get("pitcher")
        if refused is not None and not KitchenPlacePlannable.same_state(
            a=KitchenPlacePlannable.snapshot(controller=controller), b=refused
        ):
            del refusals["pitcher"]
        return set() if "pitcher" in refusals else {(PLACE_PLANNABLE, ("robot", "pitcher"))}
