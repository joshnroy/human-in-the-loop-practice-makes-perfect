"""`PickPlannable` for RackToServing3D: Pouring3D's remembered pick refusal, for food.

The rule, its reasons and its memory are Pouring3D's (`pouring3d/kitchen/pick_plannable.py`),
reused here: the atom is false iff this environment instance has seen a dispatched
PickFood refuse to *plan* for that food at its current pose, with the plate where it
was, and it is forgotten once either moves (5 mm / 5 degrees). PickFood requires it;
the human batch return does not, and makes it true for every food, so a food the robot
cannot pick leaves the costed human reset as the planner's route. The memory lives on
the environment instance: kept across practice sessions, empty in a fresh evaluation
environment, forgotten between evaluation tasks.

The 5088ea2 dry run motivated it: 9/10 evaluation failures ended in PickFood refusing
at 0 steps on food stranded in the 1 cm gap between the plate and the rack's front edge,
where every grasp's descent puts a pad into the rack's wires.

What records a refusal, after a dispatched PickFood fails:

- the error is PickFood's planning refusal (`_plan_pick`: "No collision-free food
  grasp: ..."), raised before any motion;
- the skill executed 0 control steps, so no finger touched anything;
- no food is held and the food is where it was.

A physical failure (a slip, a lost grasp, a lift blocked by the rim) moves the robot and
never records one: it is an ordinary failed attempt for competence.
"""

from collections.abc import Callable
from typing import Any

import numpy as np

from hitl_pmp.environments.pouring3d.kitchen.pick_plannable import (
    PICK_PLANNABLE,
    KitchenPickPlannable,
)
from hitl_pmp.environments.pouring3d.kitchen.types import PickRefusal, Pose

from .predicates import AbstractAtom

PLANNING_PREFIXES = ("No collision-free food grasp:",)
PLATE = "box_0"
PoseOf = Callable[..., Pose]


class RackPickPlannable:
    """A static-method container, never instantiated."""

    @staticmethod
    def live_pose(*, backend: Any) -> PoseOf:
        def pose(*, name: str) -> Pose:
            position, wxyz = backend.core().get_object(name).get_pose()
            w, x, y, z = (float(v) for v in np.asarray(wxyz))
            px, py, pz = (float(v) for v in np.asarray(position))
            return Pose(position=(px, py, pz), quat=(x, y, z, w))

        return pose

    @staticmethod
    def before(*, backend: Any, name: str) -> PickRefusal:
        pose = RackPickPlannable.live_pose(backend=backend)
        return PickRefusal(dumpling=pose(name=name), pan=pose(name=PLATE))

    @staticmethod
    def refusal(
        *,
        backend: Any,
        name: str,
        before: PickRefusal,
        error: str | None,
        steps: int,
        pose: PoseOf | None = None,
    ) -> PickRefusal | None:
        """The refusal to record after a failed dispatched PickFood, or None."""
        if error is None or steps != 0:
            return None
        if not KitchenPickPlannable.is_planning_refusal(error=error, prefixes=PLANNING_PREFIXES):
            return None
        if any(backend.held(name=food) for food in backend.cube_names):
            return None
        pose = RackPickPlannable.live_pose(backend=backend) if pose is None else pose
        if not KitchenPickPlannable.same_pose(a=pose(name=name), b=before.dumpling):
            return None
        return before.model_copy(update={"error": error})

    @staticmethod
    def atoms(
        *, backend: Any, refusals: dict[str, PickRefusal], pose: PoseOf | None = None
    ) -> set[AbstractAtom]:
        """Forget every refusal whose food or plate moved; the rest make the atom false."""
        pose = RackPickPlannable.live_pose(backend=backend) if pose is None else pose
        plate = pose(name=PLATE)
        for name in list(refusals):
            refused = refusals[name]
            if not (
                KitchenPickPlannable.same_pose(a=pose(name=name), b=refused.dumpling)
                and KitchenPickPlannable.same_pose(a=plate, b=refused.pan)
            ):
                del refusals[name]
        return {(PICK_PLANNABLE, ("robot", n)) for n in backend.cube_names if n not in refusals}
