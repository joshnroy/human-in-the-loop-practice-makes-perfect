"""`PickPlannable`: false exactly where the dispatched dumpling pick already refused to plan.

Overnight kitchen practice (2026-11-01) retried `PickShrimpFromTray` on one dumpling 120
times in a row against the identical refusal "No feasible continuation from prepared arm
postures": the pick's planner refuses deterministically in that state, and nothing in the
symbolic model said so.

## A remembered refusal, not a dry run

The pick's planning is deterministic (IK seeds come from `default_rng(0)`, the base move
is physics from the same state), so one refusal is exact evidence about that state: a
second dispatch from it would refuse again, word for word. So the atom is **history-
dependent**: it is false iff this environment instance has seen the dispatched pick refuse
to plan for this dumpling at its current pose, with the pan where it was. That costs one
real attempt per refusing state, instead of a dry run of the planner (median 49 s) at
every observation. It is still a predicate evaluated when predicates are evaluated; the
planner only reads it.

## What is recorded

After a dispatched pick (`PickShrimpFromTray` / `Floor` / `Surface`) fails, a refusal is
recorded for its dumpling only if all of these hold:

- the final error is a planning refusal (`is_planning_refusal`): the retry loop's own two
  planning messages, or an IK/clearance rejection in a phase before the fingers reach the
  dumpling (`PLANNING_PHASES`);
- no finger touched the dumpling at any control step of the skill;
- the hand is empty and the dumpling is where it was, within 5 mm and 5 degrees.

A physical failure (a slip, a lost or incomplete grasp, a tracking fault, contact-motion
IK in "descend to shrimp body") does not record a refusal the first time: it is an ordinary
failed attempt for competence. When the next dispatched pick of that dumpling fails with the
identical error from the same dumpling and pan poses (the tolerances below), it is recorded
as a refusal: EES seed 1 (2026-11-05) dispatched one pick 23 times into the same tracking
failure.

## When it is forgotten

At every observation a refusal is dropped once its dumpling or the pan has moved beyond
tolerance: a human pour-back, a successful pick, any displacement. The memory lives on
the environment instance, so practice keeps it across sessions and a fresh evaluation
environment starts empty. Every state never tried, and every imagined planner state,
carries the atom (the same default as Tossing3D).

The human bulk pour-back does not require this atom, so a dumpling the robot cannot pick
leaves the costed human reset as the planner's route.
"""

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from hitl_pmp.environments.pouring3d.types import AbstractAtom

from .types import PickRefusal, Pose

PICK_PLANNABLE = "PickPlannable"
# `pick_payload_with_retry` already treats exactly these as planning (it skips the
# physical retreat after them): no motion beyond the aisle base move was executed.
PLANNING_MESSAGES = (
    "No feasible continuation from prepared arm postures",
    "No collision-free empty-arm preparation path",
)
# Phases of the pick before the fingers reach the dumpling. An IK or clearance
# rejection here is a refusal of the planned target, not a contact outcome.
PLANNING_PHASES = (
    "prepare arm in clear aisle",
    "position for shrimp pick",
    "approach above raised shrimp",
    "approach shrimp",
)
PLANNING_PREFIXES = PLANNING_MESSAGES + tuple(
    f"{kind} in {phase}:"
    for kind in ("IK infeasible", "Insufficient obstacle clearance")
    for phase in PLANNING_PHASES
)
POSITION_TOLERANCE_M = 0.005
ANGLE_TOLERANCE_DEG = 5.0


class KitchenPickPlannable:
    """A static-method container, never instantiated."""

    @staticmethod
    def is_planning_refusal(*, error: str, prefixes: tuple[str, ...] = PLANNING_PREFIXES) -> bool:
        """Whether `error` is a planning refusal; another domain passes its own skill's
        `prefixes` (RackToServing3D's PickFood, `racktoserving3d/pick_plannable.py`)."""
        return error.startswith(prefixes)

    @staticmethod
    def pose(*, controller: Any, name: str) -> Pose:
        position, rotation = controller.body_pose(name=name)
        return Pose(position=tuple(position.tolist()), quat=tuple(rotation.as_quat().tolist()))

    @staticmethod
    def same_pose(*, a: Pose, b: Pose) -> bool:
        if np.linalg.norm(np.subtract(a.position, b.position)) > POSITION_TOLERANCE_M:
            return False
        turn = Rotation.from_quat(a.quat) * Rotation.from_quat(b.quat).inv()
        return bool(turn.magnitude() <= np.deg2rad(ANGLE_TOLERANCE_DEG))

    @staticmethod
    def refusal(
        *, controller: Any, name: str, before: PickRefusal, error: str | None, touched: bool
    ) -> PickRefusal | None:
        """The refusal to record after a failed dispatched pick, or None."""
        c = controller
        if error is None or touched or c.held is not None:
            return None
        if not KitchenPickPlannable.is_planning_refusal(error=error):
            return None
        now = KitchenPickPlannable.pose(controller=c, name=name)
        if not KitchenPickPlannable.same_pose(a=now, b=before.dumpling):
            return None
        return before.model_copy(update={"error": error})

    @staticmethod
    def repeated(
        *, controller: Any, before: PickRefusal, error: str | None, previous: PickRefusal | None
    ) -> PickRefusal | None:
        """The refusal to record when a pick fails again exactly as it last failed, or None.

        The pick is deterministic, so a second identical failure from the same poses will
        recur on every dispatch, whatever kind of failure it is.
        """
        if error is None or previous is None or controller.held is not None:
            return None
        if error != previous.error:
            return None
        same = KitchenPickPlannable.same_pose
        if not (
            same(a=before.dumpling, b=previous.dumpling) and same(a=before.pan, b=previous.pan)
        ):
            return None
        return before.model_copy(update={"error": error})

    @staticmethod
    def atoms(*, controller: Any, refusals: dict[str, PickRefusal]) -> set[AbstractAtom]:
        """Forget every refusal whose dumpling or pan moved; the rest make the atom false."""
        c = controller
        pan = KitchenPickPlannable.pose(controller=c, name="tray")
        for name in list(refusals):
            refused = refusals[name]
            dumpling = KitchenPickPlannable.pose(controller=c, name=name)
            if not (
                KitchenPickPlannable.same_pose(a=dumpling, b=refused.dumpling)
                and KitchenPickPlannable.same_pose(a=pan, b=refused.pan)
            ):
                del refusals[name]
        return {(PICK_PLANNABLE, ("robot", n)) for n in c.scene.cube_names if n not in refusals}
