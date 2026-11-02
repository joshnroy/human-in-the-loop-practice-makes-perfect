"""Whether the dumpling pick's own planner finds a plan: the rule behind `PickPlannable`.

The same design as Tossing3D's `PickPlannable`, for the same reason. Overnight kitchen
practice (2026-11-01) retried `PickShrimpFromTray` on one dumpling 120 times in a row
against the identical refusal "No feasible continuation from prepared arm postures":
the pickup's planner refuses deterministically in that state, and nothing in the
symbolic model said so. The controller's planner is the thing being predicted, so it
is asked directly. A robot may always query its own motion planner before acting.

## Why the verdict agrees with dispatch

`PotstickerController.payload_pick_plan` runs the dispatched pick itself --
`pick_shrimp` -> `pick_payload_with_retry`, every grasp candidate in order, with its
base move to the aisle, hover IK and approach/descend/lift continuation -- and stops it
with `PickPlanned` once a plan exists. That planning is not kinematic: the base move is
physics, and the arm is planned from wherever the motors left it, so a kinematic
re-implementation would plan from a different start. Instead the dry run is made on
the live simulation and every byte of it is restored afterwards. Every planning call is
deterministic (IK seeds come from `default_rng(0)`), so the dry run and the dispatch
run the same code on the same input: the plan the dry run finds is the plan dispatch
executes, and a refusal is the refusal dispatch raises, word for word.

Dispatch can still fail *after* planning -- a lost pinch, a tracking fault, a dumpling
that slides -- and then retries the next candidate from the changed state. That is an
ordinary failed attempt the planners learn from, not a refusal, and this atom does not
describe it.

## The cases

| state | dry run | atom |
| --- | --- | --- |
| the dumpling is held, or in the jug | not made | present |
| the hand holds something else | not made | present |
| the dumpling is on the floor | not made | present |
| in the pan or on a worktop, hand empty | made, cached per state | present iff planned |

Where there is no pick to plan the atom is carried present, as Tossing3D carries it
while the cube is held: carried absent, it would make every imagined later pick
impossible. The floor pick has no planning stage at all -- its motion is not preceded
by a search that can refuse -- so there is nothing to ask. With something in hand the
planner cannot be asked (an empty hand is its first requirement), and the pick is not
applicable anyway; the atom is re-read once the hand is empty.

The human bulk pour-back does not require this atom. A dumpling the robot cannot pick
leaves the costed human reset as the planner's route, chosen by cost.
"""

import hashlib
from collections import OrderedDict
from typing import Any

import mujoco
import numpy as np

from hitl_pmp.environments.pouring3d.types import AbstractAtom, PayloadPickPlan

PICK_PLANNABLE = "PickPlannable"


class KitchenPickPlannable:
    """A static-method container, never instantiated."""

    @staticmethod
    def atoms(
        *, controller: Any, cache: OrderedDict[tuple[int, str, bytes], PayloadPickPlan]
    ) -> set[AbstractAtom]:
        c = controller
        atoms: set[AbstractAtom] = set()
        locations = c.scene.cube_locations()
        for index, (name, location) in enumerate(zip(c.scene.cube_names, locations, strict=True)):
            if c.held is None and location in {"tray", "surface"}:
                key = (index, location, KitchenPickPlannable.state_key(controller=c))
                if key not in cache:
                    cache[key] = c.payload_pick_plan(index=index, location=location)
                    while len(cache) > 64:
                        cache.popitem(last=False)
                if not cache[key].plannable:
                    continue
            atoms.add((PICK_PLANNABLE, ("robot", name)))
        return atoms

    @staticmethod
    def state_key(*, controller: Any) -> bytes:
        """Everything the pick's planning reads: physics plus the controller's targets."""
        c = controller
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(c.model, spec))
        mujoco.mj_getState(c.model, c.data, state, spec)
        digest = hashlib.sha256(state.tobytes())
        for value in (c.last_q, c.base_target, np.array([c.posture, c.grip])):
            digest.update(np.asarray(value, dtype=float).tobytes())
        return digest.digest()
