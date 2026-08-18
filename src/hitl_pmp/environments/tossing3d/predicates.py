"""Tossing3D's five predicates -- **named** here, computed by KINDER.

This module used to be 578 lines of hand-written classifiers. It is now a list of five
names and a lookup, because the classifiers are upstream's:
`kinder_models.dynamic3d.tossing.state_abstractions` ships `HandEmpty`, `OnGround`,
`Holding`, `MovableInGoalRegion` and `MovableIsDownX`, and
`adapters/kinder/abstraction.py` exposes whatever its abstractor reports as
`core.Predicate`s. What is left to this domain is *which* predicates its operators are
written over, which is genuine domain knowledge.

## Why the port was deleted rather than kept in step

The old module opened by saying **"KINDER ships no symbolic model for Tossing3D"**. That
was true when it was written and is not true now -- `f12326c` carries both the state
abstractions and a bilevel model over them. So the six classifiers here were a port with
no original to track, and the docstring's own list of deviations shows the cost:
`Holding` dropped upstream's forward-kinematics conjunct, because a `core.Predicate.holds`
had no simulator handle, leaving a predicate that "can call a cube 'held' that is
closed-gripper and airborne without being grasped".

That conjunct is now evaluated, because the bridge hands the abstractor a full state and
the abstractor has its own `PyBulletSim`.

## What was deleted, and why each deletion is safe

- **`RobotAtSuccessfulThrowPose` and its whole calibration** -- `THROW_RANGE`,
  `THROW_RANGE_MIN`/`MAX`, `THROW_OVERSHOOT_MARGIN`, `THROW_SHORTFALL_MARGIN`,
  `THROW_POSE_LATERAL_TOLERANCE`, `THROW_STANDOFF_BOUNDS`,
  `WORST_BARRIER_COLLISION_STANDOFF`, `BARRIER_COLLISION_MARGIN`. All of it existed to
  describe the pose *between* a base move and a throw. Upstream fuses those into one
  controller -- *"Composed rather than split so that no predicate has to name the pose
  between them"* -- so there is no intermediate state left to characterise. This is
  obsolete work, not pending work: the throw band does not need re-deriving.
- **`Reachable`** -- replaced one-for-one by upstream's `MovableIsDownX(?cube, ?barrier)`,
  which asks the same question (is the cube still on the robot's side of the barrier?)
  by comparing the two live x positions, exactly as the port did.
- **`InBin`** -- replaced by `MovableInGoalRegion`. The old predicate tested containment
  in a box smuggled through the `State` on the bin object, under a stated assumption that
  "the bin's interior *is* the scored region". Upstream's reads the region off the live
  simulator through `Region.check_in_region`, which is the same call `_check_goals()`
  makes, so the assumption is no longer needed and the bin is no longer overloaded.
- **`HandEmpty`, `Holding`, `OnGround`** -- same names, now upstream's implementations.

## The consequence worth stating plainly

These predicates are **no longer pure functions of a `core.State`**, and they can no
longer be evaluated without a simulator. `Holding` runs forward kinematics;
`MovableInGoalRegion` reads the live scene. That is the trade CLAUDE.md describes as the
reason KINDER became a required dependency and CI installs it: the alternative is
"six classifiers kept in agreement with upstream's by test rather than by construction",
which is the duplication this project explicitly does not want.
"""

from hitl_pmp.adapters.kinder.abstraction import KinderAbstraction
from hitl_pmp.core.problem.tasks.types import Predicate

# Upstream's own names, from `kinder_models.dynamic3d.tossing.state_abstractions`. Written
# out here rather than imported so this module needs no simulator to state what the domain
# is about; `KinderBackend.api` imports the real `Predicate` objects, and the abstraction
# raises if the abstractor ever reports one this list does not cover.
HAND_EMPTY = "HandEmpty"
ON_GROUND = "OnGround"
HOLDING = "Holding"
MOVABLE_IN_GOAL_REGION = "MovableInGoalRegion"
MOVABLE_IS_DOWN_X = "MovableIsDownX"

# The order a `SkillProvider` reports them in.
PREDICATE_NAMES = (HAND_EMPTY, ON_GROUND, HOLDING, MOVABLE_IN_GOAL_REGION, MOVABLE_IS_DOWN_X)


class Tossing3DPredicates:
    """This domain's predicates, resolved out of a live abstraction.

    A static-method container, never instantiated, same as every other business-logic
    class in this project. The predicates themselves cannot be module constants any more:
    each one closes over the abstraction of one live scene, so two `Tossing3DEnvironment`
    instances hold two sets. That is why everything below takes the abstraction rather
    than reaching for a global.
    """

    @staticmethod
    def all(*, abstraction: KinderAbstraction) -> tuple[Predicate, ...]:
        """The five, in `PREDICATE_NAMES` order."""
        return tuple(abstraction.predicate(name=name) for name in PREDICATE_NAMES)

    @staticmethod
    def get(*, abstraction: KinderAbstraction, name: str) -> Predicate:
        if name not in PREDICATE_NAMES:
            raise KeyError(f"{name!r} is not one of this domain's predicates: {PREDICATE_NAMES}")
        return abstraction.predicate(name=name)
