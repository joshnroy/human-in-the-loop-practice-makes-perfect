"""The `core.State` this domain carries, and the abstract atoms travelling on it.

## Why the state carries its own symbolic abstraction

Tossing3D's six predicates are kinder-baselines', not ours (see `predicates.py`). Four
of upstream's six classifiers are pure functions of a KINDER `ObjectCentricState`, but
two are not:

- `Holding` adds a forward-kinematics conjunct, evaluated through a live `PyBulletSim`.
- `MovableInGoalRegion` reads the scored region off the live MuJoCo env's ground
  fixture -- upstream's own comment on `state_abstractor` says so: *"Not pure: poses
  come from `state`, the goal region from the live simulator."*

`core.Predicate.holds` is a positional `(state, objects)` callable with no simulator
handle -- that is the whole reason planning can happen off-simulator -- so those two
cannot be evaluated at `holds` time at all.

The resolution is to abstract **once, at the boundary**: `KinderBackend` owns upstream's
`Tossing3DStateAbstractor`, and every state this domain builds carries the atom set that
abstractor derived *from exactly that state*. `holds` is then a pure set membership test.

## Why this is honest rather than a cache that goes stale

Upstream's `state_abstractor` opens with `self._pybullet_sim.set_state(state)`: it
re-points the kinematics sim at the state it was handed before doing any FK. The sim is a
calculator, not the data source. Measured directly rather than assumed -- abstract a state
from episode A, reset the live simulator to a different episode, then re-abstract the same
captured state: the two atom sets are identical.

So a stale `Tossing3DState` carries the answers that were true *for it*, which is exactly
the property a `core.State` has to have. The failure mode this design exists to avoid is
the opposite one: a classifier that ignores its `state` argument and reads the live
simulator, which is correct only while every call site happens to pass the current state
and silently wrong the moment one does not.

**The one residual impurity, stated rather than hidden.** `MovableInGoalRegion`'s region
geometry still comes from the live env at abstraction time, not from `state`. It is a
static scene fixture -- measured identical (`[1.85, -0.15, 0.0, 2.15, 0.15, 0.15]`) across
seeds -- so it does not vary in practice, and the stale-state check above passes. But it is
a genuine dependence on there being a live scene, and it is why abstraction happens at the
boundary rather than lazily.

## Why the abstractor is not on the state

`Tossing3DEnvironment` deep-copies states (`snapshot.state.model_copy(deep=True)`), and
deep-copying a `Tossing3DStateAbstractor` would clone a PyBullet client. The abstractor
lives on `KinderBackend` for its whole lifetime; only its *output* travels.
"""

from typing import Any

# `np` and `Object` are imported for pydantic's benefit, not this module's: `State.data`
# is annotated `dict[Object, np.ndarray]` under `from __future__ import annotations`, so
# both halves of that forward reference are resolved against the *subclass's* module
# namespace. Without these imports, constructing a `Tossing3DState` raises
# "`Tossing3DState` is not fully defined".
import numpy as np  # noqa: F401

from hitl_pmp.core.problem.environment.types import Object, State  # noqa: F401

# One abstract atom: a predicate name and the object names it is applied to, using
# *this domain's* object names rather than KINDER's. The two differ in exactly one
# place -- KINDER resolves the robot's name from the robot config at reset, while this
# domain's `Object` is the literal `"robot"` -- and `KinderBackend` does that mapping so
# nothing downstream has to know about it.
AbstractAtom = tuple[str, tuple[str, ...]]


class Tossing3DState(State):
    """A `core.State` that also carries the KINDER state it was translated from, and
    the abstract atoms upstream's own abstractor derived from that KINDER state.

    Both fields are optional, and a state built with neither is still a perfectly good
    `core.State` -- the flat feature vectors are unaffected. What such a state cannot do
    is answer a `Predicate`, which is deliberate: silently returning `False` for every
    predicate because nobody attached an abstraction is precisely the kind of quiet
    wrongness this domain has shipped before.
    """

    # A copy of the KINDER `ObjectCentricState` this state was translated from. Typed
    # `Any` because naming it would import KINDER at module scope, and this module is
    # imported by the offline half of the domain.
    object_centric: Any = None

    # What upstream's `Tossing3DStateAbstractor` said held, for this state.
    abstract_atoms: frozenset[AbstractAtom] | None = None
