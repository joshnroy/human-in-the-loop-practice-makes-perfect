"""The `core.State` this domain carries, and the abstract atoms travelling on it.

## Why the state carries its own symbolic abstraction

Tossing3D's five predicates are kinder-baselines', not ours (see `predicates.py`). Three
of upstream's five classifiers are pure functions of a KINDER `ObjectCentricState`, but
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

from collections.abc import Callable
from typing import Any, Literal, Protocol, runtime_checkable

# `np` and `Object` are imported for pydantic's benefit, not this module's: `State.data`
# is annotated `dict[Object, np.ndarray]` under `from __future__ import annotations`, so
# both halves of that forward reference are resolved against the *subclass's* module
# namespace. Without these imports, constructing a `Tossing3DState` raises
# "`Tossing3DState` is not fully defined".
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from hitl_pmp.core.problem.environment.types import Object, State  # noqa: F401

# One abstract atom: a predicate name and the object names it is applied to, using
# *this domain's* object names rather than KINDER's. The two differ in exactly one
# place -- KINDER resolves the robot's name from the robot config at reset, while this
# domain's `Object` is the literal `"robot"` -- and `KinderBackend` does that mapping so
# nothing downstream has to know about it.
AbstractAtom = tuple[str, tuple[str, ...]]


class PlanarCollisionBox(BaseModel):
    """A controller's overhead collision rectangle, expressed without KINDER types."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    name: str
    center: tuple[float, float]
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    yaw: float


class TossParameterSet(BaseModel):
    """The box the toss's three learned parameters are drawn from and checked against.

    Standoff in metres, speed in joint-path deg/s, release in milliseconds from the start
    of the swing. An environment carries one (`Tossing3DEnvironment.toss_parameters`):
    the simulator's is `wide_long_range_proposal.SIM_TOSS_PARAMETERS`, the real swing's
    `REAL_TOSS_PARAMETERS`. Every draw, standoff band and refusal follows the set the
    environment carries, so a real run cannot draw a sim-only throw.
    """

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    standoff_bounds: tuple[float, float]
    speed_bounds: tuple[float, float]
    release_ms_bounds: tuple[float, float]

    def model_post_init(self, __context: object) -> None:
        for name, (low, high) in self._named():
            if not low < high:
                raise ValueError(f"{name} bounds must be an interval low < high, got {(low, high)}")

    def violation(self, *, params: tuple[float, float, float]) -> str | None:
        """Why `(standoff, speed, release_ms)` is outside this set, or `None`."""
        for (name, (low, high)), value in zip(self._named(), params, strict=True):
            if not low <= value <= high:
                return f"{name}={value} outside [{low}, {high}]"
        return None

    def _named(self) -> tuple[tuple[str, tuple[float, float]], ...]:
        return (
            ("standoff", self.standoff_bounds),
            ("speed", self.speed_bounds),
            ("release_ms", self.release_ms_bounds),
        )


class TossFeasibilityGeometry(BaseModel):
    """Geometry from one explicit state for a conservative toss-parameter check."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    robot_pose: tuple[float, float, float]
    robot_size: tuple[float, float]
    robot_state_dtype: Literal["float32", "float64"] = "float64"
    bin_pose: tuple[float, float, float]
    obstacles: tuple[PlanarCollisionBox, ...]
    sampling_x_bounds: tuple[float, float]
    sampling_y_bounds: tuple[float, float]


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


# The observed pickup-infeasibility marker the environment injects into a state's
# abstract atom set after a dispatched pick was REFUSED by the grasp planner. It is
# named here, in this module's data layer, because both the producer (the
# environment) and the consumer (predicates.PICKUP_UNBLOCKED) need the one name and
# importing either from the other would cycle.
KB_PICKUP_BLOCKED = "PickupBlocked"


class Tossing3DObservation(BaseModel):
    """One scene observation, flattened to plain floats -- a KINDER `ObjectCentricState`
    on the simulator backend, a robot `observe()` on the real one.

    This is the boundary type: every `Tossing3DBackend` produces it, `Tossing3DEnvironment`
    consumes it, and nothing about it requires KINDER to construct -- which is what lets
    a test build one by hand and exercise the translation offline.

    `goal_region` is the **live** `Region.bbox` of `blocks_goal_region`, read back from
    the compiled model rather than re-derived from the task JSON. The JSON range is
    inflated by `MujocoObject`'s per-object placement threshold (1cm per side) before it
    becomes a region, so the literal in the file is not the box that scores. Reading it
    back is what lets this domain's `InBin` agree with `_check_goals()` exactly. It
    reaches the state carried on the **bin** object -- the region is attached to the
    bin's own body (`blocks_goal_region.target` is `bin_0`), under this domain's
    assumption that the bin's interior contains this region -- see `predicates.py`'s
    module docstring. Because the region moves with the bin, this box is not fixed
    across seeds once `bin_init_region` samples a position: it tracks wherever the bin
    actually landed.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    features: dict[str, dict[str, float]]
    goal_region: tuple[float, float, float, float, float, float]
    solved: bool

    def get(self, *, name: str, feature: str) -> float:
        """One named feature of one named KINDER object, or a loud error.

        Loud because the alternative is a silent zero: every feature this package reads
        is named in `Tossing3DEnvironment`'s own `Type` declarations, so a miss means
        upstream renamed something and the translation is now wrong, not merely empty.
        """
        if name not in self.features:
            raise KeyError(f"no object named {name!r} in this observation: {sorted(self.features)}")
        obj = self.features[name]
        if feature not in obj:
            raise KeyError(f"object {name!r} has no feature {feature!r}: {sorted(obj)}")
        return obj[feature]


class ControllerRun(BaseModel):
    """What one upstream controller execution did.

    `steps` is how many `env.step` calls it took to terminate -- the quantity the
    validation record reports as `71 / 23 / 16 / 18` for the oracle's four controller
    executions, so it is returned rather than discarded.

    `error` is non-`None` when the controller raised. That is not exceptional: KINDER's
    motion planners `assert plan is not None`, so an unreachable grasp or an
    unplannable arm trajectory surfaces as an `AssertionError` from `reset()`. A
    `core.Environment.take_action` must be total over its action space, so the caller
    turns this into a no-op transition rather than propagating -- see
    `Tossing3DEnvironment.take_action`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    steps: int
    terminated: bool
    error: str | None = None


@runtime_checkable
class Tossing3DBackend(Protocol):
    """What `Tossing3DEnvironment` needs from whatever executes its skills.

    `KinderBackend` (the simulator) is one implementation and the real-robot backend is
    the other. Everything above this seam -- the state translation, the predicates, the
    skills, the methods -- sees only `Tossing3DObservation`s and abstract atom sets, so
    a backend swap changes where the numbers come from and nothing about what they
    mean. `snapshot()` is opaque: each backend's own handle, carried on the state as
    `Tossing3DState.object_centric` and read back only by that backend's own code.
    """

    def reset(self, *, seed: int) -> Tossing3DObservation: ...

    def close(self) -> None: ...

    def observe(self) -> Tossing3DObservation: ...

    def snapshot(self) -> Any: ...

    def restore(self, *, snapshot: Any) -> Tossing3DObservation: ...

    def abstract_atoms(self, *, state: Any = None) -> frozenset[AbstractAtom]: ...

    def abstraction_diagnostics(self, *, state: Any = None) -> dict[str, Any]: ...

    def check_goals(self) -> bool: ...

    def reset_cube_and_bin(
        self, *, bin_region: dict[str, object] | None = None
    ) -> Tossing3DObservation: ...

    def run_pick_cube(self) -> ControllerRun: ...

    def run_open_gripper(self) -> ControllerRun: ...

    def run_move_to_toss_location_and_toss(
        self,
        *,
        distance: float,
        rotation: float,
        release_speed_deg_s: float,
        gripper_release_ms: float,
    ) -> ControllerRun: ...

    def set_substep_recording(self, *, enabled: bool) -> None: ...

    def drain_substep_frames(self) -> list[np.ndarray]: ...

    def drain_substep_states(self) -> list[Any]: ...

    def set_control_step_observers(
        self, *, before: Callable[[], None] | None, after: Callable[[], None] | None
    ) -> None: ...

    def snapshot_to_plain(self, *, snapshot: Any) -> dict[str, list[float]]: ...

    def plain_to_snapshot(
        self, *, plain: dict[str, list[float]], dtype: type[np.floating] = np.float32
    ) -> Any: ...

    def live_state(self) -> Any: ...

    def render(self, *, follow_robot: bool = False) -> np.ndarray: ...

    def render_fps(self) -> int: ...
