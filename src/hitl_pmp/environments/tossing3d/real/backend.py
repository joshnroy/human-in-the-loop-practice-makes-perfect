"""`RealTossingBackend`: the real TidyBot behind `Tossing3DBackend`.

`Tossing3DEnvironment` and everything above it -- predicates, skills, operator model,
the toss pipeline, both methods -- run unchanged on this backend. What it supplies is
what `KinderBackend` supplies for the simulator: one `observe()` reply becomes a
`Tossing3DObservation` in the sim's feature schema, in the domain frame
(`real/frame.py`), plus the abstract atoms the sim's classifiers would have produced;
and one skill is one robot call (`real/contract.py`).

## The predicate mapping

| atom | sim | real, from the reply |
| --- | --- | --- |
| `HandEmpty(robot)` | command within 1e-3 of open | `gripper_command` within 1e-3 of 0 |
| `Holding(robot, cube)` | closed, lifted, finger contact | `holding` |
| `OnGround(cube)` | resting on a face | seen, `cube_upright`, not held |
| `MovableInGoalRegion(cube)` | centre in the scored box | `cube_in_bin` |
| `MovableIsDownX(cube, barrier)` | `cube.x < barrier.x` | the same, in the domain frame |
| `BinOnGround(bin)` | upright and on the floor | a bin fit is present |
| `PickPlannable(robot, cube)` | gate, else a planner dry run | gate, else `can_pick()` |

The side atoms (`RobotAtSide`, `CubeAtSide`, `BinAtSide`) and the rest are the sim's own
predicates over the flat state. Anything not seen is NaN, never a guess, and no
predicate holds of a NaN.

## One skill is one robot call, charged its average sim tick count

`before` runs once before the robot moves (an exhausted budget stops the skill from
starting) and `after` once per charged step when it returns (`RealSkillStepCosts`). A
refused skill -- outside the parameter set, or refused by the robot side, which owns
every geometry and safety check -- moved nothing and is charged nothing.

## A human reset is a prompt

`reset` (a task's scene) and `reset_cube_and_bin` (the planner's paid reset) tell the
person where to put the cube and the bin, in map coordinates, wait for Enter, and
observe what was actually placed. The robot does not move first and nothing is checked
or retried: the planner sees the scene as placed.
"""

import math
from collections.abc import Callable
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from hitl_pmp.environments.tossing3d.bin_on_ground import KB_BIN_ON_GROUND
from hitl_pmp.environments.tossing3d.pick_plannable import KB_PICK_PLANNABLE, PickPlannableGate
from hitl_pmp.environments.tossing3d.sides import Tossing3DSide
from hitl_pmp.environments.tossing3d.types import (
    AbstractAtom,
    ControllerRun,
    FixedTossDirection,
    Tossing3DObservation,
    TossParameterSet,
)

from .contract import CanPickReply, ObserveReply, PickReply, TossingRobot, TossReply
from .frame import BIN_SPOTS, CUBE_SPOT, ROOM_FRAME, BinSpot, RealRoomFrame

ROBOT_NAME = "robot"
CUBE_NAME = "cube_0"
BIN_NAME = "bin_0"
BARRIER_NAME = "cuboid_barrier"

# Upstream's `GRIPPER_OPEN_COMMAND_TOLERANCE`, copied so this path imports no KINDER.
GRIPPER_OPEN_COMMAND_TOLERANCE = 1e-3
CUBE_HALF_SIZE_M = 0.025

_UNKNOWN = math.nan


class RealSkillStepCosts(BaseModel):
    """What one real skill costs against the step budget: its average sim tick count.

    Measured from the per-tick state logs of the 18 completed runs of
    `hitl-pmp-results/coded-sampler-20261014/full` (barrier layout, canonical seed 125),
    pooled over every execution and rounded: `PickCube` 118.53 (n=12695),
    `MoveToTossLocationAndToss` 90.06 (n=12651), `OpenGripper` 1.00 (n=43). A sim
    controller change moves these means; re-measure then.
    """

    model_config = ConfigDict(frozen=True)

    pick_cube: int = Field(default=119, gt=0)
    move_to_toss_location_and_toss: int = Field(default=90, gt=0)
    open_gripper: int = Field(default=1, gt=0)


class PlacementRequest(BaseModel):
    """Where a person is asked to put the cube and the bin (map frame)."""

    model_config = ConfigDict(frozen=True)

    kind: str
    destination: Tossing3DSide
    cube: tuple[float, float]
    bin: BinSpot

    def instructions(self) -> str:
        return "\n".join((
            f"[{self.kind}] Please place the cube and the bin (robot map frame, metres):",
            "  cube: anywhere on the floor on the robot's side of the tape, resting on a face",
            f"  bin:  centred at ({self.bin.center[0]:.2f}, {self.bin.center[1]:.2f}), "
            f"upright: {self.bin.description}",
            "  Do not move the robot.",
        ))


class TerminalPerson(BaseModel):
    """The person at this terminal: shown the request, answers with Enter."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    write: Callable[[str], None] = print
    read: Callable[[str], str] = input

    def place(self, *, request: PlacementRequest) -> None:
        self.write(request.instructions())
        self.read("Press Enter when the cube and the bin are in place: ")
        self.placed(request=request)

    def placed(self, *, request: PlacementRequest) -> None:
        """After Enter. A real person has already done it; a fake one does it here."""
        del request

    def confirm(self, *, action: str) -> None:
        """Before a skill moves the robot: anything but y ends the run, as Ctrl-C would."""
        if self.read(f"{action} -- run it? [y/N] ").strip().lower() != "y":
            raise KeyboardInterrupt(f"the person declined: {action}")


class RealSceneSnapshot(FixedTossDirection):
    """What a real state carries as `object_centric`: the reply it was built from."""

    model_config = ConfigDict(frozen=True)

    reply: ObserveReply
    can_pick: bool | None
    frame: RealRoomFrame

    def toss_azimuth(self) -> float | None:
        """The robot side's fixed direction for the bin's spot (from +x of the bin on
        the robot's side, from -y across the tape), turned into the domain frame."""
        if self.reply.bin is None or self.reply.toss_direction is None:
            return None
        return self.frame.yaw_to_domain(yaw=self.reply.toss_direction)


class RealTossingBackend(BaseModel):
    """`Tossing3DBackend` over a `TossingRobot`. See the module docstring."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    robot: TossingRobot
    person: Any
    parameters: TossParameterSet
    frame: RealRoomFrame = ROOM_FRAME
    step_costs: RealSkillStepCosts = RealSkillStepCosts()
    # The person says y at the terminal before each skill moves the robot.
    confirm_each_skill: bool = False

    _last: RealSceneSnapshot | None = PrivateAttr(default=None)
    _ticks: list[RealSceneSnapshot] = PrivateAttr(default_factory=list)
    _before: Callable[[], None] | None = PrivateAttr(default=None)
    _after: Callable[[], None] | None = PrivateAttr(default=None)

    # --- the scene, set by a person ---------------------------------------------------

    def reset(self, *, seed: int) -> Tossing3DObservation:
        """A task's scene: the robot at home and the bin across the tape, as in the sim.
        There is one spot per side, so `seed` names the same scene for every task."""
        del seed
        # The robot starts each task at its home, the close-side toss stand, as the
        # sim's does; it drives there before the person sets the scene around it.
        if self.confirm_each_skill:
            self.person.confirm(action="GoHome (drive to the close-side toss stand)")
        self.robot.go_home()
        return self._placed(kind="scene", destination=Tossing3DSide.OPPOSITE)

    def reset_cube_and_bin(
        self, *, bin_region: dict[str, object] | None = None
    ) -> Tossing3DObservation:
        """The human reset. `bin_region` is `RealTossing3DEnvironment.bin_reset_region`'s
        `{"destination": side}`; `None` is the task's own, across the tape."""
        destination = (
            Tossing3DSide.OPPOSITE
            if bin_region is None
            else Tossing3DSide(str(bin_region["destination"]))
        )
        return self._placed(kind="human reset", destination=destination)

    def _placed(self, *, kind: str, destination: Tossing3DSide) -> Tossing3DObservation:
        self.person.place(
            request=PlacementRequest(
                kind=kind, destination=destination, cube=CUBE_SPOT, bin=BIN_SPOTS[destination]
            )
        )
        return self.observe()

    # --- observation and abstraction ----------------------------------------------------

    def observe(self) -> Tossing3DObservation:
        reply = ObserveReply.model_validate(self.robot.observe())
        if reply.barrier is None:
            raise ValueError(
                "the robot reports no barrier: the real experiment needs the tape enabled "
                "in the skill server's config"
            )
        self._last = RealSceneSnapshot(
            reply=reply, can_pick=self._pick_verdict(reply=reply), frame=self.frame
        )
        return self.to_observation(snapshot=self._last)

    def to_observation(self, *, snapshot: RealSceneSnapshot) -> Tossing3DObservation:
        reply = snapshot.reply
        frame = snapshot.frame
        assert reply.barrier is not None
        robot_x, robot_y = frame.to_domain(x=reply.robot[0], y=reply.robot[1])
        if reply.cube is not None:
            cube_x, cube_y = frame.to_domain(x=reply.cube[0], y=reply.cube[1])
            cube_yaw = frame.yaw_to_domain(yaw=reply.cube[2])
        elif reply.holding:
            # In the gripper and out of view: it is where the robot is.
            cube_x, cube_y, cube_yaw = robot_x, robot_y, _UNKNOWN
        else:
            cube_x, cube_y, cube_yaw = _UNKNOWN, _UNKNOWN, _UNKNOWN
        resting = reply.cube is not None and reply.cube_upright and not reply.holding
        barrier_x, barrier_y = frame.to_domain(x=reply.barrier.x, y=reply.barrier.y)
        bin_ = reply.bin
        bin_x, bin_y = (_UNKNOWN, _UNKNOWN) if bin_ is None else frame.to_domain(x=bin_.x, y=bin_.y)
        return Tossing3DObservation(
            features={
                ROBOT_NAME: {
                    "pos_base_x": robot_x,
                    "pos_base_y": robot_y,
                    "pos_base_rot": frame.yaw_to_domain(yaw=reply.robot[2]),
                    "pos_gripper": reply.gripper_command,
                },
                CUBE_NAME: {
                    "x": cube_x,
                    "y": cube_y,
                    "z": CUBE_HALF_SIZE_M if resting else _UNKNOWN,
                    "qx": 0.0 if resting else _UNKNOWN,
                    "qy": 0.0 if resting else _UNKNOWN,
                    "qz": math.sin(cube_yaw / 2) if resting else _UNKNOWN,
                    "qw": math.cos(cube_yaw / 2) if resting else _UNKNOWN,
                    "bb_x": 2 * CUBE_HALF_SIZE_M,
                    "bb_y": 2 * CUBE_HALF_SIZE_M,
                    "bb_z": 2 * CUBE_HALF_SIZE_M,
                },
                BIN_NAME: {"x": bin_x, "y": bin_y, "z": _UNKNOWN if bin_ is None else 0.0},
                BARRIER_NAME: {
                    "x": barrier_x,
                    "y": barrier_y,
                    "z": 0.0,
                    "bb_x": reply.barrier.thickness,
                },
            },
            goal_region=self._bin_box(bin_x=bin_x, bin_y=bin_y, snapshot=snapshot),
            solved=reply.cube_in_bin,
        )

    @staticmethod
    def _bin_box(
        *, bin_x: float, bin_y: float, snapshot: RealSceneSnapshot
    ) -> tuple[float, float, float, float, float, float]:
        """The bin's axis-aligned outer bound in the domain frame, for the state's bin
        box features. Scoring is the robot side's `cube_in_bin`, not this box."""
        bin_ = snapshot.reply.bin
        if bin_ is None:
            return (_UNKNOWN,) * 6
        yaw = snapshot.frame.yaw_to_domain(yaw=bin_.yaw)
        cosine, sine = abs(math.cos(yaw)), abs(math.sin(yaw))
        half_x = (cosine * bin_.length + sine * bin_.width) / 2
        half_y = (sine * bin_.length + cosine * bin_.width) / 2
        return (bin_x - half_x, bin_y - half_y, 0.0, bin_x + half_x, bin_y + half_y, bin_.height)

    def _pick_verdict(self, *, reply: ObserveReply) -> bool | None:
        """`can_pick()`, only where the sim's `PickPlannable` gate would run its dry run."""
        if reply.holding or RealAtoms.cube_across_barrier(reply=reply, frame=self.frame):
            return None
        return CanPickReply.model_validate(self.robot.can_pick()).feasible

    def snapshot(self) -> RealSceneSnapshot:
        if self._last is None:
            raise RuntimeError("RealTossingBackend.reset() has not run yet; there is no scene.")
        return self._last

    def abstract_atoms(self, *, state: Any = None) -> frozenset[AbstractAtom]:
        subject = self.snapshot() if state is None else state
        assert isinstance(subject, RealSceneSnapshot)
        return RealAtoms.of(snapshot=subject)

    def abstraction_diagnostics(self, *, state: Any = None) -> dict[str, Any]:
        subject = self.snapshot() if state is None else state
        assert isinstance(subject, RealSceneSnapshot)
        return {"reply": subject.reply.model_dump(mode="json"), "can_pick": subject.can_pick}

    def check_goals(self) -> bool:
        return self._last is not None and self._last.reply.cube_in_bin

    # --- skills -------------------------------------------------------------------------

    def run_pick_cube(self) -> ControllerRun:
        def act() -> ControllerRun:
            reply = PickReply.model_validate(self.robot.pick())
            if reply.refused:
                return ControllerRun(
                    steps=0, terminated=True, error=f"pick refused: {reply.reason}"
                )
            error = None if reply.held else f"pick did not hold: {reply.note}"
            return ControllerRun(steps=self.step_costs.pick_cube, terminated=True, error=error)

        return self._skill(action="PickCube", act=act, cost=self.step_costs.pick_cube)

    def run_open_gripper(self) -> ControllerRun:
        def act() -> ControllerRun:
            self.robot.open_gripper()
            return ControllerRun(steps=self.step_costs.open_gripper, terminated=True)

        return self._skill(action="OpenGripper", act=act, cost=self.step_costs.open_gripper)

    def run_move_to_toss_location_and_toss(
        self,
        *,
        distance: float,
        rotation: float,
        release_speed_deg_s: float,
        gripper_release_ms: float,
    ) -> ControllerRun:
        """Throw from `distance` along `rotation`, the domain-frame azimuth bin -> stand
        that `Tossing3DToss` put in the action from `RealSceneSnapshot.toss_azimuth`."""
        refusal = self.parameters.violation(
            params=(distance, release_speed_deg_s, gripper_release_ms)
        )
        if refusal is not None:
            return ControllerRun(steps=0, terminated=True, error=f"toss refused: {refusal}")
        cost = self.step_costs.move_to_toss_location_and_toss

        def act() -> ControllerRun:
            reply = TossReply.model_validate(
                self.robot.toss(
                    standoff=distance,
                    speed=release_speed_deg_s,
                    release_ms=gripper_release_ms,
                    direction=self.frame.yaw_to_map(yaw=rotation),
                )
            )
            if not reply.tossed:
                return ControllerRun(
                    steps=0, terminated=True, error=f"toss not executed: {reply.note}"
                )
            return ControllerRun(steps=cost, terminated=True)

        action = (
            f"MoveToTossLocationAndToss(standoff {distance:.2f} m, "
            f"{release_speed_deg_s:.0f} deg/s, release {gripper_release_ms:.0f} ms)"
        )
        return self._skill(action=action, act=act, cost=cost)

    def _skill(self, *, action: str, act: Callable[[], ControllerRun], cost: int) -> ControllerRun:
        """One robot skill, one fresh observation, then its charged steps.

        A `ValueError` is the robot side refusing before it moved: uncharged. Any other
        error from the call (a stopped or busy skill, a lost connection) may have come
        mid-motion, so it is the action's failure, charged in full, never retried.
        """
        if self.confirm_each_skill:
            self.person.confirm(action=action)
        if self._before is not None:
            self._before()
        try:
            run = act()
        except ValueError as error:
            run = ControllerRun(steps=0, terminated=True, error=f"refused: {error}")
        except Exception as error:  # noqa: BLE001 -- recorded as the skill's failure
            run = ControllerRun(
                steps=cost, terminated=False, error=f"aborted: {type(error).__name__}: {error}"
            )
        # Observed before the steps are charged, so a goal check made at each charged
        # step (evaluation's) sees the scene the skill left, as a sim tick would.
        self.observe()
        self._ticks.append(self.snapshot())
        if self._after is not None:
            for _ in range(run.steps):
                self._after()
        return run

    # --- recording, and what a real room cannot do --------------------------------------

    def set_substep_recording(self, *, enabled: bool) -> None:
        del enabled

    def drain_substep_frames(self) -> list[np.ndarray]:
        return []

    def drain_substep_states(self) -> list[Any]:
        """One tick per skill: the scene observed when it returned."""
        ticks, self._ticks = self._ticks, []
        return ticks

    def set_control_step_observers(
        self, *, before: Callable[[], None] | None, after: Callable[[], None] | None
    ) -> None:
        self._before = before
        self._after = after

    def snapshot_to_plain(self, *, snapshot: Any) -> dict[str, list[float]]:
        assert isinstance(snapshot, RealSceneSnapshot)
        observation = self.to_observation(snapshot=snapshot)
        return {name: list(features.values()) for name, features in observation.features.items()}

    def live_state(self) -> Any:
        return self.snapshot()

    def restore(self, *, snapshot: Any) -> Tossing3DObservation:
        del snapshot
        raise NotImplementedError("the real room cannot be rewound to a snapshot")

    def plain_to_snapshot(
        self, *, plain: dict[str, list[float]], dtype: type[np.floating] = np.float32
    ) -> Any:
        del plain, dtype
        raise NotImplementedError("the real room cannot be rewound to a logged tick")

    def render(self, *, follow_robot: bool = False) -> np.ndarray:
        del follow_robot
        raise NotImplementedError("the real backend has no renderer")

    def render_fps(self) -> int:
        return 1

    def close(self) -> None:
        """Nothing to release: the robot is idle between skills."""


class RealAtoms:
    """The predicate mapping in the module docstring. A static-method container."""

    @staticmethod
    def of(*, snapshot: RealSceneSnapshot) -> frozenset[AbstractAtom]:
        reply = snapshot.reply
        atoms: set[AbstractAtom] = set()
        if math.isclose(reply.gripper_command, 0.0, abs_tol=GRIPPER_OPEN_COMMAND_TOLERANCE):
            atoms.add(("HandEmpty", (ROBOT_NAME,)))
        if reply.holding:
            atoms.add(("Holding", (ROBOT_NAME, CUBE_NAME)))
        if reply.cube is not None and reply.cube_upright and not reply.holding:
            atoms.add(("OnGround", (CUBE_NAME,)))
        if reply.cube_in_bin:
            atoms.add(("MovableInGoalRegion", (CUBE_NAME,)))
        if reply.cube is not None and reply.barrier is not None:
            cube_x, _ = snapshot.frame.to_domain(x=reply.cube[0], y=reply.cube[1])
            barrier_x, _ = snapshot.frame.to_domain(x=reply.barrier.x, y=reply.barrier.y)
            if cube_x < barrier_x:
                atoms.add(("MovableIsDownX", (CUBE_NAME, BARRIER_NAME)))
        if reply.bin is not None:
            atoms.add((KB_BIN_ON_GROUND, (BIN_NAME,)))
        if reply.holding:
            plannable = True
        elif RealAtoms.cube_across_barrier(reply=reply, frame=snapshot.frame):
            plannable = False
        else:
            plannable = bool(snapshot.can_pick)
        if plannable:
            atoms.add((KB_PICK_PLANNABLE, (ROBOT_NAME, CUBE_NAME)))
        return frozenset(atoms)

    @staticmethod
    def cube_across_barrier(*, reply: ObserveReply, frame: RealRoomFrame) -> bool:
        """The sim's `PickPlannableGate`, in the domain frame. An unseen cube is not
        known to be across, so its verdict is the dry run's."""
        if reply.cube is None or reply.barrier is None:
            return False
        cube_x, _ = frame.to_domain(x=reply.cube[0], y=reply.cube[1])
        robot_x, _ = frame.to_domain(x=reply.robot[0], y=reply.robot[1])
        barrier_x, _ = frame.to_domain(x=reply.barrier.x, y=reply.barrier.y)
        return PickPlannableGate.cube_across_barrier(
            robot_x=robot_x, cube={"x": cube_x}, barrier={"x": barrier_x}
        )
