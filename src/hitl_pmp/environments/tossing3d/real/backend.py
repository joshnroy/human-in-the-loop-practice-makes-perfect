"""`RealTossingBackend`: the real TidyBot behind `Tossing3DBackend`.

`Tossing3DEnvironment` and everything above it -- predicates, skills, the toss pipeline,
both methods -- run unchanged on this backend. What it supplies is the translation the
KINDER backend supplies for the simulator: one `observe()` reply becomes a
`Tossing3DObservation` in the sim's feature schema plus the abstract atom set the sim's
classifiers would have produced, and one skill becomes one robot call.

## The predicate mapping

Every atom the sim's `KinderBackend.abstract_atoms` emits for the barrier layout is
emitted here, from the contract's fields:

| atom | sim (upstream or local) | real, from the contract |
| --- | --- | --- |
| `HandEmpty(robot)` | command within 1e-3 of open | `gripper_command` within 1e-3 of 0 |
| `Holding(robot, cube)` | closed, lifted, finger contact | `holding` |
| `OnGround(cube)` | on a face: z at half height, tilt < 0.05 | seen, `cube_upright`, not held |
| `MovableInGoalRegion(cube)` | centre in the scored box | `cube_in_bin` |
| `MovableIsDownX(cube, barrier)` | `cube.x < barrier.x` | the same, on observed x |
| `BinOnGround(bin)` | upright and on the floor | a bin fit is present |
| `PickPlannable(robot, cube)` | gate, else a planner dry run | gate, else `can_pick()` |

`holding` is the robot side's: command closed, measured opening in the held band, and
the cube not on the floor. `cube_in_bin` is the cube's centre inside the bin interior
shrunk by a margin, so the scored region is strictly inside the bin, as in the sim. The
gate is the sim's `PickPlannableGate`: present while held, absent when the cube is
across the barrier from the robot, and only otherwise the dry run.

`OnGround` adds "not held" because the sim's height test is false for a lifted cube and
`cube_upright` alone does not say the cube is down. `CubeAtSide`/`BinAtSide`/
`RobotAtSide`, `GraspClear`, `NotHolding`, `ClosedEmpty` and `PickupUnblocked` are
computed by `predicates.py` from the flat state exactly as in the sim; this module only
has to put the right numbers in it (the barrier's `bb_x` is the rectangle's x extent).

## One skill is one robot call, and one control step

The step protocol counts control steps through the observers `set_control_step_observers`
installs. A real skill has no ticks to count, so each robot skill is exactly one step:
`before` runs before the robot moves -- so an exhausted budget stops a skill from
starting rather than cutting one short -- and `after` once it returns. Budgets for a real
run are therefore counted in skill executions.

## One robot, two environments

Practice and evaluation are separate `Problem`s, so evaluation cannot hand practice a
free reset. On the real robot they share one room. `RealWorld` is the shared handle:
when an environment acts after the other one has, the human is first asked to put back
the scene this environment last saw (`PlacementRequest(kind="restore")`). A restore of a
scene in which the robot held the cube cannot be done by hand, and is logged as such.
"""

import json
import logging
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr

from hitl_pmp.environments.tossing3d.bin_on_ground import KB_BIN_ON_GROUND
from hitl_pmp.environments.tossing3d.pick_plannable import KB_PICK_PLANNABLE, PickPlannableGate
from hitl_pmp.environments.tossing3d.types import (
    AbstractAtom,
    ControllerRun,
    Tossing3DObservation,
    TossParameterSet,
)

from .geometry import BARRIER_NAME, BIN_NAME, RealTossGeometry
from .types import (
    BinFit,
    CanPickReply,
    PickReply,
    PlacementRequest,
    PlanarRegion,
    RealSceneLayout,
    RealScenePlacer,
    RealSceneSnapshot,
    RobotObservation,
    RobotPose,
    TossingRobot,
    TossReply,
)

logger = logging.getLogger(__name__)

ROBOT_NAME = "robot"
CUBE_NAME = "cube_0"

# Upstream's `GRIPPER_OPEN_COMMAND_TOLERANCE` (kinder_models.dynamic3d.utils), copied so
# the real path imports no KINDER; `test_real_backend.py` pins it to upstream's.
GRIPPER_OPEN_COMMAND_TOLERANCE = 1e-3

_UNKNOWN = float("nan")


class RealWorld(BaseModel):
    """The one robot and the one room, shared by every real environment in a run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    robot: TossingRobot
    placer: RealScenePlacer
    # Every placement (scene, reset, restore) is appended here as one JSON line.
    event_log: Path | None = None

    _owner: str | None = PrivateAttr(default=None)
    _events: list[dict[str, Any]] = PrivateAttr(default_factory=list)
    _notes: tuple[str, ...] = PrivateAttr(default=())
    _generation: int = PrivateAttr(default=0)

    def owner(self) -> str | None:
        return self._owner

    def generation(self) -> int:
        """Bumped whenever anything may have moved: a robot skill or a placement. An
        observation taken at the current generation is still the scene, so it is not
        taken again -- `observe()` costs seconds on the real robot."""
        return self._generation

    def moved(self) -> None:
        self._generation += 1

    def events(self) -> list[dict[str, Any]]:
        return list(self._events)

    def enter(self, *, owner: str, restore: RobotObservation | None) -> None:
        """Make `owner` the environment acting on the robot.

        When another environment acted last and `owner` has a scene to go back to, the
        human restores it first. `restore=None` means the caller is about to rearrange
        the scene itself (a scene setup or a reset), so nothing is restored.
        """
        previous = self._owner
        self._owner = owner
        if previous is None or previous == owner or restore is None:
            return
        cube = restore.cube
        bin_ = restore.bin
        request = PlacementRequest(
            kind="restore",
            owner=owner,
            cube_target=None if cube is None else (cube.x, cube.y),
            bin_target=None if bin_ is None else (bin_.x, bin_.y),
            was_holding=restore.holding,
        )
        if restore.holding:
            logger.warning(
                "restoring %s's scene, in which the robot was holding the cube: a human "
                "cannot put it back in the gripper, so the restored scene differs",
                owner,
            )
        self.robot.go_home()
        self.moved()
        self.place(request=request, previous_owner=previous)

    def note_for_next_placement(self, *, notes: tuple[str, ...]) -> None:
        """Attach `notes` to the next placement request, for the person to read."""
        self._notes = notes

    def place(self, *, request: PlacementRequest, previous_owner: str | None = None) -> None:
        if self._notes:
            request = request.model_copy(update={"notes": self._notes})
            self._notes = ()
        started = time.monotonic()
        self.placer.place(request=request)
        self.moved()
        self.log(
            event="placement",
            **request.model_dump(mode="json"),
            previous_owner=previous_owner,
            seconds=time.monotonic() - started,
        )

    def log(self, **record: Any) -> None:
        record = {"time": time.time(), **record}
        self._events.append(record)
        if self.event_log is not None:
            self.event_log.parent.mkdir(parents=True, exist_ok=True)
            with self.event_log.open("a") as handle:
                handle.write(json.dumps(record) + "\n")


class RealTossingBackend(BaseModel):
    """`Tossing3DBackend` over a `TossingRobot`. See the module docstring."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    world: RealWorld
    layout: RealSceneLayout
    parameters: TossParameterSet
    owner: str = "practice"

    _last: RobotObservation | None = PrivateAttr(default=None)
    _last_generation: int | None = PrivateAttr(default=None)
    _can_pick: bool | None = PrivateAttr(default=None)
    # A person's answer about a cube the bin's walls hid after a toss; see `_hidden`.
    _hidden_in_bin: bool | None = PrivateAttr(default=None)
    _rng: np.random.Generator = PrivateAttr(default_factory=lambda: np.random.default_rng(0))
    _ticks: list[RealSceneSnapshot] = PrivateAttr(default_factory=list)
    _replies: list[dict[str, Any]] = PrivateAttr(default_factory=list)
    _before: Callable[[], None] | None = PrivateAttr(default=None)
    _after: Callable[[], None] | None = PrivateAttr(default=None)

    # --- scene setup ------------------------------------------------------------------

    def reset(self, *, seed: int) -> Tossing3DObservation:
        """Have the human set up the scene for task `seed`, then observe it.

        The seed fixes where the human is asked to put things -- the cube in the cube
        reset region, the bin across the barrier, as every sim task starts -- so a seed
        names the same requested scene on every run. What is observed afterwards is
        what the human actually did, which is what the state reports.
        """
        self._rng = np.random.default_rng(seed)
        request = self.scene_request(seed=seed)
        self.world.enter(owner=self.owner, restore=None)
        self.world.robot.go_home()
        self._hidden_in_bin = None
        self.world.place(request=request)
        return self.observe()

    def scene_request(self, *, seed: int) -> PlacementRequest:
        rng = np.random.default_rng(seed)
        return PlacementRequest(
            kind="scene",
            owner=self.owner,
            seed=seed,
            cube_target=self._draw(rng=rng, region=self.layout.cube_reset_region),
            cube_region=self.layout.cube_reset_region,
            bin_target=self._draw(rng=rng, region=self.layout.opposite_side_bin_region),
            bin_region=self.layout.opposite_side_bin_region,
        )

    def reset_cube_and_bin(
        self, *, bin_region: dict[str, object] | None = None
    ) -> Tossing3DObservation:
        """The human reset: robot home, cube back in its region, bin where asked.

        `bin_region` is a `PlanarRegion` dump (`RealTossing3DEnvironment.reset_movables`
        picks the side's region); `None` is the task's own, across the barrier.
        """
        region = (
            self.layout.opposite_side_bin_region
            if bin_region is None
            else PlanarRegion.model_validate(bin_region)
        )
        destination = None if bin_region is None else str(bin_region.get("destination", ""))
        self.world.enter(owner=self.owner, restore=None)
        self.world.robot.go_home()
        self._hidden_in_bin = None
        self.world.place(
            request=PlacementRequest(
                kind="reset",
                owner=self.owner,
                cube_target=self._draw(rng=self._rng, region=self.layout.cube_reset_region),
                cube_region=self.layout.cube_reset_region,
                bin_target=self._draw(rng=self._rng, region=region),
                bin_region=region,
                destination=destination or None,
            )
        )
        return self.observe()

    # --- observation and abstraction ----------------------------------------------------

    def observe(self) -> Tossing3DObservation:
        """The scene now: observed afresh only if something may have moved since."""
        self.world.enter(owner=self.owner, restore=self._last)
        if self._last is not None and self._last_generation == self.world.generation():
            return self.to_observation(observation=self._last)
        observation = RobotObservation.parse(reply=self.world.robot.observe())
        if observation.cube is not None or observation.holding or observation.bin is None:
            self._hidden_in_bin = None
        observation = self._hidden(observation=observation)
        self._last = observation
        self._last_generation = self.world.generation()
        self._can_pick = self._pick_verdict(observation=observation)
        return self.to_observation(observation=observation)

    def _hidden(self, *, observation: RobotObservation) -> RobotObservation:
        """A cube hidden by the bin's walls, as a person reported it.

        After a toss the robot side can see the bin but not the cube, and then reports
        `cube_in_bin` False with no cube. That is unknown, not a miss. Once a person has
        said (`inspect_cube_in_bin`), the answer stands for as long as the scene still
        looks that way: in the bin, the cube is at the bin's centre, upright and scored,
        as a scoring toss leaves it in the sim; not in the bin, it stays unseen, so no
        side and no pick apply and only the human reset recovers it -- as for a cube
        lost from view anywhere else.
        """
        if self._hidden_in_bin is None or observation.bin is None:
            return observation
        if not self._hidden_in_bin:
            return observation
        return observation.model_copy(
            update={
                "cube": RobotPose(x=observation.bin.x, y=observation.bin.y, yaw=0.0),
                "cube_upright": True,
                "cube_in_bin": True,
            }
        )

    def to_observation(self, *, observation: RobotObservation) -> Tossing3DObservation:
        """The sim's feature schema, filled from one reply. Unknown is NaN, never a guess:
        an unseen cube has no position, and no predicate is computed from a NaN."""
        robot = observation.robot
        cube = observation.cube
        if cube is not None:
            cube_x, cube_y, cube_yaw = cube.x, cube.y, cube.yaw
        elif observation.holding:
            # In the gripper and out of view: it is where the robot is.
            cube_x, cube_y, cube_yaw = robot.x, robot.y, _UNKNOWN
        else:
            cube_x, cube_y, cube_yaw = _UNKNOWN, _UNKNOWN, _UNKNOWN
        resting = cube is not None and observation.cube_upright and not observation.holding
        half = self.layout.cube_half_size_m
        qz, qw = (
            (math.sin(cube_yaw / 2), math.cos(cube_yaw / 2)) if resting else (_UNKNOWN, _UNKNOWN)
        )
        bin_ = observation.bin
        barrier = observation.barrier
        return Tossing3DObservation(
            features={
                ROBOT_NAME: {
                    "pos_base_x": robot.x,
                    "pos_base_y": robot.y,
                    "pos_base_rot": robot.yaw,
                    "pos_gripper": observation.gripper_command,
                },
                CUBE_NAME: {
                    "x": cube_x,
                    "y": cube_y,
                    "z": half if resting else _UNKNOWN,
                    "qx": 0.0 if resting else _UNKNOWN,
                    "qy": 0.0 if resting else _UNKNOWN,
                    "qz": qz,
                    "qw": qw,
                    "bb_x": 2 * half,
                    "bb_y": 2 * half,
                    "bb_z": 2 * half,
                },
                BIN_NAME: {
                    "x": _UNKNOWN if bin_ is None else bin_.x,
                    "y": _UNKNOWN if bin_ is None else bin_.y,
                    "z": _UNKNOWN if bin_ is None else 0.0,
                },
                BARRIER_NAME: {
                    "x": barrier.x,
                    "y": barrier.y,
                    "z": 0.0,
                    "bb_x": barrier.x_extent(),
                },
            },
            goal_region=self.scored_box(bin_=bin_),
            solved=observation.cube_in_bin,
        )

    def scored_box(self, *, bin_: BinFit | None) -> tuple[float, float, float, float, float, float]:
        """The axis-aligned bound of the bin's interior shrunk by the scoring margin."""
        if bin_ is None:
            return (_UNKNOWN,) * 6
        inset = self.layout.bin_wall_thickness_m + self.layout.scoring_margin_m
        half_length = max(bin_.length / 2 - inset, 0.0)
        half_width = max(bin_.width / 2 - inset, 0.0)
        cosine, sine = abs(math.cos(bin_.yaw)), abs(math.sin(bin_.yaw))
        extent_x = cosine * half_length + sine * half_width
        extent_y = sine * half_length + cosine * half_width
        return (
            bin_.x - extent_x,
            bin_.y - extent_y,
            0.0,
            bin_.x + extent_x,
            bin_.y + extent_y,
            bin_.height,
        )

    def snapshot(self) -> RealSceneSnapshot:
        if self._last is None:
            raise RuntimeError("RealTossingBackend.reset() has not run yet; there is no scene.")
        return RealSceneSnapshot(
            observation=self._last, layout=self.layout, can_pick=self._can_pick
        )

    def abstract_atoms(self, *, state: Any = None) -> frozenset[AbstractAtom]:
        subject = self.snapshot() if state is None else state
        assert isinstance(subject, RealSceneSnapshot)
        return RealAtoms.of(snapshot=subject)

    def abstraction_diagnostics(self, *, state: Any = None) -> dict[str, Any]:
        subject = self.snapshot() if state is None else state
        assert isinstance(subject, RealSceneSnapshot)
        return {
            "observation": subject.observation.model_dump(mode="json"),
            "can_pick": subject.can_pick,
            "replies": list(self._replies),
        }

    def check_goals(self) -> bool:
        return self._last is not None and self._last.cube_in_bin

    # --- skills -------------------------------------------------------------------------

    def run_pick_cube(self) -> ControllerRun:
        def act() -> ControllerRun:
            reply = PickReply.model_validate(self.world.robot.pick())
            self._replies.append({"skill": "pick", **reply.model_dump()})
            return ControllerRun(steps=1, terminated=True)

        return self._skill(act=act)

    def run_open_gripper(self) -> ControllerRun:
        def act() -> ControllerRun:
            self.world.robot.open_gripper()
            self._replies.append({"skill": "open_gripper"})
            return ControllerRun(steps=1, terminated=True)

        return self._skill(act=act)

    def run_move_to_toss_location_and_toss(
        self,
        *,
        distance: float,
        rotation: float,
        release_speed_deg_s: float,
        gripper_release_ms: float,
    ) -> ControllerRun:
        """Throw from the stand the shared selector chose, then retract the arm.

        Refused without moving -- and without charging a control step -- when a
        parameter is outside this run's parameter set or the bin is not localized; the
        robot's own `ValueError` is the backstop for the same rule.
        """
        refusal = self.parameters.violation(
            params=(distance, release_speed_deg_s, gripper_release_ms)
        )
        if refusal is None and (self._last is None or self._last.bin is None):
            refusal = "the bin is not localized"
        if refusal is not None:
            self._replies.append({"skill": "toss", "refused": refusal})
            return ControllerRun(steps=0, terminated=True, error=f"toss refused: {refusal}")
        assert self._last is not None and self._last.bin is not None
        direction = RealTossGeometry.direction_azimuth(
            bin_yaw=self._last.bin.yaw, rotation=rotation
        )

        def act() -> ControllerRun:
            try:
                raw = self.world.robot.toss(
                    standoff=distance,
                    speed=release_speed_deg_s,
                    release_ms=gripper_release_ms,
                    direction=direction,
                )
            except ValueError as error:
                self._replies.append({"skill": "toss", "refused": str(error)})
                return ControllerRun(steps=0, terminated=True, error=f"toss refused: {error}")
            reply = TossReply.model_validate(raw)
            self._replies.append({"skill": "toss", "direction": direction, **reply.model_dump()})
            if not reply.tossed:
                return ControllerRun(
                    steps=0, terminated=True, error=f"toss not executed: {reply.note}"
                )
            self.world.robot.retract()
            self.world.moved()
            self._inspect_if_hidden()
            return ControllerRun(steps=1, terminated=True)

        return self._skill(act=act)

    def _skill(self, *, act: Callable[[], ControllerRun]) -> ControllerRun:
        """One robot skill: one control step, then one fresh observation.

        The client's `SkillStopped` (a `stop()` interrupted the skill) and `SkillBusy`
        (another motion skill was running) are `RuntimeError`s. Either is the action's
        failure, recorded as its error -- never retried here. A stopped skill had
        started moving, so it is charged its step; a refused busy one is not.
        """
        self.world.enter(owner=self.owner, restore=self._last)
        self._replies = []
        if self._before is not None:
            self._before()
        generation = self.world.generation()
        try:
            run = act()
        except RuntimeError as error:
            name = type(error).__name__
            self._replies.append({"aborted": name, "message": str(error)})
            self.world.log(event="skill_aborted", owner=self.owner, error=name, message=str(error))
            run = ControllerRun(
                steps=0 if name == "SkillBusy" else 1,
                terminated=False,
                error=f"aborted: {name}: {error}",
            )
        if self.world.generation() == generation:
            self.world.moved()
        if self._after is not None and run.steps > 0:
            self._after()
        self.observe()
        self._ticks.append(self.snapshot())
        return run

    # --- recording and the parts of the protocol a real room cannot do -----------------

    def set_substep_recording(self, *, enabled: bool) -> None:
        del enabled

    def drain_substep_frames(self) -> list[np.ndarray]:
        return []

    def drain_substep_states(self) -> list[Any]:
        """One tick per skill: the scene observed when it returned."""
        ticks, self._ticks = self._ticks, []
        return list(ticks)

    def set_control_step_observers(
        self, *, before: Callable[[], None] | None, after: Callable[[], None] | None
    ) -> None:
        self._before = before
        self._after = after

    def snapshot_to_plain(self, *, snapshot: Any) -> dict[str, list[float]]:
        assert isinstance(snapshot, RealSceneSnapshot)
        observation = self.to_observation(observation=snapshot.observation)
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
        self.world.robot.stop()

    # --- helpers ------------------------------------------------------------------------

    def _inspect_if_hidden(self) -> None:
        """Ask a person whether a cube the robot cannot see after a toss is in the bin."""
        observation = RobotObservation.parse(reply=self.world.robot.observe())
        if observation.cube is None and not observation.holding and observation.bin is not None:
            answer = self.world.placer.inspect_cube_in_bin(owner=self.owner)
            self._hidden_in_bin = answer
            self.world.log(event="hidden_cube_inspected", owner=self.owner, in_bin=answer)
        else:
            self._hidden_in_bin = None
        # This observation is the scene after the toss; `observe()` reuses it.
        self._last = self._hidden(observation=observation)
        self._last_generation = self.world.generation()
        self._can_pick = self._pick_verdict(observation=self._last)

    def _pick_verdict(self, *, observation: RobotObservation) -> bool | None:
        """`can_pick()` only where the sim's `PickPlannable` gate would run its dry run."""
        if observation.holding:
            return None
        if RealAtoms.cube_across_barrier(observation=observation):
            return None
        reply = CanPickReply.model_validate(self.world.robot.can_pick())
        return reply.feasible

    @staticmethod
    def _draw(*, rng: np.random.Generator, region: PlanarRegion) -> tuple[float, float]:
        return (
            float(rng.uniform(region.x_min, region.x_max)),
            float(rng.uniform(region.y_min, region.y_max)),
        )


class RealAtoms:
    """The predicate mapping in the module docstring. A static-method container."""

    @staticmethod
    def of(*, snapshot: RealSceneSnapshot) -> frozenset[AbstractAtom]:
        observation = snapshot.observation
        atoms: set[AbstractAtom] = set()
        if math.isclose(observation.gripper_command, 0.0, abs_tol=GRIPPER_OPEN_COMMAND_TOLERANCE):
            atoms.add(("HandEmpty", (ROBOT_NAME,)))
        if observation.holding:
            atoms.add(("Holding", (ROBOT_NAME, CUBE_NAME)))
        cube = observation.cube
        if cube is not None and observation.cube_upright and not observation.holding:
            atoms.add(("OnGround", (CUBE_NAME,)))
        if observation.cube_in_bin:
            atoms.add(("MovableInGoalRegion", (CUBE_NAME,)))
        if cube is not None and cube.x < observation.barrier.x:
            atoms.add(("MovableIsDownX", (CUBE_NAME, BARRIER_NAME)))
        if observation.bin is not None:
            atoms.add((KB_BIN_ON_GROUND, (BIN_NAME,)))
        if observation.holding:
            plannable = True
        elif RealAtoms.cube_across_barrier(observation=observation):
            plannable = False
        else:
            plannable = bool(snapshot.can_pick)
        if plannable:
            atoms.add((KB_PICK_PLANNABLE, (ROBOT_NAME, CUBE_NAME)))
        return frozenset(atoms)

    @staticmethod
    def cube_across_barrier(*, observation: RobotObservation) -> bool:
        """The sim's `PickPlannableGate`, on observed x. An unseen cube is not known to
        be across, so its verdict is the dry run's, as in the sim for any open cube."""
        cube = observation.cube
        if cube is None:
            return False
        return PickPlannableGate.cube_across_barrier(
            robot_x=observation.robot.x,
            cube={"x": cube.x},
            barrier={"x": observation.barrier.x},
        )
