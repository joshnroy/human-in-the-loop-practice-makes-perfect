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
person, in one line, which spot the bin goes to (`SAME`, the robot's side, or `FAR`)
and that the cube goes anywhere on the robot's side; the coordinates go to the session
log only (`real/session_log.py`). They wait for y and observe what was actually placed.
Nothing is checked or retried: the planner sees the scene as placed. A human reset moves
no robot; a task's scene drives home first and opens the gripper, as the sim's reset
restores it open.

An evaluation in the middle of practice (`--measure-every-cycles`) uses the room, so
practice's scene is remembered first (`practice_scene`: the cube's map spot, or the
robot's if it was held, and the bin's side) and put back after
(`restore_practice_scene`): home and open as a task's scene does, then one
`restore practice:` prompt with the cube's spot. It is the person's job, not a reset the
method chose, and is charged nothing.
"""

import math
import time
from collections.abc import Callable
from typing import Any, Final

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
from .session_log import RealSessionLog

ROBOT_NAME = "robot"
CUBE_NAME = "cube_0"
BIN_NAME = "bin_0"
BARRIER_NAME = "cuboid_barrier"

# Upstream's `GRIPPER_OPEN_COMMAND_TOLERANCE`, copied so this path imports no KINDER.
GRIPPER_OPEN_COMMAND_TOLERANCE = 1e-3
# The only answers a y/n prompt takes; anything else is asked again.
_ANSWERS: Final[dict[str, bool]] = {"y": True, "yes": True, "n": False, "no": False}
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


class PracticeProgress(BaseModel):
    """Where practice is as an action starts (`StepProtocolEnvironment.practice_progress`)."""

    model_config = ConfigDict(frozen=True)

    cycle: int
    action: int
    max_actions: int
    steps: int
    budget: int

    def line(self, *, width: int = 10) -> str:
        """`cycle 0 [######----] action 13/20 · steps 1050/1500`: the bar is the cycle."""
        filled = min(width, width * self.action // max(self.max_actions, 1))
        bar = "#" * filled + "-" * (width - filled)
        return (
            f"cycle {self.cycle} [{bar}] action {self.action}/{self.max_actions} · "
            f"steps {self.steps}/{self.budget}"
        )


class EvaluationProgress(BaseModel):
    """Which evaluation is running, and its task (`StepProtocolEnvironment.evaluation_progress`)."""

    model_config = ConfigDict(frozen=True)

    index: int
    # The learning cycle whose update it measures; `None` for the run's final one.
    after_cycle: int | None
    task: int
    num_tasks: int

    def line(self) -> str:
        """`eval 3 after cycle 3: task 2/5`, or `eval 4 (final): task 1/5`."""
        when = "(final)" if self.after_cycle is None else f"after cycle {self.after_cycle}"
        return f"eval {self.index} {when}: task {self.task}/{self.num_tasks}"


class PracticeScene(BaseModel):
    """Practice's scene as last observed, kept while an in-process evaluation uses the
    room, so the person can put it back (`RealTossingBackend.restore_practice_scene`).
    Plain data in the map frame, so it can be logged and saved."""

    model_config = ConfigDict(frozen=True)

    # The cube's map position; where the robot was when it was holding it; `None` if
    # the last look saw no cube.
    cube: tuple[float, float] | None
    holding: bool
    # The bin's side of the tape (SAME or FAR), `None` if the last look saw no bin.
    bin_side: Tossing3DSide | None

    @staticmethod
    def of(*, snapshot: "RealSceneSnapshot") -> "PracticeScene":
        reply, frame = snapshot.reply, snapshot.frame
        cube: tuple[float, float] | None = None
        if reply.holding:
            cube = (reply.robot[0], reply.robot[1])
        elif reply.cube is not None:
            cube = (reply.cube[0], reply.cube[1])
        side = None
        if reply.bin is not None:
            bin_x, _ = frame.to_domain(x=reply.bin.x, y=reply.bin.y)
            side = Tossing3DSide.ROBOT if bin_x < 0.0 else Tossing3DSide.OPPOSITE
        return PracticeScene(cube=cube, holding=reply.holding, bin_side=side)


# The bin's spot as the person is told it: on the robot's side of the tape, or across.
SPOT_NAMES: Final[dict[Tossing3DSide, str]] = {
    Tossing3DSide.ROBOT: "SAME",
    Tossing3DSide.OPPOSITE: "FAR",
}


class PlacementRequest(BaseModel):
    """Where a person is asked to put the cube and the bin (map frame)."""

    model_config = ConfigDict(frozen=True)

    kind: str
    destination: Tossing3DSide
    cube: tuple[float, float]
    bin: BinSpot
    # Said when the cube may still be in the gripper (`RealTossingBackend.reset`).
    gripper_note: str | None = None
    # The cube goes at `cube` exactly (a restore), not anywhere on the robot's side.
    cube_exact: bool = False

    def instructions(self) -> str:
        """One line: the bin's spot by name, and the cube anywhere on the robot's side
        (or, for a restore, at its spot)."""
        spot = "the close spot, our side of the tape"
        if self.destination is Tossing3DSide.OPPOSITE:
            spot = "the far spot, across the tape"
        note = "" if self.gripper_note is None else f"; {self.gripper_note}"
        cube = "anywhere"
        if self.cube_exact:
            cube = f"at ({self.cube[0]:.2f}, {self.cube[1]:.2f})"
        return (
            f"{self.kind}: {SPOT_NAMES[self.destination]} (bin at {spot}); "
            f"cube {cube} on our side, on a face; do not move the robot{note}"
        )

    def details(self) -> str:
        """The coordinates, for the session log only."""
        return (
            f"{self.kind} {SPOT_NAMES[self.destination]} (robot map frame, metres): bin "
            f"centred at ({self.bin.center[0]:.2f}, {self.bin.center[1]:.2f}), yaw "
            f"{self.bin.yaw:.2f}: {self.bin.description}; cube nominally at "
            f"({self.cube[0]:.2f}, {self.cube[1]:.2f})"
        )


class TerminalPerson(BaseModel):
    """The person at this terminal: shown each request, answers y. Every line and
    prompt goes through the session log (`RealSessionLog`), timestamped and flushed."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    write: Callable[[str], None] = RealSessionLog.say
    read: Callable[[str], str] = RealSessionLog.ask

    def place(self, *, request: PlacementRequest) -> None:
        RealSessionLog.detail(line=request.details())
        self.write(request.instructions())
        self._yes(question="Are the cube and the bin in place?", declined=request.kind)
        self.write("observing the scene (the cameras take up to ~30 s) ...")
        self.placed(request=request)

    def placed(self, *, request: PlacementRequest) -> None:
        """After y. A real person has already done it; a fake one does it here."""
        del request

    def confirm(self, *, action: str) -> None:
        """Before a skill moves the robot (see `_yes`)."""
        self._yes(question=f"{action} -- run it?", declined=action)

    def _yes(self, *, question: str, declined: str) -> None:
        """y (or yes) goes on; n (or no) ends the run, as Ctrl-C would. Anything else,
        a stray Enter or a typo, asks again: only a clear answer moves or stops."""
        while (answer := self.read(f"{question} [y/n] ").strip().lower()) not in _ANSWERS:
            if answer:
                self.write(f"{answer!r}: please type y or n.")
        if not _ANSWERS[answer]:
            raise KeyboardInterrupt(f"the person declined: {declined}")


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
    # The person says y at the terminal before the robot drives home at a task's start
    # (each placement is a y too); with `confirm_each_skill`, also before every skill.
    confirm_start: bool = False
    confirm_each_skill: bool = False

    _last: RealSceneSnapshot | None = PrivateAttr(default=None)
    # The last look, reused until a skill or a placement changes the scene: a look is
    # ~18 s of camera work on the robot.
    _seen: Tossing3DObservation | None = PrivateAttr(default=None)
    _ticks: list[RealSceneSnapshot] = PrivateAttr(default_factory=list)
    _before: Callable[[], None] | None = PrivateAttr(default=None)
    _after: Callable[[], None] | None = PrivateAttr(default=None)
    _progress: PracticeProgress | EvaluationProgress | None = PrivateAttr(default=None)

    # --- the scene, set by a person ---------------------------------------------------

    def reset(self, *, seed: int) -> Tossing3DObservation:
        """A task's scene: the robot at home and the bin across the tape, as in the sim.
        There is one spot per side, so `seed` names the same scene for every task."""
        del seed
        self._say_progress()
        self._home_and_open()
        return self._placed(
            kind="scene", destination=Tossing3DSide.OPPOSITE, gripper_note=self._gripper_note()
        )

    def _home_and_open(self) -> None:
        # The robot starts each task at its home, the close-side toss stand, as the
        # sim's does; it drives there before the person sets the scene around it.
        if self.confirm_start:
            self.person.confirm(action="GoHome (drive to the close-side toss stand)")
        self._say(line="driving home ...")
        self.robot.go_home()
        # The sim's scene reset restores the gripper open. Practice can end on a pick
        # with the cube held, and a gripper still commanded closed leaves `HandEmpty`
        # false for the whole task (run 9's evaluation: no plan at step 0).
        self._say(line="opening the gripper ...")
        self.robot.open_gripper()

    def practice_scene(self) -> PracticeScene:
        """Practice's scene as last observed, to put back after an evaluation."""
        return PracticeScene.of(snapshot=self.snapshot())

    def restore_practice_scene(self, *, scene: PracticeScene) -> Tossing3DObservation:
        """After an in-process evaluation: home and open, as a task's scene starts, then
        the person puts the cube and the bin back as practice left them. The person's
        job, not a reset the method chose: nothing is charged."""
        self._home_and_open()
        notes = []
        if scene.holding:
            notes.append(
                "the robot was holding the cube: take it out of the gripper and put it on "
                "the floor there, where the robot was"
            )
        elif scene.cube is None:
            notes.append("the last look saw no cube: put it where it was")
        if scene.bin_side is None:
            notes.append("the last look saw no bin: put it at the far spot")
        return self._placed(
            kind="restore practice",
            destination=scene.bin_side or Tossing3DSide.OPPOSITE,
            cube=scene.cube,
            gripper_note="; ".join(notes) or None,
        )

    def _gripper_note(self) -> str | None:
        """What to tell the person about a cube that may still be in the gripper. A
        backend that has not looked yet (evaluation's, built after practice ended) does
        not know, so it asks the person to check."""
        if self._last is None:
            return "if the cube is still in the gripper, take it out"
        if self._last.reply.holding:
            return "the robot was holding the cube: take it out of the gripper"
        return None

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
        self._say_progress()
        return self._placed(kind="human reset", destination=destination)

    def _placed(
        self,
        *,
        kind: str,
        destination: Tossing3DSide,
        gripper_note: str | None = None,
        cube: tuple[float, float] | None = None,
    ) -> Tossing3DObservation:
        """`cube` is a restore's exact spot; otherwise the cube goes anywhere on the
        robot's side (nominally `CUBE_SPOT`)."""
        self.person.place(
            request=PlacementRequest(
                kind=kind,
                destination=destination,
                cube=CUBE_SPOT if cube is None else cube,
                bin=BIN_SPOTS[destination],
                gripper_note=gripper_note,
                cube_exact=cube is not None,
            )
        )
        self._seen = None
        observation = self.observe()
        self._say(line="planning the next skill ...")
        return observation

    # --- observation and abstraction ----------------------------------------------------

    def observe(self) -> Tossing3DObservation:
        if self._seen is not None:
            return self._seen
        reply = ObserveReply.model_validate(self.robot.observe())
        if reply.barrier is None:
            raise ValueError(
                "the robot reports no barrier: the real experiment needs the tape enabled "
                "in the skill server's config"
            )
        self._last = RealSceneSnapshot(
            reply=reply, can_pick=self._pick_verdict(reply=reply), frame=self.frame
        )
        self._seen = self.to_observation(snapshot=self._last)
        return self._seen

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
        if reply.can_pick is not None:
            return reply.can_pick.feasible
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
        self._say_progress()
        if self.confirm_each_skill:
            self.person.confirm(action=action)
        self._say(line=f"running {action} ...")
        started = time.monotonic()
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
        outcome = "ok" if run.error is None else run.error
        self._say(line=f"{action}: {outcome} ({time.monotonic() - started:.0f} s); observing ...")
        self._seen = None
        self.observe()
        self._say(line="planning the next skill ...")
        self._ticks.append(self.snapshot())
        if self._after is not None:
            for _ in range(run.steps):
                self._after()
        return run

    def practice_progress(
        self, *, cycle: int, action: int, max_actions: int, steps: int, budget: int
    ) -> None:
        """Practice's place as its next action starts, shown before that action runs.
        Only practice's backend is told; evaluation's is told `evaluation_progress`."""
        self._progress = PracticeProgress(
            cycle=cycle, action=action, max_actions=max_actions, steps=steps, budget=budget
        )

    def evaluation_progress(
        self, *, index: int, after_cycle: int | None, task: int, num_tasks: int
    ) -> None:
        """Which evaluation and task is running, shown before its scene and each skill."""
        self._progress = EvaluationProgress(
            index=index, after_cycle=after_cycle, task=task, num_tasks=num_tasks
        )

    def _say_progress(self) -> None:
        if self._progress is not None:
            self._say(line=self._progress.line())
            self._progress = None

    def _say(self, *, line: str) -> None:
        """A progress line for the person, at once, so a slow step never looks hung."""
        write = getattr(self.person, "write", None)
        if write is not None:
            write(line)

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
