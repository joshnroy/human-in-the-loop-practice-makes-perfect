"""The real human: a person at a prompt, and the `HumanOracle` that asks them.

`TerminalScenePlacer` is the `RealScenePlacer` for a person: it says what to put where
and waits for Enter. Its input and output are injectable, so a test scripts the person.

`RealHumanOracle` is the real counterpart of `UnconditionalHumanOracle` -- same
pricing, same commands -- but it does not trust a reset blindly. After the person
confirms, the scene is re-observed and checked against what the reset skill promises
the planner (`Tossing3DSkillProvider._cube_bin_reset_groundings`' add effects): cube on
the floor on the robot's side, pickable, not held, the bin upright on the requested
side. Anything unmet is shown and the person asked again, up to `max_attempts`. If the
last attempt still fails, the run pauses and asks the person, through the same prompt:
continue anyway -- carrying on from the observed scene, since a planner replans from
what it observes -- or end the session (`HumanEndedSession`, which the step protocol
ends practice on). The shortfall and the answer are logged either way.

**Cost is accounted exactly as in the sim.** Nothing here charges anything: the harness
charges the reset skill's practice cost and records the intervention
(`step_protocol.py`, `Metrics.record_human_intervention`), as it does for the sim's
reset. What this adds is the real time the person took, logged per reset beside the
placements in the run's event log.
"""

import time
from collections.abc import Callable
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from hitl_pmp.core.problem.environment.environment import Environment
from hitl_pmp.core.problem.human.human import HumanEndedSession, HumanOracle
from hitl_pmp.core.problem.human.types import (
    CommandGoalDescription,
    CommandStartStateDescription,
    Cost,
)
from hitl_pmp.environments.tossing3d.predicates import (
    BIN_AT_SIDE,
    BIN_ON_GROUND,
    CUBE_AT_SIDE,
    HOLDING,
    ON_GROUND,
    PICK_PLANNABLE,
)
from hitl_pmp.environments.tossing3d.sides import Tossing3DSides
from hitl_pmp.humans.oracle import UnconditionalHumanOracle

from .environment import RealTossing3DEnvironment
from .types import PlacementRequest

_SIDE_WORDS = {"robot_side": "robot's side", "opposite_side": "far side"}


class TerminalScenePlacer(BaseModel):
    """Prompts a person to arrange the scene, and waits until they press Enter."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_fn: Callable[[str], str] = input
    output_fn: Callable[[str], None] = print

    def place(self, *, request: PlacementRequest) -> None:
        for line in TerminalScenePlacer.describe(request=request):
            self.output_fn(line)
        self.input_fn("Press Enter when the scene is set. ")

    def inspect_cube_in_bin(self, *, owner: str) -> bool:
        """Ask the person to look: the bin's walls can hide a cube right after a toss."""
        self.output_fn(f"[{owner}] The robot cannot see the cube after the toss, but sees the bin.")
        while True:
            answer = self.input_fn("Is the cube in the bin? [y/n] ").strip().lower()
            if answer in {"y", "yes"}:
                return True
            if answer in {"n", "no"}:
                return False

    def continue_after_unverified_reset(
        self, *, owner: str, unmet: tuple[str, ...], attempts: int
    ) -> bool:
        """Pause and ask: the reset failed its check every time it was tried."""
        self.output_fn(f"[{owner}] The reset still does not check out after {attempts} attempts:")
        for reason in unmet:
            self.output_fn(f"  NOT YET: {reason}")
        while True:
            answer = self.input_fn("Continue anyway, or end the session? [c/e] ").strip().lower()
            if answer in {"c", "continue"}:
                return True
            if answer in {"e", "end"}:
                return False

    @staticmethod
    def at(*, point: tuple[float, float] | None) -> str:
        return "anywhere" if point is None else f"near ({point[0]:.2f}, {point[1]:.2f})"

    @staticmethod
    def describe(*, request: PlacementRequest) -> list[str]:
        heading = {
            "scene": f"Set up the task scene (seed {request.seed}).",
            "reset": "Human reset requested by the robot.",
            "restore": f"The other run used the robot: put back the {request.owner} scene.",
        }[request.kind]
        lines = [f"[{request.owner}] {heading} The robot has gone home."]
        lines.append(
            f"  cube: upright on the floor, {TerminalScenePlacer.at(point=request.cube_target)}"
            + ("" if request.cube_region is None else f" (in {request.cube_region})")
        )
        if request.cube_avoid_y is not None:
            low, high = request.cube_avoid_y
            lines.append(
                f"    keep the cube out of map y in [{low:.2f}, {high:.2f}]: the ceiling "
                "cameras overlap there"
            )
        side = _SIDE_WORDS.get(request.destination or "", "")
        lines.append(
            f"  bin: upright, {TerminalScenePlacer.at(point=request.bin_target)}"
            + (f", on the {side} of the barrier" if side else "")
            + ("" if request.bin_region is None else f" (in {request.bin_region})")
        )
        lines.append("  gripper: empty; keep the robot's footprint clear.")
        if request.was_holding:
            lines.append(
                "  NOTE: the robot was holding the cube. It cannot be put back in the "
                "gripper -- place it on the floor where shown; this is logged."
            )
        lines.extend(f"  NOT YET: {note}" for note in request.notes)
        return lines


class RealHumanOracle(HumanOracle):
    """A person who resets the real scene, checked rather than trusted.

    A static-method container like every `HumanOracle`: the person, the prompt and the
    log all belong to the environment's `RealWorld`, which is handed in per call.
    """

    intervention_cost: ClassVar[Cost] = UnconditionalHumanOracle.intervention_cost
    max_attempts: ClassVar[int] = 3

    @staticmethod
    def calculate_cost_for_human_command(
        *,
        command_start_state_description: CommandStartStateDescription,
        command_goal_description: CommandGoalDescription,
    ) -> Cost:
        return UnconditionalHumanOracle.calculate_cost_for_human_command(
            command_start_state_description=command_start_state_description,
            command_goal_description=command_goal_description,
        )

    @staticmethod
    def execute_human_command(
        *,
        command_start_state_description: CommandStartStateDescription,
        command_goal_description: CommandGoalDescription,
        env: Environment,
    ) -> None:
        """A target state is a task's initial scene: `set_state` asks the person for it."""
        RealHumanOracle._real(env=env)
        UnconditionalHumanOracle.execute_human_command(
            command_start_state_description=command_start_state_description,
            command_goal_description=command_goal_description,
            env=env,
        )

    @staticmethod
    def execute_movables_reset(*, env: Environment, destination: str | None = None) -> None:
        real = RealHumanOracle._real(env=env)
        started = time.monotonic()
        unmet: list[str] = []
        attempts = 0
        for attempts in range(1, RealHumanOracle.max_attempts + 1):  # noqa: B007
            real.world.note_for_next_placement(notes=tuple(unmet))
            real.reset_movables(destination=destination)
            unmet = RealHumanOracle.unmet(env=real, destination=destination)
            if not unmet:
                break
        after_unverified = None
        if unmet:
            carry_on = real.world.placer.continue_after_unverified_reset(
                owner=real.owner, unmet=tuple(unmet), attempts=attempts
            )
            after_unverified = "continue" if carry_on else "end"
        real.world.log(
            event="human_reset",
            owner=real.owner,
            destination=destination,
            attempts=attempts,
            unmet=unmet,
            after_unverified=after_unverified,
            seconds=time.monotonic() - started,
        )
        if after_unverified == "end":
            raise HumanEndedSession(
                f"the reset failed its check {attempts} times ({'; '.join(unmet)}) "
                "and the person ended the session"
            )

    @staticmethod
    def unmet(*, env: RealTossing3DEnvironment, destination: str | None) -> list[str]:
        """Which of the reset skill's promises the observed scene does not keep."""
        state = env.get_current_state()
        robot, cube, bin_, barrier = env.robot, env.cube, env.bin, env.barrier
        checks = [
            (ON_GROUND.holds(state, (cube,)), "the cube is not upright on the floor"),
            (
                CUBE_AT_SIDE.holds(state, (cube, barrier, Tossing3DSides.robot)),
                "the cube is not on the robot's side, clear of the barrier",
            ),
            (not HOLDING.holds(state, (robot, cube)), "the robot is still holding the cube"),
            (BIN_ON_GROUND.holds(state, (bin_,)), "the bin is not seen upright"),
            (PICK_PLANNABLE.holds(state, (robot, cube)), "the robot cannot plan a pick"),
        ]
        if destination is not None:
            side = Tossing3DSides.parse(name=destination)
            checks.append((
                BIN_AT_SIDE.holds(state, (bin_, barrier, side)),
                f"the bin is not on the {_SIDE_WORDS[destination]} of the barrier",
            ))
        return [message for held, message in checks if not held]

    @staticmethod
    def _real(*, env: Environment) -> RealTossing3DEnvironment:
        if not isinstance(env, RealTossing3DEnvironment):
            raise TypeError(
                f"RealHumanOracle resets a RealTossing3DEnvironment, not {type(env).__name__}"
            )
        return env
