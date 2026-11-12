"""Two routes to one goal: no physics, a scripted learning curve per route."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import numpy as np
from gymnasium.spaces import Box
from pydantic import Field, PrivateAttr

from hitl_pmp.core.problem.environment.environment import Environment
from hitl_pmp.core.problem.environment.types import Action, Object, State, Type

ROUTE_A = "RouteA"
ROUTE_B = "RouteB"
RESET_A = "ResetA"
HUMAN_RESET_B = "human_reset_b"
# A robot skill's id is its index here; the human's reset is no robot action.
ROBOT_SKILL_NAMES = (ROUTE_A, ROUTE_B, RESET_A)
NOOP_ID = -1.0


class TwoRoutesEnvironment(Environment):
    """One task with two ways to finish it. Route A is learned quickly and the robot
    puts itself back after an attempt; Route B is learned slowly and only a human puts
    the robot back. Which route finished is part of the state, so a reward can depend
    on it.

    Learning is scripted, not simulated: a route attempted `n` times in practice
    succeeds with probability `min(competence_cap, start + rate * n)`. The attempt
    counts live here and not in `State`, because a practice state is what a human
    reset restores and a reset must not undo learning. An evaluation environment
    (`learns=False`) never counts its own attempts; it adopts practice's
    (`adopt_practice_attempts`), read from the log practice keeps, because evaluations
    run in another process.
    """

    route_a_start: float = Field(default=0.2, ge=0.0, le=1.0)
    route_a_rate: float = Field(default=0.10, ge=0.0)
    route_b_start: float = Field(default=0.1, ge=0.0, le=1.0)
    route_b_rate: float = Field(default=0.05, ge=0.0)
    competence_cap: float = Field(default=0.95, ge=0.0, le=1.0)
    # What finishing by each route is worth. The environment only carries them: a
    # task is solved or not, and the reward is read off which route solved it.
    route_a_reward: float = Field(default=1.0, gt=0.0)
    route_b_reward: float = Field(default=3.0, gt=0.0)
    seed: int = 0
    learns: bool = True
    attempts: dict[str, int] = Field(default_factory=lambda: {ROUTE_A: 0, ROUTE_B: 0})
    # One line per robot invocation, written by a practice environment.
    attempts_log: Path | None = None
    last_error: str | None = None

    robot_type: ClassVar[Type] = Type(
        name="robot", feature_names=("ready", "done", "via_a", "via_b", "after_a", "after_b")
    )
    robot: ClassVar[Object] = Object(name="robot", type=robot_type)
    action_space: ClassVar[Box] = Box(NOOP_ID, len(ROBOT_SKILL_NAMES) - 1, (1,))

    _rng: np.random.Generator = PrivateAttr()
    # An evaluation environment's run directory, where practice's attempt log lives.
    _practice_output: Path | None = PrivateAttr(default=None)
    _invocations: int = PrivateAttr(default=0)
    _before: Callable[[], None] | None = PrivateAttr(default=None)
    _after: Callable[[], None] | None = PrivateAttr(default=None)

    def model_post_init(self, __context: object) -> None:
        self._rng = np.random.default_rng(self.seed)

    def success_probability(self, *, skill: str) -> float:
        start, rate = {
            ROUTE_A: (self.route_a_start, self.route_a_rate),
            ROUTE_B: (self.route_b_start, self.route_b_rate),
        }[skill]
        return min(self.competence_cap, start + rate * self.attempts[skill])

    def action_for(self, *, skill: str) -> Action:
        if skill not in ROBOT_SKILL_NAMES:
            raise ValueError(f"Unknown skill: {skill}")
        return np.array([float(ROBOT_SKILL_NAMES.index(skill))])

    def build_state(self, **flags: bool) -> State:
        return State(
            data={
                self.robot: np.array([
                    float(flags.get(name, False)) for name in self.robot_type.feature_names
                ])
            }
        )

    def flag(self, *, name: str) -> bool:
        return bool(self.get_current_state().get(obj=self.robot, feature_name=name))

    def set_control_step_observers(
        self, *, before: Callable[[], None] | None, after: Callable[[], None] | None
    ) -> None:
        self._before, self._after = before, after

    def take_action(self, *, action: Action) -> State:
        skill_id = int(round(float(action[0])))
        if skill_id == int(NOOP_ID):
            return self.get_current_state()
        skill = ROBOT_SKILL_NAMES[skill_id]
        self._invocations += 1
        try:
            # A step the budget refuses raises here, before anything has changed.
            if self._before is not None:
                self._before()
            self.last_error = None
            if not self.flag(name="after_a" if skill == RESET_A else "ready"):
                self.last_error = f"{skill}: preconditions do not hold"
            elif skill == RESET_A:
                self.set_state(state=self.build_state(ready=True))
            else:
                succeeded = bool(self._rng.random() < self.success_probability(skill=skill))
                if self.learns:
                    self.attempts[skill] += 1
                suffix = skill[-1].lower()
                self.set_state(
                    state=self.build_state(**{
                        f"after_{suffix}": True,
                        f"via_{suffix}": succeeded,
                        "done": succeeded,
                    })
                )
                if not succeeded:
                    self.last_error = f"{skill}: attempt failed"
            if self._after is not None:
                self._after()
        finally:
            self.log_attempts()
        return self.get_current_state()

    def log_attempts(self) -> None:
        if self.attempts_log is None:
            return
        with self.attempts_log.open("a") as stream:
            stream.write(
                json.dumps({"invocation": self._invocations, "attempts": self.attempts}) + "\n"
            )

    def adopt_practice_attempts(self, *, log: Path, robot_invocations: int) -> None:
        """Take the attempt counts practice had reached after `robot_invocations` robot
        actions. The log may be growing: a last line still being written is skipped."""
        attempts = {ROUTE_A: 0, ROUTE_B: 0}
        if log.exists():
            for line in log.read_text().splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    break
                if record["invocation"] > robot_invocations:
                    break
                attempts = dict(record["attempts"])
        self.attempts = attempts

    def reset_movables(self, *, destination: str | None = None) -> bool:
        """The human's reset: back to ready, after a Route B attempt only."""
        if destination != HUMAN_RESET_B or not self.flag(name="after_b"):
            return False
        self.set_state(state=self.build_state(ready=True))
        return True

    def get_valid_actions(self) -> list[Action]:
        return [self.action_for(skill=skill) for skill in ROBOT_SKILL_NAMES]

    def noop_action(self) -> Action:
        return np.array([NOOP_ID])

    def hard_reset(self) -> None:
        self.set_state(state=self.build_state(ready=True))
