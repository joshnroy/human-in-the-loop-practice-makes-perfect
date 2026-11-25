"""The 3D Serving scene behind the chained toy's interface (`toy_interface.py`).

The rack scene, its controllers and the measured classifiers are the Serving
environment's own; what this class changes is the action set, the symbolic state and
the price. A planner action is one of the toy's skills:

* sweep3(d), gather(s, delta), sweep5(d): each picks the scraper up first when the
  hand does not already hold it (PickScraper's controller), then runs the split
  controllers' Sweep, Gather or Sweep (`RackControllers.sweep` / `.gather`).
* robot_reset: a held scraper is put home (ReturnScraper's controller), then every
  dumpling out of place is picked and set back on its slot (`RackRobotBatchReturn`).
* human_reset (a human skill, `reset_movables`): the person puts the scraper home and
  returns the food to a fresh draw of the start layout. The robot is never moved
  (Josh, 2026-11-25): getting it back to where its skills work is robot_reset's job.

Which atoms hold afterwards follows the toy (`ServingToyTransitions.after`); only
whether the skill succeeded is measured, after it returns
(`ServingToyTransitions.succeeded`). Only robot_reset is priced, once per execution:
its first control step is the one `cost_observation` marks `resetting`, which the
toy's own cost function (`hitl_pmp.environments.tworoutes.costs:robot_step`) charges
`robot_reset_price`."""

from typing import Any

import numpy as np
from pydantic import BaseModel, Field, PrivateAttr

from hitl_pmp.core.problem.environment.types import Action, State
from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import SkillFailed
from hitl_pmp.environments.sweepoffshelf3d.types import HumanEvent, SweepOffShelf3DState

from .environment import HUMAN_RETURN_SETTLE_STEPS, RackServingEnvironment
from .food import RackFoodController
from .objects import RackServing3DObjects as Objects
from .predicates import GATHERED, RackServing3DAtoms
from .robot_batch_return import RackRobotBatchReturn
from .toy_interface import (
    GATHER,
    HUMAN_RESET,
    ROBOT_RESET,
    SWEEP3,
    SWEEP5,
    TOY_SKILL_IDS,
    ServingToyTransitions,
)

TOY_SKILL_NAMES = {skill_id: name for name, skill_id in TOY_SKILL_IDS.items()}


class ServingToyMeasurement(BaseModel):
    plate_count: int
    gathered: bool
    full_rack: bool


class RackServingToyEnvironment(RackServingEnvironment):
    # What one robot_reset costs (the toy's `--robot-reset-price`); no other robot
    # skill is priced.
    robot_reset_price: float = Field(default=0.15, ge=0.0)
    # What serving three by sweep3, and five by sweep5, is worth (the toy's Route A
    # and Route B rewards).
    sweep3_reward: float = Field(default=3.0, gt=0.0)
    sweep5_reward: float = Field(default=4.0, gt=0.0)
    # Opt-in (Josh, 2026-11-24): gather succeeds only when, beyond the blade-band
    # Gathered classifier, it leaves the food clustered tightly enough for sweep5
    # (`RackServing3DAtoms.clustered_for_sweep`). None keeps the blade band alone.
    gather_success_half_width: float | None = Field(default=None, gt=0.0)
    gather_success_depth: float | None = Field(default=None, gt=0.0)

    _toy_flags: frozenset[str] = PrivateAttr(default_factory=ServingToyTransitions.initial)
    # The toy skill running now and the backend control-step count it started at:
    # `cost_observation` prices its first step.
    _toy_skill: str | None = PrivateAttr(default=None)
    _toy_skill_start: int = PrivateAttr(default=0)
    # The last skill's measured outcome, for logs.
    _toy_measurement: ServingToyMeasurement | None = PrivateAttr(default=None)

    def toy_flags(self) -> frozenset[str]:
        return self._toy_flags

    def toy_measurement(self) -> ServingToyMeasurement | None:
        return self._toy_measurement

    def running_toy_skill(self) -> str | None:
        return self._toy_skill

    def reset_to_seed(self, *, seed: int) -> SweepOffShelf3DState:
        self._toy_flags = ServingToyTransitions.initial()
        self._toy_skill = None
        self._toy_measurement = None
        return super().reset_to_seed(seed=seed)

    def _observed_state(self, *, seed: int, steps_taken: int) -> SweepOffShelf3DState:
        backend = self.backend()
        features = backend.features()
        data = {obj: features[obj.name] for obj in Objects.all_objects()}
        data[Objects.scene] = np.array(
            [seed, steps_taken, backend.ledger.elapsed_seconds, backend.ledger.human_cost],
            dtype=float,
        )
        return SweepOffShelf3DState(
            data=data,
            abstract_atoms=frozenset((name, (Objects.robot.name,)) for name in self._toy_flags),
        )

    def measure(self) -> ServingToyMeasurement:
        backend, controllers = self.backend(), self.rack_controllers()
        batch = RackRobotBatchReturn(controllers=controllers)
        return ServingToyMeasurement(
            plate_count=backend.plate_count(),
            gathered=(GATHERED.name, ()) in self._gathered() and self._clustered_for_sweep(),
            full_rack=all(batch.in_place(name=name) for name in backend.cube_names),
        )

    def _clustered_for_sweep(self) -> bool:
        if self.gather_success_half_width is None:
            return True
        backend, controllers = self.backend(), self.rack_controllers()
        geometry = controllers.stroke_geometry()
        return RackServing3DAtoms.clustered_for_sweep(
            unserved_corners=[
                backend.corners(name=name)
                for name in backend.cube_names
                if backend.location(name=name) != "OnTray"
            ],
            center_y=geometry["plate_center_y"],
            rack_front=geometry["rack_front"],
            half_width=self.gather_success_half_width,
            depth=self.gather_success_depth,
        )

    def cost_observation(self) -> dict[str, Any]:
        """A control step is `resetting` only when it is the first of a robot_reset."""
        first = (
            self._toy_skill == ROBOT_RESET
            and self.backend().control_steps() == self._toy_skill_start
        )
        return {"resetting": first, "robot_reset_price": self.robot_reset_price}

    def take_action(self, *, action: Action) -> SweepOffShelf3DState:
        seed, count = self._bookkeeping()
        values = np.asarray(action, dtype=float)
        current = self.get_current_state()
        assert isinstance(current, SweepOffShelf3DState)
        if values.shape == (5,) and values[0] == -1:
            return current
        self._last_skill_error = None
        self._last_action = values.copy()
        self._atoms_before = current.abstract_atoms or frozenset()
        backend, controllers = self.backend(), self.rack_controllers()
        controllers.shelf.ticks = 0
        start = backend.control_steps()
        skill = None
        if values.shape == (5,) and np.isfinite(values).all():
            skill = TOY_SKILL_NAMES.get(int(values[0])) if values[0] == int(values[0]) else None
        before = self._toy_flags
        applicable = skill is not None and ServingToyTransitions.applicable(
            skill=skill, flags=before
        )
        self._toy_skill, self._toy_skill_start = (skill if applicable else None), start
        self._open_motion_skill(action=values)
        try:
            if skill is None:
                raise SkillFailed("unknown robot skill; only the toy's skills are actions")
            if not applicable:
                raise SkillFailed(f"{skill}: preconditions do not hold")
            self._run(skill=skill, values=values)
        except (SkillFailed, ValueError) as error:
            self._last_skill_error = str(error)
        finally:
            self._last_controller_steps = backend.control_steps() - start
            controllers.shelf.ticks = self._last_controller_steps
            if applicable and skill is not None:
                self._finish(skill=skill, before=before)
            self._toy_skill = None
            self.current_state = self._observed_state(seed=seed, steps_taken=count + 1)
        return self.current_state

    def _finish(self, *, skill: str, before: frozenset[str]) -> None:
        """The toy's transition, its success measured from the scene as it is now."""
        self._toy_measurement = self.measure()
        succeeded = ServingToyTransitions.succeeded(skill=skill, measured=self._toy_measurement)
        self._toy_flags = ServingToyTransitions.after(
            skill=skill, succeeded=succeeded, flags=before
        )
        if not succeeded and self._last_skill_error is None:
            self._last_skill_error = f"{skill}: attempt failed"

    def _run(self, *, skill: str, values: np.ndarray) -> None:
        backend, controllers = self.backend(), self.rack_controllers()
        if skill == ROBOT_RESET:
            if backend.held(name=Objects.scraper.name):
                controllers.return_scraper()
            RackRobotBatchReturn(controllers=controllers).run()
            return
        if not backend.held(name=Objects.scraper.name):
            controllers.pick_scraper()
        if skill == GATHER:
            controllers.gather(s=float(values[2]), delta=float(values[3]))
        elif skill in (SWEEP3, SWEEP5):
            controllers.sweep(d=float(values[4]))

    def adopt_interrupted_skill(self, *, action: Action, reason: str) -> State:
        """The skill's transition was taken as it was cut off (`take_action`)."""
        error = self._last_skill_error
        state = super().adopt_interrupted_skill(action=action, reason=reason)
        if error is not None:
            self._last_skill_error = f"{error}; interrupted: {reason}"
        return state

    def reset_movables(self, *, destination: str | None = None) -> bool:
        """The human's reset, offered from any end (`NeedsReset`), mid-route included,
        as the toy offers it: the scraper home, then the food back from the start
        layout; the robot stays exactly where it is. It succeeds when the full rack is
        measured back; otherwise the toy's flags are left as they were."""
        if destination != HUMAN_RESET:
            raise ValueError(f"unknown human intervention: {destination!r}")
        if not ServingToyTransitions.applicable(skill=HUMAN_RESET, flags=self._toy_flags):
            return False
        backend, controllers = self.backend(), self.rack_controllers()
        seed, count = self._bookkeeping()
        if backend.held(name=Objects.scraper.name):
            controllers.tool.initialize_rest()
            backend.grip = controllers.shelf.grip
        backend.batch_return()
        RackFoodController(controllers=controllers).restore_gripper_force()
        controllers.held_food = None
        controllers.shelf.held_cube = None
        controllers.shelf.hold(count=HUMAN_RETURN_SETTLE_STEPS, label="human_reset_settle")
        self._finish(skill=HUMAN_RESET, before=self._toy_flags)
        self._last_skill_error = None
        self._human_events.append(
            HumanEvent(skill=HUMAN_RESET, moved=backend.cube_names, steps_taken=count)
        )
        self.current_state = self._observed_state(seed=seed, steps_taken=count + 1)
        return True

    def _open_motion_skill(self, *, action: np.ndarray) -> None:
        self._motion_steps = 0
        if self._motion_sink is None or not action.size or not np.isfinite(action.flat[0]):
            return
        name = TOY_SKILL_NAMES.get(int(action.flat[0]))
        if name is not None:
            self._motion_sink.record_skill(name=name, objects=(), params=(), controller={})
