"""Counter-height variant facade; original SweepOffShelf3D remains unchanged."""

from collections.abc import Callable
from typing import ClassVar

import numpy as np
from pydantic import Field, PrivateAttr

from hitl_pmp.core.problem.environment.types import Action, Object, State
from hitl_pmp.environments.pouring3d.kitchen.types import PickRefusal
from hitl_pmp.environments.sweepoffshelf3d.environment import SweepOffShelf3DEnvironment
from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import SkillFailed
from hitl_pmp.environments.sweepoffshelf3d.types import HumanEvent, SkillNote, SweepOffShelf3DState

from .backend import RackServingBackend
from .config import RackServingConfig
from .controllers import RackControllers
from .food import RackFoodController
from .objects import RackServing3DObjects as Objects
from .pick_plannable import RackPickPlannable
from .predicates import AbstractAtom, RackServing3DAtoms
from .skills import SKILL_IDS

HUMAN_RETURN_SETTLE_STEPS = 20


class RackServingEnvironment(SweepOffShelf3DEnvironment):
    config: RackServingConfig = Field(default_factory=RackServingConfig)
    scene_bg: bool = False
    robot: ClassVar[Object] = Objects.robot
    cubes: ClassVar[tuple[Object, ...]] = Objects.food
    wiper: ClassVar[Object] = Objects.scraper
    box: ClassVar[Object] = Objects.tray
    scene: ClassVar[Object] = Objects.scene

    _backend: RackServingBackend | None = PrivateAttr(default=None)
    _rack_controllers: RackControllers | None = PrivateAttr(default=None)
    # PickFood planning refusals by food name, kept for the life of this instance
    # (see `pick_plannable.py`).
    _pick_refusals: dict[str, PickRefusal] = PrivateAttr(default_factory=dict)
    # `two_route_rewards` only: the outcome facts of the last Sweep or GatherAndSweep,
    # taken as that skill returned and kept until the person's batch return or the next
    # episode (`RackServing3DAtoms.served_verdict`).
    _bulk_verdict: frozenset[AbstractAtom] = PrivateAttr(default=frozenset())

    def bulk_verdict(self) -> frozenset[AbstractAtom]:
        return self._bulk_verdict

    def backend(self) -> RackServingBackend:
        if self._backend is None:
            self._backend = RackServingBackend(config=self.config, scene_bg=self.scene_bg)
        return self._backend

    def rack_controllers(self) -> RackControllers:
        if self._rack_controllers is None:
            raise RuntimeError("reset the rack environment before requesting controllers")
        return self._rack_controllers

    def reset_to_seed(self, *, seed: int) -> SweepOffShelf3DState:
        backend = self.backend()
        backend.reset(seed=seed)
        self._bulk_verdict = frozenset()
        self._rack_controllers = RackControllers(backend=backend, step_limit=self.skill_step_limit)
        self.rack_controllers().initialize()
        backend._time_origin = float(backend.data().time)
        self._human_events = []
        self.current_state = self._observed_state(seed=seed, steps_taken=0)
        return self.current_state

    def _observed_state(self, *, seed: int, steps_taken: int) -> SweepOffShelf3DState:
        backend = self.backend()
        atoms = backend.abstract_atoms() | RackPickPlannable.atoms(
            backend=backend, refusals=self._pick_refusals
        )
        if self.config.two_route_rewards:
            atoms |= self._bulk_verdict
        features = backend.features()
        data = {obj: features[obj.name] for obj in Objects.all_objects()}
        data[Objects.scene] = np.array(
            [seed, steps_taken, backend.ledger.elapsed_seconds, backend.ledger.human_cost],
            dtype=float,
        )
        return SweepOffShelf3DState(data=data, abstract_atoms=atoms)

    def _bookkeeping(self) -> tuple[int, int]:
        state = self.get_current_state()
        return (
            int(state.get(obj=Objects.scene, feature_name="seed")),
            int(state.get(obj=Objects.scene, feature_name="steps_taken")),
        )

    def set_state(self, *, state: State) -> None:
        if state.get(obj=Objects.scene, feature_name="steps_taken") != 0:
            raise ValueError("only episode-initial states may be restored")
        self.reset_to_seed(seed=int(state.get(obj=Objects.scene, feature_name="seed")))

    def take_action(self, *, action: Action) -> SweepOffShelf3DState:
        seed, count = self._bookkeeping()
        values = np.asarray(action, dtype=float)
        self._last_skill_error = None
        self._last_action = values.copy()
        self._atoms_before = (
            getattr(self.get_current_state(), "abstract_atoms", None) or frozenset()
        )
        controllers = self.rack_controllers()
        controllers.shelf.ticks = 0
        backend = self.backend()
        start_control_steps = backend.control_steps()
        self._open_motion_skill(action=values)
        refused_from = self._pick_food_before(values=values)
        try:
            if values.shape != (5,) or not np.isfinite(values).all():
                raise SkillFailed("action must have five finite entries")
            skill_id, food_index = values[:2]
            if skill_id != int(skill_id) or food_index != int(food_index):
                raise SkillFailed("skill and food indices must be integers")
            if not 0 <= food_index < len(Objects.food):
                raise SkillFailed("food index outside the scene")
            name = Objects.food[int(food_index)].name
            if skill_id == SKILL_IDS["PickScraper"]:
                controllers.pick_scraper()
            elif skill_id == SKILL_IDS["GatherAndSweep"]:
                controllers.gather_and_sweep(
                    s=float(values[2]), delta=float(values[3]), d=float(values[4])
                )
            elif skill_id == SKILL_IDS["Sweep"]:
                if not self.config.sweep_skill:
                    raise SkillFailed("Sweep is not enabled (RackServingConfig.sweep_skill)")
                controllers.sweep(d=float(values[4]))
            elif skill_id == SKILL_IDS["ReturnScraper"]:
                controllers.return_scraper()
            elif skill_id == SKILL_IDS["PickFood"]:
                controllers.pick_food(name=name)
            elif skill_id == SKILL_IDS["PlaceFoodOnTray"]:
                controllers.place_food(name=name, destination="box_0")
            elif skill_id == SKILL_IDS["ReturnFoodToRack"]:
                controllers.place_food(name=name, destination="rack_0")
            elif skill_id == SKILL_IDS["RobotBatchReturn"]:
                if not self.config.robot_batch_return:
                    raise SkillFailed(
                        "RobotBatchReturn is not enabled (RackServingConfig.robot_batch_return)"
                    )
                plate_count = backend.plate_count()
                try:
                    controllers.robot_batch_return()
                finally:
                    # As the person's return withdraws a bulk skill's outcome; a return
                    # that failed before it took anything off the plate withdraws nothing.
                    if backend.plate_count() < plate_count:
                        self._bulk_verdict = frozenset()
            elif skill_id != -1:
                raise SkillFailed("unknown robot skill; human operations are not robot actions")
        except (SkillFailed, ValueError) as error:
            self._last_skill_error = str(error)
        finally:
            self._last_controller_steps = backend.control_steps() - start_control_steps
            controllers.shelf.ticks = self._last_controller_steps
        if refused_from is not None:
            name, before = refused_from
            refusal = RackPickPlannable.refusal(
                backend=backend,
                name=name,
                before=before,
                error=self._last_skill_error,
                steps=self._last_controller_steps,
            )
            if refusal is not None:
                self._pick_refusals[name] = refusal
        bulk = {SKILL_IDS[name]: name for name in ("Sweep", "GatherAndSweep")}
        if self.config.two_route_rewards and values.shape == (5,) and values[0] in bulk:
            self._bulk_verdict = RackServing3DAtoms.served_verdict(
                skill=bulk[int(values[0])], plate_count=backend.plate_count()
            )
        self.current_state = self._observed_state(seed=seed, steps_taken=count + 1)
        return self.current_state

    def _pick_food_before(self, *, values: np.ndarray) -> tuple[str, PickRefusal] | None:
        """The food and poses a dispatched PickFood starts from, else None."""
        if values.shape != (5,) or not np.isfinite(values).all():
            return None
        if values[0] != SKILL_IDS["PickFood"] or not 0 <= values[1] < len(Objects.food):
            return None
        name = Objects.food[int(values[1])].name
        return name, RackPickPlannable.before(backend=self.backend(), name=name)

    def forget_pick_refusals(self) -> None:
        """Start an evaluation task as a fresh environment would (`pick_plannable.py`)."""
        self._pick_refusals.clear()
        if self._rack_controllers is not None:
            seed, count = self._bookkeeping()
            self.current_state = self._observed_state(seed=seed, steps_taken=count)

    def noop_action(self) -> Action:
        """[skill ID, food index, s, delta, d]; -1 is the no-op skill ID."""
        return np.array([-1.0, 0.0, 0.0, 0.0, 0.0])

    def adopt_interrupted_skill(self, *, action: Action, reason: str) -> State:
        """Adopt measured state and grasp bookkeeping without issuing another control step."""
        seed, count = self._bookkeeping()
        self._last_action = np.asarray(action, dtype=float).copy()
        self._last_skill_error = f"interrupted: {reason}"
        backend = self.backend()
        controllers = self.rack_controllers()
        shelf = controllers.shelf
        backend.grip = shelf.grip
        controllers.held_food = next(
            (name for name in backend.cube_names if backend.held(name=name)), None
        )
        shelf.held_cube = controllers.held_food
        shelf.in_hand = (
            np.linalg.inv(shelf.site_pose()) @ controllers.tool.wiper_pose()
            if backend.held(name=Objects.scraper.name)
            else None
        )
        self.current_state = self._observed_state(seed=seed, steps_taken=count + 1)
        return self.current_state

    def reset_movables(self, *, destination: str | None = None) -> bool:
        backend = self.backend()
        seed, count = self._bookkeeping()
        if destination not in ("human_batch_return", "human_tool_recovery"):
            raise ValueError("unknown human intervention")
        interventions = backend.ledger.human_interventions
        moved: tuple[str, ...] = ()
        try:
            if destination == "human_batch_return":
                backend.batch_return()
                self._bulk_verdict = frozenset()
                moved = backend.cube_names
                RackFoodController(controllers=self.rack_controllers()).restore_gripper_force()
                self.rack_controllers().held_food = None
                self.rack_controllers().shelf.held_cube = None
                # The return places each dumpling just above the rack; until it settles
                # into contact, the measured state has it on nothing. The robot is held
                # still, as `initialize_rest` holds it after an episode's own placement.
                self.rack_controllers().shelf.hold(
                    count=HUMAN_RETURN_SETTLE_STEPS, label="human_batch_return_settle"
                )
            else:
                backend.sync_clock()
                backend.ledger = backend.ledger.charge_human(intervention="human_tool_recovery")
                if any(backend.held(name=name) for name in backend.cube_names):
                    raise RuntimeError("place held food before requesting human tool recovery")
                self.rack_controllers().tool.initialize_rest()
                backend.grip = self.rack_controllers().shelf.grip
                moved = ("wiper_0",)
        finally:
            if backend.ledger.human_interventions > interventions:
                self._human_events.append(
                    HumanEvent(skill=destination, moved=moved, steps_taken=count)
                )
                self.current_state = self._observed_state(seed=seed, steps_taken=count + 1)
        return True

    def last_skill_attempt(
        self,
    ) -> (
        tuple[np.ndarray, frozenset[AbstractAtom], frozenset[AbstractAtom], list[SkillNote]] | None
    ):
        if self._last_action is None:
            return None
        after = getattr(self.get_current_state(), "abstract_atoms", None) or frozenset()
        return self._last_action, self._atoms_before, after, []

    def set_control_step_observers(
        self, *, before: Callable[[], None] | None = None, after: Callable[[], None] | None = None
    ) -> None:
        self.backend().set_step_observers(before=before, after=after)

    def _open_motion_skill(self, *, action: np.ndarray) -> None:
        """As the shelf's `_begin_motion`, named from this domain's own skill IDs, so the
        motion ticks that follow belong to this action rather than the previous one."""
        self._motion_steps = 0
        if self._motion_sink is None or not action.size or not np.isfinite(action.flat[0]):
            return
        name = {index: skill for skill, index in SKILL_IDS.items()}.get(int(action.flat[0]))
        if name is not None:
            self._motion_sink.record_skill(name=name, objects=(), params=(), controller={})
