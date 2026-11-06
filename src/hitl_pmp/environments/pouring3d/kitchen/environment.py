"""Opt-in kitchen variant. Episode initialization is never a practice action."""

from typing import Any, ClassVar, Literal

import numpy as np
from gymnasium.spaces import Box
from pydantic import Field, PrivateAttr

from hitl_pmp.core.problem.environment.types import Action, State
from hitl_pmp.environments.pouring3d.practice.environment import FrozenPourEnvironment
from hitl_pmp.environments.pouring3d.practice.skills import RESET_NAME as LEGACY_RESET
from hitl_pmp.environments.pouring3d.practice.skills import SKILL_NAMES
from hitl_pmp.environments.pouring3d.practice.types import FrozenPourState

from .config import KitchenGeometry
from .pick_plannable import KitchenPickPlannable
from .predicates import KitchenPredicates
from .types import ClosedLoopPourParameters, PickRefusal
from .types import KitchenObjects as O

HUMAN_RESET = "human_bulk_pour_back"


class KitchenEnvironment(FrozenPourEnvironment):
    payload_kind: Literal["potsticker"] = "potsticker"
    payload_count: Literal[3] = 3
    layout: Literal["stations"] = "stations"
    receiving_vessel: Literal["skillet"] = "skillet"
    geometry: KitchenGeometry = Field(default_factory=KitchenGeometry)
    # Dumpling geoms: governs dumpling-jug, dumpling-dumpling and worktop/floor contact.
    # Low so the pours can meter pieces out of the jug. Pads are an explicit 0.6 pair.
    surface_mu: float = Field(default=0.1, ge=0, allow_inf_nan=False)
    # Skillet geoms, at a contact priority above the dumplings: frozen dough on dry
    # seasoned cast iron, a judgement in 0.3-0.5, not a measurement.
    skillet_mu: float = Field(default=0.4, ge=0, allow_inf_nan=False)
    deployment: bool = False
    human_interruption_cost: float = Field(default=5, gt=0, allow_inf_nan=False)
    human_interventions: int = 0
    interruption_cost_total: float = 0
    episode_boundaries: list[dict[str, Any]] = Field(default_factory=list)
    _started: bool = PrivateAttr(default=False)
    # Planning refusals of dispatched dumpling picks, by dumpling name; kept for the
    # life of this instance (see `pick_plannable.py`).
    _pick_refusals: dict[str, PickRefusal] = PrivateAttr(default_factory=dict)
    # While a dumpling pick runs: its dumpling, starting poses and any finger contact.
    _pick_watch: dict[str, Any] | None = PrivateAttr(default=None)
    # Each dumpling's last failed pick that was not a refusal: its starting poses and error.
    _pick_failures: dict[str, PickRefusal] = PrivateAttr(default_factory=dict)
    # [skill, dumpling index, requested count, Pour ramp rate (deg/s), 0, 0]. Slots 4-5
    # held the open-loop pour's wiggle and duration; the closed-loop Pour fixes both.
    action_space: ClassVar[Box] = Box(
        low=np.array([0, 0, 1, 0, 0, 0], dtype=float),
        high=np.array(
            [len(SKILL_NAMES) - 1, 2, 3, ClosedLoopPourParameters.LIMITS[1], 0, 0], dtype=float
        ),
        dtype=np.float64,
    )

    def objects(self) -> O:
        return O(count=3, payload_kind="potsticker")

    def run_pour(self, *, controller: Any, parameters: np.ndarray, requested: int) -> dict:
        """The closed-loop Pour reads only its ramp rate, in native action slot 3."""
        if np.any(parameters[1:] != 0):
            raise ValueError(
                "The kitchen Pour reads only its ramp rate (slot 3); slots 4 and 5 must be 0"
            )
        return controller.run_closed_loop_pour(
            parameters=ClosedLoopPourParameters(ramp_rate_deg_s=parameters[0]),
            requested_count=requested,
        )

    def forget_pick_refusals(self) -> None:
        """Start a held-out task as a fresh instance would: no refusal remembered."""
        self._pick_refusals.clear()
        self._pick_failures.clear()

    def reset_to_seed(self, *, seed: int, requested: int) -> FrozenPourState:
        if self._started:
            raise ValueError(
                "No free in-task reset; use a costed cold reset or an explicit harness boundary"
            )
        if not 1 <= requested <= 3:
            raise ValueError("Invalid task count")
        from .native import KitchenController

        self.close()
        self._seed, self._steps, self.requested_count = seed, 0, requested
        self._controller = KitchenController.build(
            geometry=self.geometry,
            deployment=self.deployment,
            requested=requested,
            seed=seed,
            scene_bg=self.scene_bg,
            surface_mu=self.surface_mu,
            skillet_mu=self.skillet_mu,
        )
        self._controller.before_step = self.before_tick
        self._controller.after_step = self.after_tick
        self._controller.callback = self._motion_sink
        self.last_info = {}
        self.current_state = self.observe()
        return self.current_state

    def initialize_harness_episode(self, *, seed: int, requested: int) -> FrozenPourState:
        """Privileged test-harness boundary, never exposed as a skill/reset option."""
        self.episode_boundaries.append({
            "seed": seed,
            "requested": requested,
            "reason": "independent harness episode",
        })
        self._started = False
        return self.reset_to_seed(seed=seed, requested=requested)

    def observe(self) -> FrozenPourState:
        state = super().observe()
        c = self.controller()
        on = c.scene.cooking_observed()
        state.data[O.hob] = np.array([float(on), float(not on)])
        atoms = set(state.abstract_atoms)
        # Settled by the rest test the drop's settle wait and ready() use: KinDER's speed
        # rule alone misses a dumpling rocking in place, and without that Settled no skill
        # (PickPitcher, or the reset once every piece is in the jug) is applicable.
        if c.ready_settled():
            atoms |= {("Settled", (name,)) for name in c.scene.cube_names if name != c.held}
        atoms |= KitchenPickPlannable.atoms(controller=c, refusals=self._pick_refusals)
        state.abstract_atoms = KitchenPredicates.translate(atoms=frozenset(atoms))
        return state

    def after_tick(self) -> None:
        super().after_tick()
        watch = self._pick_watch
        if watch is not None and not watch["touched"]:
            watch["touched"] = bool(self.controller().finger_contacts(name=watch["name"]))

    def watched_pick(self, *, action: Action, index: int) -> State:
        """Dispatch a dumpling pick; remember it if it refused to plan (`pick_plannable`)."""
        c = self.controller()
        if not 0 <= index < len(c.scene.cube_names):
            return super().take_action(action=action)
        name = c.scene.cube_names[index]
        before = PickRefusal(
            dumpling=KitchenPickPlannable.pose(controller=c, name=name),
            pan=KitchenPickPlannable.pose(controller=c, name="tray"),
        )
        watch = self._pick_watch = {"name": name, "touched": False}
        try:
            state = super().take_action(action=action)
        finally:
            self._pick_watch = None
        error = self.last_info.get("error")
        refusal = KitchenPickPlannable.refusal(
            controller=c, name=name, before=before, error=error, touched=watch["touched"]
        ) or KitchenPickPlannable.repeated(
            controller=c, before=before, error=error, previous=self._pick_failures.get(name)
        )
        if refusal is None:
            if error is not None:
                self._pick_failures[name] = before.model_copy(update={"error": error})
            return state
        self._pick_failures.pop(name, None)
        self._pick_refusals[name] = refusal
        self.current_state = self.observe()
        return self.current_state

    def take_action(self, *, action: Action) -> State:
        values = np.asarray(action, dtype=float)
        self.controller()
        self._started = True
        if values.shape != (6,) or not np.isfinite(values).all():
            return super().take_action(action=action)
        skill_id, index = int(values[0]), int(values[1])
        picks = values[0] == skill_id and 0 <= skill_id < len(SKILL_NAMES)
        if picks and SKILL_NAMES[skill_id].startswith("PickShrimp"):
            return self.watched_pick(action=action, index=index)
        return super().take_action(action=action)

    def reset_movables(self, *, destination: str | None = None) -> bool:
        if self.deployment:
            raise ValueError("Human actions are forbidden in deployment")
        if destination not in (None, HUMAN_RESET):
            raise ValueError("Unknown human reset")
        if self.controller().scene.cooking_observed():
            raise ValueError("Repeated reset practice requires cold ingredients")
        self._started = True
        # The learner charges the same price, from the same costs file, through its
        # ordinary human action path. This ledger audits direct calls; it is not
        # charged twice.
        self.human_interventions += 1
        self.interruption_cost_total += self.human_interruption_cost
        self.last_info = {"skill": HUMAN_RESET, "success": False, "error": None}
        c = self.controller()
        food_only = (
            c.held is None
            and c.vessel_home(name="pitcher")
            and c.scene.cube_locations() != ["pitcher"] * self.payload_count
        )
        if food_only:
            result = super().reset_movables(destination=LEGACY_RESET)
        else:
            result = self.reset_from_any_state()
        self.last_info.update(
            skill=HUMAN_RESET,
            success=result,
            human_interventions=1,
            interruption_cost=self.human_interruption_cost,
        )
        return result

    def reset_from_any_state(self) -> bool:
        """The human reset from a state the food-only pour-back does not cover (Josh,
        2026-11-06): the human takes the jug or a dumpling from the hand and stands the
        jug at home, and returns the dumplings to it if any is out or the jug moved. The
        skillet is welded to the stove, so it is never displaced. What is already in
        place is not touched, so from the fully reset state this changes nothing. Same
        price, same success test (`ready()`) and zero robot ticks."""
        c = self.controller()
        jug_moved = c.held is not None or not c.vessel_home(name="pitcher")
        if jug_moved:
            c.human_takes_jug_home()
        if jug_moved or c.scene.cube_locations() != ["pitcher"] * self.payload_count:
            c.scene.respawn_payload_in_pitcher()
        self._steps += 1
        self.last_info = {
            "skill": LEGACY_RESET,
            "success": c.ready(),
            "error": None,
            "human_interventions": 1,
            "seconds": 0,
            "controller_steps": 0,
        }
        self.current_state = self.observe()
        return bool(self.last_info["success"])

    def finish_human_handoff(self) -> None:
        """Declare task completion before a human removes food between tasks."""
        done = {
            (f"Split{self.requested_count}", ()),
            ("HandEmpty", ("robot",)),
            ("PitcherHome", ("pitcher",)),
        }
        if not done <= self.observe().abstract_atoms:
            raise ValueError("Handoff requires completed deployment")
        self.episode_boundaries.append({"reason": "human removes cooked food between tasks"})
        self._started = False
