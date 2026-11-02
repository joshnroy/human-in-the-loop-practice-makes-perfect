"""Opt-in kitchen lifecycle. Episode initialization is never a practice action."""

from collections import OrderedDict
from typing import Any, ClassVar, Literal

import numpy as np
from gymnasium.spaces import Box
from pydantic import Field, PrivateAttr

from hitl_pmp.core.problem.environment.types import Action, State
from hitl_pmp.environments.pouring3d.practice.environment import FrozenPourEnvironment
from hitl_pmp.environments.pouring3d.practice.skills import RESET_NAME as LEGACY_RESET
from hitl_pmp.environments.pouring3d.practice.types import FrozenPourState
from hitl_pmp.environments.pouring3d.types import PayloadPickPlan

from .config import KitchenGeometry
from .pick_plannable import KitchenPickPlannable
from .types import KitchenObjects as O

HUMAN_RESET = "human_bulk_pour_back"


class KitchenEnvironment(FrozenPourEnvironment):
    payload_kind: Literal["potsticker"] = "potsticker"
    payload_count: Literal[3] = 3
    layout: Literal["stations"] = "stations"
    receiving_vessel: Literal["skillet"] = "skillet"
    geometry: KitchenGeometry = Field(default_factory=KitchenGeometry)
    # Frozen dough on dry seasoned cast iron, a judgement in 0.3-0.5, not a measurement.
    # The legacy 0.1 is ice-like. Pad-dumpling friction is a separate explicit 0.6 pair.
    surface_mu: float = Field(default=0.4, ge=0, allow_inf_nan=False)
    deployment: bool = False
    human_interruption_cost: float = Field(default=5, gt=0, allow_inf_nan=False)
    human_interventions: int = 0
    interruption_cost_total: float = 0
    episode_boundaries: list[dict[str, Any]] = Field(default_factory=list)
    _started: bool = PrivateAttr(default=False)
    _pick_plans: OrderedDict[tuple[int, str, bytes], PayloadPickPlan] = PrivateAttr(
        default_factory=OrderedDict
    )
    action_space: ClassVar[Box] = Box(
        low=np.array([0, 0, 1, 0, 0, 0], dtype=float),
        high=np.array([9, 2, 3, 130, 0.025, 60], dtype=float),
        dtype=np.float64,
    )

    def objects(self) -> O:
        return O(count=3, payload_kind="potsticker")

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
        atoms.add(("HobOn" if on else "HobOff", ("hob",)))
        if not on:
            atoms.add(("ColdIngredients", ()))
        atoms |= KitchenPickPlannable.atoms(controller=c, cache=self._pick_plans)
        away = c.base_home[:2].copy()
        away[0] -= self.geometry.retreat_distance_m
        if np.linalg.norm(c.data.qpos[c.base_q][:2] - away) < 0.025:
            atoms.add(("RobotAway", ("robot",)))
            if on:
                for k in range(1, 4):
                    if ("Goal", (f"request_{k}",)) in atoms:
                        atoms.add(("MealStarted", (f"request_{k}",)))
        state.abstract_atoms = frozenset(atoms)
        return state

    def take_action(self, *, action: Action) -> State:
        values = np.asarray(action, dtype=float)
        self.controller()
        self._started = True
        if values.shape != (6,) or not np.isfinite(values).all():
            return super().take_action(action=action)
        action_id = values[0]
        on = self.controller().scene.cooking_observed()
        if action_id not in (8, 9) and not on:
            return super().take_action(action=action)
        c = self.controller()
        self._skill_start = c.steps
        start = float(c.data.time)
        self.last_info = {
            "skill": {8: "ActivateCooking", 9: "MoveAway"}.get(action_id, "Rejected"),
            "success": False,
            "error": None,
        }
        try:
            if (
                any(values[i] != int(values[i]) for i in range(3))
                or not 0 <= values[1] < 3
                or not 1 <= values[2] <= 3
            ):
                raise ValueError("Invalid discrete action fields")
            if not self.deployment:
                raise ValueError("Practice uses cold ingredients; cooking is deployment only")
            if action_id == 8:
                atoms = self.observe().abstract_atoms
                if on or ("Goal", (f"request_{self.requested_count}",)) not in atoms:
                    raise ValueError(
                        "Cooking requires the observed requested split, no spills, and jug put down"
                    )
                if self._skill_sink:
                    self._skill_sink(
                        name="ActivateCooking",
                        objects=[],
                        params=[],
                        controller={"interface": "simulated"},
                    )
                c.pause(phase="simulated hob command", seconds=0.5)
                c.scene.set_cooking(enabled=True)
                self.last_info["success"] = c.scene.cooking_observed()
            elif action_id == 9 and on:
                if self._skill_sink:
                    self._skill_sink(name="MoveAway", objects=[], params=[], controller={})
                c.move_away(distance=self.geometry.retreat_distance_m)
                self.last_info["success"] = (
                    "RobotAway",
                    ("robot",),
                ) in self.observe().abstract_atoms
            else:
                raise ValueError(
                    "Hot-task manipulation/reset is not implemented; "
                    "only move away after activation"
                )
        except ValueError as exc:
            self.last_info["error"] = str(exc)
        self._steps += 1
        self.last_info.update(
            seconds=float(c.data.time) - start,
            controller_steps=c.steps - self._skill_start,
            locations=c.scene.cube_locations(),
            held=c.held,
        )
        self.current_state = self.observe()
        return self.current_state

    def reset_movables(self, *, destination: str | None = None) -> bool:
        if self.deployment:
            raise ValueError("Human actions are forbidden in deployment")
        if destination not in (None, HUMAN_RESET):
            raise ValueError("Unknown human reset")
        if self.controller().scene.cooking_observed():
            raise ValueError("Repeated reset practice requires cold ingredients")
        self._started = True
        # The learner charges the same practice_cost through its ordinary human
        # action path. This ledger audits direct calls; it is not charged twice.
        self.human_interventions += 1
        self.interruption_cost_total += self.human_interruption_cost
        self.last_info = {"skill": HUMAN_RESET, "success": False, "error": None}
        result = super().reset_movables(destination=LEGACY_RESET)
        self.last_info.update(
            skill=HUMAN_RESET,
            success=result,
            human_interventions=1,
            interruption_cost=self.human_interruption_cost,
        )
        return result

    def finish_human_handoff(self) -> None:
        """Declare task completion before a human removes food between tasks."""
        if (
            "MealStarted",
            (f"request_{self.requested_count}",),
        ) not in self.observe().abstract_atoms:
            raise ValueError("Handoff requires completed deployment")
        self.episode_boundaries.append({"reason": "human removes cooked food between tasks"})
        self._started = False
