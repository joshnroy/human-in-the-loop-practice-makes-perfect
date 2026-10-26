"""Native KinDER simulation adapter; all robot skills use native motor actions."""

from collections.abc import Callable
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .native_controller import NativePouringController, PourParameters


class PouringSimulation(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    env: Any
    controller: NativePouringController
    elapsed_seconds: float = 0.0
    human_interventions: int = 0
    events: list[dict[str, Any]] = Field(default_factory=list)

    @classmethod
    def create(
        cls, *, seed: int = 0, scene_bg: bool = True, requested_count: int = 5
    ) -> "PouringSimulation":
        from kinder.envs.dynamic3d.pitcher_pour3d import PitcherPour3DEnv

        env = PitcherPour3DEnv(
            scene_bg=scene_bg, render_mode="rgb_array", requested_count=requested_count
        )
        env.reset(seed=seed)
        return cls(env=env, controller=NativePouringController.create(env=env))

    @property
    def model(self) -> Any:
        return self.controller.model

    @property
    def data(self) -> Any:
        return self.controller.data

    @property
    def held(self) -> str | None:
        return self.controller.held

    @property
    def gripper_closed(self) -> bool:
        return self.controller.grip == 1

    def locations(self) -> list[str]:
        result = self.controller.scene.cube_locations()
        if self.held is not None and self.held.startswith("cube_"):
            result[int(self.held[-1])] = "held"
        return result

    def settled(self) -> bool:
        return self.controller.scene.cubes_settled()

    def ready(self) -> bool:
        return self.controller.ready()

    def abstract_atoms(self) -> frozenset[tuple[str, tuple[str, ...]]]:
        atoms: set[tuple[str, tuple[str, ...]]] = set()
        locations = self.locations()
        if self.held is None:
            atoms.add(("HandEmpty", ("robot",)))
        elif self.held == "pitcher":
            atoms.add(("HoldingPitcher", ("robot", "pitcher")))
        else:
            atoms.add(("HoldingCube", ("robot", self.held)))
        for i, location in enumerate(locations):
            name = f"cube_{i}"
            if location in {"pitcher", "tray"}:
                atoms.add(("InPitcher" if location == "pitcher" else "InTray", (name, location)))
            elif location == "floor":
                atoms.add(("OnFloor", (name,)))
            joint = int(self.model.body_jntadr[self.model.body(name).id])
            dof = self.model.jnt_dofadr[joint]
            if (
                np.linalg.norm(self.data.qvel[dof : dof + 3]) < 0.025
                and np.linalg.norm(self.data.qvel[dof + 3 : dof + 6]) < 0.25
            ):
                atoms.add(("Settled", (name,)))
        for vessel in ("pitcher", "tray"):
            _, rotation = self.controller.body_pose(name=vessel)
            if rotation.apply([0, 0, 1])[2] > np.cos(np.deg2rad(5)):
                atoms.add(("Upright", (vessel,)))
            if self.controller.vessel_home(name=vessel):
                atoms.add(("AtHome", (vessel,)))
            if vessel not in locations:
                atoms.add(("Empty", (vessel,)))
            if locations == [vessel] * 5:
                atoms.add(("AllInPitcher" if vessel == "pitcher" else "AllInTray", (vessel,)))
        if "tray" in locations:
            atoms.add(("AnyInTray", ("tray",)))
        if all(location in {"pitcher", "tray"} for location in locations):
            atoms.add(("NoSpills", ()))
        requested = self.controller.scene.requested_count
        if locations.count("tray") == requested:
            atoms.add(("RequestedCountInTray", ()))
        if locations.count("pitcher") == 5 - requested:
            atoms.add(("RequestedRemainderInPitcher", ()))
        if self.ready():
            atoms.add(("ReadyToPour", ()))
        if self.controller.scene._check_goals():
            atoms.add(("Goal", ()))
        return frozenset(atoms)

    def snapshot(self) -> dict[str, Any]:
        return {
            **self.controller.scene.physical_snapshot(),
            "held": self.held,
            "gripper_closed": self.gripper_closed,
            "locations": self.locations(),
            "ready": self.ready(),
        }

    def check_preconditions(
        self, *, skill: str, cube: int | None = None, parameters: dict[str, float] | None = None
    ) -> None:
        if cube is not None and (not isinstance(cube, int) or not 0 <= cube < 5):
            raise ValueError("Invalid cube ID")
        valid = False
        if skill == "PickPitcher":
            valid = self.ready()
        elif skill == "Pour":
            PourParameters(**(parameters or {}))
            valid = self.held == "pitcher"
        elif skill == "PlacePitcherAtHome":
            valid = self.held == "pitcher"
        elif skill in {"PickCubeFromTray", "PickCubeFromFloor"}:
            valid = (
                cube is not None
                and self.held is None
                and self.locations()[cube] == ("tray" if skill == "PickCubeFromTray" else "floor")
            )
        elif skill == "DropCubeIntoPitcher":
            valid = self.held == f"cube_{cube}" and self.controller.vessel_home(name="pitcher")
        elif skill == "Wait":
            valid = True
        if not valid:
            raise ValueError(f"Preconditions unmet for {skill}")

    def execute(
        self, *, skill: str, cube: int | None = None, parameters: dict[str, float] | None = None
    ) -> dict[str, Any]:
        self.check_preconditions(skill=skill, cube=cube, parameters=parameters)
        event = self.controller.run_skill(
            skill=skill,
            cube=0 if cube is None else cube,
            parameters=PourParameters(**(parameters or {})) if skill == "Pour" else None,
        )
        event["cost_seconds"] = event["seconds"]
        event["human_interventions"] = 0
        self.elapsed_seconds += event["seconds"]
        self.events.append(event)
        return event

    def reset_payload(self) -> dict[str, Any]:
        """Privileged human reset; preserve robot/vessels, time, and motor history.

        The caller owns intervention pricing, as for other reset_movables domains.
        """
        if (
            self.held is not None
            or self.locations() == ["pitcher"] * 5
            or not self.controller.vessel_home(name="pitcher")
            or not self.controller.vessel_home(name="tray")
        ):
            raise ValueError("Human reset requires cubes outside pitcher, home vessels, empty hand")
        before = self.controller.scene.physical_snapshot()
        self.controller.scene.respawn_payload_in_pitcher()
        self.human_interventions += 1
        event = {
            "skill": "reset_cubes",
            "cube": None,
            "success": self.ready(),
            "error": None,
            "parameters": None,
            "seconds": 0.0,
            "cost_seconds": 0.0,
            "human_interventions": 1,
            "simulated_human_reset": True,
            "before": before,
            "after": self.controller.scene.physical_snapshot(),
        }
        self.events.append(event)
        return event

    def render(self) -> np.ndarray:
        return self.env.render()

    def set_frame_callback(self, *, callback: Callable[..., None] | None) -> None:
        self.controller.callback = callback

    def close(self) -> None:
        self.env.close()
