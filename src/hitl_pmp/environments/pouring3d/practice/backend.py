"""Native scene construction, measured abstraction, and registered Gym variants."""

from pathlib import Path
from typing import Any

import numpy as np

from .types import FrozenPourState, PourObjects


class PourBackend:
    @staticmethod
    def create(*, count: int, requested: int, mu: float, scene_bg: bool, seed: int) -> Any:
        import gymnasium as gym

        from .controller import PracticePourController

        PourBackend.register()
        env = gym.make(
            f"kinder/FrozenShrimpPour3D-o{count}-v0",
            requested_count=requested,
            surface_mu=mu,
            scene_bg=scene_bg,
            render_mode="rgb_array",
        )
        env.reset(seed=seed)
        return PracticePourController.create(env=env)

    @staticmethod
    def register() -> None:
        import gymnasium as gym

        from .native import PracticeFiveEnv, PracticeThreeEnv

        assets = Path(__file__).resolve().parents[1] / "assets"
        for count, cls, filename in (
            (3, PracticeThreeEnv, "frozen-shrimp-pour.json"),
            (5, PracticeFiveEnv, "five-frozen-shrimp-pour.json"),
        ):
            key = f"kinder/FrozenShrimpPour3D-o{count}-v0"
            if key not in gym.registry:
                gym.register(
                    id=key,
                    entry_point=cls,
                    kwargs={"task_config_path": str(assets / filename)},
                    disable_env_checker=True,
                    order_enforce=False,
                )

    @staticmethod
    def observe(*, controller: Any, objects: PourObjects, seed: int, steps: int) -> FrozenPourState:
        c, o = controller, objects
        data = {
            o.scene: np.array([seed, steps, c.scene.requested_count, c.scene.surface_mu]),
            o.robot: np.r_[c.data.qpos[c.base_q], c.grip],
        }
        for obj in (*o.shrimp(), o.pitcher, o.tray):
            body = c.model.body(obj.name).id
            joint = c.model.body_jntadr[body]
            dof = c.model.jnt_dofadr[joint]
            data[obj] = np.r_[c.data.xpos[body], c.data.xquat[body], c.data.qvel[dof : dof + 6]]
        for count, obj in enumerate(o.requests(), 1):
            data[obj] = np.array([float(count)])
        atoms: set[tuple[str, tuple[str, ...]]] = set()
        if c.held is None:
            atoms.add(("HandEmpty", ("robot",)))
        elif c.held == "pitcher":
            atoms.add(("HoldingPitcher", ("robot", "pitcher")))
        else:
            atoms.add(("HoldingShrimp", ("robot", c.held)))
        locations = c.scene.cube_locations()
        for obj, location in zip(o.shrimp(), locations, strict=True):
            if obj.name == c.held:
                continue
            if location in {"pitcher", "tray"}:
                atoms.add((
                    "InPitcher" if location == "pitcher" else "InTray",
                    (obj.name, location),
                ))
            elif location == "floor":
                atoms.add(("OnFloor", (obj.name,)))
            if np.linalg.norm(data[obj][7:10]) < 0.025 and np.linalg.norm(data[obj][10:13]) < 0.25:
                atoms.add(("Settled", (obj.name,)))
        for vessel in ("pitcher", "tray"):
            if c.vessel_home(name=vessel):
                atoms.add(("PitcherHome" if vessel == "pitcher" else "TrayHome", (vessel,)))
        for a, b in zip(o.shrimp(), o.shrimp()[1:], strict=False):
            atoms.add(("PayloadOrder", (a.name, b.name)))
        request = f"request_{c.scene.requested_count}"
        atoms.add(("ActiveRequest", (request,)))
        if all(x in {"tray", "pitcher"} for x in locations) and c.held not in c.scene.cube_names:
            atoms.add(("NoSpills", ()))
        if locations != ["pitcher"] * o.count:
            atoms.add(("NeedsReset", ()))
        if c.ready():
            atoms.add(("ReadyToPour", ()))
        if c.scene._check_goals() and c.held not in c.scene.cube_names:
            atoms.add(("SplitCorrect", (request,)))
            if c.held is None and c.vessel_home(name="pitcher") and c.vessel_home(name="tray"):
                atoms.add(("Goal", (request,)))
        return FrozenPourState(data=data, abstract_atoms=frozenset(atoms))
