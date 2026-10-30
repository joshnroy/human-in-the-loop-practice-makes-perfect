"""Native scene construction, measured abstraction, and registered Gym variants."""

from pathlib import Path
from typing import Any

import numpy as np

from .types import FrozenPourState, PourObjects


class PourBackend:
    @staticmethod
    def create(
        *,
        count: int,
        requested: int,
        mu: float,
        scene_bg: bool,
        seed: int,
        receiving_vessel: str = "skillet",
        layout: str = "stations",
        payload_kind: str = "shrimp",
    ) -> Any:
        import gymnasium as gym

        from .controller import PracticePourController

        if payload_kind == "potsticker":
            from hitl_pmp.environments.pouring3d.potsticker_controller import PotstickerController

            if count != 3 or layout != "stations" or receiving_vessel != "skillet":
                raise ValueError("Dumpling practice requires three pieces and station skillet")
            PourBackend.register()
            env = gym.make(
                "kinder/FrozenPotstickerPour3D-o3-v0",
                requested_count=requested,
                surface_mu=mu,
                scene_bg=scene_bg,
                render_mode="rgb_array",
            )
            env.reset(seed=seed)
            controller: Any = PotstickerController.create(env=env)
            controller.mode = "joint_lateral"
            return controller
        if payload_kind != "shrimp":
            raise ValueError("Unknown pouring payload")

        PourBackend.register()
        if receiving_vessel not in {"tray", "skillet"}:
            raise ValueError("Unknown receiving vessel")
        if layout not in {"stations", "floor"}:
            raise ValueError("Unknown pouring layout")
        if layout == "stations" and receiving_vessel != "skillet":
            raise ValueError("Station layout requires the skillet; legacy tray uses layout=floor")
        version = 2 if layout == "stations" else (1 if receiving_vessel == "skillet" else 0)
        env = gym.make(
            f"kinder/FrozenShrimpPour3D-o{count}-v{version}",
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

        from hitl_pmp.environments.pouring3d.potsticker_native import PotstickerPourEnv

        from .native import PracticeFiveEnv, PracticeThreeEnv

        assets = Path(__file__).resolve().parents[1] / "assets"
        key = "kinder/FrozenPotstickerPour3D-o3-v0"
        if key not in gym.registry:
            gym.register(
                id=key,
                entry_point=PotstickerPourEnv,
                kwargs={
                    "task_config_path": str(assets / "potsticker-pour3-stations.json"),
                    "payload_count": 3,
                },
                disable_env_checker=True,
                order_enforce=False,
            )
        for count, cls, filename in (
            (3, PracticeThreeEnv, "frozen-shrimp-pour.json"),
            (5, PracticeFiveEnv, "five-frozen-shrimp-pour.json"),
        ):
            for version, asset in (
                (0, filename),
                (1, filename.replace(".json", "-skillet.json")),
                (2, filename.replace(".json", "-stations.json")),
            ):
                key = f"kinder/FrozenShrimpPour3D-o{count}-v{version}"
                if key not in gym.registry:
                    gym.register(
                        id=key,
                        entry_point=cls,
                        kwargs={"task_config_path": str(assets / asset)},
                        disable_env_checker=True,
                        order_enforce=False,
                    )

    @staticmethod
    def observe(*, controller: Any, objects: PourObjects, seed: int, steps: int) -> FrozenPourState:
        c, o = controller, objects
        data = {
            o.scene: np.array([seed, steps, c.scene.requested_count, c.scene.surface_mu]),
            o.robot: np.r_[
                c.data.qpos[c.base_q], c.grip, c.data.qpos[c.arm_q], c.data.qvel[c.arm_v]
            ],
        }
        for obj in (*o.shrimp(), o.pitcher, o.tray):
            body = c.model.body(obj.name).id
            joint = c.model.body_jntadr[body]
            dof = c.model.jnt_dofadr[joint]
            data[obj] = np.r_[c.data.xpos[body], c.data.xquat[body], c.data.qvel[dof : dof + 6]]
        for count, obj in enumerate(o.requests(), 1):
            data[obj] = np.array([float(count)])
        atoms: set[tuple[str, tuple[str, ...]]] = {
            (f"Count{n}", (obj.name,)) for n, obj in enumerate(o.requests(), 1)
        }
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
            elif location in {"floor", "surface"}:
                atoms.add(("OnFloor" if location == "floor" else "OnSurface", (obj.name,)))
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
        # Success describes the measured split, independently of the task goal
        # and the count the robot chose to attempt.
        actual_count = locations.count("tray")
        if (
            1 <= actual_count <= o.count
            and locations.count("pitcher") == o.count - actual_count
            and c.scene.cubes_settled()
            and c.held not in c.scene.cube_names
        ):
            request = f"request_{actual_count}"
            atoms.add(("SplitCorrect", (request,)))
            if c.held is None and c.vessel_home(name="pitcher") and c.vessel_home(name="tray"):
                atoms.add(("Goal", (request,)))
        return FrozenPourState(data=data, abstract_atoms=frozenset(atoms))
