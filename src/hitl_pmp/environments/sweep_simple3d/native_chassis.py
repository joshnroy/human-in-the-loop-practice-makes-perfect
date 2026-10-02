"""Exact native chassis/environment contact checks for Simple base routes."""

from collections.abc import Sequence

import mujoco
import numpy as np


class NativeChassisClearance:
    """Use native collision masks/exclusions on scratch data, without physics steps."""

    def __init__(self, *, model: mujoco.MjModel, live_data: mujoco.MjData) -> None:
        self.model = model
        self.live_data = live_data
        self.data = mujoco.MjData(model)
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot_base_link")
        self.geoms = {
            g
            for g in range(model.ngeom)
            if body >= 0
            and model.geom_bodyid[g] == body
            and (model.geom_contype[g] or model.geom_conaffinity[g])
        }
        if not self.geoms:
            raise ValueError("Missing native chassis collision geometry")
        self.addresses = []
        for axis in ("x", "y", "th"):
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot_joint_{axis}")
            if joint < 0:
                raise ValueError("Missing native base joint")
            self.addresses.append(int(model.jnt_qposadr[joint]))

    def contacts(
        self,
        *,
        base: Sequence[float],
        held_wiper: bool = False,
    ) -> list[dict[str, object]]:
        """Reject native penetration, including drawers, handles and current cubes.

        A held wiper's predicted pose is checked by the existing carried-tool
        planner; its stationary live pose must not obstruct a hypothetical base.
        Robot self collisions likewise remain the existing arm/chassis check.
        """
        values = np.asarray(base, dtype=float)
        if values.shape != (3,) or not np.isfinite(values).all():
            raise ValueError("Expected finite base x/y/yaw")
        self.data.qpos[:] = self.live_data.qpos
        self.data.qpos[self.addresses] = values
        mujoco.mj_fwdPosition(self.model, self.data)
        result = []
        for contact in self.data.contact:
            pair = (int(contact.geom1), int(contact.geom2))
            if contact.dist >= 0.0 or not self.geoms.intersection(pair):
                continue
            other = pair[1] if pair[0] in self.geoms else pair[0]
            name = (
                mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, self.model.geom_bodyid[other]
                )
                or "world"
            )
            if name.startswith("robot_") or (held_wiper and name == "wiper_0"):
                continue
            result.append({"geoms": list(pair), "body": name, "distance": float(contact.dist)})
        return result

    def first_route_rejection(
        self,
        *,
        path: Sequence[Sequence[float]],
        held_wiper: bool = False,
    ) -> dict[str, object] | None:
        """Check endpoints and interpolation at <=1 cm and <=.025 rad spacing."""
        if not path:
            raise ValueError("Expected a nonempty base path")
        previous = np.asarray(path[0], dtype=float)
        for index, point in enumerate(path):
            target = np.asarray(point, dtype=float)
            delta = target - previous
            delta[2] = (delta[2] + np.pi) % (2 * np.pi) - np.pi
            steps = max(
                1,
                int(np.ceil(np.linalg.norm(delta[:2]) / 0.01)),
                int(np.ceil(abs(delta[2]) / 0.025)),
            )
            fractions = [0.0] if index == 0 else np.linspace(0.0, 1.0, steps + 1)[1:]
            for fraction in fractions:
                base = previous + fraction * delta
                contacts = self.contacts(base=base, held_wiper=held_wiper)
                if contacts:
                    return {
                        "segment": index,
                        "fraction": float(fraction),
                        "base": base.tolist(),
                        "contacts": contacts,
                    }
            previous = target
        return None
