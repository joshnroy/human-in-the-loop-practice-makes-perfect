"""Native cooked-shrimp shape prototype, with explicitly uncalibrated contacts.

These are small peeled, tail-off shrimp approximated by rigid curved capsules.
They have no deformation, adhesion, water, or food-surface calibration. The
compound collider preserves the curl; no box collider or welded grasp is used.
"""

import xml.etree.ElementTree as ET
from functools import lru_cache
from typing import Any, cast

import mujoco
import numpy as np
from kinder.envs.dynamic3d.objects.base import register_object
from kinder.envs.dynamic3d.objects.primitive_objects import Bin, Cuboid
from kinder.envs.dynamic3d.pitcher_pour3d import (
    ObjectCentricPitcherPour3DEnv,
    PitcherPour3DEnv,
)
from scipy.spatial.transform import Rotation

from .lateral_wiggle import LateralPouringController


@register_object
class CookedShrimp(Cuboid):
    """Seven tapered capsule segments sharing one freely moving rigid body."""

    @staticmethod
    @lru_cache(maxsize=1)
    def segments() -> list[tuple[np.ndarray, np.ndarray, float]]:
        angles = np.deg2rad(np.linspace(-125, 125, 8))
        points = np.column_stack((
            0.011 * np.cos(angles) - 0.003,
            0.011 * np.sin(angles),
            np.zeros(8),
        ))
        return [
            (points[i], points[i + 1], float(radius))
            for i, radius in enumerate(np.linspace(0.0055, 0.0025, 7))
        ]

    def _create_xml_element(self) -> ET.Element:
        body = ET.Element("body", name=self.name)
        ET.SubElement(body, "freejoint", name=self.joint_name)
        segments = self.segments()
        volumes = np.array([
            np.pi * radius**2 * np.linalg.norm(end - start) + 4 / 3 * np.pi * radius**3
            for start, end, radius in segments
        ])
        # Normalize the overlapping primitive masses to the chosen total mass.
        # This approximates the inertia; it does not infer food density.
        for index, ((start, end, radius), weight) in enumerate(
            zip(segments, volumes / volumes.sum(), strict=True)
        ):
            color = "0.98 0.67 0.54 1" if index % 2 == 0 else "0.99 0.82 0.71 1"
            ET.SubElement(
                body,
                "geom",
                name=f"{self.name}_segment_{index}",
                type="capsule",
                fromto=" ".join(map(str, np.r_[start, end])),
                size=str(radius),
                mass=str(self.mass * weight),
                rgba=color,
            )
        return body


class ShrimpPourScene(ObjectCentricPitcherPour3DEnv):
    """Three named shrimp, shape-aware containment, and exact split goals."""

    # Compatibility name used by the existing native motor controller. These
    # names refer to shrimp bodies, and no cube bodies are present in the model.
    cube_names = tuple(f"shrimp_{i}" for i in range(3))

    def __init__(self, *args: Any, surface_mu: float = 0.3, **kwargs: Any) -> None:
        if not np.isfinite(surface_mu) or surface_mu < 0:
            raise ValueError("surface_mu must be finite and nonnegative")
        requested_count = kwargs.get("requested_count", 3)
        if type(requested_count) is not int or not 1 <= requested_count <= 3:
            raise ValueError("Shrimp requests must be 1, 2, or 3")
        self.surface_mu = surface_mu
        kwargs["requested_count"] = requested_count
        kwargs["cube_material"] = None
        super().__init__(*args, **kwargs)

    def _create_scene_xml(self) -> str:
        root = ET.fromstring(super()._create_scene_xml())
        contact = root.find("contact")
        if contact is None:
            contact = ET.SubElement(root, "contact")
        groups = []
        for name in self.cube_names:
            body = root.find(f".//body[@name='{name}']")
            assert body is not None
            geoms = body.findall("geom")
            assert len(geoms) == 7
            for geom in geoms:
                # Priority 2 selects these contact properties over the vessels'
                # priority 1 and the floor's defaults. No torsional/rolling drag.
                geom.set("priority", "2")
                geom.set("condim", "3")
                geom.set("friction", f"{self.surface_mu} 0 0")
                geom.set("solref", "0.008 1")
                geom.set("solimp", "0.95 0.99 0.001 0.5 2")
            groups.append([geom.attrib["name"] for geom in geoms])
        for index, group in enumerate(groups):
            for other_group in groups[index + 1 :]:
                for left in group:
                    for right in other_group:
                        ET.SubElement(
                            contact,
                            "pair",
                            geom1=left,
                            geom2=right,
                            condim="3",
                            friction=f"{self.surface_mu} {self.surface_mu} 0 0 0",
                            solref="0.008 1",
                            solimp="0.95 0.99 0.001 0.5 2",
                        )
        return ET.tostring(root, encoding="unicode")

    def respawn_payload_in_pitcher(self) -> None:
        pitcher = cast(Bin, self.get_object("pitcher"))
        origin = np.asarray(pitcher.get_position(), dtype=float)
        quat = np.asarray(pitcher.get_orientation(), dtype=float)
        rotation = Rotation.from_quat(quat[[1, 2, 3, 0]])
        for name, angle in zip(
            self.cube_names, np.linspace(0, 2 * np.pi, 3, endpoint=False), strict=True
        ):
            jitter = self.np_random.uniform(-0.0005, 0.0005, 2)
            local = np.r_[
                0.024 * np.array([np.cos(angle), np.sin(angle)]) + jitter,
                pitcher.wall_thickness + 0.0055,
            ]
            shrimp = self.get_object(name)
            shrimp.set_pose(origin + rotation.apply(local), quat)
            shrimp.set_velocity([0.0] * 3, [0.0] * 3)
        model, data = self._model_and_data()
        mujoco.mj_forward(model, data)

    def cube_corners(self, name: str) -> np.ndarray:  # noqa: PLR0917
        """World-axis extrema of every capsule, including exact floor height."""
        model, data = self._model_and_data()
        body = model.body(name).id
        rotation = data.xmat[body].reshape(3, 3)
        axes = np.r_[np.eye(3), -np.eye(3)]
        points = np.concatenate([
            np.vstack([
                rotation @ endpoint + data.xpos[body] + axes * radius for endpoint in (start, end)
            ])
            for start, end, radius in CookedShrimp.segments()
        ])
        return points

    @staticmethod
    @lru_cache(maxsize=4)
    def pitcher_planes(
        *, length: float, width: float, height: float, wall: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Tapered cavity core half-spaces, conservatively excluding spout flare."""
        angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        directions = np.column_stack((np.cos(angles), np.sin(angles)))
        outer = directions * [length / 2, width / 2]
        bottom = np.column_stack((outer * 11 / 14 - wall * directions, np.zeros(32)))
        top = np.column_stack((outer - wall * directions, np.full(32, height)))
        normals = np.cross(np.roll(bottom, -1, axis=0) - bottom, top - bottom)
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        return normals, np.sum(normals * bottom, axis=1)

    def cube_in_vessel(self, cube: str, vessel: str) -> bool:  # noqa: PLR0917
        """Capsule support against all cavity planes; exact for the box tray."""
        model, data = self._model_and_data()
        body = model.body(cube).id
        vessel_body = model.body(vessel).id
        vessel_rotation = data.xmat[vessel_body].reshape(3, 3)
        rotation = data.xmat[body].reshape(3, 3)
        parts = CookedShrimp.segments()
        endpoints = np.array([p for start, end, _ in parts for p in (start, end)])
        radii = np.repeat([r for _, _, r in parts], 2)
        local = (
            endpoints @ rotation.T + data.xpos[body] - data.xpos[vessel_body]
        ) @ vessel_rotation
        geometry = cast(Bin, self.get_object(vessel))
        tolerance = 0.002
        if (local[:, 2] - radii).min() < geometry.wall_thickness - tolerance:
            return False
        if (local[:, 2] + radii).max() > geometry.height + tolerance:
            return False
        if vessel == "tray":
            interior = np.array([geometry.length, geometry.width]) / 2 - geometry.wall_thickness
            return bool(np.all(np.abs(local[:, :2]) + radii[:, None] <= interior + tolerance))
        normals, offsets = self.pitcher_planes(
            length=geometry.length,
            width=geometry.width,
            height=geometry.height,
            wall=geometry.wall_thickness,
        )
        return bool(np.all(local @ normals.T + radii[:, None] <= offsets + tolerance))

    def _check_goals(self) -> bool:
        locations = self.cube_locations()
        return (
            locations.count("tray") == self.requested_count
            and locations.count("pitcher") == 3 - self.requested_count
            and self.cubes_settled()
        )

    def physical_snapshot(self) -> dict[str, Any]:
        snapshot = super().physical_snapshot()
        snapshot.pop("cube_material")
        snapshot.update(
            payload="rigid cooked-shrimp proxy",
            payload_count=3,
            surface_mu=self.surface_mu,
        )
        return snapshot


class ShrimpPourEnv(PitcherPour3DEnv):
    """Actual native TidyBot physics and rendering with three shrimp colliders."""

    def _create_object_centric_env(self, *args: Any, **kwargs: Any) -> ShrimpPourScene:
        return ShrimpPourScene(*args, **kwargs)

    def _create_variant_specific_description(self) -> str:
        scene = cast(ShrimpPourScene, self._object_centric_env)
        return (
            f"Pour exactly {scene.requested_count} of three shrimp proxies into "
            "the tray; retain the remainder, settle all pieces, and avoid spills."
        )


class ShrimpPourController(LateralPouringController):
    """The approved fixed-center lateral pour with canonical initialization."""

    @classmethod
    def create(cls, *, env: Any) -> "ShrimpPourController":
        scene = env.unwrapped._object_centric_env
        model, data = scene._model_and_data()
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        before = np.empty(mujoco.mj_stateSize(model, spec))
        after = np.empty_like(before)
        mujoco.mj_getState(model, data, before, spec)
        mujoco.mj_forward(model, data)
        mujoco.mj_getState(model, data, after, spec)
        assert np.array_equal(before, after)
        return cast(ShrimpPourController, super().create(env=env))

    def ready(self) -> bool:
        return (
            self.held is None
            and self.scene.cube_locations() == ["pitcher"] * 3
            and self.scene.cubes_settled()
            and self.vessel_home(name="pitcher")
            and self.vessel_home(name="tray")
        )
