"""MuJoCo-side distances for the kitchen planning tests (not a test module)."""

from typing import Any

import mujoco
import numpy as np

# The bar for every planned kitchen skill: over its planned phases, MuJoCo's arm and
# held jug never come closer to the scene than this. Planning keeps 1.5 cm in PyBullet,
# which test_kitchen_pybullet_scene measured to be up to about 1 cm more optimistic than
# MuJoCo (at the bracelet/gripper coupling): 1.5 - 1.0 = 0.5 cm.
MUJOCO_CLEARANCE_M = 0.005
SCENE = (
    "prep_table",
    "stovetop",
    "pitcher",
    "tray",
    "potsticker_0",
    "potsticker_1",
    "potsticker_2",
)


class Mujoco:
    """Signed distances measured in a MuJoCo copy of the live model."""

    @staticmethod
    def subtree(*, model, root: int) -> set[int]:
        out = set()
        for body in range(model.nbody):
            ancestor = body
            while ancestor and ancestor != root:
                ancestor = int(model.body_parentid[ancestor])
            if ancestor == root:
                out.add(body)
        return out

    @staticmethod
    def geoms(*, model, names) -> list[int]:
        bodies = set().union(*(Mujoco.subtree(model=model, root=model.body(n).id) for n in names))
        return [
            g
            for g in range(model.ngeom)
            if int(model.geom_bodyid[g]) in bodies
            and (model.geom_contype[g] or model.geom_conaffinity[g])
        ]

    @staticmethod
    def arm_geoms(*, model) -> list[int]:
        # The arm and gripper; the mobile base is planned as a 2-D footprint.
        return [
            g
            for g in range(model.ngeom)
            if (model.body(int(model.geom_bodyid[g])).name or "").startswith("robot_")
            and model.body(int(model.geom_bodyid[g])).name != "robot_base_link"
            and (model.geom_contype[g] or model.geom_conaffinity[g])
        ]

    @staticmethod
    def distance(*, model, data, a: list[int], b: list[int], cap: float) -> float:
        best = cap
        for ga in a:
            for gb in b:
                gap = np.linalg.norm(data.geom_xpos[ga] - data.geom_xpos[gb])
                if gap - model.geom_rbound[ga] - model.geom_rbound[gb] >= best:
                    continue
                best = min(best, Mujoco.pair(model=model, data=data, ga=ga, gb=gb, cap=best))
        return float(best)

    @staticmethod
    def pair(*, model, data, ga: int, gb: int, cap: float) -> float:
        """`mj_geomDistance`, checked against its own witness points.

        On MuJoCo 3.3.7 it sometimes returns 0 for a mesh pair several centimetres
        apart, with witness points nowhere near either geom (measured 2026-11-02:
        2293 such pair queries in 1500 sampled arm configurations). Those are
        recomputed from the vertices instead.
        """
        fromto = np.zeros(6)
        value = mujoco.mj_geomDistance(model, data, ga, gb, cap, fromto)
        if value >= cap or abs(np.linalg.norm(fromto[3:] - fromto[:3]) - abs(value)) <= 1e-4:
            return float(value)
        pa, pb = (
            Mujoco.points(model=model, data=data, g=ga),
            Mujoco.points(model=model, data=data, g=gb),
        )
        if pb is None or (
            pa is not None and int(model.geom_type[gb]) != mujoco.mjtGeom.mjGEOM_MESH
        ):
            return min(cap, Mujoco.to_primitive(model=model, data=data, g=gb, points=pa))
        if pa is None or int(model.geom_type[ga]) != mujoco.mjtGeom.mjGEOM_MESH:
            return min(cap, Mujoco.to_primitive(model=model, data=data, g=ga, points=pb))
        from scipy.spatial import cKDTree

        # Two meshes: the closest vertex pair, an upper bound within vertex spacing.
        return min(cap, float(cKDTree(pb).query(pa)[0].min()))

    @staticmethod
    def points(*, model, data, g: int) -> np.ndarray | None:
        rotation, origin = data.geom_xmat[g].reshape(3, 3), data.geom_xpos[g]
        kind = int(model.geom_type[g])
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = int(model.geom_dataid[g])
            start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            local = model.mesh_vert[start : start + count]
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
            local = signs * model.geom_size[g]
        else:
            return None
        return local @ rotation.T + origin

    @staticmethod
    def to_primitive(*, model, data, g: int, points: np.ndarray) -> float:
        """Exact distance from points to a box, cylinder, capsule or sphere (0 inside)."""
        local = (points - data.geom_xpos[g]) @ data.geom_xmat[g].reshape(3, 3)
        size = model.geom_size[g]
        kind = int(model.geom_type[g])
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            outside = np.maximum(np.abs(local) - size, 0)
            return float(np.linalg.norm(outside, axis=1).min())
        if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            radial = np.maximum(np.linalg.norm(local[:, :2], axis=1) - size[0], 0)
            axial = np.maximum(np.abs(local[:, 2]) - size[1], 0)
            return float(np.hypot(radial, axial).min())
        if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            axial = np.clip(local[:, 2], -size[1], size[1])
            offset = local - np.c_[np.zeros((len(local), 2)), axial]
            return float(np.maximum(np.linalg.norm(offset, axis=1) - size[0], 0).min())
        if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            return float(np.maximum(np.linalg.norm(local, axis=1) - size[0], 0).min())
        raise ValueError(f"No fallback distance for geom type {kind}")


class Clearance:
    """Every control tick of a planned phase: MuJoCo distance from the arm (and the
    held jug, and nothing riding in it) to the rest of the scene, and the jug's tilt."""

    @staticmethod
    def record(*, env: Any) -> list[tuple[str, float, float, int]]:
        c = env.controller()
        m = c.model
        arm = Mujoco.arm_geoms(model=m)
        jug = Mujoco.geoms(model=m, names=["pitcher"])
        seen: list[tuple[str, float, float, int]] = []
        sink = c.callback

        def tick(**kwargs):  # noqa: PLR0917 -- controller callback signature
            if sink is not None:
                sink(**kwargs)
            if not c.phase.startswith("planned "):
                return
            locations = c.scene.cube_locations()
            if c.held == "pitcher":
                riders = [
                    n
                    for n, at in zip(c.scene.cube_names, locations, strict=True)
                    if at == "pitcher"
                ]
                scene = [n for n in SCENE if n != "pitcher" and n not in riders]
                moving = arm + jug
            else:
                scene, moving = list(SCENE), arm
            distance = Mujoco.distance(
                model=m, data=c.data, a=moving, b=Mujoco.geoms(model=m, names=scene), cap=0.15
            )
            _, rotation = c.body_pose(name="pitcher")
            tilt = float(np.rad2deg(np.arccos(np.clip(rotation.apply([0, 0, 1])[2], -1, 1))))
            seen.append((c.phase, distance, tilt, locations.count("pitcher")))

        c.callback = tick
        return seen

    @staticmethod
    def check(*, seen: list[tuple[str, float, float, int]]) -> None:
        """Print the closest approach (vertex fallback for spurious zeros) and hold it
        to the bar."""
        assert seen, "no planned phase was recorded"
        phase, closest, _, _ = min(seen, key=lambda r: r[1])
        print(f"closest MuJoCo approach over planned phases: {closest * 100:.2f} cm in {phase}")
        assert closest >= MUJOCO_CLEARANCE_M, (phase, closest)
