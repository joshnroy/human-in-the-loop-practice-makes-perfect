"""MuJoCo-side distances for the kitchen planning tests (not a test module)."""

import mujoco
import numpy as np


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
