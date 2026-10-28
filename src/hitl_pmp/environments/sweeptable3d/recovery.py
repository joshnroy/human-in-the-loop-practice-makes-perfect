"""Table-only contact-tolerant grasp planning with unchanged physical collisions."""

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from hitl_pmp.environments.sweepoffshelf3d.floor_pick import FloorPicks
from hitl_pmp.environments.sweepoffshelf3d.kinder_skills import (
    WIPER_OBSTACLE_MARGIN,
    KinderSkills,
)
from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import SkillFailed


class TableFloorPicks(FloorPicks):
    def obstacle_geoms(self, *, cube: str) -> list[int]:
        geoms = super().obstacle_geoms(cube=cube)
        if getattr(self.kinder, "contact_target", None) != cube:
            return geoms
        model = self.shelf.model
        return [
            geom
            for geom in geoms
            if not (model.body(model.geom_bodyid[geom]).name or "").startswith("cube_")
        ]


class TableKinderSkills(KinderSkills):
    contact_target: str | None = None

    @contextmanager
    def allow_ball_contact(self, *, cube: str) -> Generator[None, None, None]:
        previous = self.contact_target
        self.contact_target = cube
        try:
            yield
        finally:
            self.contact_target = previous

    @staticmethod
    def filtered_bodies(*, bodies: set[int], named: dict[str, int], target: str) -> set[int]:
        return bodies - {
            body for name, body in named.items() if name.startswith("cube_") and name != target
        }

    def planning_sim(self, *, target: str | None = None) -> Any:
        sim = super().planning_sim(target=target)
        self._replace_table_geometry(sim=sim)
        contact_target = self.contact_target
        if contact_target is not None:
            original = sim.get_collision_bodies

            def filtered(*args: Any, **kwargs: Any) -> set[int]:
                return self.filtered_bodies(
                    bodies=original(*args, **kwargs),
                    named=sim.named_collision_bodies(),
                    target=contact_target,
                )

            sim.get_collision_bodies = filtered
        return sim

    def _replace_table_geometry(self, *, sim: Any) -> None:
        """Replace only the stock cupboard body, in the frame set_state will use."""
        import pybullet

        model, data = self.shelf.model, self.shelf.data
        geoms = sorted(
            geom
            for geom in self.shelf.cupboard_geoms
            if model.geom_contype[geom] or model.geom_conaffinity[geom]
        )
        if not geoms or any(int(model.geom_type[geom]) != 6 for geom in geoms):
            raise SkillFailed("Unsupported or missing live table collision geometry")
        old_body = sim._cupboard1_shelf_id
        if old_body is None:
            raise SkillFailed("Planning scene has no table body to replace")
        state = self.state()
        table = state.get_object_from_name("cupboard_1")
        position = np.array([state.get(table, axis) for axis in ("x", "y", "z")])
        rotation = Rotation.from_quat([state.get(table, axis) for axis in ("qx", "qy", "qz", "qw")])
        shape = pybullet.createCollisionShapeArray(
            shapeTypes=[pybullet.GEOM_BOX] * len(geoms),
            halfExtents=model.geom_size[geoms].tolist(),
            collisionFramePositions=rotation.inv().apply(data.geom_xpos[geoms] - position).tolist(),
            collisionFrameOrientations=(
                rotation.inv() * Rotation.from_matrix(data.geom_xmat[geoms].reshape(-1, 3, 3))
            )
            .as_quat()
            .tolist(),
            physicsClientId=sim.physics_client_id,
        )
        body = pybullet.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=shape,
            basePosition=position.tolist(),
            baseOrientation=rotation.as_quat().tolist(),
            physicsClientId=sim.physics_client_id,
        )
        pybullet.removeBody(old_body, physicsClientId=sim.physics_client_id)
        sim._cupboard1_shelf_id = body
        sim._cupboard1_surface_ids = {-1}

    def add_wiper_obstacle(self, *, sim: Any) -> None:
        """Keep the live capsule axis and inflate each primitive by the shared margin."""
        import pybullet

        model, data = self.shelf.model, self.shelf.data
        for geom in self.shelf.wiper_geoms:
            size = model.geom_size[geom]
            if int(model.geom_type[geom]) == 3:
                shape = pybullet.createCollisionShape(
                    shapeType=pybullet.GEOM_CAPSULE,
                    radius=float(size[0]) + WIPER_OBSTACLE_MARGIN,
                    height=2 * float(size[1]),
                    physicsClientId=sim.physics_client_id,
                )
            elif int(model.geom_type[geom]) == 6:
                shape = pybullet.createCollisionShape(
                    shapeType=pybullet.GEOM_BOX,
                    halfExtents=(size + WIPER_OBSTACLE_MARGIN).tolist(),
                    physicsClientId=sim.physics_client_id,
                )
            else:
                raise SkillFailed("Unsupported live wiper collision geometry")
            body = pybullet.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=shape,
                basePosition=data.geom_xpos[geom].tolist(),
                baseOrientation=Rotation
                .from_matrix(data.geom_xmat[geom].reshape(3, 3))
                .as_quat()
                .tolist(),
                physicsClientId=sim.physics_client_id,
            )
            sim._static_colliders[f"wiper_0/geom{geom}"] = body

    def cube_between_pads(self, *, base: np.ndarray, arm: np.ndarray, cube: str) -> str | None:
        if self.contact_target == cube:
            return None
        return super().cube_between_pads(base=base, arm=arm, cube=cube)

    def touches_other_cube(self, *, base: np.ndarray, arm: np.ndarray, cube: str) -> bool:
        if self.contact_target == cube:
            return False
        return super().touches_other_cube(base=base, arm=arm, cube=cube)

    def pick_cube_from_floor(self, *, cube: str) -> None:
        if self.shelf.in_hand is not None or self.shelf.held_cube is not None:
            raise SkillFailed("pick_cube_from_floor: the hand is not empty")
        with self.allow_ball_contact(cube=cube):
            TableFloorPicks(kinder=self).pick(cube=cube, label="pick_cube_from_floor")
        if not self.held(name=cube):
            raise SkillFailed(f"pick_cube_from_floor: {cube} did not come with the hand")
        if self.other_held_cubes(cube=cube):
            raise SkillFailed(f"pick_cube_from_floor: {cube} came with another cube")
