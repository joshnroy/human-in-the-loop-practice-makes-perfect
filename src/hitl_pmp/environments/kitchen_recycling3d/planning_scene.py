"""The kitchen's PyBullet planning scene, built once and handed back as if new.

`KitchenKinderBackend._build_pick_collision_sim` costs over a second (the vision-bracelet
robot, the tapered bin's panels, every static collider), and the kitchen used to pay it
for every pick, every toss and every `PickPlannable` dry run. A `PlanningScenePool`
builds one, and when a use ends puts it back exactly as it was built:

- PyBullet's own snapshot of the world taken right after the build (`saveState`): every
  body's base pose and velocity and every joint, so the robot is back at its build-time
  configuration rather than wherever the last plan left it;
- the bin where the new state has it, as the build would have placed it, and the bin
  pose `bin_wall_clearance` reads;
- no held-object attachment.

A use may only move bodies and set those two attributes; one that adds or removes a body
(as the open-gripper clearance check does to its own, separately built scene) leaves a
scene the pool will not take back. A scene is only handed out for a state that agrees
with the one it was built from in everything the build reads beyond the bin's pose --
the static colliders and the cubes, bytes and dtype -- so a float32 observation and a
float64 restored snapshot each get a scene of their own.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np

from hitl_pmp.environments.vision_bracelet_robot import VisionBraceletPyBulletSim


class PooledPlanningScene(VisionBraceletPyBulletSim):
    """A planning scene whose `close` returns it to its pool rather than disconnecting.

    Every caller that ends a use calls `close` (the pick dry run does, in shared
    Tossing3D code), so it is the one place a use can be seen to end."""

    def __init__(self, initial_state: Any, *, pool: PlanningScenePool) -> None:  # noqa: PLR0917
        super().__init__(initial_state)
        self.pool: PlanningScenePool | None = pool
        self.key = b""
        self.fresh_state: int | None = None
        self.fresh_bodies = 0

    def close(self) -> None:
        if self.pool is not None:
            self.pool.release(scene=self)
        else:
            self.dispose()

    def dispose(self) -> None:
        self.pool = None
        super().close()


class PlanningScenePool:
    """One backend's reusable planning scenes: as many as are in use at once."""

    # Idle scenes kept, over all keys; the least recently used goes first.
    MAX_IDLE: ClassVar[int] = 4

    def __init__(
        self, *, furnish: Callable[[Any, Any], None], bin_name: str, pose_of: Callable[..., Any]
    ) -> None:
        self.furnish = furnish
        self.bin_name = bin_name
        self.pose_of = pose_of
        self.idle: list[PooledPlanningScene] = []
        self.lent: list[PooledPlanningScene] = []
        self.built = 0
        self.reused = 0

    @staticmethod
    def scene_key(*, state: Any) -> bytes:
        """Everything a build reads from `state` but the bin's pose and the movables'
        poses, which `checkout` resets: the static colliders, and the cubes' names and
        sizes."""
        parts = []
        for obj in sorted(state, key=lambda obj: obj.name):
            if obj.type.name == "mujoco_static_collider":
                parts.append(obj.name.encode() + b"\0" + np.asarray(state[obj]).tobytes())
            elif "cube" in obj.name:
                size = [state.get(obj, feature) for feature in ("bb_x", "bb_y", "bb_z")]
                parts.append(obj.name.encode() + b"\0" + np.asarray(size).tobytes())
            else:
                parts.append(obj.name.encode())
        return b"\1".join(parts)

    def checkout(self, *, state: Any) -> PooledPlanningScene:
        import pybullet as p
        from pybullet_helpers.geometry import set_pose

        key = self.scene_key(state=state)
        matching = [scene for scene in self.idle if scene.key == key]
        if matching:
            scene = matching[-1]
            self.idle.remove(scene)
            client = scene.physics_client_id
            p.restoreState(stateId=scene.fresh_state, physicsClientId=client)
            pose = self.pose_of(state=state)
            set_pose(scene._bins[self.bin_name], pose, client)  # noqa: SLF001
            scene._bin_poses[self.bin_name] = pose  # noqa: SLF001
            scene.base_link_to_held_obj = None
            scene.pool = self
            self.reused += 1
        else:
            scene = PooledPlanningScene(state, pool=self)
            scene.key = key
            self.furnish(scene, state)
            client = scene.physics_client_id
            scene.fresh_state = p.saveState(physicsClientId=client)
            scene.fresh_bodies = p.getNumBodies(physicsClientId=client)
            self.built += 1
        self.lent.append(scene)
        return scene

    def release(self, *, scene: PooledPlanningScene) -> None:
        import pybullet as p

        if scene not in self.lent:
            return
        self.lent.remove(scene)
        unchanged = scene.fresh_bodies == p.getNumBodies(physicsClientId=scene.physics_client_id)
        if not (unchanged and scene.pool is self):
            scene.dispose()
            return
        self.idle.append(scene)
        while len(self.idle) > self.MAX_IDLE:
            self.idle.pop(0).dispose()

    def release_since(self, *, mark: list[PooledPlanningScene]) -> None:
        """End every use begun since `mark` (a copy of `lent`)."""
        for scene in [scene for scene in self.lent if scene not in mark]:
            self.release(scene=scene)

    def dispose(self) -> None:
        """Disconnect every idle scene. One still lent is disconnected when its user
        closes it, or collected."""
        for scene in self.idle:
            scene.dispose()
        self.idle = []
        for scene in self.lent:
            scene.pool = None
        self.lent = []
