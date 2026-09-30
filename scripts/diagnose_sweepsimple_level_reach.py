"""Diagnose collision-checked leveling reach without executing physical actions."""

import json
from itertools import product
from pathlib import Path

import mujoco
import numpy as np
from probe_sweepsimple_pick import PickupProbe
from pybullet_helpers.geometry import Pose, multiply_poses
from pybullet_helpers.ikfast.utils import ikfast_closest_inverse_kinematics
from scipy.spatial.transform import Rotation

from hitl_pmp.environments.sweep_simple3d.controllers import FloorPrimitives
from hitl_pmp.environments.sweep_simple3d.session import SweepSimpleSession


class LevelReachDiagnosis:
    @staticmethod
    def run() -> None:
        source = Path("scratchpad/sweepsimple3d/small-level-west-v117")
        session = SweepSimpleSession(seed=0)
        primitive = None
        rows = []
        try:
            PickupProbe.restore_recorded(session=session, source=source, final=True)
            primitive = FloorPrimitives.create(session=session, distance=0.7, heading_offset=0.0)
            model, data = session.mj_model, session.mj_data
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wiper_0")
            blade = max(
                (g for g in range(model.ngeom) if model.geom_bodyid[g] == body),
                key=lambda g: float(model.geom_size[g][0]),
            )
            observed = Rotation.from_matrix(data.xmat[body].reshape(3, 3))
            yaw = float(np.arctan2(observed.as_matrix()[1, 0], observed.as_matrix()[0, 0]))
            delta = (Rotation.from_euler("z", yaw) * observed.inv()).as_rotvec()
            angle = float(np.linalg.norm(delta))
            held = multiply_poses(
                primitive.scene.ee_now().invert(),
                Pose(tuple(data.xpos[body]), tuple(observed.as_quat())),
            )
            world = np.array([
                data.geom_xpos[blade]
                + data.geom_xmat[blade].reshape(3, 3) @ (model.geom_size[blade] * signs)
                for signs in product((-1.0, 1.0), repeat=3)
            ])
            local = (world - data.xpos[body]) @ observed.as_matrix()
            support = int(np.argmin(world[:, 2]))
            bodies = primitive.scene.bodies(without_cubes=tuple(f"cube_{i}" for i in range(5)))
            for turn in (0.08, 0.04, 0.02):
                rotation = Rotation.from_rotvec(delta * min(1.0, turn / angle)) * observed
                origin = world[support] - rotation.apply(local[support])
                origin[2] += max(0.0, 0.001 - float((rotation.apply(local) + origin)[:, 2].min()))
                for dx, dy in ((0, 0), (0.02, 0), (-0.02, 0), (0, 0.02), (0, -0.02)):
                    position = origin + [dx, dy, 0]
                    target = multiply_poses(
                        Pose(tuple(position), tuple(rotation.as_quat())), held.invert()
                    )
                    primitive.scene.sync()
                    solutions = ikfast_closest_inverse_kinematics(
                        primitive.scene.robot, world_from_target=target
                    )
                    valid = [q for q in solutions if primitive.scene.within_arm_limits(arm=q[:7])]
                    valid.sort(
                        key=lambda q: float(np.linalg.norm(np.asarray(q[:7]) - session.arm()))
                    )
                    clear = [
                        q
                        for q in valid
                        if not primitive.scene.in_collision(
                            joints=primitive.scene.planning_fingers(arm=q, state=0.5),
                            bodies=bodies,
                            held=primitive.scene.wiper_body,
                            held_tf=held,
                        )
                    ]
                    path = primitive.scene.floor_descent(
                        start=session.arm(), target=target, bodies=bodies, held_tf=held
                    )
                    clear_ranks = [
                        i for i, q in enumerate(valid) if any(np.array_equal(q, c) for c in clear)
                    ]
                    row = dict(
                        turn=turn,
                        dx=dx,
                        dy=dy,
                        solutions=len(solutions),
                        clear_ranks=clear_ranks,
                        native_limits=len(valid),
                        clear=len(clear),
                        path_found=path is not None,
                        waypoints=None if path is None else len(path),
                    )
                    rows.append(row)
                    print(json.dumps(row), flush=True)
            (source / "level-reach-offsets.json").write_text(
                json.dumps(
                    {
                        "diagnostic_only": True,
                        "physical_ticks": session.ticks,
                        "rows": rows,
                    },
                    indent=2,
                )
            )
        finally:
            if primitive is not None:
                primitive.scene._sim.close()
            session.close()


if __name__ == "__main__":
    LevelReachDiagnosis.run()
