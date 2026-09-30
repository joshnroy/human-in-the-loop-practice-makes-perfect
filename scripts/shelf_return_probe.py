"""Individual return-to-ground diagnostic from an actual completed retrieval."""

import argparse, json, pickle, traceback
from pathlib import Path
import numpy as np
import mujoco
from shelf_physical_demo import Demo
from shelf_reverse_controllers import ReturnCubeToStartController
from kinder_models.dynamic3d.utils import (
    get_target_robot_pose_from_parameters,
    get_overhead_object_se2_pose,
)

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
saved = pickle.loads((args.source / "reverse-final.pkl").read_bytes())
demo = Demo(output=args.output, seed=1, max_actions=1)
demo.env.set_state(saved["state"])
demo.state = saved["state"]
demo.initial = saved["initial"]
demo.records = saved["actions"]
demo.ticks = saved["ticks"]
cube = demo.state.get_object_from_name("cube1")
model = demo.env._robot_env.sim.model.mj_model
limits = []
for index in range(1, 8):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{demo.robot.name}_joint_{index}")
    limits.append(tuple(model.jnt_range[jid]) if model.jnt_limited[jid] else None)
rows = []


def supported_in_initial_region(state):
    position = np.array([state.get(cube, f) for f in ("x", "y", "z")])
    if not demo.env._ground_fixture.check_in_region(
        position, "ground_1_object_init_region", demo.env._robot_env
    ):
        return False
    data = demo.env._robot_env.sim.data.mj_data
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cube.name)
    for contact in data.contact[: data.ncon]:
        a, b = int(contact.geom1), int(contact.geom2)
        if (
            model.geom_bodyid[a] == body_id and model.geom_type[b] == mujoco.mjtGeom.mjGEOM_PLANE
        ) or (
            model.geom_bodyid[b] == body_id and model.geom_type[a] == mujoco.mjtGeom.mjGEOM_PLANE
        ):
            if abs(contact.frame[2]) > 0.95 and contact.dist <= 0.001:
                return True
    return False


try:
    native_pick = demo.records[0]["params"]
    for distance, heading in [native_pick, [0.55, 0.0], [0.6, 0.0], [0.5, 0.0]]:
        pose = get_target_robot_pose_from_parameters(
            get_overhead_object_se2_pose(demo.initial, cube), distance, heading
        )
        target = [demo.initial.get(cube, f) for f in ("x", "y", "z", "qx", "qy", "qz", "qw")]
        target[2] += 0.003
        params = np.array([pose.x, pose.y, pose.theta(), *target])
        controller = ReturnCubeToStartController(objects=[demo.robot, cube], pybullet_sim=demo.sim)
        controller.physical_joint_limits = limits
        controller.supported_in_initial_region = supported_in_initial_region
        row = {"params": params.tolist()}
        try:
            controller.reset(demo.state, params)
            row["planned"] = True
            rows.append(row)
            demo.execute("return_cube_to_start", cube, controller, params)
            break
        except Exception as error:
            row["error"] = str(error)
            row["traceback"] = traceback.format_exc()
            rows.append(row)
finally:
    (args.output / "return-planning.json").write_text(json.dumps(rows, indent=2))
    demo.save()
    demo.snapshot("return-final")
    demo.writer.close()
    demo.env.close()
