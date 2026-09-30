"""Inspect fixed native starts without changing distribution or accepting exceptions."""

import json
from pathlib import Path
import kinder
from kinder.envs.dynamic3d.envs import ObjectCentricTidyBot3DEnv
from kinder_models.dynamic3d.utils import get_overhead_kinematic2ds, geom2ds_intersect

task = Path(kinder.__file__).parent / "envs/dynamic3d/tasks/Shelf3D/Shelf3D-o8.json"
env = ObjectCentricTidyBot3DEnv(task_config_path=str(task), num_objects=8, allow_state_access=True)
rows = []
for seed in [1016164991, *range(10)]:
    state, _ = env.reset(seed=seed)
    geoms = get_overhead_kinematic2ds(state)
    robot = state.get_object_from_name(env.robot_name)
    overlaps = [
        name
        for name, geometry in geoms.items()
        if name != robot.name and geom2ds_intersect(geoms[robot.name], geometry)
    ]
    contacts = {f"cube{i}": env.get_gripper_object_contacts(f"cube{i}") for i in range(1, 9)}
    row = {
        "seed": seed,
        "robot": {
            f: float(state.get(robot, f)) for f in ("pos_base_x", "pos_base_y", "pos_base_rot")
        },
        "native_footprint_overlaps": overlaps,
        "native_physical_robot_contacts": contacts,
    }
    rows.append(row)
    print(json.dumps(row), flush=True)
Path("scratchpad/shelf-forward-reset/native-start-audit.json").write_text(
    json.dumps(rows, indent=2)
)
env.close()
