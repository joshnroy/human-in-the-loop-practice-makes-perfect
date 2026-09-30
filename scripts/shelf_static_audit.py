"""Read-only audit of the exact native Shelf planning collision geometry."""

import json
import pickle
from pathlib import Path

from kinder.envs.dynamic3d.envs import ObjectCentricTidyBot3DEnv  # EGL initialization
from kinder_models.dynamic3d.utils import get_overhead_kinematic2ds, get_bounding_box
from kinder.envs.dynamic3d.object_types import MujocoTidyBotRobotObjectType
from kinder_models.dynamic3d.utils import geom2ds_intersect

root = Path("scratchpad/shelf-forward-reset")
rows = []
for folder in ["native-v1", "native-seed0-v1"]:
    state = pickle.loads((root / folder / "initial.pkl").read_bytes())["state"]
    geoms = get_overhead_kinematic2ds(state)
    (robot,) = state.get_objects(MujocoTidyBotRobotObjectType)
    collisions = []
    for name, geometry in geoms.items():
        if name != robot.name and geom2ds_intersect(geoms[robot.name], geometry):
            obj = state.get_object_from_name(name)
            collisions.append({
                "name": name,
                "state": {
                    f: float(state.get(obj, f))
                    for f in ("x", "y", "z", "bb_x", "bb_y", "bb_z", "qw", "qx", "qy", "qz")
                },
            })
    rows.append({"folder": folder, "initial_collisions": collisions})
(root / "initial-collision-audit.json").write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2))
