"""Read native SweepSimple3D starts and goal semantics; never run a learner."""

import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


class SweepSimpleAudit:
    @staticmethod
    def run() -> None:
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "reference/kindergarden/src"))
        sys.path.insert(0, str(root / "reference/kinder-baselines/kinder-models/src"))
        from hitl_pmp.environments.sweep_simple3d.physical.session import KinderImports

        kinder = KinderImports.load()
        destination = root / "scratchpad/sweepsimple3d"
        destination.mkdir(parents=True, exist_ok=True)
        records = []
        for seed in [0, 1, 2, *range(10000, 10010)]:
            env = kinder.make(
                "kinder/SweepSimple3D-o5-sweep_the_blocks_to_the_left_side_of_the_kitchen_island-v0",
                render_mode="rgb_array",
                allow_state_access=True,
            )
            try:
                observation, _ = env.reset(seed=seed)
                state = env.observation_space.devectorize(observation)
                core = env.unwrapped._object_centric_env
                checks = {}
                strict_regions = {}
                poses = {}
                for _, name, region in core.task_config["initial_state"]:
                    obj = state.get_object_from_name(name)
                    if name == "robot":
                        position = np.array(
                            [state.get(obj, "pos_base_x"), state.get(obj, "pos_base_y"), 0],
                            dtype=np.float32,
                        )
                        yaw = float(state.get(obj, "pos_base_rot"))
                    else:
                        position = np.array(
                            [state.get(obj, c) for c in ("x", "y", "z")], dtype=np.float32
                        )
                        quaternion = [state.get(obj, c) for c in ("qx", "qy", "qz", "qw")]
                        yaw = float(Rotation.from_quat(quaternion).as_euler("zyx")[0])
                    config = core.task_config["regions"][region]
                    assert config["target"] == "ground"
                    strict = bool(
                        core._ground_fixture.check_in_region(position, region, core._robot_env)
                    )
                    strict_regions[name] = strict
                    accepted = strict
                    if name == "wiper_0" and not strict:
                        for support_delta in (-0.01, 0.01):
                            shifted = position.copy()
                            shifted[2] += support_delta
                            accepted |= bool(
                                core._ground_fixture.check_in_region(
                                    shifted, region, core._robot_env
                                )
                            )
                    checks[name + ":region"] = accepted
                    degrees = float(np.rad2deg(yaw) % 360)
                    checks[name + ":yaw"] = any(
                        high - low >= 360
                        or any(
                            low <= candidate <= high
                            for candidate in (degrees - 360, degrees, degrees + 360)
                        )
                        for low, high in config.get("yaw_ranges", [[0, 360]])
                    )
                    poses[name] = [*position.tolist(), degrees]
                before = env.observation_space.vectorize(core._get_current_state()).copy()
                native_goal = bool(core._check_goals())
                checks["goal_read_is_pure"] = bool(
                    np.array_equal(
                        before, env.observation_space.vectorize(core._get_current_state())
                    )
                )
                core.set_state(state.copy())
                checks["initial_state_roundtrip"] = bool(
                    np.allclose(
                        before,
                        env.observation_space.vectorize(core._get_current_state()),
                        atol=1e-7,
                        rtol=0,
                    )
                )
                if seed == 0:
                    Image.fromarray(env.render()).save(destination / "native-seed0.png")
                row = {
                    "seed": seed,
                    "strict_native_region_checks": strict_regions,
                    "wiper_support_tolerance_m": 0.01,
                    "checks": checks,
                    "poses": poses,
                    "native_goal": native_goal,
                    "valid_start_with_shared_support_tolerance": all(
                        v for k, v in checks.items() if k.endswith(":region") or k.endswith(":yaw")
                    ),
                }
                records.append(row)
                print(json.dumps(row), flush=True)
                (destination / "native-audit.json").write_text(
                    json.dumps({"kinder_path": kinder.__file__, "records": records}, indent=2)
                )
            finally:
                env.close()


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    SweepSimpleAudit.run()
