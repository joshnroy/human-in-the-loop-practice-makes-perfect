"""Privileged passive-food diagnostic, not robot transfer or hardware evidence."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from hitl_pmp.environments.racktoserving3d.config import RackServingConfig
from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--contact-dimension", type=int, choices=(3, 4, 6), default=6)
    parser.add_argument("--rolling-friction", type=float, default=0.003)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new evidence path")
    config = RackServingConfig(
        dumpling_contact_dimension=args.contact_dimension,
        support_contact_dimension=args.contact_dimension,
        dumpling_friction=(0.45, 0.005, args.rolling_friction),
        rack_friction=(0.6, 0.005, args.rolling_friction),
        tray_friction=(0.6, 0.005, args.rolling_friction),
    )
    env = RackServingEnvironment(config=config, scene_bg=False)
    report = {
        "privileged_setup": True,
        "production_model_modified": False,
        "config": config.model_dump(),
        "cases": [],
    }
    try:
        env.reset_to_seed(seed=0)
        backend = env.backend()
        model, data = backend.model(), backend.data()
        body = model.body("cube_0").id
        address = model.jnt_dofadr[model.body_jntadr[body]]
        food_geom = next(iter(backend.body_geoms(name="cube_0")))
        tray = backend.support(name="box_0")
        report["geometry_contacts"] = {
            model.geom(geom).name: {
                "dimension": int(model.geom_condim[geom]),
                "priority": int(model.geom_priority[geom]),
                "friction": model.geom_friction[geom].tolist(),
            }
            for geom in backend.body_geoms(name="cube_0") | backend.body_geoms(name="box_0")
        }
        initial_qpos, initial_qvel = data.qpos.copy(), data.qvel.copy()
        for tilt, drop, speed in [(0, 0.001, 0.03), (15, 0.013, 0.03), (45, 0.013, 0.03)]:
            data.qpos[:] = initial_qpos
            data.qvel[:] = initial_qvel
            data.qacc_warmstart[:] = 0
            data.time = 0
            rotation = Rotation.from_euler("y", tilt, degrees=True)
            height = float(
                np.sqrt(np.sum((rotation.as_matrix()[2] * model.geom_size[food_geom]) ** 2))
            )
            position = np.array([tray.x_max - 0.05, 0, tray.top_z + height + drop])
            backend._commit(placements={"cube_0": (position, rotation.as_quat()[[3, 0, 1, 2]])})
            data.qvel[address] = -speed
            trace = []
            for tick in range(int(5 / model.opt.timestep)):
                backend.sim().step()
                if tick % 200 == 0:
                    trace.append({
                        "seconds": tick * float(model.opt.timestep),
                        "position": data.xpos[body].tolist(),
                        "velocity": data.qvel[address : address + 6].tolist(),
                        "contact_dimensions": sorted({
                            int(contact.dim)
                            for contact in data.contact[: data.ncon]
                            if food_geom in (contact.geom1, contact.geom2)
                        }),
                    })
            report["cases"].append({
                "tilt_degrees": tilt,
                "drop_m": drop,
                "initial_speed_m_s": speed,
                "final_location": backend.location(name="cube_0"),
                "trace": trace,
            })
        expected = args.contact_dimension
        report["contact_dimensions_verified"] = all(
            contact_dim == expected
            for case in report["cases"]
            for sample in case["trace"]
            for contact_dim in sample["contact_dimensions"]
        )
        if not report["contact_dimensions_verified"]:
            raise RuntimeError("compiled contact dimensions differ from the requested model")
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        env.close()


if __name__ == "__main__":
    main()
