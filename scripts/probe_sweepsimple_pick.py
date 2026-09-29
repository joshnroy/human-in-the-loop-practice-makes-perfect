"""One explicit native-scene pickup feasibility probe, not a learning run."""

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path


class PickupProbe:
    @staticmethod
    def run() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--tag", required=True)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--pick-distance", type=float, default=0.7)
        parser.add_argument("--sweep-distance", type=float, default=0.55)
        parser.add_argument("--sweep-angle", type=float, default=0.0)
        parser.add_argument("--pick-only", action="store_true")
        parser.add_argument("--grasp-offset", type=float)
        parser.add_argument("--grasp-height", type=float, default=0.0)
        parser.add_argument("--grasp-mode", choices=("handle", "blade"), default="handle")
        parser.add_argument("--tilt-limit", type=float, default=1.1)
        parser.add_argument("--stroke-length", type=float, default=0.132)
        parser.add_argument("--contact-step", type=float, default=0.003)
        parser.add_argument("--narrow-contact", action="store_true")
        args = parser.parse_args()
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "reference/kindergarden/src"))
        sys.path.insert(0, str(root / "reference/kinder-baselines/kinder-models/src"))
        from hitl_pmp.environments.sweep_simple3d.controllers import FloorPrimitives
        from hitl_pmp.environments.sweep_simple3d.session import SweepSimpleSession

        output = root / "scratchpad/sweepsimple3d" / args.tag
        if output.exists():
            raise FileExistsError(output)
        output.mkdir(parents=True, exist_ok=True)
        source_dir = output / "source"
        source_dir.mkdir()
        shutil.copy2(__file__, source_dir / "probe_driver.py")
        hashes = {}
        for source in [
            *root.glob("src/hitl_pmp/environments/sweep_simple3d/*.py"),
            *[
                root / "src/hitl_pmp/environments/sweep_drawer3d" / name
                for name in ("primitives.py", "session.py", "motion.py", "planning_scene.py")
            ],
        ]:
            label = source.parent.name + "__" + source.name
            shutil.copy2(source, source_dir / label)
            hashes[label] = hashlib.sha256(source.read_bytes()).hexdigest()
        (output / "probe_manifest.json").write_text(
            json.dumps({"arguments": vars(args), "source_sha256": hashes}, indent=2)
        )
        session = SweepSimpleSession(
            seed=args.seed, log_path=output / "state.jsonl", replay_path=output / "replay.jsonl"
        )
        primitive = FloorPrimitives.create(
            session=session, distance=args.pick_distance, heading_offset=0.0
        )
        primitive.scene.max_tool_tilt = args.tilt_limit
        primitive.contact_stroke_length = args.stroke_length
        primitive.contact_step = args.contact_step
        primitive.narrow_contact = args.narrow_contact
        if args.grasp_mode == "blade":
            import mujoco
            import numpy as np

            def blade_geometry(self):  # noqa: PLR0917 -- installed as a bound probe method
                model = self.session.mj_model
                body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wiper_0")
                blade = max(
                    (g for g in range(model.ngeom) if model.geom_bodyid[g] == body),
                    key=lambda g: float(model.geom_size[g][0]),
                )
                return blade, 0

            FloorPrimitives.wiper_handle_geometry = blade_geometry
            FloorPrimitives.wiper_approach_angles = lambda self: (0.0, 0.4, 0.7)
            FloorPrimitives.wiper_grasp_yaw = lambda self, *, axis: float(
                np.arctan2(axis[1], axis[0]) + np.pi / 2
            )
        if args.grasp_height:
            import numpy as np

            FloorPrimitives.wiper_grasp_point = lambda self, *, center, axis, along: (
                center + along * axis + np.array([0.0, 0.0, args.grasp_height])
            )
        if args.grasp_offset is not None:
            FloorPrimitives.wiper_grasp_offsets = lambda self: (args.grasp_offset,)
        import mujoco

        started = time.monotonic()
        session.begin(name="PickFloorWiper", kind="PickFloorWiper", phase="feasibility")
        error = ""
        success = False
        try:
            note = primitive.recover_wiper()
            import mujoco

            contacts = []
            for contact in session.mj_data.contact:
                names = [
                    mujoco.mj_id2name(
                        session.mj_model, mujoco.mjtObj.mjOBJ_BODY, session.mj_model.geom_bodyid[g]
                    )
                    for g in (contact.geom1, contact.geom2)
                ]
                if "wiper_0" in names:
                    contacts.append(names)
            print(
                json.dumps({
                    "phase": "after_pick",
                    "contacts": contacts,
                    "pads": [p.tolist() for p in primitive.pad_centers()],
                    "tool": session.position(name="wiper_0").tolist(),
                }),
                flush=True,
            )
            if not args.pick_only:
                session.end(success=True, note=note)
                session.begin(name="SweepCubeToGoal", kind="SweepCubeToGoal", phase="feasibility")
                note = primitive.sweep_cube(
                    cube="cube_0",
                    region="sweep_region",
                    distance=args.sweep_distance,
                    heading_offset=args.sweep_angle,
                )
            success = True
        except Exception as exc:
            error = repr(exc)
            note = error
        finally:
            session.end(success=success, note=note)
            result = {
                "success": success,
                "error": error,
                "note": note,
                "elapsed": time.monotonic() - started,
                "ticks": session.ticks,
                "wiper_position": session.position(name="wiper_0").tolist(),
                "final_contacts": [
                    [
                        mujoco.mj_id2name(
                            session.mj_model,
                            mujoco.mjtObj.mjOBJ_BODY,
                            session.mj_model.geom_bodyid[g],
                        )
                        for g in (contact.geom1, contact.geom2)
                    ]
                    for contact in session.mj_data.contact
                    if any(
                        mujoco.mj_id2name(
                            session.mj_model,
                            mujoco.mjtObj.mjOBJ_BODY,
                            session.mj_model.geom_bodyid[g],
                        )
                        == "wiper_0"
                        for g in (contact.geom1, contact.geom2)
                    )
                ],
            }
            (output / "result.json").write_text(json.dumps(result, indent=2))
            print(json.dumps(result), flush=True)
            primitive.scene._sim.close()
            session.close()


if __name__ == "__main__":
    PickupProbe.run()
