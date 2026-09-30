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
        parser.add_argument(
            "--resume-pick", help="Development replay tag; excluded from end-to-end readiness"
        )
        parser.add_argument("--grasp-yaw-offset", type=float, default=0.0)
        args = parser.parse_args()
        import faulthandler

        faulthandler.dump_traceback_later(120, repeat=True)
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
        if args.resume_pick:
            PickupProbe.restore_pick(
                session=session, source=root / "scratchpad/sweepsimple3d" / args.resume_pick
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
        if args.grasp_yaw_offset:
            original_yaw = FloorPrimitives.wiper_grasp_yaw
            FloorPrimitives.wiper_grasp_yaw = lambda self, *, axis: (
                original_yaw(self, axis=axis) + args.grasp_yaw_offset
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
        error_traceback = ""
        success = False
        try:
            note = (
                "Recorded picked-state continuation; not an end-to-end native-start trial"
                if args.resume_pick
                else primitive.recover_wiper()
            )
            primitive.require_handle(phase="verified pickup or recorded continuation")
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
            import traceback

            error = repr(exc)
            error_traceback = traceback.format_exc()
            print(error_traceback, flush=True)
            note = error
        finally:
            session.end(success=success, note=note)
            result = {
                "success": success,
                "end_to_end_native_start": args.resume_pick is None,
                "resumed_pick": args.resume_pick,
                "error": error,
                "traceback": error_traceback,
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
            faulthandler.cancel_dump_traceback_later()

    @staticmethod
    def restore_pick(*, session, source: Path) -> None:
        """Restore a logged physical pickup for labeled controller debugging only."""
        import mujoco

        source_manifest = json.loads((source / "probe_manifest.json").read_text())
        if source_manifest["arguments"]["seed"] != session.seed:
            raise ValueError("Replay and requested native seeds differ")
        picked = None
        subsequent_sweep = False
        for line in (source / "state.jsonl").open():
            record = json.loads(line)
            if record.get("kind") == "tick" and record.get("step") == "PickFloorWiper":
                picked = record
            if record.get("step") == "SweepCubeToGoal":
                subsequent_sweep = True
        if picked is None or not subsequent_sweep:
            raise ValueError("Source does not establish a completed pickup followed by a sweep")
        state = session.state.copy()
        for name, values in picked["state"].items():
            obj = state.get_object_from_name(name)
            for key, value in zip(state.type_features[obj.type], values, strict=True):
                is_velocity = key.startswith("vel_") or key in {"vx", "vy", "vz", "wx", "wy", "wz"}
                state.set(obj, key, 0.0 if is_velocity else value)
        session.env.unwrapped._object_centric_env.set_state(state)
        session._state = state
        frame = None
        for line in (source / "replay.jsonl").open():
            record = json.loads(line)
            if record.get("t") == picked["t"]:
                frame = record
                break
        if frame is None:
            raise ValueError("Native joint replay frame is missing")
        session.mj_data.qpos[:] = frame["qpos"]
        session.mj_data.qvel[:] = 0
        mujoco.mj_forward(session.mj_model, session.mj_data)
        session._write(
            record={
                "kind": "development_replay_resume",
                "source": str(source),
                "source_tick": picked["t"],
                "source_manifest_sha256": hashlib.sha256(
                    (source / "probe_manifest.json").read_bytes()
                ).hexdigest(),
                "velocities": "zeroed; complete native velocities were not recorded",
                "end_to_end_native_start": False,
            }
        )
        session._tick()


if __name__ == "__main__":
    PickupProbe.run()
