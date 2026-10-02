"""Bounded final-state blade-leveling replay; never a native-start readiness trial."""

import argparse
import faulthandler
import hashlib
import json
import shutil
import time
import traceback
from itertools import product
from pathlib import Path

import mujoco
import numpy as np
from probe_sweepsimple_pick import PickupProbe

from hitl_pmp.environments.sweep_simple3d.controllers import FloorGrip, FloorPrimitives
from hitl_pmp.environments.sweep_simple3d.session import SweepSimpleSession


class LevelProbe:
    @staticmethod
    def measure(*, primitive, cube):
        session = primitive.session
        model, data = session.mj_model, session.mj_data
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wiper_0")
        blade = max(
            (g for g in range(model.ngeom) if model.geom_bodyid[g] == body),
            key=lambda g: float(model.geom_size[g][0]),
        )
        cube_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cube)
        cube_geom = next(g for g in range(model.ngeom) if model.geom_bodyid[g] == cube_body)
        rotation = data.geom_xmat[blade].reshape(3, 3)
        cube_rotation = data.geom_xmat[cube_geom].reshape(3, 3)
        corners = np.array([
            data.geom_xpos[cube_geom] + cube_rotation @ (model.geom_size[cube_geom] * signs)
            for signs in product((-1.0, 1.0), repeat=3)
        ])
        local = (corners - data.geom_xpos[blade]) @ rotation
        half = model.geom_size[blade]
        lower = max(-half[1], float(local[:, 1].min()))
        upper = min(half[1], float(local[:, 1].max()))
        front = float(np.sign(local[:, 0].mean())) * half[0]
        end_height = (
            float(
                data.geom_xpos[blade][2]
                + rotation[2, 0] * front
                + max(rotation[2, 1] * lower, rotation[2, 1] * upper)
                - rotation[2, 2] * half[2]
            )
            if lower <= upper
            else None
        )
        ceiling = primitive.cube_contact_ceiling(cube=cube)
        return {
            "tick": session.ticks,
            "cube_facing_end_height": end_height,
            "cube_ceiling": ceiling,
            "vertical_overlap_attained": end_height is not None and end_height <= ceiling,
            "lateral_interval": [lower, upper],
            "blade_minimum_height": primitive.blade_minimum_height(),
            "controller_edge_height": primitive.blade_bottom_height(cube=cube),
            "tool_tilt": float(np.arccos(np.clip(data.xmat[body].reshape(3, 3)[2, 2], -1, 1))),
            "bilateral_grip": FloorGrip.has_bilateral_contact(session=session),
            "native_qpos": data.qpos.tolist(),
        }

    @staticmethod
    def run():
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--source", default="small-level-west-v117")
        parser.add_argument("--tag", required=True)
        parser.add_argument("--cube", choices=[f"cube_{i}" for i in range(5)], default="cube_0")
        parser.add_argument("--iterations", type=int, choices=range(1, 7), default=6)
        parser.add_argument("--operation", choices=("level", "ground_clearance"), default="level")
        args = parser.parse_args()
        root = Path(__file__).resolve().parents[1]
        source = root / "scratchpad/sweepsimple3d" / args.source
        output = root / "scratchpad/sweepsimple3d" / args.tag
        output.mkdir(parents=True, exist_ok=False)
        snapshot = output / "source"
        snapshot.mkdir()
        files = [
            Path(__file__),
            root / "scripts/probe_sweepsimple_pick.py",
            root / "src/hitl_pmp/environments/sweep_simple3d/controllers.py",
            root / "src/hitl_pmp/environments/sweep_simple3d/physical/motion.py",
        ]
        hashes = {}
        for path in files:
            shutil.copy2(path, snapshot / path.name)
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        seed = json.loads((source / "probe_manifest.json").read_text())["arguments"]["seed"]
        manifest = {
            "arguments": vars(args),
            "seed": seed,
            "source_sha256": hashes,
            "end_to_end_native_start": False,
            "resume_phase": "final_recorded_state",
            "purpose": "leveling diagnostics only; no science or readiness counts",
        }
        (output / "probe_manifest.json").write_text(json.dumps(manifest, indent=2))
        faulthandler.enable()
        faulthandler.dump_traceback_later(120, repeat=True)
        session = SweepSimpleSession(
            seed=seed, log_path=output / "state.jsonl", replay_path=output / "replay.jsonl"
        )
        primitive = None
        started = time.monotonic()
        report = {
            "end_to_end_native_start": False,
            "resume_phase": "final_recorded_state",
            "resumed_final": args.source,
            "purpose": manifest["purpose"],
            "attempts": [],
            "success": False,
            "error": None,
        }
        try:
            PickupProbe.restore_recorded(session=session, source=source, final=True)
            primitive = FloorPrimitives.create(session=session, distance=0.7, heading_offset=0.0)
            primitive.require_handle(phase="restored leveling diagnostic")
            session.begin(
                name="DiagnosticBladeLevel", kind="DiagnosticBladeLevel", phase="diagnostic"
            )
            bodies = primitive.scene.bodies(without_cubes=tuple(f"cube_{i}" for i in range(5)))
            for index in range(args.iterations):
                attempt = {
                    "iteration": index,
                    "before": LevelProbe.measure(primitive=primitive, cube=args.cube),
                }
                report["attempts"].append(attempt)
                progress = (
                    primitive.level_blade(bodies=bodies)
                    if args.operation == "level"
                    else primitive.raise_blade_clear_of_ground(bodies=bodies)
                )
                attempt.update(
                    progress=progress, after=LevelProbe.measure(primitive=primitive, cube=args.cube)
                )
                session._write(record={"kind": "level_probe_iteration", **attempt})
                print(json.dumps(attempt), flush=True)
                attained = (
                    attempt["after"]["vertical_overlap_attained"]
                    if args.operation == "level"
                    else attempt["after"]["blade_minimum_height"] >= 0.005
                )
                if attained:
                    report["success"] = True
                    report["stop_reason"] = (
                        "selected native geometric objective attained; diagnostic only"
                    )
                    break
                if not progress:
                    report["stop_reason"] = "selected controller returned no progress"
                    break
            else:
                report["stop_reason"] = "bounded iteration limit"
        except Exception as error:
            report.update(
                error=repr(error), traceback=traceback.format_exc(), stop_reason="exception"
            )
        finally:
            report.update(ticks=session.ticks, elapsed=time.monotonic() - started)
            if primitive is not None:
                report["final"] = LevelProbe.measure(primitive=primitive, cube=args.cube)
            if session._current is not None:
                session.end(success=report["success"], note=report.get("stop_reason", "diagnostic"))
            (output / "result.json").write_text(json.dumps(report, indent=2))
            print(json.dumps(report), flush=True)
            if primitive is not None:
                primitive.scene._sim.close()
            session.close()
            faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    LevelProbe.run()
