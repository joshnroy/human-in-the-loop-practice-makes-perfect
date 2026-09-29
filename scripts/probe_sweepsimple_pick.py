"""One explicit native-scene pickup feasibility probe, not a learning run."""

import json
import sys
import time
from pathlib import Path


class PickupProbe:
    @staticmethod
    def run() -> None:
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "reference/kindergarden/src"))
        sys.path.insert(0, str(root / "reference/kinder-baselines/kinder-models/src"))
        from hitl_pmp.environments.sweep_simple3d.controllers import FloorPrimitives
        from hitl_pmp.environments.sweep_simple3d.session import SweepSimpleSession

        output = root / "scratchpad/sweepsimple3d/forward-seed0-v15"
        output.mkdir(parents=True, exist_ok=True)
        session = SweepSimpleSession(
            seed=0, log_path=output / "state.jsonl", replay_path=output / "replay.jsonl"
        )
        primitive = FloorPrimitives.create(session=session, distance=0.7, heading_offset=0.0)
        started = time.monotonic()
        session.begin(name="PickFloorWiper", kind="PickFloorWiper", phase="feasibility")
        error = ""
        success = False
        try:
            note = primitive.recover_wiper()
            session.end(success=True, note=note)
            session.begin(name="SweepCubeToGoal", kind="SweepCubeToGoal", phase="feasibility")
            note = primitive.sweep_cube(
                cube="cube_0", region="sweep_region", distance=0.55, heading_offset=0.0
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
            }
            (output / "result.json").write_text(json.dumps(result, indent=2))
            print(json.dumps(result), flush=True)
            primitive.scene._sim.close()
            session.close()


if __name__ == "__main__":
    PickupProbe.run()
