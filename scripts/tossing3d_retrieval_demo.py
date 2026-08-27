"""Physical controller regression, explicitly scripted (not an EES demonstration)."""

import argparse
import json
from pathlib import Path

import imageio.v2 as iio
import numpy as np

from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.layout import Tossing3DLayout
from hitl_pmp.environments.tossing3d.predicates import HOLDING, IN_BIN, REACHABLE


class RetrievalDemo:
    @staticmethod
    def capture(*, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        env = Tossing3DEnvironment(layout=Tossing3DLayout.SAME_SIDE, canonical_seed=125)
        records = []
        try:
            env.set_substep_recording(enabled=True)
            env.hard_reset()
            with iio.get_writer(
                output_dir / "retrieval.mp4", fps=env.backend().render_fps()
            ) as video:
                for frame in env.drain_substep_frames():
                    video.append_data(frame)
                for name, action in (
                    ("initial floor pick", [0, 0, 0, 0, 0]),
                    ("throw 1", [1, 1.35, 0, 130, 792]),
                    ("retrieve 1", None),
                    ("throw 2", [1, 1.35, 0, 115, 700]),
                    ("retrieve 2", None),
                ):
                    state = env.get_current_state()
                    if action is None:
                        action = [3 if IN_BIN.holds(state, (env.cube, env.bin)) else 0, 0, 0, 0, 0]
                    env.take_action(action=np.asarray(action, dtype=float))
                    frames = env.drain_substep_frames()
                    for frame in frames:
                        video.append_data(frame)
                    state = env.get_current_state()
                    record = {
                        "step": name,
                        "action": action,
                        "frames": len(frames),
                        "holding": HOLDING.holds(state, (env.robot, env.cube)),
                        "in_bin": IN_BIN.holds(state, (env.cube, env.bin)),
                        "reachable": REACHABLE.holds(state, (env.cube, env.barrier)),
                        "error": env.last_skill_error(),
                        "cube": {
                            axis: float(state.get(obj=env.cube, feature_name=axis))
                            for axis in ("x", "y", "z")
                        },
                    }
                    records.append(record)
                    (output_dir / "results.json").write_text(json.dumps(records, indent=2) + "\n")
                    print(json.dumps(record), flush=True)
                    if action[0] in (0, 3) and not record["holding"]:
                        raise RuntimeError(f"Physical pickup failed: {record}")
        finally:
            env.close()

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--output-dir", type=Path, required=True)
        args = parser.parse_args()
        RetrievalDemo.capture(output_dir=args.output_dir)


if __name__ == "__main__":
    RetrievalDemo.main()
