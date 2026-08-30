"""Capture initial scenes and measured poses, without executing or resetting a policy."""

import argparse
import json
from pathlib import Path

import imageio.v3 as iio

from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.layout import Tossing3DLayout


class LayoutDemo:
    @staticmethod
    def capture(*, output_dir: Path, seed: int) -> list[dict]:
        output_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for layout in Tossing3DLayout:
            env = Tossing3DEnvironment(layout=layout, canonical_seed=seed)
            try:
                env.hard_reset()
                state = env.get_current_state()
                objects = {}
                for obj in (env.robot, env.cube, env.bin, env.barrier):
                    objects[obj.name] = {
                        axis: float(
                            state.get(
                                obj=obj,
                                feature_name=f"pos_base_{axis}" if obj == env.robot else axis,
                            )
                        )
                        for axis in ("x", "y")
                    }
                records.append({"layout": layout.value, "seed": seed, "objects": objects})
                iio.imwrite(output_dir / f"{layout.value}.png", env.backend().render())
            finally:
                env.close()
        (output_dir / "poses.json").write_text(json.dumps(records, indent=2) + "\n")
        return records

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.add_argument("--seed", type=int, default=125)
        args = parser.parse_args()
        LayoutDemo.capture(output_dir=args.output_dir, seed=args.seed)


if __name__ == "__main__":
    LayoutDemo.main()
