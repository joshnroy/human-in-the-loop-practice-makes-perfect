"""Run standard EES in the same-side scene, with continuous practice and recording.

No actions, throw parameters, skill targets, or recovery decisions are prescribed.
The output's period_videos/practice files are the practice footage; episode.mp4
is a separate evaluation episode and must not be presented as autonomous practice.
"""

import argparse
from pathlib import Path

from hitl_pmp.cli import Cli


class AutonomousDemo:
    @staticmethod
    def arguments(*, output_dir: Path, cycles: int, steps: int, seed: int) -> list[str]:
        if cycles <= 0 or steps <= 0:
            raise ValueError("cycles and steps must be positive")
        return [
            "--env",
            "tossing3d",
            "--method",
            "ees",
            "--layout",
            "same-side",
            "--seed",
            str(seed),
            "--canonical-seed",
            "125",
            "--practice-reset-policy",
            "never",
            "--num-cycles",
            str(cycles),
            "--max-steps-per-interaction",
            str(steps),
            "--num-test-tasks",
            "1",
            "--goal-pursuit-horizon",
            "2",
            "--sampler-max-train-iters",
            "200",
            "--record-sampler-draws",
            "--record-skill-competence",
            "--output-dir",
            str(output_dir),
        ]

    @staticmethod
    def run(*, output_dir: Path, cycles: int, steps: int, seed: int) -> None:
        Cli.main(
            argv=AutonomousDemo.arguments(
                output_dir=output_dir, cycles=cycles, steps=steps, seed=seed
            )
        )

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.add_argument("--cycles", type=int, default=1)
        parser.add_argument("--steps", type=int, default=16)
        parser.add_argument("--seed", type=int, default=0)
        args = parser.parse_args()
        AutonomousDemo.run(
            output_dir=args.output_dir, cycles=args.cycles, steps=args.steps, seed=args.seed
        )


if __name__ == "__main__":
    AutonomousDemo.main()
