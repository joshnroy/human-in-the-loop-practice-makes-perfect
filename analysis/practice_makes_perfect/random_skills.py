"""Generates the Random Skills baseline's evaluation-success-rate curve on Light
Switch, directly comparable to Figure 4's Light Switch panel from the "Practice
Makes Perfect" paper (see CLAUDE.md's Notion link). The paper reports Random
Skills flat at 0% across the entire x-axis there ("Like the Random Skills
baseline, MAPLE-Q fails to solve any evaluation tasks"); this reproduction
should match.

Pure output-dir -> plots/table transform: runs the experiment by invoking
`python -m hitl_pmp.cli --env lightswitch --method random-skills ...` once per
seed (with --output-dir and --gif), then reads each run's stats.json back in --
no simulation logic of its own. This mirrors how a real "run the experiment,
then analyze the results" pipeline should be split: methods/practice_makes_
perfect/cli.py's RandomSkillsCli owns wiring/running/writing, this script only
ever reads what it wrote.

Run with: python -m analysis.practice_makes_perfect.random_skills
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

_NUM_SEEDS = 3  # matches this reproduction's established seed count (see methods/README.md)
_GRID_SIZE = 100  # matches the paper's own Light Switch default (LightSwitchEnvironment.grid_size)
# TODO(scale): num_cycles/num_test_tasks are reduced well below what the paper
# itself uses, purely so this script finishes in a few minutes rather than
# ~20+ -- SkillGrounder's brute-force grounding cost, and so per-episode cost,
# grows steeply with grid_size (one full unsolved episode measured ~2.8s at
# grid_size=100 vs. ~0.04s at grid_size=20, so keeping grid_size faithful to
# the paper means the *other* knobs have to shrink instead). Random Skills
# never learns, so this shouldn't qualitatively change the near-0% result --
# fewer checkpoints/test tasks just means a coarser, noisier read on a curve
# that's flat regardless.
_NUM_CYCLES = 5
_STEPS_PER_CYCLE = 20
_NUM_TEST_TASKS = 5
_RESULTS_DIR = Path("analysis/practice_makes_perfect/results")
_PLOT_PATH = _RESULTS_DIR / "random_skills_light_switch.png"
_GIF_PATH = _RESULTS_DIR / "random_skills_episode.gif"


class RandomSkillsAnalysis:
    """Static-method container, never instantiated, same as every other
    business-logic class in this project -- runs the CLI once per seed, reads
    each run's --output-dir back in, prints a metrics table, and saves a
    comparison plot + episode gif (see module docstring for why this is a pure
    output-dir transform rather than a simulation)."""

    @staticmethod
    def _run_cli(*, seed: int, output_dir: Path) -> None:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "hitl_pmp.cli",
                "--env",
                "lightswitch",
                "--method",
                "random-skills",
                "--grid-size",
                str(_GRID_SIZE),
                "--seed",
                str(seed),
                "--num-cycles",
                str(_NUM_CYCLES),
                "--steps-per-cycle",
                str(_STEPS_PER_CYCLE),
                "--num-test-tasks",
                str(_NUM_TEST_TASKS),
                "--output-dir",
                str(output_dir),
                "--gif",
            ],
            check=True,
            capture_output=True,
        )

    @staticmethod
    def _load_curve(*, output_dir: Path) -> list[tuple[int, float]]:
        stats = json.loads((output_dir / "stats.json").read_text())
        curve: list[tuple[int, float]] = [tuple(pair) for pair in stats["task_training_curve"]]  # type: ignore[misc]
        return curve

    @staticmethod
    def _print_metrics_table(*, curves: list[list[tuple[int, float]]]) -> None:
        header = [
            "Online Transitions",
            *(f"Seed {s} % Solved" for s in range(_NUM_SEEDS)),
            "Mean",
            "Std",
        ]
        print("| " + " | ".join(header) + " |")
        print("|" + "---|" * len(header))
        transitions = [t for t, _ in curves[0]]
        for row, num_transitions in enumerate(transitions):
            percents = [curves[s][row][1] * 100 for s in range(_NUM_SEEDS)]
            mean, std = float(np.mean(percents)), float(np.std(percents))
            cells = " | ".join(f"{p:.1f}%" for p in percents)
            print(f"| {num_transitions} | {cells} | {mean:.1f}% | {std:.1f}% |")

    @staticmethod
    def _plot(*, curves: list[list[tuple[int, float]]]) -> None:
        transitions = [t for t, _ in curves[0]]
        percents = np.array([[pct * 100 for _, pct in curve] for curve in curves])
        mean, std = percents.mean(axis=0), percents.std(axis=0)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(transitions, mean, label="Random Skills", color="tab:blue")
        ax.fill_between(transitions, mean - std, mean + std, alpha=0.2, color="tab:blue")
        ax.set_xlabel("Number of Online Transitions")
        ax.set_ylabel("% Evaluation Tasks Solved")
        ax.set_ylim(-5, 100)
        ax.set_title(f"Light Switch: Random Skills ({_NUM_SEEDS} seeds) -- cf. paper Fig. 4")
        ax.legend()
        fig.tight_layout()
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(_PLOT_PATH, dpi=150)
        plt.close(fig)
        print(f"Saved plot to {_PLOT_PATH}")

    @staticmethod
    def run() -> None:
        curves: list[list[tuple[int, float]]] = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            for seed in range(_NUM_SEEDS):
                seed_output_dir = Path(tmp_dir) / f"seed_{seed}"
                RandomSkillsAnalysis._run_cli(seed=seed, output_dir=seed_output_dir)
                curves.append(RandomSkillsAnalysis._load_curve(output_dir=seed_output_dir))
                if seed == 0:
                    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(seed_output_dir / "episode.gif", _GIF_PATH)
                    print(f"Saved episode gif to {_GIF_PATH}")

        RandomSkillsAnalysis._print_metrics_table(curves=curves)
        RandomSkillsAnalysis._plot(curves=curves)


if __name__ == "__main__":  # pragma: no cover
    RandomSkillsAnalysis.run()
