"""Does the tightened `RobotAtSuccessfulThrowPose` band actually solve at 100%?

Post-run analysis only; it reads the JSON `scripts/tossing3d_oracle_demo.py --seeds
--results-json` writes back in and never drives the simulator itself -- that script
does, oracle-style (`Pick -> MoveToThrowPose(standoff) -> Toss`), the same methodology
PR #105 and the 48-episode grid behind `THROW_RANGE` used.

## Why this experiment, and what it does not need

`predicates.py`'s `RobotAtSuccessfulThrowPoseClassifier` was tightened from the full
geometric band `[1.125, 1.425]` to `[1.150, 1.375]` -- the 5/5 core PR #105's finer sweep
(5 scene seeds, 0.025 m resolution) found, rather than the wider band that only partially
solves at its own edges (2/5, 3/5, 2/5). That tightening changes a *label*
`MoveToThrowPose`'s sampler is trained against; it says nothing new about the physics PR
#105 already measured. This experiment re-measures the physics directly, on a fresh set of
ten scene seeds, to confirm the new band actually is where the throw solves reliably --
without retraining EES or touching the sampler at all.

## Which standoffs, and why

Not a literal "4 corners": standoff is the domain's one continuous throw parameter
(rotation is pinned to 0), so there is no 2D corner set to speak of. The chosen points are:

- **The new band's two endpoints**, `1.150` and `1.375` -- these *are* the claim, so they
  get the most seeds.
- **The old band's two discarded edges**, `1.125` and `1.425` -- positive evidence the
  tightening is correctly placed, not just conservative: PR #105 measured these at 2/5.
- **`1.400`** -- inside the old band, outside the new one, and PR #105's most informative
  miss (3/5): the cleanest single point for showing the trim is where reliability
  actually drops, not an arbitrary margin.
- **Three random interior draws**, from `np.random.default_rng(42).uniform(1.150, 1.375,
  size=3)` -- fixed, never drawn ad hoc -- standing in for "a random sampling from the
  middle".

Ten fixed scene seeds (`0`-`9`) per standoff, all run with `--task-config
coincident-bin-goal` (the config every number in `predicates.py`'s docstring is measured
on).

**Note, 2026-08-12: that flag no longer exists, and its scene is now the only scene.**
`kindergarden` PR #126 moved `bin_init_region` back to x = 2.0 in `Tossing3D-o1.json`
itself, so upstream's `o1` became the geometry this sweep was run against and the
stock/coincident choice was retired. The numbers below are **not** recomputed and are not
stale on that account -- they were taken on the same bin position the domain runs today.
What changed is only how you would ask for it: there is no flag to pass.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# The tightened band this experiment confirms, and the geometric band it replaced --
# restated here rather than imported, since this script must still describe what it
# measured even if `predicates.py`'s constants change again later.
NEW_BAND = (1.150, 1.375)
OLD_BAND = (1.125, 1.425)

SOLVED_COLOR = "#2166ac"
UNSOLVED_COLOR = "#b2182b"
NEW_BAND_COLOR = "#1a9850"
OLD_EDGE_COLOR = "#666666"


class Tossing3DThrowBandSweep:
    """Post-run analysis for the `(standoff, seed)` grid `tossing3d_oracle_demo.py
    --seeds --results-json` writes."""

    @staticmethod
    def load(*, results_json: Path) -> list[dict]:
        return list(json.loads(results_json.read_text()))

    @staticmethod
    def per_standoff_counts(*, results: list[dict]) -> dict[float, tuple[int, int]]:
        """`(solved, total)` per standoff, `x/y` never a bare fraction -- the fraction
        alone hides that every standoff here was tested on the same ten seeds."""
        by_standoff: dict[float, list[dict]] = {}
        for result in results:
            by_standoff.setdefault(result["standoff"], []).append(result)
        return {
            standoff: (sum(1 for r in cell if r["solved"]), len(cell))
            for standoff, cell in by_standoff.items()
        }

    @staticmethod
    def report(*, results_json: Path) -> None:
        results = Tossing3DThrowBandSweep.load(results_json=results_json)
        counts = Tossing3DThrowBandSweep.per_standoff_counts(results=results)
        print(f"loaded {len(results)} (standoff, seed) cells from {results_json}\n")
        print("standoff  solved  in new band [1.150, 1.375]")
        for standoff in sorted(counts):
            solved, total = counts[standoff]
            in_band = NEW_BAND[0] <= standoff <= NEW_BAND[1]
            print(f"{standoff:<9.4f} {solved}/{total:<6} {in_band}")
        total_solved = sum(solved for solved, _ in counts.values())
        total_cells = sum(total for _, total in counts.values())
        print(f"\n{total_solved}/{total_cells} cells solved overall")

        in_band_results = [r for r in results if NEW_BAND[0] <= r["standoff"] <= NEW_BAND[1]]
        in_band_solved = sum(1 for r in in_band_results if r["solved"])
        print(
            f"{in_band_solved}/{len(in_band_results)} solved strictly inside the new "
            "band (endpoints + interior draws)"
        )

    @staticmethod
    def render(*, results_json: Path, output: Path) -> None:
        results = Tossing3DThrowBandSweep.load(results_json=results_json)
        counts = Tossing3DThrowBandSweep.per_standoff_counts(results=results)
        standoffs = sorted(counts)

        fig, axes = plt.subplots(figsize=(9, 4.5))
        axes.axvspan(
            NEW_BAND[0],
            NEW_BAND[1],
            color=NEW_BAND_COLOR,
            alpha=0.12,
            label=f"tightened band [{NEW_BAND[0]:.3f}, {NEW_BAND[1]:.3f}]",
            zorder=0,
        )
        for edge in OLD_BAND:
            axes.axvline(
                edge,
                color=OLD_EDGE_COLOR,
                linestyle="--",
                linewidth=1.2,
                zorder=1,
                label="old geometric band edge" if edge == OLD_BAND[0] else None,
            )

        bar_width = 0.014
        for standoff in standoffs:
            solved, total = counts[standoff]
            fraction = solved / total
            axes.bar(
                standoff,
                fraction,
                width=bar_width,
                color=SOLVED_COLOR,
                alpha=0.35,
                zorder=2,
                edgecolor="none",
            )
            axes.text(
                standoff,
                min(fraction + 0.07, 0.95),
                f"{solved}/{total}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#222222",
                zorder=5,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1},
            )
            # Per-seed spread, not just the aggregate bar: one jittered dot per cell,
            # at y=1 if solved and y=0 if not, so a bar built from a bimodal split
            # (e.g. 3/5) is visibly two clusters of dots rather than only "0.6".
            cell = [r for r in results if r["standoff"] == standoff]
            rng = np.random.default_rng(0)
            jitter = rng.uniform(-bar_width * 0.4, bar_width * 0.4, size=len(cell))
            for offset, result in zip(jitter, cell, strict=True):
                y = 1.0 if result["solved"] else 0.0
                axes.scatter(
                    standoff + offset,
                    y,
                    s=14,
                    color=SOLVED_COLOR if result["solved"] else UNSOLVED_COLOR,
                    alpha=0.85,
                    zorder=3,
                    linewidths=0,
                )

        axes.set_xlabel("standoff (m)")
        axes.set_ylabel("solved per seed")
        axes.set_ylim(-0.08, 1.15)
        axes.set_title(
            "Tossing3D: solved per (standoff, seed) cell, 10 scene seeds each (coincident config)"
        )
        axes.grid(alpha=0.25, linewidth=0.6, axis="y")
        axes.legend(fontsize=8, loc="lower left", framealpha=0.95)
        fig.tight_layout()
        fig.savefig(output, dpi=150)
        plt.close(fig)
        print(f"wrote {output}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    Tossing3DThrowBandSweep.report(results_json=args.results_json)
    if args.output is not None:
        Tossing3DThrowBandSweep.render(results_json=args.results_json, output=args.output)


if __name__ == "__main__":
    main()
