"""Post-run analysis for Tossing3D's `Pick`/`MoveToThrowPose` success-box sweep: a dense,
controlled grid over each skill's *entire* sampling parameter space, labelled by the real
classifiers EES trains against (`HoldingClassifier`, `RobotAtSuccessfulThrowPoseClassifier`
-- see `hitl_pmp.environments.tossing3d.predicates`), never a hand-rolled check.

## Why this exists

Every number this domain had on record before `scripts/tossing3d_skill_parameter_sweep.py`
ran was one of: a coarse bin, a handful of spot-checks, or pooled EES-practice data, which
mixes parameters non-uniformly since it is whatever the sampler happened to draw rather
than a controlled grid. Two prior findings motivated running the real thing:

- **`Pick`'s own non-stationarity.** A fixed `rotation=0.65` measured 5/30 success across
  30 different scene seeds -- even `rotation=0.0`, the tested-safest point, got 29/30, not
  30/30. Pooled EES-practice data showed success correlating with `|rotation|` (99% near
  0, ~75% at the +-pi/4 extremes), but with an unresolved sign asymmetry against the
  oracle's own fixed point.
- **`MoveToThrowPose`'s residual informed-mode failures** cluster at the edges of
  `RobotAtSuccessfulThrowPoseClassifier`'s accepted standoff band (`[1.150, 1.375]` after
  the classifier was tightened to the 5/5 core PR #105's finer sweep found), and
  separately, low standoffs in the old `THROW_STANDOFF_BOUNDS=(0.45, 1.75)` were suspected
  of driving the base into scene geometry with no collision checking.

Reads only already-produced sweep output (CLAUDE.md's `analysis/` convention): the two flat
JSON lists `scripts/tossing3d_skill_parameter_sweep.py --which both` writes. Never drives
the simulator itself.

## What the `MoveToThrowPose` figure had to become, and why

`RobotAtSuccessfulThrowPoseClassifier` reads only `pos_base_x` (plus a lateral conjunct)
after `move_to_target` terminates, and that controller lands the base within `WAYPOINT_TOL`
of its own commanded pose -- so the classifier's labelled outcome is close to a step
function of the commanded standoff alone, confirmed by this sweep's own data (see
`move_cell_counts`). A single success-rate curve against standoff would therefore show the
band and nothing else: it cannot reveal what happens physically at standoffs the classifier
was never designed to distinguish, because the classifier does not look at what the base hit
on the way.

So `scripts/tossing3d_skill_parameter_sweep.py` records more than the label per cell: the
bin's and the barrier's own position before and after the skill sequence. A genuine collision
shows up as a nonzero delta on either, independent of whatever the classifier says. The figure
is therefore two panels sharing an x axis -- the classifier curve on top (the trainable label),
the physical collision signal underneath (the mechanism) -- rather than one curve asked to
carry both stories.

## Every count is `x/y`, never a bare percentage (CLAUDE.md)

Every printed line and every figure title states a denominator. `n=12 seeds/cell` in a
panel title is a denominator statement, not decoration.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering -- no GUI backend needed/available in CI

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from hitl_pmp.environments.tossing3d.skill_oracle_policy import (  # noqa: E402
    ORACLE_PICK_DISTANCE,
    ORACLE_PICK_ROTATION,
)
from hitl_pmp.environments.tossing3d.skills import (  # noqa: E402
    PICK_DISTANCE_BOUNDS,
    PICK_ROTATION_BOUNDS,
)

# The tightened band `RobotAtSuccessfulThrowPoseClassifier` accepts on the coincident
# config (`predicates.py`'s `THROW_OVERSHOOT_MARGIN`/`THROW_SHORTFALL_MARGIN` applied to
# the live goal box) -- restated here rather than imported, matching the precedent already
# set by the sibling `tossing3d_throw_band_sweep.py`: this module must still describe what
# it measured even if `predicates.py`'s constants change again later. The bin's own
# placement range is ~1 mm wide (`bin_init_region`), so this band is effectively constant
# across scene seeds on this config, not merely at the moment this was measured.
THROW_BAND = (1.150, 1.375)
# The two `THROW_STANDOFF_BOUNDS` this domain has used for the *sampler's* draw range --
# not the classifier's accepted band. `OLD` is what `skills.py` uses as of this sweep;
# `NEW` is a separate, in-flight, not-yet-merged PR's proposed tightening, included only
# as an annotation so this figure does not need to be regenerated the moment that PR lands.
THROW_STANDOFF_BOUNDS_OLD = (0.45, 1.75)
THROW_STANDOFF_BOUNDS_NEW_PROPOSED = (1.10, 1.75)

# A displacement below this (metres) is treated as "did not move" -- upstream reports
# base/object poses in float32 (`kinder_backend.py`'s own `restore` docstring notes a
# ~1e-7-to-1e-4 round-trip noise floor at a much finer scale than this), and ordinary
# episode-to-episode positioning noise is on the order of centimetres elsewhere in this
# domain (`WAYPOINT_TOL`, 4 cm). 5 mm is comfortably above float noise and comfortably
# below the ~5-25 cm shoves this sweep actually measures at colliding standoffs.
DISPLACEMENT_NOISE_FLOOR_M = 0.005

_SUCCESS_COLOR = "#2166ac"  # matches tossing3d_throw_band_sweep's SOLVED_COLOR
_FAILURE_COLOR = "#b2182b"  # matches tossing3d_throw_band_sweep's UNSOLVED_COLOR
_BAND_COLOR = "#1a9850"  # matches tossing3d_throw_band_sweep's NEW_BAND_COLOR
_OLD_EDGE_COLOR = "#666666"
_BIN_COLOR = "#8c510a"  # brown -- the object that gets shoved
_BARRIER_COLOR = "#762a83"  # purple -- the wall that gets driven into
_CANVAS = "white"


class Tossing3DSkillParameterSweep:
    """A static-method container, never instantiated, same as every other business-logic
    class in this project."""

    # ---- loading ----------------------------------------------------------------

    @staticmethod
    def load(*, results_json: Path) -> list[dict]:
        return list(json.loads(results_json.read_text()))

    # ---- Pick aggregation ---------------------------------------------------------

    @staticmethod
    def pick_grid_axes(*, results: list[dict]) -> tuple[list[float], list[float]]:
        """The sorted, deduplicated distance and rotation values actually swept.

        Read from the data rather than re-derived from `PICK_DISTANCE_BOUNDS`/
        `PICK_ROTATION_BOUNDS` and a step count, so a heatmap regenerated against a
        differently-resolved sweep draws its own grid rather than a stale one.
        """
        distances = sorted({row["distance"] for row in results})
        rotations = sorted({row["rotation"] for row in results})
        return distances, rotations

    @staticmethod
    def pick_cell_counts(*, results: list[dict]) -> dict[tuple[float, float], tuple[int, int]]:
        """`(successes, total)` per `(distance, rotation)` cell."""
        counts: dict[tuple[float, float], tuple[int, int]] = {}
        for row in results:
            key = (row["distance"], row["rotation"])
            successes, total = counts.get(key, (0, 0))
            counts[key] = (successes + int(row["success"]), total + 1)
        return counts

    @staticmethod
    def pick_seed_breakdown(*, results: list[dict]) -> dict[int, tuple[int, int]]:
        """`(successes, total)` per scene seed, pooled over every grid cell.

        The same seeds are reused at every cell (paired, not independent -- see the
        sweep script's own docstring), so a single pathological seed would shift every
        cell by the same `1/num_seeds` and show up here as one seed's rate sitting far
        from the rest, rather than being smeared invisibly across the heatmap.
        """
        counts: dict[int, tuple[int, int]] = {}
        for row in results:
            seed = row["seed"]
            successes, total = counts.get(seed, (0, 0))
            counts[seed] = (successes + int(row["success"]), total + 1)
        return counts

    @staticmethod
    def pick_overall(*, results: list[dict]) -> tuple[int, int]:
        return (sum(int(row["success"]) for row in results), len(results))

    @staticmethod
    def pick_rotation_marginal(*, results: list[dict]) -> dict[float, tuple[int, int]]:
        """`(successes, total)` per rotation value, pooled over every distance and seed.

        The marginal that the session's earlier pooled-data finding was about: success
        correlating with `|rotation|`. This is the controlled version of that same
        question -- every rotation value here was tested at every distance and every
        seed, unlike the pooled EES-practice draws it is checked against.
        """
        counts: dict[float, tuple[int, int]] = {}
        for row in results:
            rotation = row["rotation"]
            successes, total = counts.get(rotation, (0, 0))
            counts[rotation] = (successes + int(row["success"]), total + 1)
        return counts

    # ---- MoveToThrowPose aggregation ----------------------------------------------

    @staticmethod
    def move_standoff_axis(*, results: list[dict]) -> list[float]:
        return sorted({row["standoff"] for row in results})

    @staticmethod
    def move_cell_counts(*, results: list[dict]) -> dict[float, tuple[int, int]]:
        """`(successes, total)` per standoff, from `RobotAtSuccessfulThrowPoseClassifier`."""
        counts: dict[float, tuple[int, int]] = {}
        for row in results:
            standoff = row["standoff"]
            successes, total = counts.get(standoff, (0, 0))
            counts[standoff] = (successes + int(row["success"]), total + 1)
        return counts

    @staticmethod
    def move_seed_breakdown(*, results: list[dict]) -> dict[int, tuple[int, int]]:
        counts: dict[int, tuple[int, int]] = {}
        for row in results:
            seed = row["seed"]
            successes, total = counts.get(seed, (0, 0))
            counts[seed] = (successes + int(row["success"]), total + 1)
        return counts

    @staticmethod
    def move_overall(*, results: list[dict]) -> tuple[int, int]:
        return (sum(int(row["success"]) for row in results), len(results))

    @staticmethod
    def move_pick_confound_count(*, results: list[dict]) -> tuple[int, int]:
        """How many cells ran from a `Pick` that did *not* hold, at the fixed oracle
        point -- confounded cells this sweep keeps rather than silently drops (see the
        sweep script's own `MoveCellResult` docstring). `x/y` of the whole sweep, so a
        reader can judge how much of the grid this could affect before reading further."""
        failed = sum(1 for row in results if not row["pick_success"])
        return (failed, len(results))

    @staticmethod
    def move_mean_bin_displacement(*, results: list[dict]) -> dict[float, float]:
        """Mean `hypot(dx, dy)` bin displacement per standoff -- the bin-shove signal
        the classifier itself never reads."""
        by_standoff: dict[float, list[float]] = {}
        for row in results:
            dx = row["bin_x"] - row["bin_x_initial"]
            dy = row["bin_y"] - row["bin_y_initial"]
            by_standoff.setdefault(row["standoff"], []).append(float(np.hypot(dx, dy)))
        return {standoff: sum(values) / len(values) for standoff, values in by_standoff.items()}

    @staticmethod
    def move_mean_barrier_displacement(*, results: list[dict]) -> dict[float, float]:
        """Mean `|dx|` barrier displacement per standoff -- the barrier only moves along
        x in this domain (`ReachableClassifier` reads only `barrier.x`), so unlike the
        bin this is a 1D delta, not a hypot of two."""
        by_standoff: dict[float, list[float]] = {}
        for row in results:
            dx = abs(row["barrier_x"] - row["barrier_x_initial"])
            by_standoff.setdefault(row["standoff"], []).append(dx)
        return {standoff: sum(values) / len(values) for standoff, values in by_standoff.items()}

    @staticmethod
    def measured_collision_boundary(
        *, displacement_by_standoff: dict[float, float]
    ) -> float | None:
        """The smallest standoff at or above which every larger standoff's mean
        displacement stays under `DISPLACEMENT_NOISE_FLOOR_M`.

        Scanned from the top down (largest standoff first) so one noisy low-standoff cell
        below an otherwise-clean region cannot pull the boundary down past it. Returns
        `None` if every standoff swept shows displacement above the floor -- i.e. this
        sweep's own range was not wide enough to find where it stops -- rather than
        guessing.
        """
        standoffs = sorted(displacement_by_standoff)
        if not standoffs:
            return None
        boundary = standoffs[0]
        for standoff in standoffs:
            if displacement_by_standoff[standoff] < DISPLACEMENT_NOISE_FLOOR_M:
                boundary = standoff
                break
        else:
            return None
        # Confirm nothing *larger* than `boundary` re-crosses the floor -- a genuine
        # boundary, not the first lucky low value in a noisy region.
        for standoff in standoffs:
            if (
                standoff >= boundary
                and displacement_by_standoff[standoff] >= DISPLACEMENT_NOISE_FLOOR_M
            ):
                return None
        return boundary

    # ---- reporting ------------------------------------------------------------

    @staticmethod
    def print_report(*, pick_results: list[dict], move_results: list[dict]) -> None:
        print("=== Pick success box ===")
        successes, total = Tossing3DSkillParameterSweep.pick_overall(results=pick_results)
        distances, rotations = Tossing3DSkillParameterSweep.pick_grid_axes(results=pick_results)
        print(
            f"  overall {successes}/{total}, grid {len(distances)} distances x "
            f"{len(rotations)} rotations, bounds distance={PICK_DISTANCE_BOUNDS} "
            f"rotation={PICK_ROTATION_BOUNDS}"
        )
        print("  rotation marginal (pooled over distance and seed):")
        marginal = Tossing3DSkillParameterSweep.pick_rotation_marginal(results=pick_results)
        for rotation in sorted(marginal):
            successes, total = marginal[rotation]
            print(f"    rotation={rotation:+.3f}  {successes}/{total}")
        print("  per-seed breakdown:")
        for seed, (successes, total) in sorted(
            Tossing3DSkillParameterSweep.pick_seed_breakdown(results=pick_results).items()
        ):
            print(f"    seed {seed}: {successes}/{total}")

        print("\n=== MoveToThrowPose success box ===")
        successes, total = Tossing3DSkillParameterSweep.move_overall(results=move_results)
        confound, confound_total = Tossing3DSkillParameterSweep.move_pick_confound_count(
            results=move_results
        )
        num_standoffs = len(Tossing3DSkillParameterSweep.move_standoff_axis(results=move_results))
        print(
            f"  overall {successes}/{total}, {num_standoffs} standoffs swept, oracle Pick "
            f"did not hold in {confound}/{confound_total} cells"
        )
        bin_disp = Tossing3DSkillParameterSweep.move_mean_bin_displacement(results=move_results)
        barrier_disp = Tossing3DSkillParameterSweep.move_mean_barrier_displacement(
            results=move_results
        )
        bin_boundary = Tossing3DSkillParameterSweep.measured_collision_boundary(
            displacement_by_standoff=bin_disp
        )
        barrier_boundary = Tossing3DSkillParameterSweep.measured_collision_boundary(
            displacement_by_standoff=barrier_disp
        )
        no_boundary = "no collision-free standoff found in range"
        bin_boundary_text = no_boundary if bin_boundary is None else f"{bin_boundary:.3f} m"
        barrier_boundary_text = (
            no_boundary if barrier_boundary is None else f"{barrier_boundary:.3f} m"
        )
        print(f"  measured bin-collision boundary: {bin_boundary_text}")
        print(f"  measured barrier-collision boundary: {barrier_boundary_text}")
        counts = Tossing3DSkillParameterSweep.move_cell_counts(results=move_results)
        print("  per-seed breakdown:")
        for seed, (successes, total) in sorted(
            Tossing3DSkillParameterSweep.move_seed_breakdown(results=move_results).items()
        ):
            print(f"    seed {seed}: {successes}/{total}")
        in_band = [s for s in counts if THROW_BAND[0] <= s <= THROW_BAND[1]]
        in_band_successes = sum(counts[s][0] for s in in_band)
        in_band_total = sum(counts[s][1] for s in in_band)
        print(
            f"  inside classifier band {THROW_BAND}: {in_band_successes}/{in_band_total} "
            f"across {len(in_band)} standoffs"
        )

    # ---- figures ----------------------------------------------------------------

    @staticmethod
    def figure_pick(*, results: list[dict]) -> Figure:
        """The `Pick` success box: a 2D heatmap of success rate over `(distance,
        rotation)`, since this is the domain's one genuinely two-parameter skill.

        Cell color is a sequential-through-diverging `RdYlGn` (0 = red, 1 = green),
        chosen per the task's own suggestion rather than this repo's training-curve
        blue/orange convention, which is for arm comparisons and does not apply here --
        there is no "arm" being compared, only one skill's parameter space.
        """
        distances, rotations = Tossing3DSkillParameterSweep.pick_grid_axes(results=results)
        counts = Tossing3DSkillParameterSweep.pick_cell_counts(results=results)
        n_per_cell = next(iter(counts.values()))[1] if counts else 0
        rate_grid = np.full((len(rotations), len(distances)), np.nan)
        for row_idx, rotation in enumerate(rotations):
            for col_idx, distance in enumerate(distances):
                successes, total = counts.get((distance, rotation), (0, 0))
                if total:
                    rate_grid[row_idx, col_idx] = successes / total

        figure, axis = plt.subplots(figsize=(8.5, 7.0), dpi=150, facecolor=_CANVAS)
        extent = (min(distances), max(distances), min(rotations), max(rotations))
        image = axis.imshow(
            rate_grid,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="RdYlGn",
            vmin=0.0,
            vmax=1.0,
        )
        colorbar = figure.colorbar(image, ax=axis)
        colorbar.set_label("success rate (HoldingClassifier)")

        axis.scatter(
            [ORACLE_PICK_DISTANCE],
            [ORACLE_PICK_ROTATION],
            marker="x",
            s=140,
            linewidths=2.5,
            color="black",
            label=(f"oracle point ({ORACLE_PICK_DISTANCE:.3f}, {ORACLE_PICK_ROTATION:.3f})"),
        )
        # The plotted extent already *is* PICK_DISTANCE_BOUNDS/PICK_ROTATION_BOUNDS --
        # Pick is gridded over exactly its own sampling bounds, nothing outside them --
        # so the axes limits themselves mark the bounds; a redundant rectangle would sit
        # exactly on the frame and add nothing.
        axis.set_xlabel("distance (m)")
        axis.set_ylabel("rotation (rad)")
        axis.set_title(
            f"Tossing3D Pick success box — n={n_per_cell} seeds/cell, "
            f"{len(distances)}x{len(rotations)} grid over the full sampling bounds",
            loc="left",
            fontsize=11,
        )
        axis.legend(loc="upper right", framealpha=0.92, fontsize=9)
        figure.tight_layout()
        return figure

    @staticmethod
    def figure_move(*, results: list[dict]) -> Figure:
        """The `MoveToThrowPose` success box: two panels sharing an x axis (standoff).

        Top: `RobotAtSuccessfulThrowPoseClassifier`'s success rate -- the trainable
        label, close to a step function of standoff (see the module docstring).
        Bottom: mean bin/barrier displacement -- the physical mechanism the classifier
        cannot see, which is what actually makes the low-standoff region unsafe.
        """
        standoffs = Tossing3DSkillParameterSweep.move_standoff_axis(results=results)
        counts = Tossing3DSkillParameterSweep.move_cell_counts(results=results)
        n_per_cell = next(iter(counts.values()))[1] if counts else 0
        bin_disp = Tossing3DSkillParameterSweep.move_mean_bin_displacement(results=results)
        barrier_disp = Tossing3DSkillParameterSweep.move_mean_barrier_displacement(results=results)

        figure, (top, bottom) = plt.subplots(
            2,
            1,
            figsize=(11.0, 8.0),
            dpi=150,
            facecolor=_CANVAS,
            sharex=True,
            gridspec_kw={"height_ratios": [2.0, 1.2]},
        )

        for axis in (top, bottom):
            axis.axvspan(
                THROW_BAND[0],
                THROW_BAND[1],
                color=_BAND_COLOR,
                alpha=0.12,
                zorder=0,
                label=f"classifier band {THROW_BAND}" if axis is top else None,
            )
            for edge in THROW_STANDOFF_BOUNDS_OLD:
                axis.axvline(
                    edge,
                    color=_OLD_EDGE_COLOR,
                    linestyle="--",
                    linewidth=1.1,
                    zorder=1,
                    label="old sampler bounds (0.45, 1.75)"
                    if axis is top and edge == THROW_STANDOFF_BOUNDS_OLD[0]
                    else None,
                )
            for edge in THROW_STANDOFF_BOUNDS_NEW_PROPOSED:
                axis.axvline(
                    edge,
                    color="#0072B2",
                    linestyle=":",
                    linewidth=1.4,
                    zorder=1,
                    label="in-flight proposed sampler bounds (1.10, 1.75)"
                    if axis is top and edge == THROW_STANDOFF_BOUNDS_NEW_PROPOSED[0]
                    else None,
                )

        rates = [counts[s][0] / counts[s][1] for s in standoffs]
        top.plot(standoffs, rates, color=_SUCCESS_COLOR, linewidth=1.6, zorder=3)
        for standoff in standoffs:
            successes, total = counts[standoff]
            color = (
                _SUCCESS_COLOR
                if successes == total
                else (_FAILURE_COLOR if successes == 0 else "#444444")
            )
            top.scatter(standoff, successes / total, s=16, color=color, zorder=4)
        top.set_ylabel("success rate (RobotAtSuccessfulThrowPose)")
        top.set_ylim(-0.05, 1.15)
        top.set_title(
            f"Tossing3D MoveToThrowPose success box — n={n_per_cell} seeds/standoff, "
            f"{len(standoffs)} standoffs, Pick fixed at the oracle's point",
            loc="left",
            fontsize=11,
        )
        top.legend(loc="upper left", framealpha=0.92, fontsize=8)
        top.grid(alpha=0.25, linewidth=0.6)

        bottom.plot(
            standoffs,
            [bin_disp[s] for s in standoffs],
            color=_BIN_COLOR,
            linewidth=1.8,
            label="mean |bin displacement| (m)",
        )
        bottom.plot(
            standoffs,
            [barrier_disp[s] for s in standoffs],
            color=_BARRIER_COLOR,
            linewidth=1.8,
            linestyle=(0, (4, 2)),
            label="mean |barrier x displacement| (m)",
        )
        bottom.axhline(
            DISPLACEMENT_NOISE_FLOOR_M,
            color="#999999",
            linewidth=0.9,
            linestyle=":",
            label=f"noise floor ({DISPLACEMENT_NOISE_FLOOR_M * 100:.1f} cm)",
        )
        bottom.set_xlabel("standoff (m)")
        bottom.set_ylabel("mean displacement (m)")
        bottom.legend(loc="upper right", framealpha=0.92, fontsize=8)
        bottom.grid(alpha=0.25, linewidth=0.6)
        for axis in (top, bottom):
            for side in ("top", "right"):
                axis.spines[side].set_visible(False)

        figure.tight_layout()
        return figure

    @staticmethod
    def plot(
        *, pick_results: list[dict], move_results: list[dict], pick_output: Path, move_output: Path
    ) -> None:
        for figure, output_path in (
            (Tossing3DSkillParameterSweep.figure_pick(results=pick_results), pick_output),
            (Tossing3DSkillParameterSweep.figure_move(results=move_results), move_output),
        ):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(output_path, bbox_inches="tight", facecolor=_CANVAS)
            plt.close(figure)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pick-json", type=Path, required=True)
    parser.add_argument("--move-json", type=Path, required=True)
    parser.add_argument("--pick-output", type=Path, default=None)
    parser.add_argument("--move-output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    pick_results = Tossing3DSkillParameterSweep.load(results_json=args.pick_json)
    move_results = Tossing3DSkillParameterSweep.load(results_json=args.move_json)
    Tossing3DSkillParameterSweep.print_report(pick_results=pick_results, move_results=move_results)
    if args.pick_output is not None and args.move_output is not None:
        Tossing3DSkillParameterSweep.plot(
            pick_results=pick_results,
            move_results=move_results,
            pick_output=args.pick_output,
            move_output=args.move_output,
        )
        print(f"wrote {args.pick_output}")
        print(f"wrote {args.move_output}")
    elif args.pick_output is not None or args.move_output is not None:
        raise SystemExit("pass both --pick-output and --move-output, or neither")


if __name__ == "__main__":
    main()
