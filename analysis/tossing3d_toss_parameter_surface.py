"""Tossing3D's `(release_speed x gripper_release_ms)` surface: figures and the two tests.

Post-run analysis only: this reads the committed grid back in and never builds an
environment or drives a skill. It exists to answer two questions that a heatmap alone
cannot, so unlike PR #234's surface script -- which this file is the generalisation of --
it does draw inference, and every claim it prints carries a p-value.

## Question 1: is the reachable set degenerate?

`Toss` has `param_dim=2`. That is only justified if the second dial reaches throws the
first cannot. If every `(speed, ms)` pair merely reproduces a distance some speed already
reaches at the default millisecond, the second dial buys redundancy rather than reach and
the parameter should be one-dimensional.

`reach_interval` answers it directly and without a model: the span of ballistic distances
reachable over the whole 2-D grid, against the span reachable along the single `ms = 720`
row. `variance_shares` then says *how* the surface is shaped -- a two-way decomposition into
a speed main effect, a millisecond main effect and their interaction. The interaction term
is not a nuisance here, it is predicted: the swing's duration is itself a function of the
speed (3100 ms at 60 deg/s down to 1700 ms at 140), so the same absolute millisecond is a
different point in the swing at each end of the speed range.

## Question 2: does the speed reversal survive?

kb#11 measured `1/16` reversals at 3.3 mm on a single seed and explicitly did not establish
whether it was real. `adjacent_speed_reversals` re-asks it with the seed axis populated: for
each adjacent pair of commanded speeds, is the mean change in ballistic distance negative?

**The test is paired, and that is not optional.** Every cell of this grid runs the same five
seeds, so a speed pair's two arms are the same five scenes throw for throw. An unpaired test
discards that structure and understates the effect. `paired_difference_test` pairs on
`(release_ms, seed)`, giving `n = 20 x 5 = 100` paired observations per speed step.

Holm-Bonferroni is applied across the 19 speed steps, because asking 19 questions at
`alpha = 0.05` expects roughly one false positive and this experiment exists partly to
decide whether a single reported reversal was one.

## Ballistic distance is primary; resting distance is drawn beside it, not instead

Resting x is contaminated by bin contact and the contamination is a **step**, not a drift:
scanning the millisecond axis at 140 deg/s gave `690 -> 1.7175`, `705 -> 1.7428`, then a jump
of 244 mm to `710 -> 1.9870`. That is the bin catching the cube versus not. Both surfaces are
plotted so a reader can see the artifact rather than be told about it, but every statistic
above is computed on the ballistic distance.

## Palette

Deliberately **not** the project's `#0072B2`/`#D55E00`. Those encode "an assistance mechanism
is available" versus "nothing intervenes", and no cell here is an arm of that comparison --
every cell is the same robot doing the same thing with two dials moved. Borrowing them would
import a distinction that does not exist. The millisecond is an **ordered continuous**
variable, so its curve family gets a sequential ramp, which is the one encoding that makes
"these curves are in order" readable at a glance; the same ramp carries the heatmaps. Cells
where nothing was thrown are hatched grey -- a third state, not a low distance.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

# Upstream's own shipped default, and the row the 1-D reachable set is measured along. Kept
# as a module constant rather than read from `predicates` because it is a *property of the
# grid being analysed* -- the row must actually exist in the axis, and a grid swept over
# different bounds will not have it. `main` reports which row it fell back to.
DEFAULT_RELEASE_MS = 720.0

# Sequential, because the millisecond is an ordered continuous variable. Pale -> saturated
# purple; see the palette note above for why the project's blue/orange is wrong here.
SEQUENTIAL = plt.get_cmap("viridis")
SURFACE_CMAP = "viridis"
NOT_THROWN_COLOUR = "#BFBFBF"

# Project style: faint per-seed traces underneath bold subgroup means.
SEED_TRACE_ALPHA = 0.16
SEED_TRACE_WIDTH = 0.8
MEAN_TRACE_WIDTH = 2.3


def load_grid(*, path: Path) -> dict[str, Any]:
    """The committed grid, with its axes and its raw rows."""
    payload = json.loads(path.read_text())
    return {
        "speeds": payload["speeds"],
        "release_ms": payload["release_ms"],
        "seeds": payload["seeds"],
        "standoffs": payload["standoffs"],
        "rows": payload["rows"],
    }


def distance_table(*, grid: dict[str, Any], field: str) -> dict[tuple[float, float, int], float]:
    """`(speed, release_ms, seed) -> distance from the robot base`, in metres.

    **From the base, not the world x.** `MoveToThrowPose` parks the base a standoff back from
    the bin, so a world-frame landing position carries wherever the base ended up; subtracting
    `base_x_before_toss` makes the number a property of the throw. Every comparison in this
    module is on this quantity, and a forgotten subtraction would shift all of them alike --
    which is exactly why it is done once, here.

    Cells where nothing was thrown are **absent** rather than zero. A missing key is a fact
    about the parameter pair; a zero would be a measurement of a throw that never happened.
    """
    table: dict[tuple[float, float, int], float] = {}
    for row in grid["rows"]:
        if not row.get("threw"):
            continue
        value = row.get(field)
        base = row.get("base_x_before_toss")
        if value is None or base is None:
            continue
        key = (row["commanded_speed_deg"], row["commanded_release_ms"], row["seed"])
        table[key] = float(value) - float(base)
    return table


def dead_cells(*, grid: dict[str, Any]) -> list[tuple[float, float, int]]:
    """Cells where the gripper never opened, so no throw happened at all.

    `gripper_release_ms` is not clamped upstream, so a millisecond at or past the end of the
    swing is a real and reachable corner rather than an error. Reported as itself so it can
    never be read as a flat measurement -- PR #231 hatched its `30/100` never-threw cells for
    the same reason.
    """
    return sorted(
        (row["commanded_speed_deg"], row["commanded_release_ms"], row["seed"])
        for row in grid["rows"]
        if not row.get("threw")
    )


def seed_mean_surface(
    *, table: dict[tuple[float, float, int], float]
) -> dict[tuple[float, float], float]:
    """`(speed, release_ms) -> mean distance over the seeds that threw`."""
    grouped: dict[tuple[float, float], list[float]] = {}
    for (speed, release_ms, _seed), value in table.items():
        grouped.setdefault((speed, release_ms), []).append(value)
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def reach_interval(
    *, table: dict[tuple[float, float, int], float], release_ms: float | None = None
) -> tuple[float, float]:
    """The span of ballistic distances the parameter set reaches, as `(min, max)`.

    With `release_ms=None` this is the 2-D reachable set: what both dials together can
    command. With a millisecond fixed it is the 1-D set: what the speed dial alone reaches
    with the second held at its default. **The comparison of those two intervals is the
    degeneracy statistic** -- if they coincide, the second dial adds no reach.

    Computed on seed means, because the question is what a caller can *command*: a single
    seed's outlier is not a distance the parameter reaches, it is scene noise.
    """
    surface = seed_mean_surface(table=table)
    values = [
        value
        for (_speed, cell_ms), value in surface.items()
        if release_ms is None or cell_ms == release_ms
    ]
    if not values:
        raise ValueError(f"no cells at release_ms={release_ms}")
    return (float(min(values)), float(max(values)))


def variance_shares(*, table: dict[tuple[float, float, int], float]) -> dict[str, float]:
    """Two-way decomposition of the seed-mean surface into speed, millisecond, interaction.

    The classical identity on a complete grid:
    `SS_total = SS_speed + SS_release_ms + SS_interaction`, exactly, so the three returned
    shares sum to 1. Reported as shares rather than raw sums of squares because the question
    is proportional -- "how much of the surface's shape does each dial account for" -- and a
    raw SS is not comparable across grids of different size.

    The **interaction** term is the one to read first here. A large interaction means the two
    dials are not separable, which is the mechanism this domain predicts: the swing's
    duration is a function of the speed, so a fixed absolute millisecond is a different point
    in the swing at each speed.

    Requires a complete rectangular grid; incomplete cells are dropped along with their whole
    row and column, since an unbalanced decomposition is not the identity above.
    """
    surface = seed_mean_surface(table=table)
    speeds = sorted({s for s, _ in surface})
    release_ms_values = sorted({m for _, m in surface})
    speeds = [s for s in speeds if all((s, m) in surface for m in release_ms_values)]
    release_ms_values = [m for m in release_ms_values if all((s, m) in surface for s in speeds)]

    matrix = np.array([[surface[(s, m)] for m in release_ms_values] for s in speeds])
    grand = matrix.mean()
    by_speed = matrix.mean(axis=1, keepdims=True)
    by_ms = matrix.mean(axis=0, keepdims=True)

    ss_total = float(((matrix - grand) ** 2).sum())
    if ss_total == 0.0:
        return {"speed": 0.0, "release_ms": 0.0, "interaction": 0.0}
    ss_speed = float(len(release_ms_values) * ((by_speed - grand) ** 2).sum())
    ss_ms = float(len(speeds) * ((by_ms - grand) ** 2).sum())
    ss_interaction = float(((matrix - by_speed - by_ms + grand) ** 2).sum())
    return {
        "speed": ss_speed / ss_total,
        "release_ms": ss_ms / ss_total,
        "interaction": ss_interaction / ss_total,
    }


def paired_difference_test(*, before: np.ndarray, after: np.ndarray) -> dict[str, float]:
    """A paired t-test on `after - before`, with the degenerate case handled explicitly.

    Paired because the two arms are the same `(release_ms, seed)` cells at two speeds -- the
    same scenes, throw for throw. An unpaired test would attribute the between-cell spread
    (which here is metres) to noise and could not see an effect of millimetres.

    `scipy.stats.ttest_rel` returns `nan` when every difference is identical, because the
    sample variance is zero. That is not "no information", it is a perfectly consistent
    effect, and letting a `nan` through would silently drop the comparison from a
    significance count. So it is resolved here: a constant non-zero difference is `p = 0`,
    a constant zero difference is `p = 1`.

    The zero test is **relative, not exact**. Differences that cancel algebraically need not
    cancel in floating point, so a genuinely constant difference can arrive with a standard
    deviation of ~1e-17 -- which slips past an `== 0.0` guard and reaches scipy, which then
    emits a catastrophic-cancellation warning and returns a t-statistic built from rounding
    noise. Scaling the threshold by the mean is what makes the guard describe the situation
    ("all the differences are the same") rather than a bit pattern.
    """
    before_array = np.asarray(before, dtype=float)
    after_array = np.asarray(after, dtype=float)
    differences = after_array - before_array
    mean_difference = float(differences.mean())
    spread = float(differences.std(ddof=1)) if len(differences) > 1 else 0.0
    if spread <= 1e-12 * max(1.0, abs(mean_difference)):
        return {
            "n": float(len(differences)),
            "mean_difference": mean_difference,
            "t_statistic": float("inf") if mean_difference > 0 else float("-inf"),
            "p_value": 0.0 if mean_difference != 0.0 else 1.0,
            "ci_low": mean_difference,
            "ci_high": mean_difference,
        }
    result = stats.ttest_rel(after_array, before_array)
    # A 95% interval on the paired mean, so a null step can be *shown* to be a null step.
    # Without it a bar of height zero and a bar whose interval spans +/- 200 mm look
    # identical, and only one of them is evidence of no effect.
    half_width = float(
        stats.t.ppf(0.975, len(differences) - 1)
        * differences.std(ddof=1)
        / np.sqrt(len(differences))
    )
    return {
        "n": float(len(differences)),
        "mean_difference": mean_difference,
        "t_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "ci_low": mean_difference - half_width,
        "ci_high": mean_difference + half_width,
    }


def holm_bonferroni(*, p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, order preserved.

    Applied because the reversal question is asked once per adjacent speed step. At 19 steps
    and `alpha = 0.05`, roughly one uncorrected false positive is expected -- and "a single
    reported reversal" is precisely the thing this analysis is deciding the reality of, so an
    uncorrected count would beg its own question.
    """
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (len(p_values) - rank) * p_values[index]
        running = max(running, min(1.0, candidate))
        adjusted[index] = running
    return adjusted


def adjacent_speed_reversals(
    *, table: dict[tuple[float, float, int], float]
) -> list[dict[str, float]]:
    """For each adjacent pair of commanded speeds, the paired change in ballistic distance.

    A **negative** `mean_difference` is a reversal: distance falling as commanded speed rises.
    Each pair is tested on the `(release_ms, seed)` cells the two speeds share, so the arms
    are the same scenes at the same millisecond and the pairing is real rather than nominal.

    Returns one record per adjacent pair, in speed order, carrying both the raw and the
    Holm-adjusted p-value so a reader can see the correction rather than only its verdict.
    """
    speeds = sorted({s for s, _, _ in table})
    records: list[dict[str, float]] = []
    for low, high in zip(speeds, speeds[1:], strict=False):
        shared = sorted(
            {(m, k) for s, m, k in table if s == low} & {(m, k) for s, m, k in table if s == high}
        )
        if not shared:
            continue
        before = np.array([table[(low, m, k)] for m, k in shared])
        after = np.array([table[(high, m, k)] for m, k in shared])
        record = paired_difference_test(before=before, after=after)
        record["speed_low"] = low
        record["speed_high"] = high
        records.append(record)
    adjusted = holm_bonferroni(p_values=[r["p_value"] for r in records])
    for record, value in zip(records, adjusted, strict=True):
        record["p_value_holm"] = value
    return records


def _panel_curve_family(  # noqa: PLR0917
    *,
    ax: Any,
    fig: Any,
    table: dict[tuple[float, float, int], float],
    seeds: list[int],
    outer: list[float],
    inner: list[float],
    outer_is_speed: bool,
    colourbar_label: str,
) -> None:
    """One curve per `outer` value, over `inner`, with faint per-seed traces underneath.

    Project style, not re-derived per figure: the per-seed traces are drawn first at
    `alpha=0.16 / lw=0.8`, the bold seed means over them at `lw=2.3`, and every legend entry
    carries its own `n=`. A bold mean over a population that splits describes none of it, and
    the faint lines are what make that visible instead of asserted.
    """
    for index, outer_value in enumerate(outer):
        colour = SEQUENTIAL(index / max(1, len(outer) - 1))
        for seed in seeds:
            xs, ys = [], []
            for inner_value in inner:
                key = (
                    (outer_value, inner_value, seed)
                    if outer_is_speed
                    else (inner_value, outer_value, seed)
                )
                if key in table:
                    xs.append(inner_value)
                    ys.append(table[key])
            ax.plot(xs, ys, color=colour, alpha=SEED_TRACE_ALPHA, lw=SEED_TRACE_WIDTH, zorder=1)
        xs, ys, counts = [], [], []
        for inner_value in inner:
            values = [
                table[k]
                for seed in seeds
                if (
                    k := (
                        (outer_value, inner_value, seed)
                        if outer_is_speed
                        else (inner_value, outer_value, seed)
                    )
                )
                in table
            ]
            if values:
                xs.append(inner_value)
                ys.append(float(np.mean(values)))
                counts.append(len(values))
        label = None
        if index in (0, len(outer) // 2, len(outer) - 1):
            unit = "deg/s" if outer_is_speed else "ms"
            label = f"{outer_value:.0f} {unit} — mean, n={min(counts) if counts else 0}"
        ax.plot(xs, ys, color=colour, lw=MEAN_TRACE_WIDTH, zorder=3, label=label)
    ax.legend(fontsize=7.6, frameon=False, loc="best")
    ax.grid(alpha=0.25, lw=0.6)
    # Only three of the twenty curves can carry a legend entry without the legend eating the
    # panel, so the ramp itself is the key for the other seventeen. Without this a reader can
    # see that the curves are ordered but cannot read *which* curve is which.
    mappable = plt.cm.ScalarMappable(
        cmap=SURFACE_CMAP, norm=plt.Normalize(vmin=min(outer), vmax=max(outer))
    )
    bar = fig.colorbar(mappable, ax=ax, fraction=0.035, pad=0.015)
    bar.set_label(colourbar_label, fontsize=8)
    bar.ax.tick_params(labelsize=7.5)


def _panel_heatmap(
    *,
    ax: Any,
    fig: Any,
    surface: dict[tuple[float, float], float],
    speeds: list[float],
    release_ms_values: list[float],
    missing: set[tuple[float, float]],
    title: str,
    label: str,
) -> None:
    """Seed-mean distance over the plane, with never-thrown cells hatched rather than shaded."""
    matrix = np.full((len(release_ms_values), len(speeds)), np.nan)
    for j, speed in enumerate(speeds):
        for i, release_ms in enumerate(release_ms_values):
            if (speed, release_ms) in surface:
                matrix[i, j] = surface[(speed, release_ms)]
    image = ax.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        cmap=SURFACE_CMAP,
        extent=(-0.5, len(speeds) - 0.5, -0.5, len(release_ms_values) - 0.5),
    )
    for j, speed in enumerate(speeds):
        for i, release_ms in enumerate(release_ms_values):
            if (speed, release_ms) in missing:
                ax.add_patch(
                    mpatches.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        facecolor=NOT_THROWN_COLOUR,
                        hatch="///",
                        edgecolor="white",
                        linewidth=0,
                    )
                )
    ax.set_xticks(np.arange(len(speeds))[::3])
    ax.set_xticklabels([f"{s:.0f}" for s in speeds[::3]], fontsize=7.5)
    ax.set_yticks(np.arange(len(release_ms_values))[::3])
    ax.set_yticklabels([f"{m:.0f}" for m in release_ms_values[::3]], fontsize=7.5)
    ax.set_xlabel("commanded release speed (joint-path deg/s)", fontsize=8.5)
    ax.set_ylabel("gripper release (ms from swing start)", fontsize=8.5)
    ax.set_title(title, fontsize=9.5)
    bar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.02)
    bar.set_label(label, fontsize=8)
    bar.ax.tick_params(labelsize=7.5)


def build_surface_figure(*, grid: dict[str, Any], output: Path) -> None:
    """The four-panel surface: two curve families and two heatmaps."""
    speeds, release_ms_values = grid["speeds"], grid["release_ms"]
    seeds = grid["seeds"]
    ballistic = distance_table(grid=grid, field="ballistic_impact_x")
    resting = distance_table(grid=grid, field="cube_x_final")
    ballistic_surface = seed_mean_surface(table=ballistic)
    resting_surface = seed_mean_surface(table=resting)
    missing = {(s, m) for s in speeds for m in release_ms_values if (s, m) not in ballistic_surface}

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 10.2))
    _panel_curve_family(
        ax=axes[0][0],
        fig=fig,
        table=ballistic,
        seeds=seeds,
        outer=release_ms_values,
        inner=speeds,
        outer_is_speed=False,
        colourbar_label="gripper release (ms)",
    )
    axes[0][0].set_xlabel("commanded release speed (joint-path deg/s)", fontsize=8.5)
    axes[0][0].set_ylabel("ballistic distance from base (m)", fontsize=8.5)
    axes[0][0].set_title(
        f"Ballistic distance vs speed, one curve per release ms\n"
        f"{len(release_ms_values)} curves over {len(speeds)} speeds, {len(seeds)} seeds each "
        "— faint traces are individual seeds",
        fontsize=9.5,
    )
    _panel_curve_family(
        ax=axes[0][1],
        fig=fig,
        table=ballistic,
        seeds=seeds,
        outer=speeds,
        inner=release_ms_values,
        outer_is_speed=True,
        colourbar_label="commanded speed (deg/s)",
    )
    axes[0][1].set_xlabel("gripper release (ms from swing start)", fontsize=8.5)
    axes[0][1].set_ylabel("ballistic distance from base (m)", fontsize=8.5)
    axes[0][1].set_title(
        f"The same surface transposed: distance vs release ms, one curve per speed\n"
        f"{len(speeds)} curves over {len(release_ms_values)} milliseconds, "
        f"{len(seeds)} seeds each — non-parallel curves are the interaction",
        fontsize=9.5,
    )
    _panel_heatmap(
        ax=axes[1][0],
        fig=fig,
        surface=ballistic_surface,
        speeds=speeds,
        release_ms_values=release_ms_values,
        missing=missing,
        title=(
            f"Ballistic distance (primary criterion), mean of {len(seeds)} seeds\n"
            "free-flight parabola extrapolated to the resting height"
        ),
        label="distance from base (m)",
    )
    _panel_heatmap(
        ax=axes[1][1],
        fig=fig,
        surface=resting_surface,
        speeds=speeds,
        release_ms_values=release_ms_values,
        missing=missing,
        title=(
            f"Resting distance (recorded alongside), mean of {len(seeds)} seeds\n"
            "contaminated by bin contact — a step, not a drift"
        ),
        label="distance from base (m)",
    )
    fig.suptitle(
        "Tossing3D: the (release speed x gripper release ms) surface — "
        f"{len(speeds)} x {len(release_ms_values)} x {len(seeds)} seeds = "
        f"{len(speeds) * len(release_ms_values) * len(seeds)} cells, standoff fixed at "
        f"{grid['standoffs'][0]:.2f} m",
        fontsize=11.5,
    )
    fig.text(
        0.5,
        0.008,
        "Sequential palette, not the project's blue/orange: those encode assistance-available "
        "vs nothing-intervenes, and no cell here is an arm of that comparison.\n"
        "The millisecond is an ordered continuous variable, so its curve family is ramped.",
        ha="center",
        va="bottom",
        fontsize=7.6,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.965))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def build_reversal_figure(*, grid: dict[str, Any], output: Path) -> None:
    """The monotonicity test: paired change in ballistic distance per adjacent speed step."""
    ballistic = distance_table(grid=grid, field="ballistic_impact_x")
    records = adjacent_speed_reversals(table=ballistic)

    fig, ax = plt.subplots(figsize=(11.2, 5.4))
    xs = np.arange(len(records))
    means = np.array([r["mean_difference"] for r in records])
    significant_negative = [r["mean_difference"] < 0 and r["p_value_holm"] < 0.05 for r in records]
    colours = ["#8B1A1A" if flag else "#4C72B0" for flag in significant_negative]
    ax.bar(xs, means * 1000.0, color=colours, width=0.72)
    # 95% intervals on every bar. The finding here is a *negative* one -- no step's apparent
    # reversal survives -- and a null claim is only readable if the reader can see how
    # tightly zero is bracketed. Bars alone leave "no effect" and "no power" looking
    # identical, which is exactly the ambiguity kb#11's single-seed `1/16` left behind.
    ax.errorbar(
        xs,
        means * 1000.0,
        yerr=[
            (means - np.array([r["ci_low"] for r in records])) * 1000.0,
            (np.array([r["ci_high"] for r in records]) - means) * 1000.0,
        ],
        fmt="none",
        ecolor="#222222",
        elinewidth=1.1,
        capsize=3,
        zorder=4,
    )
    ax.axhline(0.0, color="#333333", lw=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"{r['speed_low']:.1f}\n→{r['speed_high']:.1f}" for r in records], fontsize=7.2
    )
    ax.set_xlabel("adjacent commanded-speed step (joint-path deg/s)", fontsize=9)
    ax.set_ylabel("paired change in ballistic distance (mm)", fontsize=9)
    n_reversals = sum(1 for r in records if r["mean_difference"] < 0)
    n_significant = sum(significant_negative)
    ax.set_title(
        f"Does ballistic distance ever fall as commanded speed rises?\n"
        f"{n_reversals}/{len(records)} steps have a negative mean; "
        f"{n_significant}/{len(records)} are negative and survive Holm correction — "
        f"each step paired on {int(records[0]['n'])} (release ms, seed) cells",
        fontsize=10,
    )
    ax.legend(
        handles=[
            mpatches.Patch(color="#8B1A1A"),
            mpatches.Patch(color="#4C72B0"),
        ],
        labels=[
            f"reversal, Holm-adjusted p < 0.05 — {n_significant}/{len(records)} steps",
            f"distance rises with speed — {len(records) - n_reversals}/{len(records)} steps",
        ],
        fontsize=8.2,
        frameon=False,
        loc="best",
    )
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--surface-output", type=Path, required=True)
    parser.add_argument("--reversal-output", type=Path, required=True)
    parser.add_argument("--default-release-ms", type=float, default=DEFAULT_RELEASE_MS)
    args = parser.parse_args()

    grid = load_grid(path=args.grid)
    ballistic = distance_table(grid=grid, field="ballistic_impact_x")
    resting = distance_table(grid=grid, field="cube_x_final")
    total_cells = len(grid["speeds"]) * len(grid["release_ms"]) * len(grid["seeds"])
    dead = dead_cells(grid=grid)

    build_surface_figure(grid=grid, output=args.surface_output)
    build_reversal_figure(grid=grid, output=args.reversal_output)
    print(f"wrote {args.surface_output}")
    print(f"wrote {args.reversal_output}")

    print(f"\ncells measured: {len(grid['rows'])}/{total_cells}")
    print(f"cells where nothing was thrown: {len(dead)}/{len(grid['rows'])}")
    if dead:
        print(f"  first few: {dead[:8]}")

    # --- Question 1: degeneracy -------------------------------------------------------
    nearest_default = min(grid["release_ms"], key=lambda m: abs(m - args.default_release_ms))
    two_d = reach_interval(table=ballistic)
    one_d = reach_interval(table=ballistic, release_ms=nearest_default)
    print("\n== Q1: is the reachable set degenerate? ==")
    print(
        f"nearest grid row to the {args.default_release_ms:.0f} ms default: {nearest_default:.1f}"
    )
    print(
        f"  2-D reach (both dials):  [{two_d[0]:.4f}, {two_d[1]:.4f}] m, "
        f"span {two_d[1] - two_d[0]:.4f}"
    )
    print(
        f"  1-D reach (speed only):  [{one_d[0]:.4f}, {one_d[1]:.4f}] m, "
        f"span {one_d[1] - one_d[0]:.4f}"
    )
    print(f"  widening factor: {(two_d[1] - two_d[0]) / (one_d[1] - one_d[0]):.2f}x")
    shares = variance_shares(table=ballistic)
    print("  variance shares of the seed-mean surface:")
    for name, value in shares.items():
        print(f"    {name:12s} {value:.4f}")

    # The paired test that the second dial does anything at all, at fixed speed.
    lowest, highest = grid["release_ms"][0], grid["release_ms"][-1]
    shared = sorted(
        {(s, k) for s, m, k in ballistic if m == lowest}
        & {(s, k) for s, m, k in ballistic if m == highest}
    )
    ms_effect = paired_difference_test(
        before=np.array([ballistic[(s, lowest, k)] for s, k in shared]),
        after=np.array([ballistic[(s, highest, k)] for s, k in shared]),
    )
    print(
        f"  paired {lowest:.0f} ms -> {highest:.0f} ms at matched (speed, seed), "
        f"n={int(ms_effect['n'])}: mean {ms_effect['mean_difference'] * 1000:+.1f} mm, "
        f"t={ms_effect['t_statistic']:.2f}, p={ms_effect['p_value']:.3g}"
    )

    # --- Question 2: the reversal -----------------------------------------------------
    records = adjacent_speed_reversals(table=ballistic)
    negative = [r for r in records if r["mean_difference"] < 0]
    significant = [r for r in negative if r["p_value_holm"] < 0.05]
    print("\n== Q2: does the speed reversal survive? ==")
    print(f"adjacent speed steps with a negative mean change: {len(negative)}/{len(records)}")
    print(f"  of those, Holm-adjusted p < 0.05: {len(significant)}/{len(records)}")
    for record in records:
        marker = "REVERSAL" if record["mean_difference"] < 0 else "        "
        star = "*" if record["p_value_holm"] < 0.05 else " "
        print(
            f"  {record['speed_low']:6.2f} -> {record['speed_high']:6.2f} deg/s  "
            f"{record['mean_difference'] * 1000:+8.2f} mm  n={int(record['n'])}  "
            f"t={record['t_statistic']:+8.2f}  p={record['p_value']:.3g}  "
            f"p_holm={record['p_value_holm']:.3g} {star} {marker}"
        )

    # --- The resting-position companion ----------------------------------------------
    print("\n== resting position, recorded alongside (not the primary criterion) ==")
    common = sorted(set(ballistic) & set(resting))
    gaps = np.array([resting[k] - ballistic[k] for k in common])
    print(f"resting minus ballistic over {len(common)}/{len(grid['rows'])} thrown cells:")
    print(
        f"  mean {gaps.mean() * 1000:+.1f} mm, sd {gaps.std(ddof=1) * 1000:.1f} mm, "
        f"min {gaps.min() * 1000:+.1f} mm, max {gaps.max() * 1000:+.1f} mm"
    )


if __name__ == "__main__":
    main()
