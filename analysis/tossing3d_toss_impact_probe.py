"""Read `scripts/tossing3d_toss_impact_probe.py`'s JSON back and draw the figure.

Post-run analysis only: this never builds an environment or drives a skill. Four panels,
because the measurement answers four separate questions and a single curve would let three
of them hide:

1. **impact range vs speed** -- the quantity the union predicate was supposed to be built
   on, with `THROW_RANGE` and PR #213's linear fit drawn as references so the reader can
   see at a glance that the constant matches neither the impact nor the resting curve;
2. **the impact-to-rest offset** -- the "roll", split by what the cube actually hit first,
   since a cube caught by the bin has no roll to measure;
3. **arrival angle** -- which turns out to co-determine whether the bin catches the cube,
   and which sawtooths in speed for the same release-quantisation reason the range does;
4. **solved x/10** -- the ground truth, showing the non-monotone band the first three
   panels explain.

Per-seed traces are drawn faint underneath the bold means throughout, because the whole
point of a 10-seed grid is that the spread is visible rather than asserted.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# The project palette. Blue is the quantity a predicate would be built on (impact);
# orange is the quantity PR #213 actually measured (resting); grey dotted is reserved for
# reference lines that are not themselves being measured here.
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREY = "#666666"

SEED_ALPHA = 0.16
SEED_LW = 0.8
MEAN_LW = 2.3

# `predicates.THROW_RANGE`, and PR #213's fit over its three-point grid, both drawn as
# references rather than recomputed here.
THROW_RANGE = 1.275
PR213_SLOPE = 0.004872
PR213_INTERCEPT = 0.6614


def _by_speed(*, rows: list[dict[str, Any]], speeds: list[float]) -> dict[float, list[dict]]:
    return {sp: [r for r in rows if r["commanded_speed_deg"] == sp] for sp in speeds}


def _seed_series(
    *, rows: list[dict[str, Any]], speeds: list[float], seeds: list[int], value: Any
) -> dict[int, np.ndarray]:
    """One trace per seed, indexed by speed, with `nan` wherever a cell is missing."""
    out: dict[int, np.ndarray] = {}
    for seed in seeds:
        series = np.full(len(speeds), np.nan)
        for i, sp in enumerate(speeds):
            match = [r for r in rows if r["seed"] == seed and r["commanded_speed_deg"] == sp]
            if match:
                v = value(r=match[0])
                if v is not None:
                    series[i] = v
        out[seed] = series
    return out


def _impact_range(*, r: dict[str, Any]) -> float | None:
    if r["ballistic_impact_x"] is None or r["base_x_before_toss"] is None:
        return None
    return r["ballistic_impact_x"] - r["base_x_before_toss"]


def _resting_range(*, r: dict[str, Any]) -> float | None:
    if r["cube_x_final"] is None or r["base_x_before_toss"] is None:
        return None
    return r["cube_x_final"] - r["base_x_before_toss"]


def _offset(*, r: dict[str, Any]) -> float | None:
    imp, rest = _impact_range(r=r), _resting_range(r=r)
    if imp is None or rest is None:
        return None
    return rest - imp


def _arrival_deg(*, r: dict[str, Any]) -> float | None:
    if r["ballistic_impact_vx"] is None or r["ballistic_impact_vz"] is None:
        return None
    return float(np.degrees(np.arctan2(-r["ballistic_impact_vz"], r["ballistic_impact_vx"])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.results.read_text())
    rows = payload["rows"]
    speeds = [float(s) for s in payload["speeds"]]
    seeds = [int(s) for s in payload["seeds"]]
    n_seeds = len(seeds)
    x = np.array(speeds)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # --- panel 1: impact and resting range ---------------------------------
    ax = axes[0][0]
    for series, colour in (
        (_seed_series(rows=rows, speeds=speeds, seeds=seeds, value=_impact_range), BLUE),
        (_seed_series(rows=rows, speeds=speeds, seeds=seeds, value=_resting_range), ORANGE),
    ):
        for trace in series.values():
            ax.plot(x, trace, color=colour, alpha=SEED_ALPHA, linewidth=SEED_LW)
    by_speed_all = _by_speed(rows=rows, speeds=speeds)
    imp_mean = np.array([
        np.nanmean([_impact_range(r=r) for r in by_speed_all[sp]]) for sp in speeds
    ])
    rest_mean = np.array([
        np.nanmean([_resting_range(r=r) for r in by_speed_all[sp]]) for sp in speeds
    ])
    ax.plot(
        x, imp_mean, color=BLUE, linewidth=MEAN_LW, label=f"ballistic impact — mean, n={n_seeds}"
    )
    ax.plot(x, rest_mean, color=ORANGE, linewidth=MEAN_LW, label=f"resting — mean, n={n_seeds}")
    ax.axhline(THROW_RANGE, color=GREY, linestyle=":", linewidth=2.0, label="THROW_RANGE = 1.275")
    ax.plot(
        x,
        PR213_SLOPE * x + PR213_INTERCEPT,
        color=GREY,
        linestyle="--",
        linewidth=1.4,
        label="PR #213 fit (3-point grid)",
    )
    ax.set_title(f"Range vs release speed (10 seeds x {len(speeds)} speeds, standoff 1.35)")
    ax.set_xlabel("commanded release speed (deg/s)")
    ax.set_ylabel("range from base (m)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    # --- panel 2: the impact-to-rest offset --------------------------------
    ax = axes[0][1]
    off_series = _seed_series(rows=rows, speeds=speeds, seeds=seeds, value=_offset)
    for trace in off_series.values():
        ax.plot(x, trace, color=BLUE, alpha=SEED_ALPHA, linewidth=SEED_LW)
    by_speed = _by_speed(rows=rows, speeds=speeds)
    floor_mean = np.full(len(speeds), np.nan)
    bin_mean = np.full(len(speeds), np.nan)
    n_floor = n_bin = 0
    for i, sp in enumerate(speeds):
        floor = [_offset(r=r) for r in by_speed[sp] if r["first_contact_body"] != "bin_0"]
        binned = [_offset(r=r) for r in by_speed[sp] if r["first_contact_body"] == "bin_0"]
        floor = [v for v in floor if v is not None]
        binned = [v for v in binned if v is not None]
        n_floor += len(floor)
        n_bin += len(binned)
        if floor:
            floor_mean[i] = float(np.mean(floor))
        if binned:
            bin_mean[i] = float(np.mean(binned))
    ax.plot(
        x,
        floor_mean,
        color=BLUE,
        linewidth=MEAN_LW,
        label=f"landed on open floor — mean, n={n_floor}",
    )
    ax.plot(
        x,
        bin_mean,
        color=ORANGE,
        linewidth=MEAN_LW,
        linestyle=(0, (4, 2)),
        label=f"caught by the bin — mean, n={n_bin}",
    )
    ax.axhline(0.0, color=GREY, linestyle=":", linewidth=1.5)
    ax.axhline(0.075, color=GREY, linestyle="--", linewidth=1.4, label="0.075 m: the asserted roll")
    ax.set_title(f"Resting minus impact — the 'roll' ({n_floor + n_bin} cells)")
    ax.set_xlabel("commanded release speed (deg/s)")
    ax.set_ylabel("resting - impact (m)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    # --- panel 3: arrival angle --------------------------------------------
    ax = axes[1][0]
    ang_series = _seed_series(rows=rows, speeds=speeds, seeds=seeds, value=_arrival_deg)
    for trace in ang_series.values():
        ax.plot(x, trace, color=BLUE, alpha=SEED_ALPHA, linewidth=SEED_LW)
    ang_mean = np.array([np.nanmean([_arrival_deg(r=r) for r in by_speed[sp]]) for sp in speeds])
    ax.plot(x, ang_mean, color=BLUE, linewidth=MEAN_LW, label=f"arrival angle — mean, n={n_seeds}")
    ax.set_title("Arrival angle below horizontal at the ground crossing")
    ax.set_xlabel("commanded release speed (deg/s)")
    ax.set_ylabel("arrival angle (deg)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    # --- panel 4: what actually solved -------------------------------------
    ax = axes[1][1]
    solved = np.array([sum(1 for r in by_speed[sp] if r["solved"]) for sp in speeds])
    ax.bar(x, solved, width=3.6, color=BLUE, alpha=0.85, label=f"solved (of {n_seeds} per speed)")
    ax.set_title(f"Seeds solved per speed (of {n_seeds}), standoff 1.35")
    ax.set_xlabel("commanded release speed (deg/s)")
    ax.set_ylabel("seeds solved")
    ax.set_ylim(0, n_seeds + 0.5)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
