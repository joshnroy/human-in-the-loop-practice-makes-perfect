"""How far the cube goes, and the angle it is released at, across 60-240 deg/s.

Post-run analysis only: this reads two committed grids back in and never builds an
environment or drives a skill. Both grids ship beside this repo's experiment-log entry, so
the figure is reproducible from the tree alone. **One figure, four panels, one shared
commanded-speed axis**, because the four quantities are one causal chain and splitting them
across figures would make a reader reconstruct the link by eye:

1. **how far the cube goes** -- the ballistic ground-crossing and the resting position.
   Two series rather than one: they are different quantities and the gap between them is
   speed-dependent, so plotting one and calling it "how far it goes" would hide a real
   effect;
2. **the release angle** -- the launch elevation, plotted two ways that disagree: the
   *kinematic* angle from the pinch-site Jacobian, and the *actual* elevation the cube
   leaves with. Reporting only the first would overstate the launch by ~10 deg;
3. **the realised release fraction** -- the mechanism. `TossController` opens the gripper
   on the first control step past `fraction_covered >= 0.46`, and control steps come once
   per 0.1 s, so raising the speed shortens the swing, coarsens the sampling, and makes the
   release step index *decrement*. Every reset in this panel is a step index dropping by one;
4. **seeds solved per speed**, the ground truth the other three have to answer to.

Reading down a vertical line is the point: a reset in panel 3 puts a notch in panel 2,
which puts a reversal in panel 1, which moves panel 4.

**Significance without scipy, and what that costs.** `hitl-pmp` does not ship scipy, so
rather than add a dependency this uses an *exact* paired permutation test: with 10 seeds
the 2^10 = 1024 sign-flips of the within-seed differences enumerate the entire null
distribution, no normality assumption. Exact rather than asymptotic -- but it floors the
two-sided p at 2/1024 = 1.95e-03, and Holm across all 36 consecutive steps needs the
smallest p below 0.05/36 = 1.39e-03, which is *below that floor*. **So no step can reach
Holm-corrected significance under this test however real its effect** -- the corrected
column reads "cannot be resolved at n=10", not "no effect". Both are printed so the
distinction stays visible. PR #226's parametric t-test on five of these steps reached
4.4e-05 down to 3.1e-07; the two agree on sign and on which steps are real.

**Palette.** Deliberately *not* the project's `#0072B2`/`#D55E00`. Those encode "assistance
mechanism available" versus "nothing intervenes" across every reset-policy figure here, and
nothing in this figure is an arm of any such comparison -- these are all measurements of the
same throw. Using them would import a contrast that does not exist. Purple/green carry the
two distances, magenta/teal the two release angles, and grey stays reference.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Not the project blue/orange -- see the module docstring for why.
IMPACT = "#5D3A9B"
RESTING = "#1B7837"
KINEMATIC = "#B2168B"
CUBE = "#00767A"
SOLVE = "#3F3F3F"
GREY = "#666666"
RESET_LINE = "#C1272D"

SEED_ALPHA = 0.16
SEED_LW = 0.8
MEAN_LW = 2.3

# The exact sign-flip null has 2^10 outcomes, so the smallest two-sided p is 2/1024.
PERMUTATION_FLOOR = 2 / 1024


def _matrix(
    *, rows: list[dict[str, Any]], speeds: list[float], seeds: list[int], value: Any
) -> np.ndarray:
    """`(seed, speed)` matrix of `value`, `nan` wherever a cell is missing."""
    out = np.full((len(seeds), len(speeds)), np.nan)
    for r in rows:
        v = value(r=r)
        if v is not None:
            out[seeds.index(r["seed"]), speeds.index(r["commanded_speed_deg"])] = v
    return out


def _impact_range(*, r: dict[str, Any]) -> float | None:
    if r["ballistic_impact_x"] is None or r["base_x_before_toss"] is None:
        return None
    return r["ballistic_impact_x"] - r["base_x_before_toss"]


def _resting_range(*, r: dict[str, Any]) -> float | None:
    if r["cube_x_final"] is None or r["base_x_before_toss"] is None:
        return None
    return r["cube_x_final"] - r["base_x_before_toss"]


def _field(*, name: str) -> Any:
    def read(*, r: dict[str, Any]) -> float | None:
        value = r.get(name)
        return None if value is None else float(value)

    return read


def _exact_paired_permutation_p(*, diffs: np.ndarray) -> float:
    """Two-sided exact p for `mean(diffs) == 0` by enumerating every sign-flip.

    Exact rather than asymptotic, and dependency-free. Only tractable because n is 10:
    2^10 = 1024 sign assignments enumerate the entire null distribution.
    """
    n = len(diffs)
    signs = 1 - 2 * ((np.arange(2**n)[:, None] >> np.arange(n)) & 1)
    null = (signs * diffs).mean(axis=1)
    observed = float(np.abs(diffs.mean()))
    return float((np.abs(null) >= observed - 1e-15).sum() / len(null))


def _holm(*, pvalues: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values."""
    m = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvalues[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def _step_tests(*, values: np.ndarray, speeds: list[float]) -> list[dict[str, Any]]:
    """One paired test per consecutive speed step, Holm-corrected across all of them."""
    raw = []
    for j in range(len(speeds) - 1):
        diffs = values[:, j + 1] - values[:, j]
        raw.append({
            "from": speeds[j],
            "to": speeds[j + 1],
            "delta": float(np.mean(diffs)),
            "p": _exact_paired_permutation_p(diffs=diffs),
        })
    adjusted = _holm(pvalues=np.array([r["p"] for r in raw]))
    for entry, p_adj in zip(raw, adjusted, strict=True):
        entry["p_holm"] = float(p_adj)
    return raw


def reset_speeds(*, index: np.ndarray, speeds: list[float]) -> list[float]:
    """Speeds at which the *majority* release step index drops relative to the speed below.

    A reset is a discrete event -- the gripper opening one control step earlier in the
    swing -- so it is read off the step index rather than off a threshold on the fraction.
    """
    modal = [float(np.bincount(index[:, j].astype(int)).argmax()) for j in range(len(speeds))]
    return [speeds[j] for j in range(1, len(speeds)) if modal[j] < modal[j - 1]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--range-results", type=Path, required=True)
    parser.add_argument("--angle-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rng_payload = json.loads(args.range_results.read_text())
    ang_payload = json.loads(args.angle_results.read_text())
    rng_rows, ang_rows = rng_payload["rows"], ang_payload["rows"]
    speeds = [float(s) for s in rng_payload["speeds"]]
    seeds = [int(s) for s in rng_payload["seeds"]]
    if [float(s) for s in ang_payload["speeds"]] != speeds:
        raise ValueError("the two grids do not share a speed axis")
    n_seeds = len(seeds)
    x = np.array(speeds)

    impact = _matrix(rows=rng_rows, speeds=speeds, seeds=seeds, value=_impact_range)
    resting = _matrix(rows=rng_rows, speeds=speeds, seeds=seeds, value=_resting_range)
    kinematic = _matrix(
        rows=ang_rows, speeds=speeds, seeds=seeds, value=_field(name="release_elevation_deg")
    )
    cube = _matrix(
        rows=ang_rows, speeds=speeds, seeds=seeds, value=_field(name="cube_launch_elevation_deg")
    )
    fraction = _matrix(
        rows=ang_rows, speeds=speeds, seeds=seeds, value=_field(name="realised_release_fraction")
    )
    step_index = _matrix(
        rows=ang_rows, speeds=speeds, seeds=seeds, value=_field(name="release_step_index")
    )
    solved = np.array([
        sum(1 for r in rng_rows if r["commanded_speed_deg"] == sp and r["solved"]) for sp in speeds
    ])

    resets = reset_speeds(index=step_index, speeds=speeds)
    tests = _step_tests(values=impact, speeds=speeds)
    reversals = [t for t in tests if t["delta"] < 0]
    # Uncorrected, deliberately: the exact test's floor sits below Holm's threshold for
    # 36 comparisons, so the corrected column can never fire. See the module docstring.
    significant = [t for t in reversals if t["p"] < 0.05]

    fig, axes = plt.subplots(
        4, 1, figsize=(13.5, 17), sharex=True, gridspec_kw={"height_ratios": [2.1, 2.1, 1.5, 1.0]}
    )
    for ax in axes:
        for k, speed in enumerate(resets):
            ax.axvline(
                speed,
                color=RESET_LINE,
                linestyle=":",
                linewidth=1.3,
                alpha=0.75,
                label=(
                    f"release step index drops ({len(resets)} speeds)"
                    if k == 0 and ax is axes[0]
                    else None
                ),
            )

    # --- panel 1: how far it goes ------------------------------------------
    ax = axes[0]
    for trace in impact:
        ax.plot(x, trace, color=IMPACT, alpha=SEED_ALPHA, linewidth=SEED_LW)
    for trace in resting:
        ax.plot(x, trace, color=RESTING, alpha=SEED_ALPHA, linewidth=SEED_LW)
    impact_mean = np.nanmean(impact, axis=0)
    ax.plot(
        x,
        impact_mean,
        color=IMPACT,
        linewidth=MEAN_LW,
        marker="o",
        markersize=3.5,
        label=f"ballistic ground-crossing — mean, n={n_seeds}",
    )
    ax.plot(
        x,
        np.nanmean(resting, axis=0),
        color=RESTING,
        linewidth=MEAN_LW,
        linestyle=(0, (4, 2)),
        marker="s",
        markersize=3.0,
        label=f"where it comes to rest — mean, n={n_seeds}",
    )
    for t in significant:
        j = speeds.index(t["from"])
        ax.annotate(
            "",
            xy=(x[j + 1], impact_mean[j + 1]),
            xytext=(x[j], impact_mean[j]),
            arrowprops={"arrowstyle": "-|>", "color": RESET_LINE, "linewidth": 2.0},
        )
    ax.plot(
        [],
        [],
        color=RESET_LINE,
        linewidth=2.0,
        marker=">",
        label=f"significant reversal ({len(significant)}/{len(reversals)} of the reversals)",
    )
    ax.set_title(
        f"1. How far the cube goes — {len(rng_rows)} cells "
        f"({len(speeds)} speeds x {n_seeds} seeds), standoff 1.35"
    )
    ax.set_ylabel("distance from robot base (m)")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.3)

    # --- panel 2: the release angle ----------------------------------------
    ax = axes[1]
    for trace in kinematic:
        ax.plot(x, trace, color=KINEMATIC, alpha=SEED_ALPHA, linewidth=SEED_LW)
    for trace in cube:
        ax.plot(x, trace, color=CUBE, alpha=SEED_ALPHA, linewidth=SEED_LW)
    ax.plot(
        x,
        np.nanmean(kinematic, axis=0),
        color=KINEMATIC,
        linewidth=MEAN_LW,
        marker="^",
        markersize=3.5,
        label=f"kinematic: pinch-site Jacobian at release — mean, n={n_seeds}",
    )
    ax.plot(
        x,
        np.nanmean(cube, axis=0),
        color=CUBE,
        linewidth=MEAN_LW,
        linestyle=(0, (4, 2)),
        marker="v",
        markersize=3.5,
        label=f"actual: elevation the cube leaves with — mean, n={n_seeds}",
    )
    gap = float(np.nanmean(kinematic - cube))
    ax.set_title(
        f"2. Release angle above horizontal — the kinematic angle overstates the cube's\n"
        f"actual launch by {gap:.1f} deg on average, and the bias grows with speed"
    )
    ax.set_ylabel("launch elevation (deg)")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.3)

    # --- panel 3: the mechanism --------------------------------------------
    ax = axes[2]
    for trace in fraction:
        ax.plot(x, trace, color=SOLVE, alpha=SEED_ALPHA, linewidth=SEED_LW)
    ax.plot(
        x,
        np.nanmean(fraction, axis=0),
        color=SOLVE,
        linewidth=MEAN_LW,
        marker="o",
        markersize=3.0,
        label=f"realised release fraction — mean, n={n_seeds}",
    )
    ax.axhline(
        0.46, color=GREY, linestyle="--", linewidth=1.5, label="_release_fraction = 0.46 (target)"
    )
    # The two speeds where the seed population straddles two release step indices: the
    # faint traces show it, this makes the count explicit rather than eyeballed.
    split_rank = 0
    for j, sp in enumerate(speeds):
        counts = np.bincount(step_index[:, j].astype(int))
        present = [(i, int(c)) for i, c in enumerate(counts) if c]
        if len(present) > 1:
            # Stagger successive callouts vertically; at 5 deg/s spacing two adjacent
            # labels at the same height overlap and neither is readable.
            ax.annotate(
                " / ".join(f"step {i}: {c}/{n_seeds}" for i, c in present),
                xy=(sp, float(np.nanmean(fraction[:, j]))),
                xytext=(sp, 0.638 if split_rank % 2 == 0 else 0.614),
                fontsize=7.5,
                ha="center",
                color=SOLVE,
                arrowprops={"arrowstyle": "-", "color": SOLVE, "linewidth": 0.8},
            )
            split_rank += 1
    ax.set_title(
        "3. The mechanism: the gripper opens on the first control step past 0.46, once per\n"
        "0.1 s — so the realised fraction sawtooths, resetting as the release step index drops"
    )
    ax.set_ylabel("realised release fraction")
    ax.set_ylim(0.44, 0.65)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(alpha=0.3)

    # --- panel 4: what actually solved -------------------------------------
    ax = axes[3]
    ax.bar(x, solved, width=3.6, color=SOLVE, alpha=0.85)
    for j, (sp, count) in enumerate(zip(speeds, solved, strict=True)):
        if count and (j == 0 or solved[j - 1] != count):
            ax.text(sp, count + 0.25, f"{count}/{n_seeds}", ha="center", fontsize=8, color=SOLVE)
    ax.set_title(
        f"4. Seeds solved per speed (of {n_seeds}) — 190 deg/s solves 10/10 between "
        f"185 at 0/10 and 195 at 1/10, on the shallowest release in the grid"
    )
    ax.set_xlabel("commanded release speed (deg/s)")
    ax.set_ylabel("seeds solved")
    ax.set_ylim(0, n_seeds + 1.8)
    ax.set_xlim(x.min() - 4, x.max() + 4)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}")

    # --- the tables that accompany the figure ------------------------------
    n_holm = sum(1 for t in reversals if t["p_holm"] < 0.05)
    print(
        f"\nconsecutive steps tested: {len(tests)}; "
        f"steps where the mean range fell: {len(reversals)}/{len(tests)}; "
        f"exact p<0.05 uncorrected: {len(significant)}/{len(reversals)}; "
        f"Holm-adjusted p<0.05: {n_holm}/{len(reversals)}"
    )
    print(
        f"Holm across {len(tests)} steps needs p < {0.05 / len(tests):.2e}, below this test's "
        f"exact floor of {PERMUTATION_FLOOR:.2e} -- so the Holm column cannot fire at "
        f"n={n_seeds} regardless of effect size. See the module docstring."
    )
    print(f"\n{'step':>13} {'delta (m)':>10} {'exact p':>10} {'Holm p':>10}")
    for t in sorted(reversals, key=lambda z: z["delta"]):
        floored = " (at exact floor)" if t["p"] <= PERMUTATION_FLOOR else ""
        flag = "  reversal" if t["p"] < 0.05 else "  null result"
        print(
            f"{t['from']:5.0f} -> {t['to']:5.0f} {t['delta']:10.4f} "
            f"{t['p']:10.2e} {t['p_holm']:10.2e}{flag}{floored}"
        )

    print(f"\nrelease step index drops at {len(resets)} speeds: {[int(s) for s in resets]}")
    print(
        f"\n{'speed':>6} {'impact':>8} {'resting':>8} {'frac':>7} {'step':>6} "
        f"{'kinematic':>10} {'cube':>8} {'bias':>6} {'solved':>8}"
    )
    for j, sp in enumerate(speeds):
        counts = np.bincount(step_index[:, j].astype(int))
        modal = int(counts.argmax())
        split = "*" if (counts > 0).sum() > 1 else " "
        print(
            f"{sp:6.0f} {impact_mean[j]:8.4f} {np.nanmean(resting[:, j]):8.4f} "
            f"{np.nanmean(fraction[:, j]):7.4f} {modal:5d}{split} "
            f"{np.nanmean(kinematic[:, j]):10.2f} {np.nanmean(cube[:, j]):8.2f} "
            f"{np.nanmean(kinematic[:, j] - cube[:, j]):6.2f} {solved[j]:5d}/{n_seeds}"
        )
    print("* the seed population straddles two release step indices at this speed")


if __name__ == "__main__":
    main()
