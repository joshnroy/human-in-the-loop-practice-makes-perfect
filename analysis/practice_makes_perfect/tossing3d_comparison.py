"""Post-run analysis for EES on Tossing3D, the KINDER MuJoCo/PyBullet domain: the learning
curve against whatever arms were run, and the standoff-response sweep that says what there
is to learn in the first place.

Deliberately arm-agnostic (`--arm label=root`, repeatable) rather than hard-coding
"ees vs random-skills" the way `tossingroom_comparison.py` does. `RandomSkillsMethod`
cannot run on this domain at all -- after a `Toss` the cube is past the barrier,
`Reachable` is false, no ground skill's preconditions hold, and
`RandomSkillsMethod.get_labeled_action` asserts rather than degrading to a no-op the way
`EesMethod` does -- so the floor here is EES's own pre-practice checkpoint and the ceiling
is `skill-oracle`.

Reads only already-produced output (CLAUDE.md's `analysis/` convention -- never runs a
simulation): `<root>/<method>/<seed>/stats.json` as written by `--output-dir`, the layout
`scripts/run_sweep.py` produces.

**Why the standoff panel exists.** The domain has three continuous parameters -- `Pick`'s
distance and rotation, and `MoveToThrowPose`'s standoff -- and only the standoff decides
success. The bin is placed from a 1 mm-wide `bin_init_region`, so it is in the same place
every episode and the thing to be learned is a **constant**, not a function of state. That
makes the uniform prior a high floor rather than a low one, and a learning curve is
uninterpretable without it. The panel is therefore drawn from the same `stats.json` files
as everything else -- one `run_sweep` per standoff, `--method skill-oracle` with
`--oracle-throw-standoff` -- rather than being quoted from a docstring.

**The standoff of a run is read from its own `config_snapshot.json`**, never parsed out of
a directory name. `ConfigSnapshot` records the *resolved* argparse namespace, so the value
in it is the value the run actually used; a directory name is a label someone typed and
can disagree with the run it names.

Arms share their seed set, so every arm-vs-arm comparison here is **paired** and is tested
as such -- an unpaired test on a paired design has already produced a wrong p-value in
this project. The test is `TossingRoomComparison`'s exact Wilcoxon signed-rank, imported
rather than re-implemented so there is exactly one hand-rolled significance test in this
package to get wrong.
"""

import argparse
import itertools
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from analysis.practice_makes_perfect.tossingroom_comparison import (  # noqa: E402
    TossingRoomComparison,
)

# Assigned in fixed order and never cycled, so an arm keeps its colour when another is
# added or dropped. Linestyle repeats the identity as a second channel, so the arms stay
# separable in greyscale and under colour-vision deficiency.
_ARM_STYLES: tuple[tuple[str, str], ...] = (
    ("tab:blue", "-"),
    ("tab:orange", "--"),
    ("tab:green", "-."),
)


class Tossing3DComparison:
    """A static-method container, never instantiated."""

    @staticmethod
    def per_seed_counts(*, root: Path, method: str) -> dict[str, dict[int, tuple[int, int]]]:
        """seed -> {transitions: (num_solved, num_total)}, copied verbatim off the
        `evaluations` triples `Metrics.record_evaluation` writes.

        Counts, not rates: 4/10 cannot be recovered from "40%" without the denominator,
        and every table this module prints is `x/y`."""
        counts: dict[str, dict[int, tuple[int, int]]] = {}
        for stats_path in sorted(root.glob(f"{method}/*/stats.json")):
            counts[stats_path.parent.name] = {
                int(transitions): (int(num_solved), int(num_total))
                for transitions, num_solved, num_total in json.loads(stats_path.read_text())[
                    "evaluations"
                ]
            }
        return counts

    @staticmethod
    def per_seed_curves(*, root: Path, method: str) -> dict[str, dict[int, float]]:
        """seed -> {transitions: % solved}. The percentage view, for plotting only."""
        return {
            seed: {
                transitions: 100.0 * solved / total
                for transitions, (solved, total) in curve.items()
            }
            for seed, curve in Tossing3DComparison.per_seed_counts(root=root, method=method).items()
        }

    @staticmethod
    def pooled_endpoints(*, root: Path, method: str) -> dict[str, tuple[int, int]]:
        """`{"first": (solved, total), "final": (solved, total)}` -- evaluation episodes
        summed across seeds, not per-seed rates averaged.

        Seeds can legitimately hold *different* checkpoint grids here: a practice period
        on this domain ends when nothing is left worth practicing (`InteractionComplete`),
        which happens after however many skills that seed's run happened to need, so the
        x-axis is data-driven rather than a fixed multiple of the cycle length. "First"
        and "final" are therefore each seed's own first and last checkpoint."""
        counts = Tossing3DComparison.per_seed_counts(root=root, method=method)
        if not counts:
            return {}
        firsts = [curve[min(curve)] for curve in counts.values()]
        finals = [curve[max(curve)] for curve in counts.values()]
        return {
            "first": (sum(s for s, _ in firsts), sum(t for _, t in firsts)),
            "final": (sum(s for s, _ in finals), sum(t for _, t in finals)),
        }

    @staticmethod
    def endpoint_rates(*, root: Path, method: str) -> dict[str, list[float]]:
        """Per-seed first/final percentages in a stable seed order, so two arms' lists
        line up element-wise and can be handed straight to a paired test."""
        counts = Tossing3DComparison.per_seed_counts(root=root, method=method)
        seeds = sorted(counts, key=int)
        return {
            "seeds": [float(int(seed)) for seed in seeds],
            "first": [
                100.0 * counts[s][min(counts[s])][0] / counts[s][min(counts[s])][1] for s in seeds
            ],
            "final": [
                100.0 * counts[s][max(counts[s])][0] / counts[s][max(counts[s])][1] for s in seeds
            ],
        }

    @staticmethod
    def mean_curve(*, root: Path, method: str) -> dict[int, tuple[float, float]]:
        """transitions -> (mean %, stderr %) across the seeds that reached it.

        Checkpoints are pooled on the exact transition count, and a checkpoint only one
        seed reached is reported with a stderr of 0 and an n of 1 -- which is why the plot
        draws the per-seed lines underneath rather than the mean alone, and why
        `mean_curve_by_checkpoint` exists for the aggregate."""
        by_transitions: dict[int, list[float]] = {}
        for curve in Tossing3DComparison.per_seed_curves(root=root, method=method).values():
            for transitions, percent in curve.items():
                by_transitions.setdefault(transitions, []).append(percent)
        return {
            transitions: (
                statistics.mean(values),
                statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0,
            )
            for transitions, values in sorted(by_transitions.items())
        }

    @staticmethod
    def mean_curve_by_checkpoint(*, root: Path, method: str) -> list[tuple[int, float, float]]:
        """`(median transitions, mean %, stderr %)` for the k-th evaluation sweep, one
        entry per k.

        **This, not `mean_curve`, is what the aggregate line should be drawn from on this
        domain.** A practice period here ends when nothing is left worth practicing, so
        two seeds reach their 7th evaluation at, say, 17 and 19 transitions. Averaging on
        the exact transition count then computes each point over whatever subset of seeds
        happened to land there -- the mean wanders because its *denominator* changes, not
        because the policy did, and n = 1 points sit on the same line as n = 10 ones.
        Averaging on the checkpoint index instead keeps every point an average over the
        same ten runs at the same stage of training; the x coordinate is the median
        transition count at that stage, so the axis still reads in transitions.

        Seeds with fewer checkpoints than the longest run contribute to the prefix only,
        and the point's own n falls accordingly -- which is visible in the stderr rather
        than hidden."""
        curves = [
            [percent for _, percent in sorted(curve.items())]
            for curve in Tossing3DComparison.per_seed_curves(root=root, method=method).values()
        ]
        grids = [
            sorted(curve)
            for curve in Tossing3DComparison.per_seed_counts(root=root, method=method).values()
        ]
        if not curves:
            return []
        out: list[tuple[int, float, float]] = []
        for index in range(max(len(curve) for curve in curves)):
            values = [curve[index] for curve in curves if index < len(curve)]
            transitions = [grid[index] for grid in grids if index < len(grid)]
            stderr = statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0
            out.append((int(statistics.median(transitions)), statistics.mean(values), stderr))
        return out

    @staticmethod
    def standoff_counts(*, root: Path) -> dict[float, tuple[int, int]]:
        """standoff (metres) -> (solved, total), pooled over every skill-oracle run under
        `root` that used that standoff.

        `root` holds one `run_sweep` results-root per standoff, so the glob is
        `*/skill-oracle/*/stats.json`. The standoff is read from each run's own
        `config_snapshot.json` (see the module docstring), and a run without one is an
        error rather than a silent omission -- a missing snapshot, a snapshot from some
        other domain, or an empty root all mean the panel would misrepresent what was
        measured, so each raises instead of quietly drawing fewer points."""
        pooled: dict[float, tuple[int, int]] = {}
        stats_paths = sorted(root.glob("*/skill-oracle/*/stats.json"))
        if not stats_paths:
            raise FileNotFoundError(
                f"{root} holds no */skill-oracle/*/stats.json, so there is no standoff sweep here."
            )
        for stats_path in stats_paths:
            snapshot_path = stats_path.parent / "config_snapshot.json"
            if not snapshot_path.exists():
                raise FileNotFoundError(
                    f"{stats_path.parent} has no config_snapshot.json, so its standoff is unknown."
                )
            args = json.loads(snapshot_path.read_text())["args"]
            if "oracle_throw_standoff" not in args:
                raise KeyError(
                    f"{snapshot_path} records no oracle_throw_standoff; this is not a "
                    "tossing3d skill-oracle run."
                )
            standoff = round(float(args["oracle_throw_standoff"]), 4)
            solved, total = 0, 0
            for _, num_solved, num_total in json.loads(stats_path.read_text())["evaluations"]:
                solved += int(num_solved)
                total += int(num_total)
            previous = pooled.get(standoff, (0, 0))
            pooled[standoff] = (previous[0] + solved, previous[1] + total)
        return dict(sorted(pooled.items()))

    @staticmethod
    def solving_band(*, counts: dict[float, tuple[int, int]]) -> tuple[float, float] | None:
        """The lowest and highest standoff at which *anything* solved, or None.

        Reported as the two measured endpoints, not as an interval the sweep established:
        the grid is coarse, so the true edge lies somewhere between the last solving point
        and the first non-solving one above it, and this returns the former."""
        solving = [standoff for standoff, (solved, _) in counts.items() if solved > 0]
        return (min(solving), max(solving)) if solving else None

    @staticmethod
    def print_standoff_table(*, counts: dict[float, tuple[int, int]]) -> None:
        print("\nStandoff response (privileged oracle, standoff swept, everything else fixed)")
        print(f"{'standoff (m)':>14} | {'solved':>9}")
        print("-" * 26)
        for standoff, (solved, total) in counts.items():
            print(f"{standoff:>14.3f} | {solved:>4}/{total:<4}")
        band = Tossing3DComparison.solving_band(counts=counts)
        if band is not None:
            print(f"solving standoffs measured between {band[0]:.3f} m and {band[1]:.3f} m")

    @staticmethod
    def print_arm_table(*, arms: list[tuple[str, Path]], method_of: dict[str, str]) -> None:
        print("\nArms (pooled evaluation episodes across seeds)")
        # Width from the labels themselves: a fixed one silently ragged-edges the table
        # as soon as an arm is called something longer than the guess.
        width = max([len("arm"), *(len(label) for label, _ in arms)])
        print(f"{'arm':>{width}} | {'seeds':>5} | {'pre-practice':>13} | {'end of training':>16}")
        print("-" * (width + 44))
        for label, root in arms:
            method = method_of[label]
            endpoints = Tossing3DComparison.pooled_endpoints(root=root, method=method)
            if not endpoints:
                print(f"{label:>{width}} | {'0':>5} | {'-':>13} | {'-':>16}")
                continue
            seeds = len(Tossing3DComparison.per_seed_counts(root=root, method=method))
            first = f"{endpoints['first'][0]}/{endpoints['first'][1]}"
            final = f"{endpoints['final'][0]}/{endpoints['final'][1]}"
            print(f"{label:>{width}} | {seeds:>5} | {first:>13} | {final:>16}")

    @staticmethod
    def print_paired_tests(*, arms: list[tuple[str, Path]], method_of: dict[str, str]) -> None:
        """Two questions, both paired over the shared seed set: did each arm improve on
        itself, and did the arms end differently?

        **Every** arm pair, not consecutive ones: `itertools.combinations`, matching
        `TossingRoomComparison.print_paired_tests`. Zipping a label list against its own
        tail silently omits first-vs-last as soon as there are three arms, which is not a
        hypothetical -- `_ARM_STYLES` carries three."""
        print("\nPaired tests (exact Wilcoxon signed-rank over seeds)")
        rates = {
            label: Tossing3DComparison.endpoint_rates(root=root, method=method_of[label])
            for label, root in arms
        }
        for label, values in rates.items():
            if not values.get("seeds"):
                continue
            Tossing3DComparison._report(
                description=f"{label}: end of training vs its own pre-practice",
                first=values["final"],
                second=values["first"],
            )
        labels = [label for label, _ in arms if rates[label].get("seeds")]
        for left, right in itertools.combinations(labels, 2):
            shared = sorted(set(rates[left]["seeds"]) & set(rates[right]["seeds"]))
            if not shared:
                print(f"  {left} vs {right}: no shared seeds, no paired test possible")
                continue
            Tossing3DComparison._report(
                description=f"{left} vs {right}: end of training",
                first=[
                    Tossing3DComparison._final_for_seed(values=rates[left], seed=seed)
                    for seed in shared
                ],
                second=[
                    Tossing3DComparison._final_for_seed(values=rates[right], seed=seed)
                    for seed in shared
                ],
            )

    @staticmethod
    def _final_for_seed(*, values: dict[str, list[float]], seed: float) -> float:
        """That seed's end-of-training rate, looked up by seed rather than by position --
        two arms can hold different seed sets, and indexing by position would then pair
        one arm's seed 3 against the other's seed 4."""
        return values["final"][values["seeds"].index(seed)]

    @staticmethod
    def _report(*, description: str, first: list[float], second: list[float]) -> None:
        differences = [a - b for a, b in zip(first, second, strict=True)]
        test = TossingRoomComparison.wilcoxon_signed_rank(first=first, second=second)
        mean = statistics.mean(differences) if differences else 0.0
        verdict = "established" if test["p"] < 0.05 else "NOT established"
        print(
            f"  {description}: mean {mean:+.1f} pp, "
            f"n = {test['n']} non-tied of {len(differences)}, p = {test['p']:.4f} -- {verdict}"
        )

    @staticmethod
    def render(
        *,
        arms: list[tuple[str, Path]],
        method_of: dict[str, str],
        standoff_root: Path | None,
        output: Path,
        title: str,
    ) -> None:
        """Three panels: the learning curves, the per-seed endpoints, and the standoff
        response. The per-seed panel is not decoration -- with ten seeds a bar chart of
        two means hides one seed driving the whole effect."""
        num_panels = 3 if standoff_root is not None else 2
        figure, axes = plt.subplots(1, num_panels, figsize=(5.5 * num_panels, 4.4))
        Tossing3DComparison._plot_curves(ax=axes[0], arms=arms, method_of=method_of)
        Tossing3DComparison._plot_per_seed(ax=axes[1], arms=arms, method_of=method_of)
        if standoff_root is not None:
            Tossing3DComparison._plot_standoff(
                ax=axes[2], counts=Tossing3DComparison.standoff_counts(root=standoff_root)
            )
        figure.suptitle(title)
        figure.tight_layout()
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=150)
        plt.close(figure)

    @staticmethod
    def _plot_curves(*, ax, arms: list[tuple[str, Path]], method_of: dict[str, str]) -> None:
        for index, (label, root) in enumerate(arms):
            colour, linestyle = _ARM_STYLES[index % len(_ARM_STYLES)]
            for curve in Tossing3DComparison.per_seed_curves(
                root=root, method=method_of[label]
            ).values():
                xs = sorted(curve)
                ax.plot(xs, [curve[x] for x in xs], color=colour, alpha=0.18, linewidth=0.9)
            mean = Tossing3DComparison.mean_curve_by_checkpoint(root=root, method=method_of[label])
            if not mean:
                continue
            if len(mean) == 1:
                # A non-learning arm (skill-oracle) evaluates once, at 0 transitions, so
                # it is a single point and would draw nothing -- leaving a legend entry
                # with no line, which reads as a plotting bug. It is a *reference level*,
                # not a curve, so draw it as one across the whole axis.
                ax.axhline(
                    mean[0][1], color=colour, linestyle=linestyle, linewidth=2.0, label=label
                )
                continue
            xs = [transitions for transitions, _, _ in mean]
            ys = [value for _, value, _ in mean]
            errors = [stderr for _, _, stderr in mean]
            ax.plot(xs, ys, color=colour, linestyle=linestyle, linewidth=2.2, label=label)
            ax.fill_between(
                xs,
                [y - e for y, e in zip(ys, errors, strict=True)],
                [y + e for y, e in zip(ys, errors, strict=True)],
                color=colour,
                alpha=0.15,
            )
        ax.set_xlabel("online transitions (skill executions during practice)")
        ax.set_ylabel("evaluation tasks solved (%)")
        ax.set_ylim(-3, 103)
        ax.set_title("Learning curves: mean +- s.e. over seeds, thin lines individual seeds")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)

    @staticmethod
    def _plot_per_seed(*, ax, arms: list[tuple[str, Path]], method_of: dict[str, str]) -> None:
        for index, (label, root) in enumerate(arms):
            colour, _ = _ARM_STYLES[index % len(_ARM_STYLES)]
            rates = Tossing3DComparison.endpoint_rates(root=root, method=method_of[label])
            if not rates.get("seeds"):
                continue
            for seed_index, (first, final) in enumerate(
                zip(rates["first"], rates["final"], strict=True)
            ):
                x = index * 2.4 + 0.06 * seed_index
                ax.plot([x, x + 1.0], [first, final], color=colour, alpha=0.5, linewidth=1.0)
                ax.plot([x, x + 1.0], [first, final], "o", color=colour, markersize=3.5)
            ax.text(
                index * 2.4 + 0.5,
                -9,
                label,
                ha="center",
                fontsize=9,
                color=colour,
            )
        ax.set_xticks([])
        ax.set_ylabel("evaluation tasks solved (%)")
        ax.set_ylim(-12, 103)
        ax.set_title("Per-seed: pre-practice (left) to end of training (right)")
        ax.grid(alpha=0.3, axis="y")

    @staticmethod
    def _plot_standoff(*, ax, counts: dict[float, tuple[int, int]]) -> None:
        standoffs = list(counts)
        rates = [100.0 * solved / total for solved, total in counts.values()]
        ax.plot(standoffs, rates, "o-", color="tab:purple")
        for standoff, (solved, total) in counts.items():
            ax.annotate(
                f"{solved}/{total}",
                (standoff, 100.0 * solved / total),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=7,
            )
        band = Tossing3DComparison.solving_band(counts=counts)
        if band is not None:
            ax.axvspan(band[0], band[1], color="tab:purple", alpha=0.12)
        ax.set_xlabel("MoveToThrowPose standoff (m)")
        ax.set_ylabel("evaluation tasks solved (%)")
        ax.set_ylim(-3, 112)
        ax.set_title("What there is to learn: the oracle's standoff response")
        ax.grid(alpha=0.3)


def _parse_arm(*, raw: str) -> tuple[str, str, Path]:
    """`label=path`, or `label:method=path` when the label is not the `--method` name.

    The two forms are told apart by the ":" in the *label*, never by counting "="
    separators, and the path is everything after the **first** "=". That matters: a
    results-root legitimately contains "=" (`.../standoff=1.35`), and a count-based parse
    reads such a path as a three-field arm and silently plots the wrong directory."""
    label_part, separator, path_part = raw.partition("=")
    if not separator or not label_part or not path_part:
        raise ValueError(f"expected 'label=path' or 'label:method=path', got {raw!r}")
    label, _, method = label_part.partition(":")
    return label, method or label, Path(path_part)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        help="Repeatable 'label=results-root', or 'label:method=results-root' when the "
        "label differs from the --method name the sweep ran under.",
    )
    parser.add_argument(
        "--standoff-root",
        type=Path,
        default=None,
        help="Directory holding one run_sweep results-root per --oracle-throw-standoff.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="EES on Tossing3D (KINDER simulator)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    parsed = [_parse_arm(raw=raw) for raw in args.arm]
    arms = [(label, root) for label, _, root in parsed]
    method_of = {label: method for label, method, _ in parsed}
    Tossing3DComparison.print_arm_table(arms=arms, method_of=method_of)
    Tossing3DComparison.print_paired_tests(arms=arms, method_of=method_of)
    if args.standoff_root is not None:
        Tossing3DComparison.print_standoff_table(
            counts=Tossing3DComparison.standoff_counts(root=args.standoff_root)
        )
    Tossing3DComparison.render(
        arms=arms,
        method_of=method_of,
        standoff_root=args.standoff_root,
        output=args.output,
        title=args.title,
    )
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
