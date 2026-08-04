"""Post-run analysis for the Tossing3D EES run: the paired arm-level comparison that
sits underneath the learning curve `practice_makes_perfect/ees.py` draws.

Reads only already-produced output (CLAUDE.md's analysis/ convention -- never runs a
simulation): `<root>/<method>/<seed>/stats.json` as written by `--output-dir`, the
layout `scripts/run_sweep.py` produces.

Two things this domain needs that a plot does not show.

**Both comparisons here are paired.** The arms share the seed set 0..9, so EES vs the
random-skills floor is paired across seeds, and so is EES's own endpoint against its
own pre-practice checkpoint. An unpaired test on a paired design has already produced a
wrong p-value in this project, so the test is `TossingRoomComparison`'s exact Wilcoxon
signed-rank -- imported rather than re-implemented, so this arithmetic keeps exactly one
home in the repo, the same reason `analysis/` reads its curves off `Metrics` instead of
recomputing them.

**EES's own start is the reference that matters here, not just the floor.** Tossing3D's
untrained sampler already draws swings uniformly over a range that contains the band
which reaches the goal region, so the pre-practice checkpoint is not near zero and a
curve can go *down*. Reporting only the endpoint against random-skills would hide that,
so `--print-table` reports first, final, and their paired difference per arm, with the
per-seed values printed whenever the spread is driven by a handful of seeds.

Where a comparison does not reach p < 0.05 it is reported as **not established**, with
the sample size a paired design would need for 80% power at the observed effect and
spread -- rather than as a trend, which is how two results here were overclaimed and
then retracted.
"""

import argparse
import math
import statistics
from pathlib import Path

from hitl_pmp.core.metrics.metrics import Metrics

from .tossingroom_comparison import TossingRoomComparison


class Tossing3DComparison:
    """A static-method container, never instantiated, same as every other
    business-logic class in this project."""

    @staticmethod
    def per_seed_curves(*, root: Path, method: str) -> dict[int, dict[int, float]]:
        """`{seed: {transitions: percent_solved}}`, computed by `Metrics` itself rather
        than recomputed here."""
        curves: dict[int, dict[int, float]] = {}
        method_dir = root / method
        if not method_dir.is_dir():
            return curves
        for stats_path in sorted(method_dir.glob("*/stats.json")):
            metrics = Metrics.model_validate_json(stats_path.read_text())
            curves[int(stats_path.parent.name)] = {
                transitions: 100.0 * fraction
                for transitions, fraction in metrics.task_training_curve()
            }
        return curves

    @staticmethod
    def endpoints(*, curves: dict[int, dict[int, float]]) -> dict[int, tuple[float, float]]:
        """`{seed: (first_checkpoint, last_checkpoint)}`. The first checkpoint is the
        pre-practice evaluation at 0 transitions, which is this domain's real baseline."""
        return {
            seed: (curve[min(curve)], curve[max(curve)]) for seed, curve in curves.items() if curve
        }

    @staticmethod
    def required_n_for_80_percent_power(*, differences: list[float]) -> int | None:
        """Paired-design sample size for 80% power at alpha = 0.05, given the observed
        mean difference and its sd -- the normal approximation
        `n = (1.96 + 0.842)^2 * sd^2 / d^2`.

        Reported only to make "not established" quantitative: it says how far the design
        was from being able to detect what it saw, not that the effect is real. `None`
        when the observed effect is exactly zero (no finite n suffices) or when fewer
        than two pairs make an sd undefined.
        """
        if len(differences) < 2:
            return None
        effect = abs(statistics.fmean(differences))
        if effect == 0.0:
            return None
        spread = statistics.stdev(differences)
        return max(2, math.ceil((1.96 + 0.842) ** 2 * spread**2 / effect**2))

    @staticmethod
    def describe_paired(*, label: str, first: list[float], second: list[float]) -> None:
        """One paired comparison, printed with its test, its per-seed differences and --
        when it fails to reach significance -- the n that would have been needed."""
        differences = [a - b for a, b in zip(first, second, strict=True)]
        result = TossingRoomComparison.wilcoxon_signed_rank(first=first, second=second)
        mean_difference = statistics.fmean(differences) if differences else 0.0
        spread = statistics.stdev(differences) if len(differences) > 1 else 0.0
        print(f"\n{label}")
        print(
            f"  mean paired difference {mean_difference:+.1f} pp "
            f"(sd {spread:.1f}, n = {len(differences)} seeds)"
        )
        print(
            f"  exact Wilcoxon signed-rank: p = {result['p']:.4f} on {result['n']} non-tied pair(s)"
        )
        if result["p"] < 0.05:
            print("  -> significant at alpha = 0.05")
        else:
            needed = Tossing3DComparison.required_n_for_80_percent_power(differences=differences)
            print("  -> NOT ESTABLISHED at alpha = 0.05. This is not evidence of an effect.")
            if needed is None:
                print("     (observed effect is exactly zero; no sample size would resolve it)")
            else:
                print(f"     {needed} paired seeds would be needed for 80% power at this effect")
        print(f"  per-seed differences: {[round(d, 1) for d in differences]}")

    @staticmethod
    def describe_trough(*, arm: str, curves: dict[int, dict[int, float]]) -> None:
        """Test the dip, rather than only pointing at it.

        EES's mean curve on this domain falls below its pre-practice checkpoint before
        it climbs, and "gets worse before it gets better" is an *effect claim* -- the
        exact kind this project has twice had to retract for being asserted off a mean
        with no test behind it. So the worst post-practice checkpoint by mean is located
        and tested, paired, against the pre-practice one.

        The checkpoint is chosen by the same data it is then tested on, which inflates
        significance; the honest reading of a p just under 0.05 here is therefore "not
        established" anyway. It is reported so the claim has a number attached at all,
        not to license a stronger one.
        """
        seeds = sorted(seed for seed, curve in curves.items() if curve)
        if len(seeds) < 2:
            return
        checkpoints = sorted(curves[seeds[0]])
        if len(checkpoints) < 2:
            return
        after_start = checkpoints[1:]
        trough = min(after_start, key=lambda t: statistics.fmean(curves[s][t] for s in seeds))
        Tossing3DComparison.describe_paired(
            label=(
                f"[{arm}] worst post-practice checkpoint ({trough} transitions) vs "
                "pre-practice -- the dip, chosen post hoc"
            ),
            first=[curves[s][trough] for s in seeds],
            second=[curves[s][checkpoints[0]] for s in seeds],
        )

    @staticmethod
    def print_report(*, root: Path, arms: list[str]) -> None:
        per_arm = {arm: Tossing3DComparison.per_seed_curves(root=root, method=arm) for arm in arms}
        header = f"{'arm':<18}{'seeds':>6}{'first':>9}{'final':>9}{'sd':>7}{'min':>7}{'max':>7}"
        print(header)
        print("-" * len(header))
        for arm, curves in per_arm.items():
            ends = Tossing3DComparison.endpoints(curves=curves)
            if not ends:
                print(f"{arm:<18}{'(no stats.json found)':>38}")
                continue
            firsts = [first for first, _ in ends.values()]
            finals = [final for _, final in ends.values()]
            print(
                f"{arm:<18}{len(ends):>6}{statistics.fmean(firsts):>9.1f}"
                f"{statistics.fmean(finals):>9.1f}"
                f"{(statistics.stdev(finals) if len(finals) > 1 else 0.0):>7.1f}"
                f"{min(finals):>7.1f}{max(finals):>7.1f}"
            )
        print("\n(first = pre-practice checkpoint at 0 transitions; final = end of training)")

        for arm, curves in per_arm.items():
            ends = Tossing3DComparison.endpoints(curves=curves)
            if len(ends) > 1:
                seeds = sorted(ends)
                print(f"\nper-seed final % solved, {arm}: " + str([ends[s][1] for s in seeds]))
                Tossing3DComparison.describe_paired(
                    label=f"[{arm}] end of training vs its own pre-practice checkpoint",
                    first=[ends[s][1] for s in seeds],
                    second=[ends[s][0] for s in seeds],
                )
                Tossing3DComparison.describe_trough(arm=arm, curves=curves)

        if len(arms) == 2:
            first_arm, second_arm = arms
            shared = sorted(
                set(Tossing3DComparison.endpoints(curves=per_arm[first_arm]))
                & set(Tossing3DComparison.endpoints(curves=per_arm[second_arm]))
            )
            if len(shared) > 1:
                ends_a = Tossing3DComparison.endpoints(curves=per_arm[first_arm])
                ends_b = Tossing3DComparison.endpoints(curves=per_arm[second_arm])
                Tossing3DComparison.describe_paired(
                    label=(
                        f"[{first_arm} vs {second_arm}] end of training, {len(shared)} shared seeds"
                    ),
                    first=[ends_a[s][1] for s in shared],
                    second=[ends_b[s][1] for s in shared],
                )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["ees", "random-skills"],
        help="Method directories under --results-root, in comparison order.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    Tossing3DComparison.print_report(root=args.results_root, arms=args.arms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
