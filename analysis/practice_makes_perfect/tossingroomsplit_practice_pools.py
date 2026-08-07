"""Post-run analysis for the Tossing Room practice-pool audit: the two *derived* tables in
`docs/experiment-logs/2026-08-06-tossingroomsplit-practice-pools.md`.

`practice_diagnostics.py` already prints the four-way `SamplerConsultation` split per
lifted skill, and that table is the primary result -- this module deliberately does not
duplicate it. What it adds is the two quantities that table cannot express, both of which
the audit's conclusions turn on:

1. **Informed draws against that same skill's own epsilon-random control.** The control has
   to come from inside the same runs: it is the uniform-draw baseline under the identical
   task distribution, so it absorbs "these tasks were easy" in a way an analytic prior
   (`0.2`, from `TossingRoomSplitSkills.sample_params`' docstring) does not. Two-sided
   Fisher exact via `TossingRoomComparison.fisher_exact_two_sided`, which stays valid at
   the small cell counts the recycling arm always has.

2. **The per-window trajectory of the informed and uninformative pools**, grouped into
   equal buckets. This is what separates *starvation* from *inability*, and it is why the
   audit rejects the decision rule carried over from #127: that rule's `inability` cell
   asks only that the informed *share* be large, with no power requirement and no check
   that the informed-success curve has plateaued. On this domain
   `ThrowRecycling`'s informed successes are still rising in the final bucket, which is
   the starvation signature, while the rule's cell fires for inability.

**Why the uninformative pool decaying to zero is the clean reading.** A permissive success
predicate -- Tossing3D's defect in #127 -- holds `UNINFORMATIVE` flat and high forever,
because the classifier never gets two classes to separate. A merely *young* classifier
shows it decaying as labels accumulate. Same number at any single checkpoint, opposite
diagnoses, and only the trajectory tells them apart.

Reads only already-produced `--output-dir` output (CLAUDE.md's `analysis/` convention):
it never drives a `Problem`/`Method`, and every count comes from `stats.json`'s
`practice_outcomes_per_cycle` by way of `practice_diagnostics.py`.

Counts, never bare percentages: every printed cell is `x/y`.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

from hitl_pmp.core.metrics.metrics import Metrics

from .practice_diagnostics import PracticeDiagnostics
from .tossingroom_comparison import TossingRoomComparison


class TossingRoomSplitPracticePools:
    """A static-method container, never instantiated, same as every other business-logic
    class in this project."""

    #: The two lifted skills that declare `param_dim = 1`. Named rather than discovered so
    #: that a domain change adding a third parameterized skill fails loudly here (the
    #: lookup raises) instead of silently dropping it from the comparison.
    THROWS: tuple[str, ...] = ("ThrowTrash", "ThrowRecycling")

    @staticmethod
    def informed_vs_random(*, runs: Sequence[Metrics], skill_name: str) -> dict:
        """One throw's informed draws against its own epsilon-random control.

        Returns the two `x/y` pairs, the difference in percentage points, and the
        two-sided Fisher exact p-value -- or `p = None` when either arm is empty, which is
        a real state (a skill practiced only before its classifier was ever fitted) and is
        reported as "no inference supported" rather than as a p of 1."""
        pooled = PracticeDiagnostics.totals(runs=runs)[skill_name]
        informed_successes = pooled.num_informed_successes
        informed_attempts = pooled.num_informed_attempts
        random_successes = pooled.num_random_successes
        random_attempts = pooled.num_random_attempts
        if informed_attempts == 0 or random_attempts == 0:
            return {
                "skill": skill_name,
                "informed": (informed_successes, informed_attempts),
                "random": (random_successes, random_attempts),
                "delta_pp": None,
                "p": None,
            }
        delta = (
            informed_successes / informed_attempts - random_successes / random_attempts
        ) * 100.0
        return {
            "skill": skill_name,
            "informed": (informed_successes, informed_attempts),
            "random": (random_successes, random_attempts),
            "delta_pp": delta,
            "p": TossingRoomComparison.fisher_exact_two_sided(
                a=informed_successes,
                b=informed_attempts - informed_successes,
                c=random_successes,
                d=random_attempts - random_successes,
            ),
        }

    @staticmethod
    def pool_trajectory(
        *, runs: Sequence[Metrics], skill_name: str, field: str, num_buckets: int
    ) -> list[int]:
        """One pool's per-window counts, summed over seeds and grouped into `num_buckets`
        equal buckets of consecutive windows.

        **Summed, never averaged**, for the reason `practice_diagnostics.py`'s own module
        docstring gives: a mean over ten seeds rounds a single seed's one informed draw to
        0.1, and the question here is whether the count is rising off zero at all.

        The trailing evaluation-only bucket is excluded via `practice_window_count`: it
        holds no practice by construction, so including it would compare the last real
        window against a structural zero and make any rising curve look like it fell."""
        series = PracticeDiagnostics.per_seed_series(runs=runs, skill_name=skill_name, field=field)
        if not runs:
            return []
        num_windows = PracticeDiagnostics.practice_window_count(metrics=runs[0])
        per_window = [
            sum(entry[index] for entry in series if index < len(entry))
            for index in range(num_windows)
        ]
        if num_windows == 0 or num_buckets <= 0:
            return []
        size = max(1, num_windows // num_buckets)
        return [sum(per_window[index : index + size]) for index in range(0, num_windows, size)]

    @staticmethod
    def print_report(*, results_root: Path, num_buckets: int) -> None:
        """Both derived tables, for every method directory under `results_root`."""
        summary = PracticeDiagnostics.summarize(results_root=results_root)
        for method, runs in summary.items():
            print(f"\n=== {method} ({len(runs)} seeds) ===")
            for skill_name in TossingRoomSplitPracticePools.THROWS:
                result = TossingRoomSplitPracticePools.informed_vs_random(
                    runs=runs, skill_name=skill_name
                )
                informed = f"{result['informed'][0]}/{result['informed'][1]}"
                random_arm = f"{result['random'][0]}/{result['random'][1]}"
                if result["p"] is None:
                    print(
                        f"{skill_name:<16} informed {informed:<10} "
                        f"eps-random {random_arm:<10} -- no inference supported"
                    )
                else:
                    print(
                        f"{skill_name:<16} informed {informed:<10} "
                        f"eps-random {random_arm:<10} "
                        f"{result['delta_pp']:+.2f}pp, Fisher exact p = {result['p']:.4f}"
                    )
            print(f"\nper-window pools, summed over seeds, in {num_buckets} buckets:")
            for skill_name in TossingRoomSplitPracticePools.THROWS:
                for field in (
                    "num_informed_attempts",
                    "num_informed_successes",
                    "num_uninformative_attempts",
                ):
                    trajectory = TossingRoomSplitPracticePools.pool_trajectory(
                        runs=runs,
                        skill_name=skill_name,
                        field=field,
                        num_buckets=num_buckets,
                    )
                    print(f"  {skill_name:<16}{field:<28}{trajectory}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Sweep directory laid out as <results-root>/<method>/<seed>/stats.json.",
    )
    parser.add_argument(
        "--num-buckets",
        type=int,
        default=5,
        help="How many equal buckets of consecutive practice windows to group into.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    TossingRoomSplitPracticePools.print_report(
        results_root=args.results_root, num_buckets=args.num_buckets
    )


if __name__ == "__main__":
    main()
