"""Post-run comparison of the published Ball-Ring `iters10k` arm (99/100) against PR
#126's re-run at nominally identical settings and the same fixed seeds 0-9 (91/100).

**The question this is for.** Two arms sharing a seed set are paired data, and the
obvious paired test -- on the ten endpoints -- cannot answer this one. 5/10 seeds are
tied, so the exact two-sided permutation test over all 2^5 sign flips of the non-tied
pairs has a **floor** of 2 x 2^-5 = 0.0625. No two-sided test on those ten numbers can
reach 0.05, whatever the effect. Reporting "p = 0.0625, not significant" therefore says
nothing about the world; it describes the design.

Two things this script does instead, both of which use data the endpoint summary throws
away:

**Divergence onset.** `evaluations` is a whole curve, not an endpoint. If two runs of the
same code at the same seed diverge at all, they diverge because something upstream of the
RNG changed -- so *when* they first differ is diagnostic in a way the final score is not.
A gap that opens only in the last few checkpoints is consistent with two converged runs
jittering; a gap at the first checkpoint after any practice at all means the two arms
were never the same computation. `divergence_onset` reports the first differing
checkpoint per seed.

**Curve area as the paired statistic.** Summing solved tasks over all 26 checkpoints
gives one number per seed that uses the entire run rather than its last 10 evaluations.
That breaks the ties the endpoint comparison suffers from, so all 10 pairs count and the
permutation floor drops from 2 x 2^-5 to 2 x 2^-10. Checkpoints within a seed are heavily
autocorrelated, so this is emphatically **not** 260 independent observations -- the test
is over the 10 seeds, exactly as the endpoint test is. What changes is only that each
seed's summary is less noisy.

**What this cannot do.** It cannot identify *which* change caused the shift. The published
arm predates `config_snapshot.json`, and its raw run directories did not survive a move
between machines, so the tree it ran at is inferred from `git log` rather than recorded.
Establishing the cause needs a re-run at candidate commits; see
`2026-08-06-ballring-placeballontable.md`.

Never runs a simulation or drives a `Problem`/`Method` (CLAUDE.md's `analysis/`
convention). The published arm is read from the committed aggregate
`2026-08-03-ballring-arms.json` -- **never restated or recomputed by hand** -- and the
re-run from `<rerun-root>/ees/<seed>/stats.json`.

Counts, never bare percentages.
"""

import argparse
import itertools
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

Arm = dict[int, list[list[int]]]


class BallRingPublishedVsRerun:
    """A static-method container, never instantiated."""

    @staticmethod
    def load_pair(
        *, published_json: Path, rerun_root: Path, arm: str = "iters10k"
    ) -> tuple[Arm, Arm]:
        """(published, rerun), each seed -> [[transitions, num_solved, num_total], ...]."""
        arms = json.loads(published_json.read_text())
        if arm not in arms:
            raise ValueError(f"{published_json} has no arm {arm!r}; got {sorted(arms)}")
        published: Arm = {int(seed): curve for seed, curve in arms[arm].items()}
        rerun: Arm = {}
        for path in sorted(rerun_root.glob("ees/*/stats.json")):
            rerun[int(path.parent.name)] = json.loads(path.read_text())["evaluations"]
        if not rerun:
            raise ValueError(f"no ees/<seed>/stats.json found under {rerun_root}")
        if sorted(published) != sorted(rerun):
            raise ValueError(
                f"arms do not share a seed set: published {sorted(published)}, "
                f"rerun {sorted(rerun)}"
            )
        return published, rerun

    @staticmethod
    def endpoint_total(*, arm: Arm) -> tuple[int, int]:
        """(solved, total) at the final checkpoint, summed over seeds."""
        return (
            sum(arm[seed][-1][1] for seed in arm),
            sum(arm[seed][-1][2] for seed in arm),
        )

    @staticmethod
    def endpoint_diffs(*, published: Arm, rerun: Arm) -> list[int]:
        """rerun - published at the final checkpoint, one per seed."""
        return [rerun[seed][-1][1] - published[seed][-1][1] for seed in sorted(published)]

    @staticmethod
    def curve_area_diffs(*, published: Arm, rerun: Arm) -> list[int]:
        """rerun - published summed over every checkpoint, one per seed."""
        return [
            sum(row[1] for row in rerun[seed]) - sum(row[1] for row in published[seed])
            for seed in sorted(published)
        ]

    @staticmethod
    def divergence_onset(*, published: Arm, rerun: Arm) -> dict[int, int | None]:
        """seed -> index of the first checkpoint whose solved counts differ, or `None`."""
        onsets: dict[int, int | None] = {}
        for seed in sorted(published):
            onsets[seed] = next(
                (
                    i
                    for i in range(len(published[seed]))
                    if published[seed][i][1] != rerun[seed][i][1]
                ),
                None,
            )
        return onsets

    @staticmethod
    def paired_permutation_p(*, diffs: list[int]) -> float:
        """Exact two-sided paired permutation test over all 2^n sign flips.

        Every pair is flipped, tied ones included: a tie contributes 0 under either sign,
        so it cannot change the statistic, but it does keep the enumeration over the full
        2^n and therefore the p-value on the scale the reader expects.
        """
        observed = abs(sum(diffs))
        hits = sum(
            1
            for signs in itertools.product((1, -1), repeat=len(diffs))
            if abs(sum(s * d for s, d in zip(signs, diffs, strict=True))) >= observed - 1e-12
        )
        return hits / 2 ** len(diffs)

    @staticmethod
    def print_report(*, published: Arm, rerun: Arm) -> None:
        p_solved, p_total = BallRingPublishedVsRerun.endpoint_total(arm=published)
        r_solved, r_total = BallRingPublishedVsRerun.endpoint_total(arm=rerun)
        print(f"published endpoint : {p_solved}/{p_total}")
        print(f"re-run   endpoint  : {r_solved}/{r_total}")

        end = BallRingPublishedVsRerun.endpoint_diffs(published=published, rerun=rerun)
        area = BallRingPublishedVsRerun.curve_area_diffs(published=published, rerun=rerun)
        onsets = BallRingPublishedVsRerun.divergence_onset(published=published, rerun=rerun)

        print("\nper seed:")
        print(
            f"  {'seed':>4} {'published':>12} {'rerun':>10} {'end diff':>9} "
            f"{'area diff':>10} {'diverges at':>12}"
        )
        for i, seed in enumerate(sorted(published)):
            pe = published[seed][-1]
            re_ = rerun[seed][-1]
            onset = onsets[seed]
            trans = published[seed][onset][0] if onset is not None else None
            print(
                f"  {seed:>4} {pe[1]:>8}/{pe[2]} {re_[1]:>6}/{re_[2]} {end[i]:>+9} "
                f"{area[i]:>+10} {str(trans):>12}"
            )

        n_end_tied = sum(1 for d in end if d == 0)
        print(
            f"\nendpoint   : {len(end) - n_end_tied}/{len(end)} pairs non-tied, "
            f"p = {BallRingPublishedVsRerun.paired_permutation_p(diffs=end):.4f} "
            f"(floor {2 / 2 ** (len(end) - n_end_tied):.4f})"
        )
        n_area_tied = sum(1 for d in area if d == 0)
        print(
            f"curve area : {len(area) - n_area_tied}/{len(area)} pairs non-tied, "
            f"p = {BallRingPublishedVsRerun.paired_permutation_p(diffs=area):.5f} "
            f"(floor {2 / 2 ** (len(area) - n_area_tied):.5f})"
        )
        print(
            f"  re-run lower in {sum(1 for d in area if d < 0)}/{len(area)} seeds, "
            f"mean {statistics.mean(area):+.1f} solved-task-checkpoints per seed"
        )

    @staticmethod
    def plot(*, published: Arm, rerun: Arm, output_path: Path) -> None:
        """Left: both arms' curves with per-seed spread. Right: per-seed curve-area diff."""
        fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.8))
        seeds = sorted(published)
        xs = [row[0] for row in published[seeds[0]]]

        for arm, colour, label in (
            (published, "tab:blue", "published iters10k"),
            (rerun, "tab:orange", "PR #126 re-run"),
        ):
            for seed in seeds:
                left.plot(
                    xs, [row[1] for row in arm[seed]], color=colour, alpha=0.18, linewidth=0.9
                )
            mean = [statistics.mean(arm[seed][i][1] for seed in seeds) for i in range(len(xs))]
            left.plot(xs, mean, color=colour, linewidth=2.4, label=label)

        onsets = BallRingPublishedVsRerun.divergence_onset(published=published, rerun=rerun)
        first = min(o for o in onsets.values() if o is not None)
        left.axvline(
            xs[first],
            color="crimson",
            linestyle="--",
            linewidth=1.2,
            label=f"first divergence ({xs[first]} transitions)",
        )
        left.set_xlabel("online transitions")
        left.set_ylabel("test tasks solved (out of 10)")
        left.set_ylim(0, 10.4)
        left.set_title(
            "The arms part company at the first practice checkpoint\n"
            "thin lines = individual seeds, thick = mean of 10"
        )
        left.legend(loc="lower right", fontsize=8)

        area = BallRingPublishedVsRerun.curve_area_diffs(published=published, rerun=rerun)
        colours = ["tab:orange" if d < 0 else "tab:blue" for d in area]
        right.bar([str(s) for s in seeds], area, color=colours)
        right.axhline(0, color="black", linewidth=0.9)
        p = BallRingPublishedVsRerun.paired_permutation_p(diffs=area)
        lower = sum(1 for d in area if d < 0)
        right.set_xlabel("seed")
        right.set_ylabel("re-run minus published, solved summed over 26 checkpoints")
        right.set_title(
            f"Whole-curve paired difference\n"
            f"re-run lower in {lower}/{len(area)} seeds, exact paired permutation p = {p:.4f}"
        )

        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--published-json", type=Path, required=True)
    parser.add_argument("--rerun-root", type=Path, required=True)
    parser.add_argument("--arm", type=str, default="iters10k")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    published, rerun = BallRingPublishedVsRerun.load_pair(
        published_json=args.published_json, rerun_root=args.rerun_root, arm=args.arm
    )
    BallRingPublishedVsRerun.print_report(published=published, rerun=rerun)
    if args.output is not None:
        BallRingPublishedVsRerun.plot(published=published, rerun=rerun, output_path=args.output)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
