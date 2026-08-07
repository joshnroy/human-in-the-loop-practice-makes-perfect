"""Exact paired tests over shared seeds, and the power question a null result owes.

**Why this is its own module.** `PairedTests` was defined twice, in two per-experiment
Tossing Room reset-policy reports (both since deleted by #141), and imported out of the
later one by four other analysis modules. Two copies of a statistics helper is a defect on
its own: the copies had already diverged, and only one of them scaled past ten seeds. It
also tied every consumer to a *domain* module, so a reader of `reset_free_ledge_curves.py`
had to know that a file named for one experiment's reset interval was where the Wilcoxon
test lived.

**Which copy this is.** The later one, which is a strict superset of the other: it adds
`minimum_detectable_effect`, and it enumerates the sign-flip null meet-in-the-middle
(`_subset_sums`) instead of iterating `itertools.product` directly. The enumeration is
exhaustive either way, so the two agree exactly; the difference is that the direct form
costs `2**n * n` Python operations and takes minutes at n = 20, where twenty paired seeds
is a size this project actually runs.

Nothing here is domain-specific: every method takes a list of per-seed differences.
"""

import itertools
import math
import statistics

import numpy as np
from pydantic import BaseModel, ConfigDict

# z_{0.975} and z_{0.80}, for the "how many seeds would 80% power need?" line that
# every non-significant result on this project is required to carry, and for the
# minimum detectable effect this design actually had.
_Z_ALPHA = 1.959964
_Z_POWER = 0.841621


class PairedTests(BaseModel):
    """Exact paired tests over the shared seeds, plus the power question every
    non-significant result on this project has to answer.

    Exact by enumeration rather than approximate, which sidesteps the normal
    approximation (badly behaved at these n), the continuity correction and the tie
    correction all at once -- and needs no special functions, hence no scipy, which
    is not a dependency of this project.

    At n = 20 the sign-flip null has 2**20 = 1,048,576 members, so the enumeration
    is done meet-in-the-middle (`_subset_sums`) rather than by iterating
    `itertools.product`: the terms are split in half, each half's 2**10 subset sums
    are enumerated, and the two are added by numpy broadcasting. Identical answer,
    seconds instead of minutes.
    """

    model_config = ConfigDict(frozen=True)

    statistic: float
    p_value: float
    num_zero_differences: int

    @staticmethod
    def wilcoxon_signed_rank(*, differences: list[float]) -> "PairedTests":
        """Two-sided exact Wilcoxon signed-rank test of differences against zero.

        Zero differences are dropped (Wilcoxon's original handling) and counted, so
        a result driven by "most seeds did not move at all" is visible rather than
        buried. Tied |differences| get average ranks, and the null is enumerated
        over sign assignments to *those* ranks, which stays exact under ties.
        """
        nonzero = [d for d in differences if d != 0.0]
        num_zero = len(differences) - len(nonzero)
        if not nonzero:
            return PairedTests(statistic=0.0, p_value=1.0, num_zero_differences=num_zero)
        ranks = PairedTests._average_ranks(values=[abs(d) for d in nonzero])
        observed = sum(rank for rank, d in zip(ranks, nonzero, strict=True) if d > 0)
        total = sum(ranks)
        # The statistic's null distribution is symmetric about total/2, so the
        # two-sided p is the mass at least as far from the centre as observed.
        distance = abs(observed - total / 2)
        sums = PairedTests._subset_sums(weights=ranks)
        extreme = int(np.count_nonzero(np.abs(sums - total / 2) >= distance - 1e-9))
        return PairedTests(
            statistic=observed,
            p_value=extreme / 2 ** len(ranks),
            num_zero_differences=num_zero,
        )

    @staticmethod
    def sign_flip(*, differences: list[float]) -> "PairedTests":
        """Two-sided exact sign-flip permutation test on the *mean* difference --
        the paired t-test's assumption-free twin, and the reason no t distribution
        (and so no incomplete beta function) is needed here.

        Flipping the sign of a subset S is the same as subtracting twice that
        subset's sum from the total, so this reuses `_subset_sums` rather than
        enumerating +-1 vectors.
        """
        num_zero = sum(1 for d in differences if d == 0.0)
        if all(d == 0.0 for d in differences):
            return PairedTests(statistic=0.0, p_value=1.0, num_zero_differences=num_zero)
        observed = abs(sum(differences))
        total = sum(differences)
        sums = PairedTests._subset_sums(weights=differences)
        extreme = int(np.count_nonzero(np.abs(total - 2.0 * sums) >= observed - 1e-9))
        return PairedTests(
            statistic=statistics.mean(differences),
            p_value=extreme / 2 ** len(differences),
            num_zero_differences=num_zero,
        )

    @staticmethod
    def _subset_sums(*, weights: list[float]) -> np.ndarray:
        """Every one of the 2**len(weights) subset sums, meet-in-the-middle.

        Enumerating directly costs 2**n * n Python operations, which at n = 20 is
        ~20 million and takes minutes per test. Splitting the weights in half and
        broadcasting the two halves' subset sums against each other costs
        2**(n/2) work plus one 2**n-element numpy add -- the same exhaustive
        enumeration, done in seconds. Exactness is preserved: every subset appears
        exactly once, as (a subset of the left half) + (a subset of the right half).
        """
        half = len(weights) // 2
        left = np.array([
            sum(combo) for combo in itertools.product(*[(0.0, w) for w in weights[:half]])
        ])
        right = np.array([
            sum(combo) for combo in itertools.product(*[(0.0, w) for w in weights[half:]])
        ])
        return (left[:, None] + right[None, :]).ravel()

    @staticmethod
    def seeds_for_80_percent_power(*, differences: list[float]) -> float:
        """Paired-sample size that would give 80% power at alpha = 0.05 two-sided
        for an effect the size of the one observed. Reported whenever p > 0.05, so
        "not established" comes with the cost of establishing it.

        Returns infinity when the observed mean difference is exactly zero: no
        sample size detects a zero effect.
        """
        mean = statistics.mean(differences)
        if mean == 0.0 or len(differences) < 2:
            return math.inf
        sd = statistics.stdev(differences)
        return (_Z_ALPHA + _Z_POWER) ** 2 * (sd / mean) ** 2

    @staticmethod
    def minimum_detectable_effect(*, differences: list[float]) -> float:
        """The smallest true effect this design had an 80% chance of detecting, at
        the spread actually observed: (z_alpha + z_power) * sd / sqrt(n).

        The companion to seeds_for_80_percent_power, and the more useful number
        when a result is null -- it says what the experiment could have found, so a
        reader can tell "no effect" from "no power".
        """
        if len(differences) < 2:
            return math.inf
        return (_Z_ALPHA + _Z_POWER) * statistics.stdev(differences) / len(differences) ** 0.5

    @staticmethod
    def bootstrap_ci(*, values: list[float], num_resamples: int = 20000) -> tuple[float, float]:
        """Percentile bootstrap 95% CI of the mean. Seeded, so the figure is
        reproducible; the CI is drawn rather than an sd bar because the per-seed
        scatter beside it already shows the spread."""
        rng = np.random.default_rng(0)
        sample = np.asarray(values, dtype=float)
        draws = rng.choice(sample, size=(num_resamples, sample.size), replace=True).mean(axis=1)
        return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))

    @staticmethod
    def _average_ranks(*, values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda index: values[index])
        ranks = [0.0] * len(values)
        position = 0
        while position < len(order):
            end = position
            while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
                end += 1
            average = (position + end) / 2 + 1
            for index in order[position : end + 1]:
                ranks[index] = average
            position = end + 1
        return ranks
