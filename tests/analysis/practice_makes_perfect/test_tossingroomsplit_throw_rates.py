"""Tests for the per-skill throw-rate analysis.

Three things here can be wrong in ways that are invisible on inspection, so all three
are pinned against values worked out by hand rather than by running the code:

1. **The consistency gate.** The report puts the sweep's `stats.json` success curves
   next to the trace collector's per-skill counts and calls them one experiment. If the
   gate that checks the two agree does not actually fire on a disagreement, that claim
   is unverified.
2. **The attempt ratio**, which is the headline number.
3. **The binomial noise floor and the minimum detectable effect**, which decide whether
   any gap reported is worth believing.
"""

import json
from pathlib import Path

import pytest

from analysis.practice_makes_perfect.tossingroomsplit_throw_rates import (
    TossingRoomSplitThrowRates,
)

_TRASH = "TrashInBin(trash, trash_bin)"
_RECYCLING = "RecyclingInBin(recycling, recycling_bin)"


def _trace_seed(*, seed: int, trash: list[int], recycling: list[int], solved: list[int]) -> dict:
    """One seed's trace with `len(trash)` cycles. `trash[i]`/`recycling[i]` are that
    period's attempts for each throw; successes are pinned at half, rounded down."""
    return {
        "seed": seed,
        "horizon": 7,
        "sweeps": [
            {
                "transitions": 100 * index,
                "solved": count,
                "total": 4,
                "families": {_TRASH: [count, 2], _RECYCLING: [0, 2]},
            }
            for index, count in enumerate(solved)
        ],
        "periods": [
            {
                "skills": {
                    "ThrowTrash": {
                        "attempts": t,
                        "successes": t // 2,
                        "random_attempts": 0,
                        "random_successes": 0,
                    },
                    "ThrowRecycling": {
                        "attempts": r,
                        "successes": r // 2,
                        "random_attempts": 0,
                        "random_successes": 0,
                    },
                }
            }
            for t, r in zip(trash, recycling, strict=True)
        ],
        "competence": [
            {
                "ThrowTrash": {"competence": 0.8, "num_groundings": 1, "num_observations": 1},
                "ThrowRecycling": {"competence": 0.2, "num_groundings": 1, "num_observations": 1},
            }
            for _ in trash
        ],
    }


def _write_traces(*, path: Path, runs: list[dict]) -> Path:
    path.write_text(json.dumps({"label": "ees", "seeds": runs}))
    return path


def _write_stats(*, root: Path, seed: int, evaluations: list[list[int]]) -> None:
    directory = root / "ees" / str(seed)
    directory.mkdir(parents=True)
    (directory / "stats.json").write_text(
        json.dumps({"evaluations": evaluations, "task_name": "default"})
    )


class TestTheConsistencyGate:
    """The traced run and the swept run must be the same run. The gate is what makes
    that a checked fact rather than an argument about determinism."""

    @staticmethod
    def test_agreeing_runs_pass(*, tmp_path: Path) -> None:
        traces = _write_traces(
            path=tmp_path / "t.json",
            runs=[_trace_seed(seed=0, trash=[6, 6], recycling=[1, 1], solved=[1, 3])],
        )
        _write_stats(root=tmp_path / "results", seed=0, evaluations=[[0, 1, 4], [100, 3, 4]])
        disagreements = TossingRoomSplitThrowRates.check_against_sweep(
            traces=[json.loads(traces.read_text())], results_root=tmp_path / "results"
        )
        assert disagreements == []

    @staticmethod
    def test_a_single_differing_count_is_reported(*, tmp_path: Path) -> None:
        """Non-vacuity for the test above: the gate has to be able to fail. One solved
        task differs and nothing else does."""
        traces = _write_traces(
            path=tmp_path / "t.json",
            runs=[_trace_seed(seed=0, trash=[6, 6], recycling=[1, 1], solved=[1, 3])],
        )
        _write_stats(root=tmp_path / "results", seed=0, evaluations=[[0, 1, 4], [100, 2, 4]])
        disagreements = TossingRoomSplitThrowRates.check_against_sweep(
            traces=[json.loads(traces.read_text())], results_root=tmp_path / "results"
        )
        assert len(disagreements) == 1
        assert "seed 0" in disagreements[0]

    @staticmethod
    def test_a_seed_missing_from_the_sweep_is_reported_rather_than_skipped(
        *, tmp_path: Path
    ) -> None:
        """Silently skipping is the dangerous failure: a gate that checks zero seeds
        passes."""
        traces = _write_traces(
            path=tmp_path / "t.json",
            runs=[_trace_seed(seed=7, trash=[6], recycling=[1], solved=[1])],
        )
        (tmp_path / "results").mkdir()
        disagreements = TossingRoomSplitThrowRates.check_against_sweep(
            traces=[json.loads(traces.read_text())], results_root=tmp_path / "results"
        )
        assert len(disagreements) == 1
        assert "seed 7" in disagreements[0]


class TestTheAttemptRatio:
    """The headline number. Reported as two totals and their quotient, never as a bare
    ratio -- 12:1 and 1200:100 are different evidence."""

    @staticmethod
    def test_totals_are_summed_over_every_period_of_every_seed(*, tmp_path: Path) -> None:
        runs = [
            _trace_seed(seed=0, trash=[6, 8], recycling=[1, 1], solved=[0, 0]),
            _trace_seed(seed=1, trash=[10, 4], recycling=[1, 0], solved=[0, 0]),
        ]
        totals = TossingRoomSplitThrowRates.attempt_totals(traces=[{"label": "ees", "seeds": runs}])
        assert totals["ThrowTrash"] == 28  # 6 + 8 + 10 + 4
        assert totals["ThrowRecycling"] == 3  # 1 + 1 + 1 + 0

    @staticmethod
    def test_the_ratio_is_trash_attempts_over_recycling_attempts(*, tmp_path: Path) -> None:
        runs = [_trace_seed(seed=0, trash=[12], recycling=[1], solved=[0])]
        assert TossingRoomSplitThrowRates.attempt_ratio(
            traces=[{"label": "ees", "seeds": runs}]
        ) == pytest.approx(12.0)

    @staticmethod
    def test_the_ratio_is_none_when_recycling_was_never_attempted() -> None:
        """A zero denominator is a real possible outcome of this domain (the ledge can
        strand the robot before any throw), and it must not come back as `inf` or a
        crash dressed up as a measurement."""
        runs = [_trace_seed(seed=0, trash=[12], recycling=[0], solved=[0])]
        assert (
            TossingRoomSplitThrowRates.attempt_ratio(traces=[{"label": "ees", "seeds": runs}])
            is None
        )

    @staticmethod
    def test_per_seed_ratios_are_reported_alongside_the_pooled_one() -> None:
        """A pooled ratio can be dominated by one seed. Per-seed values are what show
        whether the effect is structural or a single run's accident."""
        runs = [
            _trace_seed(seed=0, trash=[12], recycling=[1], solved=[0]),
            _trace_seed(seed=1, trash=[24], recycling=[2], solved=[0]),
        ]
        per_seed = TossingRoomSplitThrowRates.per_seed_attempts(
            traces=[{"label": "ees", "seeds": runs}]
        )
        assert per_seed[0] == {"ThrowTrash": 12, "ThrowRecycling": 1}
        assert per_seed[1] == {"ThrowTrash": 24, "ThrowRecycling": 2}


class TestTheNoiseFloor:
    """`sqrt(0.25/n_a + 0.25/n_b)`, and the effect size the design can actually
    detect. Both computed by hand below."""

    @staticmethod
    def test_the_floor_matches_the_closed_form() -> None:
        # n_a = n_b = 140 -> sqrt(0.25/140 + 0.25/140) = sqrt(1/280) = 0.0597614...
        assert TossingRoomSplitThrowRates.noise_floor(n_first=140, n_second=140) == pytest.approx(
            0.05976143, abs=1e-8
        )

    @staticmethod
    def test_an_imbalanced_split_has_a_worse_floor_than_a_balanced_one_of_the_same_size() -> None:
        """The case that matters here: the two throws do not get the same number of
        attempts, and at a FIXED total the smaller arm dominates the floor. 1400/140
        gives 4.43pp against 2.55pp for the same 1540 observations split evenly, so
        piling more trash attempts on cannot buy resolution the recycling side does not
        have.

        (An earlier version of this test asserted the imbalanced floor exceeds the
        *equal-140* floor, which is simply false -- 4.43pp < 5.98pp, because raising one
        arm from 140 to 1400 does shrink that arm's term. The floor is driven by the
        smaller n, not by the imbalance as such.)"""
        imbalanced = TossingRoomSplitThrowRates.noise_floor(n_first=1400, n_second=140)
        balanced = TossingRoomSplitThrowRates.noise_floor(n_first=770, n_second=770)
        assert imbalanced > balanced
        # And bounded below by the smaller arm alone, whatever the larger one does.
        assert imbalanced > TossingRoomSplitThrowRates.noise_floor(n_first=10**9, n_second=140) * (
            1 - 1e-6
        )

    @staticmethod
    def test_the_minimum_detectable_effect_is_2_8_standard_errors() -> None:
        """(z_{0.025} + z_{0.20}) = 1.959964 + 0.841621 = 2.801585, the standard 80%-power
        two-sided constant."""
        floor = TossingRoomSplitThrowRates.noise_floor(n_first=140, n_second=140)
        assert TossingRoomSplitThrowRates.minimum_detectable_effect(
            n_first=140, n_second=140
        ) == pytest.approx(2.801585 * floor, rel=1e-6)


class TestTheCurves:
    """What gets plotted. Cumulative, because the question is how much practice each
    skill has had by a given point on the shared transition axis."""

    @staticmethod
    def test_cumulative_attempts_accumulate_across_periods() -> None:
        runs = [_trace_seed(seed=0, trash=[6, 8, 4], recycling=[1, 1, 1], solved=[0, 0, 0])]
        series = TossingRoomSplitThrowRates.cumulative_attempts(
            traces=[{"label": "ees", "seeds": runs}], skill="ThrowTrash"
        )
        assert [mean for _transitions, mean, _stderr in series] == [6.0, 14.0, 18.0]

    @staticmethod
    def test_the_transition_axis_comes_from_the_sweeps_not_from_the_period_index() -> None:
        """Periods are counted; transitions are measured. A period that ends early
        (`InteractionComplete`) contributes fewer than `max_steps_per_interaction`, so
        plotting against the period index would silently mis-scale the x-axis."""
        runs = [_trace_seed(seed=0, trash=[6, 8], recycling=[1, 1], solved=[0, 0])]
        series = TossingRoomSplitThrowRates.cumulative_attempts(
            traces=[{"label": "ees", "seeds": runs}], skill="ThrowTrash"
        )
        # sweeps are at transitions 0 and 100; a period's attempts are credited to the
        # sweep that measured it, i.e. the one AFTER it.
        assert [transitions for transitions, _mean, _stderr in series] == [0, 100]

    @staticmethod
    def test_a_skill_absent_from_a_period_counts_as_zero_attempts_not_as_missing() -> None:
        """A period in which a throw was never reached is a real, informative zero. Left
        as missing it would be dropped from the mean and inflate the curve."""
        runs = [
            {
                "seed": 0,
                "horizon": 7,
                "sweeps": [
                    {"transitions": 0, "solved": 0, "total": 4, "families": {}},
                    {"transitions": 100, "solved": 0, "total": 4, "families": {}},
                ],
                "periods": [
                    {"skills": {"ThrowTrash": {"attempts": 6, "successes": 3}}},
                    {"skills": {}},
                ],
                "competence": [{}, {}],
            }
        ]
        series = TossingRoomSplitThrowRates.cumulative_attempts(
            traces=[{"label": "ees", "seeds": runs}], skill="ThrowRecycling"
        )
        assert [mean for _transitions, mean, _stderr in series] == [0.0, 0.0]


class TestTheStructuralClaim:
    """ "Recycling buys at most one attempt per practice period, ever" is the domain's
    defining asymmetry. It is a claim about a DISTRIBUTION, not a mean: a skill averaging
    0.8 attempts per period could be 0-or-1 every time, or 0 four times and 4 once, and
    only the first is the structural story. So the histogram is what gets reported."""

    @staticmethod
    def test_the_histogram_counts_periods_by_attempts() -> None:
        runs = [
            _trace_seed(seed=0, trash=[0, 3, 12], recycling=[0, 1, 1], solved=[0, 0, 0]),
            _trace_seed(seed=1, trash=[12, 12, 0], recycling=[1, 1, 0], solved=[0, 0, 0]),
        ]
        traces = [{"label": "ees", "seeds": runs}]
        assert TossingRoomSplitThrowRates.attempts_per_period_histogram(
            traces=traces, skill="ThrowRecycling"
        ) == {0: 2, 1: 4}
        assert TossingRoomSplitThrowRates.attempts_per_period_histogram(
            traces=traces, skill="ThrowTrash"
        ) == {0: 2, 3: 1, 12: 3}

    @staticmethod
    def test_a_period_with_no_record_for_the_skill_counts_as_a_zero_period() -> None:
        """A period the skill never came up in is a real zero. Dropped, the histogram
        would claim recycling is attempted in every period."""
        runs = [
            {
                "seed": 0,
                "horizon": 7,
                "sweeps": [{"transitions": 0, "solved": 0, "total": 4, "families": {}}],
                "periods": [{"skills": {"ThrowTrash": {"attempts": 5, "successes": 1}}}],
                "competence": [{}],
            }
        ]
        assert TossingRoomSplitThrowRates.attempts_per_period_histogram(
            traces=[{"label": "ees", "seeds": runs}], skill="ThrowRecycling"
        ) == {0: 1}


class TestGreedyVersusRandom:
    """At the paper's epsilon = 0.5 roughly half of every practice attempt's parameters
    come from a coin flip. Pooling those with the learned ones measures how often a coin
    flip works, so the two are always reported apart."""

    @staticmethod
    def test_greedy_counts_subtract_the_random_ones() -> None:
        runs = [
            {
                "seed": 0,
                "horizon": 7,
                "sweeps": [{"transitions": 0, "solved": 0, "total": 4, "families": {}}],
                "periods": [
                    {
                        "skills": {
                            "ThrowTrash": {
                                "attempts": 10,
                                "successes": 7,
                                "random_attempts": 4,
                                "random_successes": 1,
                            }
                        }
                    }
                ],
                "competence": [{}],
            }
        ]
        traces = [{"label": "ees", "seeds": runs}]
        assert TossingRoomSplitThrowRates.greedy_success_totals(traces=traces)["ThrowTrash"] == (
            6,
            6,
        )
        assert TossingRoomSplitThrowRates.random_success_totals(traces=traces)["ThrowTrash"] == (
            1,
            4,
        )


class TestLandingsVersusScoredSuccesses:
    """A throw scored a success by EES is not necessarily a throw that landed -- see
    `SkillTally`. The analysis has to report both, because on this domain the gap between
    them is large on one side and exactly zero on the other."""

    @staticmethod
    def test_landing_totals_are_reported_separately_from_scored_successes() -> None:
        runs = [
            {
                "seed": 0,
                "horizon": 7,
                "sweeps": [{"transitions": 0, "solved": 0, "total": 2, "families": {}}],
                "periods": [
                    {
                        "skills": {
                            "ThrowTrash": {
                                "attempts": 10,
                                "successes": 9,
                                "random_attempts": 0,
                                "random_successes": 0,
                                "landed": 3,
                                "landed_random": 0,
                                "prefilled": 6,
                            }
                        }
                    }
                ],
                "competence": [{}],
            }
        ]
        traces = [{"label": "ees", "seeds": runs}]
        assert TossingRoomSplitThrowRates.landing_totals(traces=traces)["ThrowTrash"] == (3, 10)
        assert TossingRoomSplitThrowRates.prefilled_totals(traces=traces)["ThrowTrash"] == (
            6,
            10,
        )

    @staticmethod
    def test_the_greedy_landing_curve_counts_landings_not_scored_successes() -> None:
        """The honest learning curve. It must track `landed`, not `successes`, or it
        reproduces exactly the artifact the audit exists to expose."""
        runs = [
            {
                "seed": 0,
                "horizon": 7,
                "sweeps": [
                    {"transitions": 0, "solved": 0, "total": 2, "families": {}},
                    {"transitions": 100, "solved": 0, "total": 2, "families": {}},
                ],
                "periods": [
                    {
                        "skills": {
                            "ThrowTrash": {
                                "attempts": 10,
                                "successes": 10,
                                "random_attempts": 4,
                                "random_successes": 4,
                                "landed": 5,
                                "landed_random": 2,
                                "prefilled": 5,
                            }
                        }
                    }
                ],
                "competence": [{}],
            }
        ]
        series = TossingRoomSplitThrowRates.greedy_landing_curve(
            traces=[{"label": "ees", "seeds": runs}], skill="ThrowTrash"
        )
        # greedy attempts 10 - 4 = 6, greedy landings 5 - 2 = 3 -> 0.5, not the 1.0 the
        # scored successes would give.
        assert [round(mean, 6) for _transitions, mean, _stderr in series] == [0.0, 0.5]

    @staticmethod
    def test_the_inflated_share_is_the_scored_successes_that_did_not_land() -> None:
        """The number the writeup quotes. 9 scored, 3 landed -> 6 of the 9 were credited
        to a throw that missed."""
        runs = [
            {
                "seed": 0,
                "horizon": 7,
                "sweeps": [{"transitions": 0, "solved": 0, "total": 2, "families": {}}],
                "periods": [
                    {
                        "skills": {
                            "ThrowTrash": {
                                "attempts": 10,
                                "successes": 9,
                                "random_attempts": 0,
                                "random_successes": 0,
                                "landed": 3,
                                "landed_random": 0,
                                "prefilled": 6,
                            }
                        }
                    }
                ],
                "competence": [{}],
            }
        ]
        assert TossingRoomSplitThrowRates.inflated_successes(
            traces=[{"label": "ees", "seeds": runs}]
        )["ThrowTrash"] == (6, 9)


class TestHowMuchSlower:
    """ "Learns slower" has to become a number. Two are reported: how many transitions
    each family needs to first reach a given success level, and the area under each
    family's own curve (paired per seed, so the Wilcoxon has something to test)."""

    @staticmethod
    def test_transitions_to_reach_returns_the_first_sweep_at_or_above_the_level() -> None:
        runs = [
            _trace_seed(seed=0, trash=[1, 1, 1, 1], recycling=[1, 1, 1, 1], solved=[0, 1, 1, 2])
        ]
        # The fixture puts every solved task in TRASH, out of 2 TRASH tasks per sweep.
        traces = [{"label": "ees", "seeds": runs}]
        assert (
            TossingRoomSplitThrowRates.transitions_to_reach(
                traces=traces, skill="ThrowTrash", level=0.5
            )
            == 100
        )
        assert (
            TossingRoomSplitThrowRates.transitions_to_reach(
                traces=traces, skill="ThrowTrash", level=1.0
            )
            == 300
        )

    @staticmethod
    def test_transitions_to_reach_is_none_when_the_level_is_never_reached() -> None:
        """Never reaching a level is a real outcome and must not be reported as the last
        sweep, which would read as "reached it right at the end"."""
        runs = [_trace_seed(seed=0, trash=[1], recycling=[1], solved=[0])]
        assert (
            TossingRoomSplitThrowRates.transitions_to_reach(
                traces=[{"label": "ees", "seeds": runs}], skill="ThrowRecycling", level=0.5
            )
            is None
        )

    @staticmethod
    def test_the_greedy_success_curve_is_cumulative_and_ignores_random_attempts() -> None:
        """The direct "is this sampler learning" signal, and the one the competence model
        cannot give: competence is a windowed estimate under a Beta(10, 1) prior, so a
        skill with few observations sits near 0.909 whatever it can actually do."""
        runs = [
            {
                "seed": 0,
                "horizon": 7,
                "sweeps": [
                    {"transitions": 0, "solved": 0, "total": 2, "families": {}},
                    {"transitions": 100, "solved": 0, "total": 2, "families": {}},
                    {"transitions": 200, "solved": 0, "total": 2, "families": {}},
                ],
                "periods": [
                    {
                        "skills": {
                            "ThrowTrash": {
                                "attempts": 4,
                                "successes": 1,
                                "random_attempts": 2,
                                "random_successes": 1,
                            }
                        }
                    },
                    {
                        "skills": {
                            "ThrowTrash": {
                                "attempts": 4,
                                "successes": 4,
                                "random_attempts": 2,
                                "random_successes": 2,
                            }
                        }
                    },
                ],
                "competence": [{}, {}],
            }
        ]
        series = TossingRoomSplitThrowRates.greedy_success_curve(
            traces=[{"label": "ees", "seeds": runs}], skill="ThrowTrash"
        )
        # sweep 0: nothing practiced yet -> 0.0. sweep 1: greedy 0/2. sweep 2: greedy 2/4.
        assert [round(mean, 6) for _transitions, mean, _stderr in series] == [0.0, 0.0, 0.5]

    @staticmethod
    def test_area_under_curve_is_the_mean_of_a_seeds_own_sweep_rates() -> None:
        """Mean rather than a trapezoid sum: sweeps are evenly spaced in transitions, so
        the mean is the trapezoid up to a constant, and it stays on the 0-1 scale a rate
        belongs on."""
        runs = [_trace_seed(seed=0, trash=[1, 1], recycling=[1, 1], solved=[0, 2])]
        # TRASH rates across the two sweeps: 0/2 and 2/2 -> mean 0.5.
        assert TossingRoomSplitThrowRates.area_under_curve(
            traces=[{"label": "ees", "seeds": runs}], skill="ThrowTrash"
        ) == [0.5]


def _family_seed(*, seed: int, trash_solved: list[int], recycling_solved: list[int]) -> dict:
    """One seed whose per-sweep family counts are given directly, out of 14 each -- the
    real composition. `periods` and `competence` are present but empty, because the
    statistics under test read only the evaluation record."""
    return {
        "seed": seed,
        "horizon": 12,
        "sweeps": [
            {
                "transitions": 100 * index,
                "solved": t + r,
                "total": 28,
                "families": {_TRASH: [t, 14], _RECYCLING: [r, 14]},
            }
            for index, (t, r) in enumerate(zip(trash_solved, recycling_solved, strict=True))
        ],
        "periods": [],
        "competence": [],
    }


class TestLearningIsASwitchNotACurve:
    """A mean-over-seeds curve that rises smoothly can be produced by seeds that never sit
    anywhere but 0/14 and 14/14 and flip between them at different times. The two are
    different findings and the pooled line cannot tell them apart, so the split has to be
    counted per (seed, checkpoint) rather than inferred from the shape of the mean."""

    @staticmethod
    def test_a_seed_checkpoint_is_extreme_at_or_beyond_either_threshold() -> None:
        runs = [
            # 0/14 and 14/14 are extreme low and high; 7/14 is the middle; 12/14 and 4/14
            # sit exactly ON the thresholds and are extreme, because the thresholds are
            # inclusive.
            _family_seed(
                seed=0,
                trash_solved=[0, 14, 7, 12, 4],
                recycling_solved=[0, 0, 0, 0, 0],
            )
        ]
        split = TossingRoomSplitThrowRates.seed_checkpoint_extremes(
            traces=[{"label": "ees", "seeds": runs}], skill="ThrowTrash", high=12, low=4
        )
        assert split == {"extreme": 4, "middle": 1, "total": 5}

    @staticmethod
    def test_every_seed_contributes_its_own_checkpoints() -> None:
        """Non-vacuity for the pooling: a version that read only the first seed would give
        the same answer on a one-seed fixture."""
        runs = [
            _family_seed(seed=0, trash_solved=[0, 0], recycling_solved=[0, 0]),
            _family_seed(seed=1, trash_solved=[7, 8], recycling_solved=[0, 0]),
        ]
        split = TossingRoomSplitThrowRates.seed_checkpoint_extremes(
            traces=[{"label": "ees", "seeds": runs}], skill="ThrowTrash", high=12, low=4
        )
        assert split == {"extreme": 2, "middle": 2, "total": 4}

    @staticmethod
    def test_a_seed_that_falls_back_out_is_reported_as_its_peak_and_its_endpoint() -> None:
        """The Tossing Room baseline found a seed that reached 10/14 mid-run and ended at
        3/14. A table of final scores alone cannot show that, so the peak is carried
        alongside the endpoint."""
        runs = [
            _family_seed(seed=0, trash_solved=[0, 0, 0], recycling_solved=[0, 10, 3]),
            _family_seed(seed=1, trash_solved=[0, 0, 0], recycling_solved=[0, 1, 2]),
        ]
        peaks = TossingRoomSplitThrowRates.per_seed_family_peaks(
            traces=[{"label": "ees", "seeds": runs}], skill="ThrowRecycling"
        )
        assert peaks == {
            0: {"peak": 10, "final": 3, "total": 14},
            1: {"peak": 2, "final": 2, "total": 14},
        }


class TestWhatTheSamplerActuallyAnswered:
    """`attempts` counts how often a sampler was asked. The mechanism claim -- a sampler
    that has convinced itself of a wrong force, with one datapoint per period to
    unconvince it -- is about WHAT it answered, and needs the forces themselves.

    That used to be countable from the choice alone: when a target force was drawn
    `U(0.5, 1.0)` at tolerance 0.1, any force below 0.4 missed *whatever* task it was
    aiming at. The throw-representation change made that vacuous -- the required force is
    now derived from the bin's `throw_distance` and the item's `weight` and spans
    `[0.1, 0.9]`, so every force in the `U(0, 1)` draw range is right for *some* task. The
    statistic is therefore per grounding now: how far the chosen force was from the force
    that grounding actually required, against a threshold of 3x the tolerance."""

    @staticmethod
    def test_draws_far_from_their_own_grounding_are_counted_against_all_greedy_draws() -> None:
        runs = [
            {
                "seed": 0,
                "horizon": 12,
                "sweeps": [{"transitions": 0, "solved": 0, "total": 2, "families": {}}],
                "periods": [
                    {
                        "skills": {
                            "ThrowRecycling": {
                                "attempts": 5,
                                "successes": 0,
                                "random_attempts": 1,
                                "random_successes": 0,
                                "landed": 0,
                                "landed_random": 0,
                                "prefilled": 0,
                                # Misses of 0.70, 0.68 and 0.31 exceed the 0.30
                                # threshold; 0.10 and 0.00 do not.
                                "greedy_forces": [0.0, 0.02, 0.39, 0.4, 0.83],
                                "greedy_targets": [0.7, 0.7, 0.7, 0.5, 0.83],
                            }
                        }
                    }
                ],
                "competence": [{}],
            }
        ]
        summary = TossingRoomSplitThrowRates.badly_missed_force_totals(
            traces=[{"label": "ees", "seeds": runs}], miss_threshold=0.30
        )
        assert summary["ThrowRecycling"] == (3, 5)

    @staticmethod
    def test_a_sampler_hitting_its_groundings_counts_zero() -> None:
        """Non-vacuity: the statistic must be able to come out at 0, or "recycling is
        pinned on a wrong answer" would be unfalsifiable."""
        runs = [
            {
                "seed": 0,
                "horizon": 12,
                "sweeps": [{"transitions": 0, "solved": 0, "total": 2, "families": {}}],
                "periods": [
                    {
                        "skills": {
                            "ThrowTrash": {
                                "attempts": 2,
                                "successes": 2,
                                "random_attempts": 0,
                                "random_successes": 0,
                                "landed": 2,
                                "landed_random": 0,
                                "prefilled": 0,
                                "greedy_forces": [0.55, 0.91],
                                "greedy_targets": [0.55, 0.91],
                            }
                        }
                    }
                ],
                "competence": [{}],
            }
        ]
        summary = TossingRoomSplitThrowRates.badly_missed_force_totals(
            traces=[{"label": "ees", "seeds": runs}], miss_threshold=0.30
        )
        assert summary["ThrowTrash"] == (0, 2)

    @staticmethod
    def test_the_longest_consecutive_run_of_all_missing_periods_is_reported_per_seed() -> None:
        """ "Stuck" is a claim about consecutive periods, not about a pooled rate: the same
        miss count spread evenly is a sampler still searching, and concentrated is a
        sampler that stopped moving. A period with no greedy attempt breaks nothing and
        contributes nothing -- it is not evidence either way."""
        runs = [
            {
                "seed": 4,
                "horizon": 12,
                "sweeps": [{"transitions": 0, "solved": 0, "total": 2, "families": {}}],
                "periods": [
                    {"skills": {"ThrowRecycling": _throw(forces=[0.02], targets=[0.7])}},
                    {"skills": {}},
                    {"skills": {"ThrowRecycling": _throw(forces=[0.01], targets=[0.8])}},
                    {"skills": {"ThrowRecycling": _throw(forces=[0.79], targets=[0.8])}},
                    {"skills": {"ThrowRecycling": _throw(forces=[0.03], targets=[0.6])}},
                ],
                "competence": [{}],
            }
        ]
        streaks = TossingRoomSplitThrowRates.longest_missing_streaks(
            traces=[{"label": "ees", "seeds": runs}], skill="ThrowRecycling"
        )
        # Periods 0 and 2 both miss and the empty period 1 does not interrupt them, so the
        # streak is 2; period 3 lands and resets it; period 4 starts a streak of 1.
        assert streaks == {4: 2}


class TestSeparatingLearnedDrawsFromTheUniformFallback:
    """A greedy draw is not automatically a learned one.

    `LearnedSkillSampler.sample` falls back to a uniform draw when its scores cannot
    rank the candidates, and reports `was_random=False` for it -- `was_random` means
    "the epsilon-greedy branch fired", nothing more. So the old greedy pool mixes a
    trained classifier's choices with draws that carry no belief, and the skill with
    fewer observations spends longer in the second state. These are the statistics that
    unpool them."""

    @staticmethod
    def _run(*, tallies: dict) -> list[dict]:
        return [
            {
                "seed": 0,
                "horizon": 12,
                "sweeps": [{"transitions": 0, "solved": 0, "total": 2, "families": {}}],
                "periods": [{"skills": tallies}],
                "competence": [{}],
            }
        ]

    @staticmethod
    def test_landings_are_counted_against_informed_draws_not_all_greedy_ones() -> None:
        tally = {
            "attempts": 10,
            "successes": 4,
            "random_attempts": 4,
            "random_successes": 1,
            "landed": 4,
            "landed_random": 1,
            "prefilled": 0,
            # Two of the six greedy draws came from a discriminating classifier, and
            # `informed_*` is a strict subset of `greedy_*` -- the last two entries here.
            # Misses of 0.60 and 0.50 exceed the 0.30 threshold; 0.00, 0.00, 0.00 and
            # 0.05 do not. Distances stay clear of the threshold on both sides, since
            # 0.9 - 0.6 is 0.30000000000000004 in binary floating point.
            "greedy_forces": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "greedy_targets": [0.1, 0.2, 0.9, 0.9, 0.5, 0.55],
            "informed_attempts": 2,
            "informed_successes": 1,
            "informed_landed": 1,
            "informed_forces": [0.5, 0.6],
            "informed_targets": [0.5, 0.55],
        }
        traces = [
            {
                "label": "ees",
                "seeds": TestSeparatingLearnedDrawsFromTheUniformFallback._run(
                    tallies={"ThrowRecycling": tally}
                ),
            }
        ]
        assert TossingRoomSplitThrowRates.informed_landing_totals(traces=traces) == {
            "ThrowRecycling": (1, 2)
        }
        # The old pool: six greedy draws, of which four were the uniform fallback.
        assert TossingRoomSplitThrowRates.uninformed_greedy_totals(traces=traces) == {
            "ThrowRecycling": (4, 6)
        }
        assert TossingRoomSplitThrowRates.per_seed_informed_draws(
            traces=traces, skill="ThrowRecycling"
        ) == {0: (2, 6)}
        # Only the informed draws are asked whether they missed -- neither did, while
        # the greedy pool they sit inside contains two that did. Reporting the pooled
        # 2/6 as "what the sampler answered" is exactly the conflation being removed.
        assert TossingRoomSplitThrowRates.informed_badly_missed_force_totals(
            traces=traces, miss_threshold=0.30
        ) == {"ThrowRecycling": (0, 2)}
        assert TossingRoomSplitThrowRates.badly_missed_force_totals(
            traces=traces, miss_threshold=0.30
        ) == {"ThrowRecycling": (2, 6)}

    @staticmethod
    def test_the_epsilon_random_control_counts_only_random_draws() -> None:
        """The control the informed rate is read against. It must come from
        `landed_random`/`random_attempts` and never pick up a greedy landing, or the
        comparison the log turns on would be the informed draws against themselves."""
        tally = {
            "attempts": 10,
            "successes": 5,
            "random_attempts": 4,
            "random_successes": 1,
            "landed": 5,
            "landed_random": 1,
            "prefilled": 0,
            "greedy_forces": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "greedy_targets": [0.1, 0.2, 0.9, 0.9, 0.5, 0.55],
            "informed_attempts": 2,
            "informed_successes": 1,
            "informed_landed": 1,
            "informed_forces": [0.5, 0.6],
            "informed_targets": [0.5, 0.55],
        }
        traces = [
            {
                "label": "ees",
                "seeds": TestSeparatingLearnedDrawsFromTheUniformFallback._run(
                    tallies={"ThrowRecycling": tally}
                ),
            }
        ]
        assert TossingRoomSplitThrowRates.random_landing_totals(traces=traces) == {
            "ThrowRecycling": (1, 4)
        }

    @staticmethod
    def test_each_informed_versus_random_row_carries_its_own_denominators_mde(
        *, capsys: pytest.CaptureFixture
    ) -> None:
        """**The defect this test exists to stop recurring.** PR #90 and the committed log
        quote a single **20.19pp** MDE for the recycling null result. That figure is the
        floor for a **310 vs 57** comparison -- trash's epsilon-random draws against
        recycling's, a row in a *different* table. The comparison actually called null is
        recycling-informed against recycling-random, **56 vs 57**, whose floor is
        `sqrt(0.25/56 + 0.25/57)` = 9.41pp and whose MDE is **26.36pp**. The quoted number
        understated the MDE by 6.17pp (the floor itself by 2.20pp), making a null result
        look better resolved than it was.

        26.36 rather than 26.34: this repo's `_MDE_CONSTANT` is the unrounded
        `z_{0.025} + z_{0.20}` = 2.801585, and 2.801585 x 9.4076pp = 26.36pp. Rounding the
        constant to 2.8 gives 26.34pp. Every other MDE on the page uses the unrounded
        constant, so this one does too.

        The mechanism was that this table printed a gap and a p-value but **no MDE**, so
        the only MDE on the page belonged to a neighbouring comparison and got borrowed.
        Every row now derives its own from its own two denominators.

        **Both skills are in the trace on purpose.** With one row, code that hardcoded a
        single constant on every row would pass -- which is the very defect class being
        guarded. Trash's own comparison is 301 vs 310, floor 4.05pp, MDE 11.34pp, so the
        two rows must differ from each other as well as from 20.19pp.
        """

        def _tally(*, informed: int, informed_landed: int, random: int, random_landed: int) -> dict:
            return {
                "attempts": informed + random,
                "successes": informed_landed + random_landed,
                "random_attempts": random,
                "random_successes": random_landed,
                "landed": informed_landed + random_landed,
                "landed_random": random_landed,
                "prefilled": 0,
                "greedy_forces": [],
                "greedy_targets": [],
                "informed_attempts": informed,
                "informed_successes": informed_landed,
                "informed_landed": informed_landed,
                "informed_forces": [],
                "informed_targets": [],
            }

        traces = [
            {
                "label": "ees",
                "seeds": TestSeparatingLearnedDrawsFromTheUniformFallback._run(
                    tallies={
                        # The exact two comparisons the log reports.
                        "ThrowRecycling": _tally(
                            informed=56, informed_landed=11, random=57, random_landed=11
                        ),
                        "ThrowTrash": _tally(
                            informed=301, informed_landed=208, random=310, random_landed=61
                        ),
                    }
                ),
            }
        ]

        assert 100 * TossingRoomSplitThrowRates.noise_floor(
            n_first=56, n_second=57
        ) == pytest.approx(9.41, abs=0.01)
        assert 100 * TossingRoomSplitThrowRates.minimum_detectable_effect(
            n_first=56, n_second=57
        ) == pytest.approx(26.36, abs=0.01)
        assert 100 * TossingRoomSplitThrowRates.noise_floor(
            n_first=301, n_second=310
        ) == pytest.approx(4.05, abs=0.01)
        assert 100 * TossingRoomSplitThrowRates.minimum_detectable_effect(
            n_first=301, n_second=310
        ) == pytest.approx(11.34, abs=0.01)

        TossingRoomSplitThrowRates.print_informed_split(traces=traces)
        printed = capsys.readouterr().out

        recycling = next(
            line
            for line in printed.splitlines()
            if "ThrowRecycling" in line and "11/56" in line and "11/57" in line
        )
        assert "9.41pp" in recycling
        assert "26.36pp" in recycling, f"row carries no own-denominator MDE: {recycling!r}"
        # And specifically not the borrowed 310-vs-57 figure.
        assert "20.19pp" not in recycling

        trash = next(
            line
            for line in printed.splitlines()
            if "ThrowTrash" in line and "208/301" in line and "61/310" in line
        )
        assert "4.05pp" in trash
        assert "11.34pp" in trash, f"row carries no own-denominator MDE: {trash!r}"
        # The decisive non-vacuity check: the two rows carry DIFFERENT floors, so no
        # single hardcoded constant satisfies both.
        assert "26.36pp" not in trash

    @staticmethod
    def test_a_p_value_below_the_printed_resolution_is_reported_as_an_inequality() -> None:
        """`p = 0.0000` claims a p-value of zero, which no test returns."""
        assert TossingRoomSplitThrowRates.format_p_value(p=1e-30) == "< 0.0001"
        assert TossingRoomSplitThrowRates.format_p_value(p=1.0) == "1.0000"
        assert TossingRoomSplitThrowRates.format_p_value(p=0.0078) == "0.0078"

    @staticmethod
    def test_traces_without_the_instrumentation_report_no_informed_draws() -> None:
        """Backwards compatibility: the committed shards from before this split have no
        `informed_*` keys, and must read as "nothing recorded" rather than raising."""
        traces = [
            {
                "label": "ees",
                "seeds": TestSeparatingLearnedDrawsFromTheUniformFallback._run(
                    tallies={"ThrowRecycling": _throw(forces=[0.5, 0.6], targets=[0.5, 0.9])}
                ),
            }
        ]
        assert TossingRoomSplitThrowRates.informed_landing_totals(traces=traces) == {
            "ThrowRecycling": (0, 0)
        }
        assert TossingRoomSplitThrowRates.uninformed_greedy_totals(traces=traces) == {
            "ThrowRecycling": (2, 2)
        }
        # A skill with no recorded informed forces is omitted rather than reported as
        # 0/0, matching `badly_missed_force_totals`' own convention for absent keys.
        assert (
            TossingRoomSplitThrowRates.informed_badly_missed_force_totals(
                traces=traces, miss_threshold=0.30
            )
            == {}
        )


def _throw(*, forces: list[float], targets: list[float]) -> dict:
    """A throw tally whose greedy draws are given, with `landed` derived from the same
    0.1 tolerance the environment uses -- so a fixture cannot claim a landing its numbers
    do not support."""
    landed = sum(1 for f, t in zip(forces, targets, strict=True) if abs(f - t) < 0.1)
    return {
        "attempts": len(forces),
        "successes": landed,
        "random_attempts": 0,
        "random_successes": 0,
        "landed": landed,
        "landed_random": 0,
        "prefilled": 0,
        "greedy_forces": forces,
        "greedy_targets": targets,
    }


class TestWhichArmProducedTheseShards:
    """`scripts/tossingroomsplit_skill_traces.py` serves two throw representations of the
    same world, and their shards are deliberately the same shape. That is what lets one
    analysis read both -- and it is also why a figure has to say which arm it came from,
    or the two arms' figures become indistinguishable.

    The label is derived from the shard's own `env` key rather than from a flag, so it
    cannot be set wrong. These pin that, and pin that pooling the two arms is refused."""

    @staticmethod
    def test_the_identity_arm_labels_itself() -> None:
        label = TossingRoomSplitThrowRates.arm_label(
            traces=[{"env": "tossingroomsplitidentity", "seeds": []}]
        )
        assert "IDENTITY" in label

    @staticmethod
    def test_the_causal_arm_labels_itself() -> None:
        label = TossingRoomSplitThrowRates.arm_label(
            traces=[{"env": "tossingroomsplit", "seeds": []}]
        )
        assert "CAUSAL" in label

    @staticmethod
    def test_a_shard_predating_the_env_key_is_the_causal_arm() -> None:
        """The committed 2026-08-05 traces were written before `--env` existed. They are
        the causal arm by definition, and must keep labelling themselves as one rather
        than becoming unlabelled."""
        assert "CAUSAL" in TossingRoomSplitThrowRates.arm_label(traces=[{"seeds": []}])

    @staticmethod
    def test_pooling_the_two_arms_is_refused() -> None:
        """They are the same world under two representations: comparable side by side,
        never summed. A pooled figure would be a category error, so it raises rather than
        silently taking one arm's label for a mixture of both."""
        with pytest.raises(ValueError, match="more than one arm"):
            TossingRoomSplitThrowRates.arm_label(
                traces=[
                    {"env": "tossingroomsplit", "seeds": []},
                    {"env": "tossingroomsplitidentity", "seeds": []},
                ]
            )
