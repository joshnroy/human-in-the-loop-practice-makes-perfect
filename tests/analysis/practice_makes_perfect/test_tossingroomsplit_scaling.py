"""Tests for the 10x-budget scaling analysis.

Everything here is pinned against a hand-built trace whose answer is known by
construction, because the page this module produces makes claims that no amount of
staring at a plot would falsify:

1. **Where a draw is dated.** A draw is credited with the transitions completed BEFORE
   its own practice period, since that is what its sampler had been fitted on. Off by
   one period and every crossover shifts.
2. **What counts as evidence.** `prior_landings` and `prior_separation` must count
   epsilon-random landings too -- they are in the classifier's training set even though
   they are the control it is scored against. Counting only informed landings would
   understate the evidence at exactly the early draws that decide the question.
3. **That the three kinds of draw are never pooled.** `fallback` is neither informed nor
   random; pooling it into `informed` is the error PR #90 corrected, and pooling it into
   `random` would flatter the control.
4. **That a crossover has to persist.** One significant window in ten at alpha = 0.05 is
   roughly what chance produces.
"""

import pytest

from analysis.practice_makes_perfect.tossingroomsplit_scaling import (
    Draw,
    TossingRoomSplitScaling,
)

_TRASH = "TrashInBin(trash, trash_bin)"
_RECYCLING = "RecyclingInBin(recycling, recycling_bin)"


def _period(*, skill: str, targets: list[float], landed: list[bool], kinds: list[str]) -> dict:
    return {
        "skills": {
            skill: {
                "attempts": len(targets),
                "successes": sum(landed),
                "random_attempts": kinds.count("random"),
                "random_successes": 0,
                "landed": sum(landed),
                "landed_random": 0,
                "prefilled": 0,
                "greedy_forces": [],
                "greedy_targets": [],
                "informed_attempts": kinds.count("informed"),
                "informed_successes": 0,
                "informed_landed": 0,
                "informed_forces": [],
                "informed_targets": [],
                "throw_targets": targets,
                "throw_landed_flags": landed,
                "throw_kinds": kinds,
            }
        }
    }


def _shard(*, seed: int, periods: list[dict], steps: int = 100) -> dict:
    return {
        "label": "ees",
        "sampler_iters": 100,
        "num_cycles": len(periods),
        "max_steps_per_interaction": steps,
        "num_test_tasks": 4,
        "seeds": [
            {
                "seed": seed,
                "horizon": 12,
                "sweeps": [
                    {
                        "transitions": steps * index,
                        "solved": 0,
                        "total": 4,
                        "families": {_TRASH: [0, 2], _RECYCLING: [0, 2]},
                    }
                    for index in range(len(periods) + 1)
                ],
                "periods": periods,
                "competence": [],
            }
        ],
    }


class TestWhenADrawIsDated:
    """A draw belongs to the evidence its sampler had, not to the evidence its own
    period went on to produce."""

    @staticmethod
    def test_a_draw_is_dated_by_the_transitions_completed_before_its_own_period() -> None:
        periods = [
            _period(skill="ThrowRecycling", targets=[0.5], landed=[False], kinds=["informed"])
            for _ in range(3)
        ]
        draws = TossingRoomSplitScaling.draws(
            traces=[_shard(seed=0, periods=periods)], skill="ThrowRecycling"
        )
        # Period 0 is the first 100 transitions, so its draw had seen none of them.
        assert [draw.transitions for draw in draws] == [0, 100, 200]

    @staticmethod
    def test_shards_that_disagree_on_period_length_are_refused_rather_than_averaged() -> None:
        periods = [_period(skill="ThrowTrash", targets=[0.5], landed=[True], kinds=["random"])]
        with pytest.raises(ValueError, match="disagree on max_steps_per_interaction"):
            TossingRoomSplitScaling.steps_per_period(
                traces=[_shard(seed=0, periods=periods), _shard(seed=1, periods=periods, steps=50)]
            )


def test_traces_predating_the_per_draw_record_are_refused_by_name() -> None:
    """The committed 2026-08-05 shards have per-period counts but no per-draw record.
    Every comparison on this page is per draw, so there is no degraded mode -- and a bare
    `KeyError: 'throw_kinds'` from inside a zip would send the next reader hunting for a
    bug in the analysis rather than recollecting the traces."""
    stale = _shard(seed=0, periods=[_period(skill="ThrowTrash", targets=[], landed=[], kinds=[])])
    for tally in stale["seeds"][0]["periods"][0]["skills"].values():
        del tally["throw_kinds"]
    with pytest.raises(KeyError, match="predates the collector"):
        TossingRoomSplitScaling.draws(traces=[stale], skill="ThrowTrash")


class TestWhatCountsAsEvidenceBehindADraw:
    @staticmethod
    def test_prior_landings_counts_epsilon_random_landings_too() -> None:
        """The control is also training data. `observe_outcome` feeds the classifier every
        attempt, so a random landing is a positive exactly like an informed one."""
        periods = [
            _period(skill="ThrowRecycling", targets=[0.2], landed=[True], kinds=["random"]),
            _period(skill="ThrowRecycling", targets=[0.9], landed=[True], kinds=["fallback"]),
            _period(skill="ThrowRecycling", targets=[0.5], landed=[False], kinds=["informed"]),
        ]
        draws = TossingRoomSplitScaling.draws(
            traces=[_shard(seed=0, periods=periods)], skill="ThrowRecycling"
        )
        assert [draw.prior_landings for draw in draws] == [0, 1, 2]

    @staticmethod
    def test_prior_separation_is_the_range_of_the_landed_targets_before_this_draw() -> None:
        periods = [
            _period(skill="ThrowRecycling", targets=[0.2], landed=[True], kinds=["random"]),
            _period(skill="ThrowRecycling", targets=[0.25], landed=[True], kinds=["random"]),
            _period(skill="ThrowRecycling", targets=[0.9], landed=[True], kinds=["random"]),
            _period(skill="ThrowRecycling", targets=[0.5], landed=[False], kinds=["informed"]),
        ]
        draws = TossingRoomSplitScaling.draws(
            traces=[_shard(seed=0, periods=periods)], skill="ThrowRecycling"
        )
        # One landing cannot define a separation at all, so the second draw sees 0.0.
        assert draws[1].prior_separation == 0.0
        assert draws[2].prior_separation == pytest.approx(0.05)
        # A miss adds nothing, and the range spans the extremes rather than the last pair.
        assert draws[3].prior_separation == pytest.approx(0.70)

    @staticmethod
    def test_a_missed_throw_never_widens_the_separation() -> None:
        periods = [
            _period(skill="ThrowRecycling", targets=[0.2], landed=[True], kinds=["random"]),
            _period(skill="ThrowRecycling", targets=[0.9], landed=[False], kinds=["random"]),
            _period(skill="ThrowRecycling", targets=[0.5], landed=[False], kinds=["informed"]),
        ]
        assert (
            TossingRoomSplitScaling.draws(
                traces=[_shard(seed=0, periods=periods)], skill="ThrowRecycling"
            )[2].prior_separation
            == 0.0
        )


class TestTheThreeKindsOfDrawAreNeverPooled:
    @staticmethod
    def test_a_fallback_draw_is_counted_as_neither_informed_nor_random() -> None:
        draws = [
            Draw(
                seed=0,
                transitions=0,
                target=0.5,
                landed=landed,
                kind=kind,
                prior_landings=0,
                prior_separation=0.0,
            )
            for kind, landed in (
                ("informed", True),
                ("random", False),
                ("fallback", True),
                ("fallback", True),
            )
        ]
        assert TossingRoomSplitScaling.rate(draws=draws, kind="informed") == (1, 1)
        assert TossingRoomSplitScaling.rate(draws=draws, kind="random") == (0, 1)
        assert TossingRoomSplitScaling.rate(draws=draws, kind="fallback") == (2, 2)

    @staticmethod
    def test_a_comparison_with_an_empty_arm_reports_no_p_value_rather_than_a_number() -> None:
        """A comparison that cannot be made must say so. Returning 0.0 or 1.0 here would
        put an unfalsifiable 'null result' into a table."""
        draws = [
            Draw(
                seed=0,
                transitions=0,
                target=0.5,
                landed=True,
                kind="informed",
                prior_landings=0,
                prior_separation=0.0,
            )
        ]
        record = TossingRoomSplitScaling.compare(draws=draws, label="no control")
        assert record["informed"] == (1, 1)
        assert record["random"] == (0, 0)
        assert record["p"] is None
        assert record["gap"] is None
        assert record["mde"] is None


class TestTheMinimumDetectableEffectComesFromItsOwnDenominators:
    @staticmethod
    def test_the_mde_is_derived_from_this_comparisons_two_sample_sizes() -> None:
        """The published null is 56 informed against 57 epsilon-random, whose floor is
        `sqrt(0.25/56 + 0.25/57)` = 9.41pp and MDE 26.36pp -- NOT the 20.19pp that
        belongs to that page's 310-vs-57 comparison. Pinned so the wrong figure cannot
        be inherited again.

        26.36 rather than 26.34 because the constant is the exact
        `z_{0.025} + z_{0.20}` = 2.801585 that `minimum_detectable_effect` already uses,
        not 2.8 rounded. Worth pinning at this precision precisely because this test
        exists to stop an MDE being carried between comparisons by hand."""
        draws = [
            Draw(
                seed=0,
                transitions=0,
                target=0.5,
                landed=index < 11,
                kind="informed",
                prior_landings=0,
                prior_separation=0.0,
            )
            for index in range(56)
        ] + [
            Draw(
                seed=0,
                transitions=0,
                target=0.5,
                landed=index < 11,
                kind="random",
                prior_landings=0,
                prior_separation=0.0,
            )
            for index in range(57)
        ]
        record = TossingRoomSplitScaling.compare(draws=draws, label="published null")
        assert record["informed"] == (11, 56)
        assert record["random"] == (11, 57)
        assert record["noise_floor"] == pytest.approx(9.41, abs=0.01)
        assert record["mde"] == pytest.approx(26.356, abs=0.01)
        assert record["gap"] == pytest.approx(0.34, abs=0.01)
        assert record["p"] == pytest.approx(1.0)


class TestWhatCountsAsACrossover:
    @staticmethod
    def _window(*, start: int, gap: float, p: float) -> dict:
        return {"start": start, "end": start + 2500, "label": f"{start}", "gap": gap, "p": p}

    @staticmethod
    def test_a_lone_significant_window_that_does_not_persist_is_not_a_crossover() -> None:
        records = [
            TestWhatCountsAsACrossover._window(start=0, gap=1.0, p=0.9),
            TestWhatCountsAsACrossover._window(start=2500, gap=30.0, p=0.001),
            TestWhatCountsAsACrossover._window(start=5000, gap=-4.0, p=0.7),
        ]
        assert TossingRoomSplitScaling.first_separating_bin(records=records) is None

    @staticmethod
    def test_the_earliest_significant_window_whose_gap_persists_is_the_crossover() -> None:
        records = [
            TestWhatCountsAsACrossover._window(start=0, gap=-2.0, p=0.9),
            TestWhatCountsAsACrossover._window(start=2500, gap=25.0, p=0.01),
            TestWhatCountsAsACrossover._window(start=5000, gap=30.0, p=0.001),
        ]
        crossover = TossingRoomSplitScaling.first_separating_bin(records=records)
        assert crossover is not None
        assert crossover["start"] == 2500

    @staticmethod
    def test_a_significant_window_in_the_wrong_direction_is_never_a_crossover() -> None:
        records = [
            TestWhatCountsAsACrossover._window(start=0, gap=-40.0, p=0.0001),
            TestWhatCountsAsACrossover._window(start=2500, gap=1.0, p=0.9),
        ]
        assert TossingRoomSplitScaling.first_separating_bin(records=records) is None

    @staticmethod
    def test_holm_bonferroni_refuses_a_p_that_only_survives_uncorrected() -> None:
        """0.04 is significant on its own and is not significant as the smallest of ten."""
        assert TossingRoomSplitScaling.holm_threshold(p_values=[0.04] + [0.9] * 9) is None
        assert TossingRoomSplitScaling.holm_threshold(p_values=[0.001] + [0.9] * 9) == 0.001

    @staticmethod
    def test_the_final_window_can_never_be_a_crossover() -> None:
        """`all([])` is True, so a naive persistence check hands back the last window --
        the one case with no persistence evidence at all, which is exactly what the rule
        exists to exclude."""
        records = [
            TestWhatCountsAsACrossover._window(start=0, gap=1.0, p=0.9),
            TestWhatCountsAsACrossover._window(start=2500, gap=40.0, p=0.0001),
        ]
        assert TossingRoomSplitScaling.first_separating_bin(records=records) is None

    @staticmethod
    def test_a_later_window_that_measured_nothing_does_not_veto_a_crossover() -> None:
        """A window with an empty arm has `gap is None`. That is an absence of evidence,
        not evidence of a contradiction, so it must not block a real crossover."""
        records = [
            TestWhatCountsAsACrossover._window(start=0, gap=-1.0, p=0.9),
            TestWhatCountsAsACrossover._window(start=2500, gap=40.0, p=0.0001),
            {"start": 5000, "end": 7500, "label": "5000", "gap": None, "p": None},
            TestWhatCountsAsACrossover._window(start=7500, gap=35.0, p=0.001),
        ]
        crossover = TossingRoomSplitScaling.first_separating_bin(records=records)
        assert crossover is not None
        assert crossover["start"] == 2500


class TestTheAttemptCapTheDesignRestsOn:
    @staticmethod
    def test_a_period_with_two_recycling_attempts_is_reported_as_a_broken_cap(
        *, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Scaling periods rather than period length is only the right design while the
        cap holds, so the report has to be able to say that it does not."""
        periods = [
            _period(
                skill="ThrowRecycling",
                targets=[0.2, 0.6],
                landed=[False, False],
                kinds=["random", "informed"],
            )
        ]
        TossingRoomSplitScaling.print_attempt_cap(traces=[_shard(seed=0, periods=periods)])
        assert "THE CAP IS BROKEN" in capsys.readouterr().out

    @staticmethod
    def test_a_capped_run_is_reported_as_holding(*, capsys: pytest.CaptureFixture[str]) -> None:
        periods = [
            _period(skill="ThrowRecycling", targets=[0.2], landed=[False], kinds=["random"]),
            _period(skill="ThrowTrash", targets=[0.2], landed=[False], kinds=["random"]),
        ]
        TossingRoomSplitScaling.print_attempt_cap(traces=[_shard(seed=0, periods=periods)])
        output = capsys.readouterr().out
        assert "the cap holds" in output
        assert "THE CAP IS BROKEN" not in output


class TestTransitionsAndAccumulatedLandingsAreNotConfounded:
    @staticmethod
    def test_the_factorial_split_reports_all_four_cells_with_their_own_denominators() -> None:
        draws = [
            Draw(
                seed=0,
                transitions=transitions,
                target=0.5,
                landed=landed,
                kind=kind,
                prior_landings=landings,
                prior_separation=0.0,
            )
            for transitions, landings, kind, landed in (
                (0, 0, "informed", False),
                (0, 5, "informed", True),
                (10000, 0, "informed", False),
                (10000, 5, "informed", True),
                (0, 0, "random", False),
                (0, 5, "random", False),
                (10000, 0, "random", False),
                (10000, 5, "random", False),
            )
        ]
        cells = TossingRoomSplitScaling.factorial_split(draws=draws, landing_threshold=2)
        assert [cell["label"] for cell in cells] == [
            "early, <2 landings",
            "early, >=2 landings",
            "late, <2 landings",
            "late, >=2 landings",
        ]
        # The gap tracks the landing count in both time halves, which is the shape that
        # would say the mechanism variable is landings rather than the clock.
        by_label = {cell["label"]: cell for cell in cells}
        assert by_label["early, <2 landings"]["gap"] == pytest.approx(0.0)
        assert by_label["early, >=2 landings"]["gap"] == pytest.approx(100.0)
        assert by_label["late, <2 landings"]["gap"] == pytest.approx(0.0)
        assert by_label["late, >=2 landings"]["gap"] == pytest.approx(100.0)


class TestTheMechanismSeries:
    @staticmethod
    def test_separation_crosses_the_tolerance_only_once_two_landings_really_span_it() -> None:
        periods = [
            _period(skill="ThrowRecycling", targets=[0.20], landed=[True], kinds=["random"]),
            _period(skill="ThrowRecycling", targets=[0.25], landed=[True], kinds=["random"]),
            _period(skill="ThrowRecycling", targets=[0.80], landed=[True], kinds=["random"]),
        ]
        traces = [_shard(seed=0, periods=periods)]
        # 0.05 apart is inside the tolerance, so the second landing does not cross it.
        assert TossingRoomSplitScaling.separation_series(traces=traces, skill="ThrowRecycling")[
            0
        ] == [(100, 0.0), (200, pytest.approx(0.05)), (300, pytest.approx(0.60))]
        assert TossingRoomSplitScaling.crosses_tolerance_at(
            traces=traces, skill="ThrowRecycling"
        ) == {0: 300}

    @staticmethod
    def test_a_seed_whose_landings_never_spread_reports_no_crossing() -> None:
        periods = [
            _period(skill="ThrowRecycling", targets=[0.50], landed=[True], kinds=["random"]),
            _period(skill="ThrowRecycling", targets=[0.52], landed=[True], kinds=["random"]),
        ]
        assert TossingRoomSplitScaling.crosses_tolerance_at(
            traces=[_shard(seed=0, periods=periods)], skill="ThrowRecycling"
        ) == {0: None}

    @staticmethod
    def test_landings_accumulate_monotonically_and_count_every_kind_of_draw() -> None:
        periods = [
            _period(skill="ThrowRecycling", targets=[0.2], landed=[True], kinds=["random"]),
            _period(skill="ThrowRecycling", targets=[0.3], landed=[False], kinds=["informed"]),
            _period(skill="ThrowRecycling", targets=[0.4], landed=[True], kinds=["fallback"]),
        ]
        assert TossingRoomSplitScaling.landings_series(
            traces=[_shard(seed=0, periods=periods)], skill="ThrowRecycling"
        )[0] == [(100, 1), (200, 1), (300, 2)]


class TestTheBinningItself:
    """The binning is the highest-risk code on the page and was the last part without a
    test: a window that silently drops or double-counts draws changes every number
    downstream and is invisible on the rendered figure."""

    @staticmethod
    def _draw(*, transitions: int, landings: int = 0, kind: str = "informed") -> Draw:
        return Draw(
            seed=0,
            transitions=transitions,
            target=0.5,
            landed=False,
            kind=kind,
            prior_landings=landings,
            prior_separation=0.0,
        )

    @staticmethod
    def test_every_draw_lands_in_exactly_one_transition_window() -> None:
        draws = [TestTheBinningItself._draw(transitions=value) for value in (0, 2499, 2500, 5001)]
        records = TossingRoomSplitScaling.binned_by_transitions(draws=draws, bin_width=2500)
        assert sum(record["informed"][1] for record in records) == len(draws)
        # Half-open windows: 2499 is in the first, 2500 opens the second.
        by_start = {record["start"]: record["informed"][1] for record in records}
        assert by_start[0] == 2
        assert by_start[2500] == 1
        assert by_start[5000] == 1

    @staticmethod
    def test_a_window_with_no_draws_is_kept_rather_than_dropped() -> None:
        """Dropping it leaves a hole the table hides and the plotted line interpolates
        straight across, drawing a measurement where none was taken."""
        draws = [TestTheBinningItself._draw(transitions=value) for value in (0, 7000)]
        records = TossingRoomSplitScaling.binned_by_transitions(draws=draws, bin_width=2500)
        assert [record["start"] for record in records] == [0, 2500, 5000, 7500]
        empty = records[1]
        assert empty["informed"] == (0, 0)
        assert empty["gap"] is None and empty["p"] is None

    @staticmethod
    def test_every_draw_lands_in_exactly_one_landings_band() -> None:
        draws = [
            TestTheBinningItself._draw(transitions=0, landings=value)
            for value in (0, 1, 2, 3, 4, 5, 9, 10, 19, 20, 99)
        ]
        records = TossingRoomSplitScaling.binned_by_prior_landings(draws=draws)
        assert sum(record["informed"][1] for record in records) == len(draws)
        assert [record["label"] for record in records] == [
            "0",
            "1",
            "2",
            "3-4",
            "5-9",
            "10-19",
            "20+",
        ]

    @staticmethod
    @pytest.mark.parametrize("edges", [(2, 5), (0, 5, 2), (0, 2, 2, 5)])
    def test_edges_that_would_drop_or_double_count_draws_are_refused(
        *, edges: tuple[int, ...]
    ) -> None:
        draws = [TestTheBinningItself._draw(transitions=0, landings=value) for value in range(6)]
        with pytest.raises(ValueError, match="must start at 0 and strictly increase"):
            TossingRoomSplitScaling.binned_by_prior_landings(draws=draws, edges=edges)

    @staticmethod
    def test_the_factorial_median_ignores_the_fallback_draws_it_never_compares() -> None:
        """Fallback draws are concentrated early by construction -- an unfitted classifier
        is what produces them -- so pooling them into the median drags the early/late cut
        earlier than the median of the draws actually being compared."""
        compared = [
            TestTheBinningItself._draw(transitions=value, kind=kind)
            for value in (1000, 2000, 3000, 4000)
            for kind in ("informed", "random")
        ]
        fallback = [TestTheBinningItself._draw(transitions=0, kind="fallback") for _ in range(100)]
        cells = TossingRoomSplitScaling.factorial_split(
            draws=compared + fallback, landing_threshold=2
        )
        # Median of the compared draws is 2500, not the ~0 that 100 early fallbacks force.
        assert cells[0]["midpoint"] == 2500
        # And no cell ever contains a fallback draw.
        assert sum(cell["informed"][1] + cell["random"][1] for cell in cells) == len(compared)


class TestTheSeparationSplitTheMechanismClaimRestsOn:
    @staticmethod
    def test_draws_are_split_by_whether_past_landings_already_spanned_the_tolerance() -> None:
        """`prior_separation` is the variable the diagnosis names, so it needs its own
        comparison rather than only being computed and plotted."""
        draws = [
            Draw(
                seed=0,
                transitions=0,
                target=0.5,
                landed=landed,
                kind=kind,
                prior_landings=2,
                prior_separation=separation,
            )
            for separation, kind, landed in (
                (0.05, "informed", False),
                (0.05, "random", False),
                (0.60, "informed", True),
                (0.60, "random", False),
            )
        ]
        narrow, wide = TossingRoomSplitScaling.split_by_separation(draws=draws)
        assert narrow["informed"] == (0, 1) and narrow["random"] == (0, 1)
        assert wide["informed"] == (1, 1) and wide["random"] == (0, 1)
        # A separation exactly at the tolerance is still consistent with one flat band.
        assert "<= 0.1" in narrow["label"] and "> 0.1" in wide["label"]
