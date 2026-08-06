"""`TossingRoomSplitFamilyOverlay` puts `TRASH` and `RECYCLING` on one axes so the gap
between them is read directly rather than reconstructed across two panels. It reuses
`TossingRoomGoalFamilyCurves` for every count, so what is pinned here is only the new
arithmetic layered on top -- and each of those is a number an experiment log quotes:

- a **threshold crossing**, which is how a "recycling gets there later" claim is stated
  at all, and which is off-by-one-checkpoint wrong if the comparison is `>` not `>=`;
- an **AUC**, which is a rate over a curve and therefore silently wrong if the
  checkpoints are treated as equally weighted when they are not; and
- the **paired per-seed endpoints**, which must stay aligned seed-to-seed, because the
  arms share seeds and an unpaired reading of them throws that structure away.

Expected values are derived on paper, not recorded from a run of the code.
"""

import json
import math
from pathlib import Path

import pytest

from analysis.practice_makes_perfect.tossingroomsplit_family_overlay import (
    TossingRoomSplitFamilyOverlay,
)

_TRASH_GOAL = "ItemInBin(trash, trash_bin)"
_RECYCLING_GOAL = "ItemInBin(recycling, recycling_bin)"


def _write_run(*, root: Path, method: str, seed: int, sweeps: list[list[tuple[str, bool]]]) -> None:
    """One stats.json whose `breakdowns` and `evaluations` agree, as Metrics guarantees."""
    directory = root / method / str(seed)
    directory.mkdir(parents=True)
    breakdowns = []
    evaluations = []
    for sweep_index, sweep in enumerate(sweeps):
        transitions = sweep_index * 100
        outcomes = [
            {"task_index": i, "goal": goal, "solved": solved}
            for i, (goal, solved) in enumerate(sweep)
        ]
        breakdowns.append({"num_online_transitions": transitions, "outcomes": outcomes})
        evaluations.append([transitions, sum(1 for _, s in sweep if s), len(sweep)])
    (directory / "stats.json").write_text(
        json.dumps({"evaluations": evaluations, "breakdowns": breakdowns, "task_name": "default"})
    )


def test_threshold_crossing_is_the_first_checkpoint_at_or_above_the_level() -> None:
    """ "Reaches 35/140" means the first checkpoint whose count is >= 35, not > 35. With
    `>` the curve below would report 200, which claims recycling got there a whole
    practice period later than it did."""
    curve = {0: (0, 140), 100: (35, 140), 200: (70, 140)}
    assert (
        TossingRoomSplitFamilyOverlay.first_transitions_at_or_above(curve=curve, solved=35) == 100
    )


def test_threshold_crossing_is_none_when_the_level_is_never_reached() -> None:
    """A level the run never reaches must be absent rather than reported as the final
    checkpoint -- "did not get there" and "got there at the end" are opposite findings,
    and this is exactly the tail a 60-cycle budget cannot see."""
    curve = {0: (0, 140), 100: (35, 140)}
    assert (
        TossingRoomSplitFamilyOverlay.first_transitions_at_or_above(curve=curve, solved=140) is None
    )


def test_threshold_crossing_ignores_a_later_dip_below_the_level() -> None:
    """The claim is "first reached", so a curve that crosses and then falls back still
    crossed. Scanning for a *sustained* level would answer a different question and
    would make the number depend on how far the run happened to continue."""
    curve = {0: (0, 140), 100: (70, 140), 200: (35, 140), 300: (70, 140)}
    assert (
        TossingRoomSplitFamilyOverlay.first_transitions_at_or_above(curve=curve, solved=70) == 100
    )


def test_mean_solved_rate_is_the_trapezoidal_area_over_the_span() -> None:
    """AUC here is a mean rate over the transition axis, so it is the trapezoidal area
    divided by the span -- not the mean of the checkpoint rates, which would weight a
    dense cluster of checkpoints more heavily than a sparse one. Rising 0/14 -> 14/14
    linearly over 100 transitions is area 100 * (0 + 1)/2 = 50, over span 100 = 0.5."""
    curve = {0: (0, 14), 100: (14, 14)}
    assert TossingRoomSplitFamilyOverlay.mean_solved_rate(curve=curve) == pytest.approx(0.5)


def test_mean_solved_rate_weights_by_transition_span_not_by_checkpoint_count() -> None:
    """Three checkpoints at 0, 100 and 500: the rate is 1.0 from 100 onward. Trapezoids
    are 100*(0+1)/2 = 50 and 400*(1+1)/2 = 400, total 450 over span 500 = 0.9. The mean
    of the three checkpoint rates would be 0.667 -- the error this pins."""
    curve = {0: (0, 14), 100: (14, 14), 500: (14, 14)}
    assert TossingRoomSplitFamilyOverlay.mean_solved_rate(curve=curve) == pytest.approx(0.9)


def test_mean_solved_rate_of_a_single_checkpoint_is_that_checkpoint() -> None:
    """A zero-length span must not divide by zero. One checkpoint is a degenerate curve
    whose mean rate is simply its own rate."""
    assert TossingRoomSplitFamilyOverlay.mean_solved_rate(curve={0: (7, 14)}) == pytest.approx(0.5)


def test_paired_endpoints_keep_the_two_families_aligned_by_seed(*, tmp_path: Path) -> None:
    """The families are measured inside the same runs, so they are paired data. The
    endpoints must come back seed-aligned -- seed 0 solving TRASH but not RECYCLING and
    seed 1 the reverse is a zero mean difference with real per-seed spread, and an
    unpaired reading cannot tell that from both seeds sitting at the mean."""
    _write_run(
        root=tmp_path,
        method="ees",
        seed=0,
        sweeps=[[(_TRASH_GOAL, True), (_RECYCLING_GOAL, False)]],
    )
    _write_run(
        root=tmp_path,
        method="ees",
        seed=1,
        sweeps=[[(_TRASH_GOAL, False), (_RECYCLING_GOAL, True)]],
    )
    paired = TossingRoomSplitFamilyOverlay.paired_final_counts(root=tmp_path, method="ees")
    assert paired == [("0", (1, 1), (0, 1)), ("1", (0, 1), (1, 1))]


def test_paired_endpoints_are_ordered_by_seed_number_not_lexically(*, tmp_path: Path) -> None:
    """Seed 10 must not sort between 1 and 2. A lexical order would silently mis-pair
    the two families' lists once a sweep passes ten seeds."""
    for seed in (0, 2, 10):
        _write_run(
            root=tmp_path,
            method="ees",
            seed=seed,
            sweeps=[[(_TRASH_GOAL, True), (_RECYCLING_GOAL, False)]],
        )
    paired = TossingRoomSplitFamilyOverlay.paired_final_counts(root=tmp_path, method="ees")
    assert [seed for seed, _, _ in paired] == ["0", "2", "10"]


def test_a_run_missing_one_of_the_two_families_fails_loudly(*, tmp_path: Path) -> None:
    """Both families must be present in every seed or the pairing is not a pairing.
    Dropping the seed instead would change the denominator of a paired test without
    saying so."""
    _write_run(root=tmp_path, method="ees", seed=0, sweeps=[[(_TRASH_GOAL, True)]])
    with pytest.raises(ValueError, match="RECYCLING"):
        TossingRoomSplitFamilyOverlay.paired_final_counts(root=tmp_path, method="ees")


def test_the_two_families_share_a_denominator_so_one_axis_can_carry_both(*, tmp_path: Path) -> None:
    """The overlay puts both families on one y axis, which is only honest if their
    denominators match -- 14 tasks each per seed. A composition that broke that has to
    fail here rather than produce a figure whose two curves mean different things."""
    _write_run(
        root=tmp_path,
        method="ees",
        seed=0,
        sweeps=[[(_TRASH_GOAL, True), (_TRASH_GOAL, False), (_RECYCLING_GOAL, True)]],
    )
    with pytest.raises(ValueError, match="denominator"):
        TossingRoomSplitFamilyOverlay.shared_total(root=tmp_path, method="ees")


def test_shared_total_returns_the_common_denominator(*, tmp_path: Path) -> None:
    """Two seeds x 2 TRASH and 2 RECYCLING tasks pools to 4 episodes per family, and
    that 4 is what every count on the figure is written over."""
    for seed in (0, 1):
        _write_run(
            root=tmp_path,
            method="ees",
            seed=seed,
            sweeps=[
                [
                    (_TRASH_GOAL, True),
                    (_TRASH_GOAL, False),
                    (_RECYCLING_GOAL, True),
                    (_RECYCLING_GOAL, False),
                ]
            ],
        )
    assert TossingRoomSplitFamilyOverlay.shared_total(root=tmp_path, method="ees") == 4


def test_dumped_json_carries_counts_and_never_a_bare_rate(*, tmp_path: Path) -> None:
    """The log's tables re-derive from a committed record, so the dump is `[solved,
    total]` pairs at every checkpoint. A committed percentage cannot be inverted to the
    count behind it."""
    _write_run(
        root=tmp_path,
        method="ees",
        seed=0,
        sweeps=[
            [(_TRASH_GOAL, False), (_RECYCLING_GOAL, False)],
            [(_TRASH_GOAL, True), (_RECYCLING_GOAL, False)],
        ],
    )
    dumped = TossingRoomSplitFamilyOverlay.as_json(root=tmp_path, method="ees")
    assert dumped["pooled"]["TRASH"]["100"] == [1, 1]
    assert dumped["pooled"]["RECYCLING"]["100"] == [0, 1]
    assert dumped["per_seed"]["0"]["TRASH"]["100"] == [1, 1]
    assert json.loads(json.dumps(dumped)) == dumped


def test_dumped_json_records_the_thresholds_the_log_tabulates(*, tmp_path: Path) -> None:
    """The "recycling gets there later" table is the one comparison this figure is
    meant to support, so its crossings ship in the dump rather than being read off a
    terminal. A level never reached is `null`, not the last checkpoint."""
    _write_run(
        root=tmp_path,
        method="ees",
        seed=0,
        sweeps=[
            [(_TRASH_GOAL, False), (_RECYCLING_GOAL, False)],
            [(_TRASH_GOAL, True), (_RECYCLING_GOAL, False)],
        ],
    )
    dumped = TossingRoomSplitFamilyOverlay.as_json(root=tmp_path, method="ees")
    assert dumped["thresholds"]["TRASH"]["1"] == 100
    assert dumped["thresholds"]["RECYCLING"]["1"] is None


def test_format_count_never_renders_a_bare_percentage() -> None:
    """Every rate this figure prints is `x/y (p%)`. The percentage may accompany the
    count, never replace it."""
    assert TossingRoomSplitFamilyOverlay.format_count(solved=140, total=140) == "140/140 (100.0%)"
    assert TossingRoomSplitFamilyOverlay.format_count(solved=0, total=14) == "0/14 (0.0%)"


def test_the_figure_is_a_single_axes_carrying_both_families(*, tmp_path: Path) -> None:
    """Josh asked for one graph, not a panel per family: the gap between the two curves
    is the finding, and reading it across two panels is exactly the reconstruction a
    figure is supposed to remove. Rendering must therefore produce one axes, and must
    write a file."""
    for seed in (0, 1):
        _write_run(
            root=tmp_path,
            method="ees",
            seed=seed,
            sweeps=[
                [(_TRASH_GOAL, False), (_RECYCLING_GOAL, False)],
                [(_TRASH_GOAL, True), (_RECYCLING_GOAL, True)],
            ],
        )
    output = tmp_path / "overlay.png"
    figure = TossingRoomSplitFamilyOverlay.render(
        root=tmp_path, method="ees", output=output, title="test"
    )
    assert len(figure.axes) == 1
    assert output.exists() and output.stat().st_size > 0


def test_rendering_draws_one_thin_line_per_seed_per_family(*, tmp_path: Path) -> None:
    """Per-seed spread is the whole point of the figure -- the crossover is precisely
    where one seed could drive everything. Two seeds x two families is four thin lines
    plus two bold pooled ones, and a mean drawn alone would pass a weaker test."""
    for seed in (0, 1):
        _write_run(
            root=tmp_path,
            method="ees",
            seed=seed,
            sweeps=[
                [(_TRASH_GOAL, False), (_RECYCLING_GOAL, False)],
                [(_TRASH_GOAL, True), (_RECYCLING_GOAL, True)],
            ],
        )
    figure = TossingRoomSplitFamilyOverlay.render(
        root=tmp_path, method="ees", output=tmp_path / "overlay.png", title="test"
    )
    (axis,) = figure.axes
    thin = [line for line in axis.get_lines() if line.get_linewidth() < 1.5]
    bold = [line for line in axis.get_lines() if line.get_linewidth() >= 2.0]
    assert len(thin) == 4
    assert len(bold) == 2


def test_axis_ticks_read_as_counts(*, tmp_path: Path) -> None:
    """A bare "100" on the axis is the denominator-hiding this project's logs forbid, so
    the ticks carry the shared denominator both families are measured over.

    Two seeds x 7 tasks per family pools to 14, the real per-seed denominator, whose
    quarter ticks are 0, 4 (3.5), 7, 10 (10.5), 14 -- rounded to whole episodes, because
    an invented "3.5/14" would name a number of episodes that cannot exist. Both halves
    round to even, which is why 3.5 goes up to 4 and 10.5 goes down to 10."""
    for seed in (0, 1):
        _write_run(
            root=tmp_path,
            method="ees",
            seed=seed,
            sweeps=[[(_TRASH_GOAL, True)] * 7 + [(_RECYCLING_GOAL, False)] * 7],
        )
    figure = TossingRoomSplitFamilyOverlay.render(
        root=tmp_path, method="ees", output=tmp_path / "overlay.png", title="test"
    )
    (axis,) = figure.axes
    assert [label.get_text() for label in axis.get_yticklabels()] == [
        "0/14",
        "4/14",
        "7/14",
        "10/14",
        "14/14",
    ]


def test_mean_solved_rate_matches_a_hand_computed_two_segment_curve() -> None:
    """A curve that is not monotone still integrates: 0/14 -> 7/14 -> 0/14 over two
    100-transition segments is 100*(0+0.5)/2 + 100*(0.5+0)/2 = 50, over span 200 = 0.25.
    Pinned because an implementation that took `max` or the endpoint would agree with
    every monotone test above and be wrong here."""
    curve = {0: (0, 14), 100: (7, 14), 200: (0, 14)}
    assert TossingRoomSplitFamilyOverlay.mean_solved_rate(curve=curve) == pytest.approx(0.25)
    assert not math.isnan(TossingRoomSplitFamilyOverlay.mean_solved_rate(curve=curve))
