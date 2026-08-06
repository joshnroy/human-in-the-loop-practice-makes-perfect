"""`TossingRoomGoalFamilyCurves` splits a run's evaluation record by goal family, and
every number it produces is a count that ends up quoted in an experiment log. Two
classes of error are invisible on inspection and are pinned here rather than trusted:

- a family **misclassification**, which silently moves tasks between denominators (14
  TRASH / 14 RECYCLING / 2 EMPTY is the whole composition, so one mislabelled goal
  string is a 1/14 error in two families at once); and
- a **noise floor** arithmetic slip, which produces a plausible number rather than a
  crash -- this project has already published one wrong statistic that way.

Expected values are derived on paper, not recorded from a run of the code.
"""

import json
import math
from pathlib import Path

import pytest

from analysis.practice_makes_perfect.tossingroom_goal_family_curves import (
    TossingRoomGoalFamilyCurves,
)

# The three goal strings TossingRoomTasks actually writes into a breakdown, verbatim.
_TRASH_GOAL = "ItemInBin(trash, trash_bin)"
_RECYCLING_GOAL = "ItemInBin(recycling, recycling_bin)"
_EMPTY_GOAL = "BinEmpty(recycling_bin) & BinEmpty(trash_bin)"


def _outcome(*, index: int, goal: str, solved: bool) -> dict:
    return {"task_index": index, "goal": goal, "solved": solved}


def _write_run(*, root: Path, method: str, seed: int, sweeps: list[list[tuple[str, bool]]]) -> None:
    """One stats.json whose `breakdowns` and `evaluations` agree, as Metrics guarantees."""
    directory = root / method / str(seed)
    directory.mkdir(parents=True)
    breakdowns = []
    evaluations = []
    for sweep_index, sweep in enumerate(sweeps):
        transitions = sweep_index * 100
        outcomes = [
            _outcome(index=i, goal=goal, solved=solved) for i, (goal, solved) in enumerate(sweep)
        ]
        breakdowns.append({"num_online_transitions": transitions, "outcomes": outcomes})
        evaluations.append([transitions, sum(1 for _, s in sweep if s), len(sweep)])
    (directory / "stats.json").write_text(
        json.dumps({"evaluations": evaluations, "breakdowns": breakdowns, "task_name": "default"})
    )


def test_family_of_reads_the_item_name_out_of_a_throw_goal() -> None:
    """A throw goal is a single `ItemInBin` atom whose first object names the item, so
    the family is that name uppercased -- read off the goal itself rather than off a
    task index, which would silently depend on draw order."""
    assert TossingRoomGoalFamilyCurves.family_of(goal=_TRASH_GOAL) == "TRASH"
    assert TossingRoomGoalFamilyCurves.family_of(goal=_RECYCLING_GOAL) == "RECYCLING"


def test_family_of_reads_the_split_domains_per_item_throw_predicates() -> None:
    """`tossingroomsplit` splits the shared `ItemInBin` into `TrashInBin` and
    `RecyclingInBin`, because there the item and bin types are split too
    (`environments/tossingroomsplit/predicates.py`). The family still comes from the
    **first object**, not from the predicate name, so one rule covers both domains --
    and without this the split domain's runs cannot be read at all."""
    assert TossingRoomGoalFamilyCurves.family_of(goal="TrashInBin(trash, trash_bin)") == "TRASH"
    assert (
        TossingRoomGoalFamilyCurves.family_of(goal="RecyclingInBin(recycling, recycling_bin)")
        == "RECYCLING"
    )


def test_family_of_recognises_the_split_domains_two_atom_empty_goal() -> None:
    """The split domain also renames the empty atoms per bin. It must still be EMPTY,
    and must not be mistaken for a throw family -- `TrashBinEmpty` contains the word
    Trash, so a rule keyed on the predicate name rather than the objects would put it in
    the TRASH denominator."""
    goal = "RecyclingBinEmpty(recycling_bin) & TrashBinEmpty(trash_bin)"
    assert TossingRoomGoalFamilyCurves.family_of(goal=goal) == "EMPTY"


def test_family_of_recognises_the_two_atom_empty_goal() -> None:
    """EMPTY is the only conjunctive goal in this domain -- one `BinEmpty` per bin since
    #74 gave each bin its own button. It must not be parsed as a throw family; doing so
    would invent a fourth family and drop 2 of every 30 tasks out of the totals."""
    assert TossingRoomGoalFamilyCurves.family_of(goal=_EMPTY_GOAL) == "EMPTY"


def test_family_of_rejects_a_goal_it_does_not_understand() -> None:
    """Silently bucketing an unknown goal would corrupt a denominator invisibly. A
    domain change that adds a family must fail here rather than be absorbed."""
    with pytest.raises(ValueError, match="unrecognised goal"):
        TossingRoomGoalFamilyCurves.family_of(goal="SomethingElse(a, b)")


def test_pooled_family_counts_sum_across_seeds_at_each_checkpoint(*, tmp_path: Path) -> None:
    """Pooled counts are episodes summed across seeds, not per-seed rates averaged.
    Two seeds, one sweep, 2 TRASH + 1 EMPTY each: TRASH solves once on seed 0 and twice
    on seed 1, so pooled TRASH is 3/4 and EMPTY is 2/2 -- derived by hand."""
    _write_run(
        root=tmp_path,
        method="ees",
        seed=0,
        sweeps=[[(_TRASH_GOAL, True), (_TRASH_GOAL, False), (_EMPTY_GOAL, True)]],
    )
    _write_run(
        root=tmp_path,
        method="ees",
        seed=1,
        sweeps=[[(_TRASH_GOAL, True), (_TRASH_GOAL, True), (_EMPTY_GOAL, True)]],
    )
    pooled = TossingRoomGoalFamilyCurves.pooled_family_counts(root=tmp_path, method="ees")
    assert pooled["TRASH"][0] == (3, 4)
    assert pooled["EMPTY"][0] == (2, 2)
    assert "RECYCLING" not in pooled


def test_pooled_counts_include_a_family_that_solved_nothing(*, tmp_path: Path) -> None:
    """A family at 0 solved must appear as `0/n`, not vanish. An absent row reads as
    "not measured"; 0/14 reads as "measured, and failed" -- the two are opposite
    findings and the composition is what tells them apart."""
    _write_run(
        root=tmp_path,
        method="ees",
        seed=0,
        sweeps=[[(_RECYCLING_GOAL, False), (_RECYCLING_GOAL, False), (_EMPTY_GOAL, True)]],
    )
    pooled = TossingRoomGoalFamilyCurves.pooled_family_counts(root=tmp_path, method="ees")
    assert pooled["RECYCLING"][0] == (0, 2)


def test_per_seed_family_counts_keep_the_seeds_apart(*, tmp_path: Path) -> None:
    """With ten seeds a pooled mean hides one seed driving the whole effect, so the
    per-seed record is kept rather than reduced. Seed 1 solves both TRASH tasks and
    seed 0 neither -- pooled that is 2/4, which describes neither seed."""
    _write_run(
        root=tmp_path,
        method="ees",
        seed=0,
        sweeps=[[(_TRASH_GOAL, False), (_TRASH_GOAL, False)]],
    )
    _write_run(
        root=tmp_path,
        method="ees",
        seed=1,
        sweeps=[[(_TRASH_GOAL, True), (_TRASH_GOAL, True)]],
    )
    per_seed = TossingRoomGoalFamilyCurves.per_seed_family_counts(root=tmp_path, method="ees")
    assert per_seed["0"]["TRASH"][0] == (0, 2)
    assert per_seed["1"]["TRASH"][0] == (2, 2)


def test_counts_track_every_checkpoint_not_just_the_endpoint(*, tmp_path: Path) -> None:
    """The learning curve is the point, so every sweep's breakdown is kept keyed by its
    own transition count -- 0 (pre-practice) and 100 here."""
    _write_run(
        root=tmp_path,
        method="ees",
        seed=0,
        sweeps=[[(_TRASH_GOAL, False)], [(_TRASH_GOAL, True)]],
    )
    pooled = TossingRoomGoalFamilyCurves.pooled_family_counts(root=tmp_path, method="ees")
    assert pooled["TRASH"] == {0: (0, 1), 100: (1, 1)}


def test_reading_a_run_without_breakdowns_fails_loudly(*, tmp_path: Path) -> None:
    """`breakdowns` is optional on Metrics, and a run recorded without it has no family
    information at all. Falling back to the pooled `evaluations` triples would produce a
    curve that looks right and answers a different question."""
    directory = tmp_path / "ees" / "0"
    directory.mkdir(parents=True)
    (directory / "stats.json").write_text(
        json.dumps({"evaluations": [[0, 1, 2]], "breakdowns": []})
    )
    with pytest.raises(ValueError, match="no per-task breakdowns"):
        TossingRoomGoalFamilyCurves.pooled_family_counts(root=tmp_path, method="ees")


def test_binomial_noise_floor_is_the_two_proportion_standard_error() -> None:
    """`sqrt(0.25/n_a + 0.25/n_b)`, in percentage points. At the pooled n = 300 of a
    ten-seed arm against another the same size that is
    `sqrt(0.25/300 + 0.25/300) = sqrt(1/600) = 0.040825`, i.e. 4.08 points."""
    floor = TossingRoomGoalFamilyCurves.binomial_noise_floor(n_a=300, n_b=300)
    assert floor == pytest.approx(100.0 * math.sqrt(1.0 / 600.0))
    assert floor == pytest.approx(4.0825, abs=1e-3)


def test_the_noise_floor_grows_as_the_smaller_arm_shrinks() -> None:
    """EMPTY is 2 tasks x 10 seeds = 20 episodes, so its floor is an order of magnitude
    worse than the pooled one: `sqrt(0.25/20 + 0.25/20) = 0.1581`, 15.81 points. Quoting
    the pooled floor beside an EMPTY comparison would understate it 4x."""
    assert TossingRoomGoalFamilyCurves.binomial_noise_floor(n_a=20, n_b=20) == pytest.approx(
        15.8114, abs=1e-3
    )


def test_minimum_detectable_effect_is_2_8_standard_errors() -> None:
    """80% power at a two-sided 5% level needs `z(0.975) + z(0.80) = 1.96 + 0.84 = 2.80`
    standard errors. The MDE is that multiple of the noise floor, so a design whose MDE
    exceeds the effect it is looking for cannot resolve it however the p-value lands."""
    floor = TossingRoomGoalFamilyCurves.binomial_noise_floor(n_a=300, n_b=300)
    assert TossingRoomGoalFamilyCurves.minimum_detectable_effect(n_a=300, n_b=300) == pytest.approx(
        2.8 * floor
    )


def test_format_count_never_renders_a_bare_percentage() -> None:
    """Every rate in this project's logs is written `x/y`, because a percentage hides
    its denominator -- "EMPTY 100%" is really 2/2 and must read that way."""
    assert TossingRoomGoalFamilyCurves.format_count(solved=2, total=2) == "2/2 (100.0%)"
    assert TossingRoomGoalFamilyCurves.format_count(solved=0, total=14) == "0/14 (0.0%)"


def test_dumped_json_carries_the_counts_the_log_quotes(*, tmp_path: Path) -> None:
    """Every table in an experiment log has to re-derive from a file committed beside
    it, so the numbers cannot drift from the runs that produced them. The dump is
    therefore counts -- `[solved, total]` pairs -- never rates, at every checkpoint and
    for every seed."""
    _write_run(
        root=tmp_path,
        method="ees",
        seed=0,
        sweeps=[
            [(_TRASH_GOAL, False), (_EMPTY_GOAL, True)],
            [(_TRASH_GOAL, True), (_EMPTY_GOAL, True)],
        ],
    )
    dumped = TossingRoomGoalFamilyCurves.as_json(root=tmp_path, method="ees")
    assert dumped["pooled"]["TRASH"]["0"] == [0, 1]
    assert dumped["pooled"]["TRASH"]["100"] == [1, 1]
    assert dumped["pooled"]["EMPTY"]["100"] == [1, 1]
    assert dumped["per_seed"]["0"]["TRASH"]["100"] == [1, 1]
    assert dumped["overall"]["100"] == [2, 2]


def test_dumped_json_round_trips_through_a_file(*, tmp_path: Path) -> None:
    """It is written to be read back by a human and by the next analysis, so it has to
    survive `json.dumps` -- which integer keys silently would not."""
    _write_run(root=tmp_path, method="ees", seed=0, sweeps=[[(_TRASH_GOAL, True)]])
    dumped = TossingRoomGoalFamilyCurves.as_json(root=tmp_path, method="ees")
    assert json.loads(json.dumps(dumped)) == dumped


def test_axis_ticks_are_labelled_as_counts_not_percentages() -> None:
    """The y axis is positioned in percent (the families share it despite having
    denominators that differ more than tenfold), but it must *read* as counts -- a bare
    "100" on an axis is exactly the denominator-hiding this project's logs forbid."""
    positions, labels = TossingRoomGoalFamilyCurves.count_ticks(total=140)
    assert positions == [0.0, 25.0, 50.0, 75.0, 100.0]
    assert labels == ["0/140", "35/140", "70/140", "105/140", "140/140"]


def test_axis_ticks_carry_the_small_empty_denominator_too() -> None:
    """EMPTY pools to 20 episodes, not 300. Reusing the pooled axis labels on its panel
    would overstate the evidence behind a flat line at the top by fifteenfold."""
    _, labels = TossingRoomGoalFamilyCurves.count_ticks(total=20)
    assert labels == ["0/20", "5/20", "10/20", "15/20", "20/20"]


def test_axis_tick_counts_round_to_whole_episodes() -> None:
    """A denominator not divisible by four must not produce a fractional episode: 14
    tasks give 0, 4 (3.5), 7, 11 (10.5), 14 -- banker's rounding is fine here, an
    invented `3.5/14` is not."""
    _, labels = TossingRoomGoalFamilyCurves.count_ticks(total=14)
    assert labels == ["0/14", "4/14", "7/14", "10/14", "14/14"]
    assert all("." not in label for label in labels)
