"""`HumanLadderCurves` turns eight reset-free sweeps into twelve figures and one report,
so what is pinned here is the arithmetic that does that -- not matplotlib.

Nine things can go wrong silently, and each has a test:

- **a family figure inherits the overall denominator.** The cross-arm curve is split three
  ways (overall x/300, TRASH x/140, RECYCLING x/140), and every level on a family panel --
  reference lines most of all -- must be that family's own. `skill-oracle` is 140/140 on
  both throw families but 300/300 overall;

- **an arm goes missing.** Seven arms cannot express the comparisons this experiment
  makes, so a missing one must raise rather than draw whichever it still can;
- **a family's denominator drifts.** TRASH and RECYCLING are 14 each per seed and EMPTY
  is 2, so a goal misfiled between families moves tasks between denominators invisibly;
- **the reset-free manipulation stops holding.** Every arm here is
  `--practice-reset-policy never`, so any `num_practice_resets` above zero means the arm
  was quietly reset for free and its whole premise is gone;
- **the human accounting stops adding up.** Cost must stay the v0 oracle's flat 1.0 per
  intervention -- otherwise a different oracle was wired and every cost number means
  something else;
- **the two zero-intervention rules get confused.** An arm with *no reachable human at
  all* (`no-human`, `two-way-ledge`, `skill-oracle`, `random-skills`) must never record
  one -- that would be a wiring error. An arm that asks and is never rescued must be
  ACCEPTED -- that is a finding, and rejecting it would hide the result. Two tests,
  because one list is not the other;
- **the timing contrast gets dropped.** `on-stuck` against a matched-target `at-random` is
  the only pair that isolates whether rescue *timing* carries information, so its absence
  from `comparisons()` would silently reduce the ladder to "help helps";
- **a never-practising arm is assumed to have a learner's checkpoint grid.**
  `skill-oracle` has no `--num-cycles` flag at all and evaluates once, at zero
  transitions, so the loader and both renderers must cope with a single checkpoint;
- **the pairing breaks, or the wrong arms get paired.** All arms share fixed seeds, so
  per-seed differences must stay aligned; and only same-world EES arms may be paired --
  `random-skills` and `skill-oracle` change the `Method`, `two-way-ledge` changes the
  world, so none of the three may appear in a comparison.

Expected values are derived on paper, not recorded from a run of the code.
"""

import json
from pathlib import Path

import pytest

from analysis.practice_makes_perfect.human_ladder_curves import HumanLadderCurves

_TRASH = "TrashInBin(trash, trash_bin)"
_RECYCLING = "RecyclingInBin(recycling, recycling_bin)"
_EMPTY = "RecyclingBinEmpty(recycling_bin) & TrashBinEmpty(trash_bin)"

_ARM_METHODS = {
    "no-human": "ees",
    "at-random-initial": "ees",
    "at-random-random": "ees",
    "stuck-initial": "ees",
    "stuck-random": "ees",
    "two-way-ledge": "ees",
    "skill-oracle": "skill-oracle",
    "random-skills": "random-skills",
}

# `skill-oracle` never practises, so it has ONE evaluation checkpoint where every other
# arm has two. Kept as a named set rather than an inline check because three helpers
# below have to agree about it.
_SINGLE_CHECKPOINT_ARMS = frozenset({"skill-oracle"})


def _sweep(*, trash_solved: int, recycling_solved: int, empty_solved: int) -> list[dict]:
    """One evaluation sweep with the domain's real 14 / 14 / 2 composition."""
    outcomes = []
    for index in range(14):
        outcomes.append({"task_index": index, "goal": _TRASH, "solved": index < trash_solved})
    for index in range(14):
        outcomes.append({
            "task_index": 14 + index,
            "goal": _RECYCLING,
            "solved": index < recycling_solved,
        })
    for index in range(2):
        outcomes.append({"task_index": 28 + index, "goal": _EMPTY, "solved": index < empty_solved})
    return outcomes


def _write_run(
    *,
    directory: Path,
    seed: int,
    sweeps: list[tuple[int, int, int]],
    num_practice_resets: int = 0,
    interventions: int = 0,
    human_cost: float | None = None,
    transitions_per_checkpoint: int = 150,
) -> None:
    run = directory / str(seed)
    run.mkdir(parents=True)
    breakdowns = [
        {
            "num_online_transitions": index * transitions_per_checkpoint,
            "outcomes": _sweep(trash_solved=trash, recycling_solved=recycling, empty_solved=empty),
        }
        for index, (trash, recycling, empty) in enumerate(sweeps)
    ]
    run.joinpath("stats.json").write_text(
        json.dumps({
            "breakdowns": breakdowns,
            "num_practice_resets": num_practice_resets,
            "num_human_interventions_recorded": interventions,
            "summed_human_cost_recorded": (
                float(interventions) if human_cost is None else human_cost
            ),
            "task_name": "default",
        })
    )


def _write_arm(
    *, root: Path, arm: str, sweeps: list[tuple[int, int, int]], interventions: int = 0, **kwargs
) -> Path:
    directory = root / arm / _ARM_METHODS[arm]
    for seed in (0, 1):
        _write_run(
            directory=directory,
            seed=seed,
            sweeps=sweeps,
            interventions=interventions,
            **kwargs,
        )
    return directory


def _write_all(*, root: Path, **overrides) -> dict[str, Path]:
    """Eight arms, two seeds each, deliberately separated so every comparison is nonzero:
    no-human ends 4 trash, stuck-initial 8, stuck-random 6, at-random-initial 5,
    at-random-random 5, two-way-ledge 12, skill-oracle 14, random-skills 1.

    The two `at-random` arms sit between the control and the two `on-stuck` arms, which is
    the shape the real experiment would have if rescue timing carried information --
    enough to make every one of the seven comparisons nonzero and signed, so a pairing
    bug cannot hide behind a tie."""
    finals = {
        "no-human": (4, 0, 0),
        "stuck-initial": (8, 0, 0),
        "stuck-random": (6, 0, 0),
        "at-random-initial": (5, 0, 0),
        "at-random-random": (5, 0, 0),
        "two-way-ledge": (12, 0, 0),
        "skill-oracle": (14, 0, 0),
        "random-skills": (1, 0, 0),
    }
    rescues = {
        "no-human": 0,
        "stuck-initial": 5,
        "stuck-random": 7,
        "at-random-initial": 4,
        "at-random-random": 4,
        "two-way-ledge": 0,
        "skill-oracle": 0,
        "random-skills": 0,
    }
    return {
        # `interventions` goes in the dict rather than as its own keyword so that an
        # override may replace it; passing it both ways is a TypeError, which would make
        # "this arm asked and was never rescued" untestable through this fixture.
        arm: _write_arm(
            root=root,
            arm=arm,
            sweeps=([finals[arm]] if arm in _SINGLE_CHECKPOINT_ARMS else [(0, 0, 0), finals[arm]]),
            **{"interventions": rescues[arm], **overrides.get(arm, {})},
        )
        for arm in finals
    }


def test_load_arms_requires_every_arm(*, tmp_path: Path) -> None:
    directories = _write_all(root=tmp_path)
    del directories["no-human"]
    with pytest.raises(ValueError, match="missing arm"):
        HumanLadderCurves.load_arms(directories=directories)


def test_load_arms_reads_every_seed(*, tmp_path: Path) -> None:
    arms = HumanLadderCurves.load_arms(directories=_write_all(root=tmp_path))
    assert sorted(arms) == [
        "at-random-initial",
        "at-random-random",
        "no-human",
        "random-skills",
        "skill-oracle",
        "stuck-initial",
        "stuck-random",
        "two-way-ledge",
    ]
    assert sorted(arms["stuck-initial"]) == [0, 1]


def test_a_free_practice_reset_anywhere_is_refused(*, tmp_path: Path) -> None:
    """Every arm is --practice-reset-policy never, so this is the check that the arms
    are what their names say."""
    directories = _write_all(root=tmp_path, **{"stuck-random": {"num_practice_resets": 4}})
    with pytest.raises(ValueError, match="num_practice_resets"):
        HumanLadderCurves.load_arms(directories=directories)


def test_the_control_arm_may_never_call_a_human(*, tmp_path: Path) -> None:
    root = tmp_path
    _write_all(root=root)
    # Rewrite the control's seed 0 as though a human had been called.
    _write_run(
        directory=root / "no-human" / "ees",
        seed=2,
        sweeps=[(0, 0, 0), (4, 0, 0)],
        interventions=1,
    )
    directories = {arm: root / arm / _ARM_METHODS[arm] for arm in _ARM_METHODS}
    with pytest.raises(ValueError, match="control"):
        HumanLadderCurves.load_arms(directories=directories)


def test_a_cost_that_is_not_the_v0_flat_rate_is_refused(*, tmp_path: Path) -> None:
    """Cost and count are proportional at v0. If they come apart, a different
    HumanOracle was wired and every cost number here means something else."""
    directories = _write_all(root=tmp_path, **{"stuck-initial": {"human_cost": 99.0}})
    with pytest.raises(ValueError, match="oracle"):
        HumanLadderCurves.load_arms(directories=directories)


def test_a_stuck_arm_that_was_never_rescued_is_not_refused(*, tmp_path: Path) -> None:
    """Zero interventions under a stuck trigger is a finding about the trigger, and
    rejecting the run would hide exactly the result worth reporting."""
    root = tmp_path
    directories = {}
    for arm in _ARM_METHODS:
        directories[arm] = _write_arm(
            root=root, arm=arm, sweeps=[(0, 0, 0), (4, 0, 0)], interventions=0
        )
    arms = HumanLadderCurves.load_arms(directories=directories)
    assert arms["stuck-initial"][0]["interventions"] == 0


def test_a_drifted_family_composition_is_refused(*, tmp_path: Path) -> None:
    root = tmp_path
    _write_all(root=root)
    bad = root / "no-human" / "ees" / "9"
    bad.mkdir(parents=True)
    bad.joinpath("stats.json").write_text(
        json.dumps({
            "breakdowns": [
                {
                    "num_online_transitions": 0,
                    "outcomes": [{"task_index": 0, "goal": _TRASH, "solved": False}],
                }
            ],
            "num_practice_resets": 0,
            "num_human_interventions_recorded": 0,
            "summed_human_cost_recorded": 0.0,
        })
    )
    directories = {arm: root / arm / _ARM_METHODS[arm] for arm in _ARM_METHODS}
    with pytest.raises(ValueError, match="composition"):
        HumanLadderCurves.load_arms(directories=directories)


def test_pooled_curve_sums_both_solved_and_total(*, tmp_path: Path) -> None:
    """Two seeds ending 8 TRASH each is 16/60 overall, not a mean of two rates."""
    arms = HumanLadderCurves.load_arms(directories=_write_all(root=tmp_path))
    assert HumanLadderCurves.pooled_curve(arm=arms["stuck-initial"], family=None) == [
        (0, 60),
        (16, 60),
    ]


def test_pooled_curve_splits_by_family(*, tmp_path: Path) -> None:
    arms = HumanLadderCurves.load_arms(directories=_write_all(root=tmp_path))
    assert HumanLadderCurves.pooled_curve(arm=arms["stuck-initial"], family="TRASH")[-1] == (16, 28)
    assert HumanLadderCurves.pooled_curve(arm=arms["stuck-initial"], family="EMPTY")[-1] == (0, 4)


def test_transitions_are_the_shared_checkpoint_grid(*, tmp_path: Path) -> None:
    arms = HumanLadderCurves.load_arms(directories=_write_all(root=tmp_path))
    assert HumanLadderCurves.transitions(arm=arms["no-human"]) == [0, 150]


def test_seeds_may_disagree_on_transitions_because_a_rescue_consumes_one(
    *,
    tmp_path: Path,
) -> None:
    """A granted rescue `continue`s its loop iteration, so a rescued seed reaches its
    checkpoints one transition earlier per rescue and the seeds genuinely do NOT share an
    x axis. Refusing that -- which an earlier version did, on the since-falsified premise
    that a rescue is never charged -- would make every asking arm unplottable. The pooled
    x is the per-checkpoint mean instead."""
    root = tmp_path
    directories = _write_all(root=root)
    _write_run(
        directory=root / "stuck-initial" / "ees",
        seed=5,
        sweeps=[(0, 0, 0), (8, 0, 0)],
        transitions_per_checkpoint=140,
    )
    arms = HumanLadderCurves.load_arms(directories=directories)
    # Seeds 0 and 1 sit at 150, seed 5 at 140: the mean of (150, 150, 140) is 146.67.
    assert HumanLadderCurves.transitions(arm=arms["stuck-initial"]) == [0, 147]


def test_a_differing_number_of_checkpoints_is_still_refused(*, tmp_path: Path) -> None:
    """Differing checkpoint *counts* is a structural mismatch -- one seed ran a different
    number of cycles -- not the transition shortfall a rescue causes, so it stays fatal."""
    root = tmp_path
    directories = _write_all(root=root)
    _write_run(
        directory=root / "no-human" / "ees",
        seed=5,
        sweeps=[(0, 0, 0), (4, 0, 0), (4, 0, 0)],
    )
    arms = HumanLadderCurves.load_arms(directories=directories)
    with pytest.raises(ValueError, match="number of evaluation checkpoints"):
        HumanLadderCurves.transitions(arm=arms["no-human"])


def test_paired_differences_are_taken_within_a_seed(*, tmp_path: Path) -> None:
    """stuck-initial ends 8 per seed and no-human 4, so every seed's difference is +4."""
    arms = HumanLadderCurves.load_arms(directories=_write_all(root=tmp_path))
    assert HumanLadderCurves.paired_final_differences(
        arms=arms, treatment="stuck-initial", control="no-human", family=None
    ) == [4.0, 4.0]


def test_paired_differences_can_be_negative(*, tmp_path: Path) -> None:
    arms = HumanLadderCurves.load_arms(directories=_write_all(root=tmp_path))
    assert HumanLadderCurves.paired_final_differences(
        arms=arms, treatment="stuck-random", control="stuck-initial", family=None
    ) == [-2.0, -2.0]


def test_only_same_world_ees_arms_are_compared() -> None:
    """Each excluded arm moves a second variable: `random-skills` and `skill-oracle`
    change the Method, `two-way-ledge` changes the world. A gap against any of them would
    be a gap in two things at once, so none may appear in a paired comparison."""
    named = {name for pair in HumanLadderCurves.comparisons() for name in pair}
    for excluded in ("random-skills", "skill-oracle", "two-way-ledge"):
        assert excluded not in named
    assert named == {
        "no-human",
        "stuck-initial",
        "stuck-random",
        "at-random-initial",
        "at-random-random",
    }


def test_rescue_timing_is_compared_against_matched_random_timing() -> None:
    """The experiment's sharpest question is whether the *timing* of a rescue carries
    information, and it is only answerable by differencing `on-stuck` against `at-random`
    at a MATCHED `--human-reset-target`. Dropping either pair would leave the ladder able
    to say "being rescued helps" but not "being rescued *when stuck* helps"."""
    comparisons = HumanLadderCurves.comparisons()
    assert ("stuck-initial", "at-random-initial") in comparisons
    assert ("stuck-random", "at-random-random") in comparisons


def test_every_treated_arm_is_compared_against_the_control() -> None:
    """All four cells of the 2x2 are differenced against `no-human`, so no cell can be
    silently dropped from the report."""
    comparisons = HumanLadderCurves.comparisons()
    for treated in ("stuck-initial", "stuck-random", "at-random-initial", "at-random-random"):
        assert (treated, "no-human") in comparisons


def test_solves_per_rescue_prices_a_gap_by_what_it_cost(*, tmp_path: Path) -> None:
    """The score gap alone is misleading when two arms rescue at different rates, which
    `on-stuck` and `at-random` measurably do. `stuck-initial` buys 8-4 = 4 extra solves
    per seed for 5 rescues (0.8 each); `at-random-initial` buys 5-4 = 1 for 4 rescues
    (0.25 each). Both derived on paper from the fixture, not read off the code."""
    arms = HumanLadderCurves.load_arms(directories=_write_all(root=tmp_path))
    assert HumanLadderCurves.solves_per_rescue(
        arms=arms, treatment="stuck-initial", control="no-human"
    ) == pytest.approx(0.8)
    assert HumanLadderCurves.solves_per_rescue(
        arms=arms, treatment="at-random-initial", control="no-human"
    ) == pytest.approx(0.25)


def test_solves_per_rescue_is_none_when_an_arm_never_rescued(*, tmp_path: Path) -> None:
    """`no-human` spends nothing, so "solves per rescue" is a division by zero rather than
    an infinite return. Reported as absent, never as a number."""
    arms = HumanLadderCurves.load_arms(directories=_write_all(root=tmp_path))
    assert (
        HumanLadderCurves.solves_per_rescue(arms=arms, treatment="no-human", control="no-human")
        is None
    )


def test_an_arm_that_asks_for_help_and_is_never_rescued_is_not_refused(*, tmp_path: Path) -> None:
    """An arm configured with `--ask-for-help` that records zero interventions is a
    measurement, not a broken run: "the human was never called" and "the human did not
    help" are different findings, and refusing the run would hide the first."""
    directories = _write_all(root=tmp_path, **{"stuck-initial": {"interventions": 0}})
    arms = HumanLadderCurves.load_arms(directories=directories)
    assert arms["stuck-initial"][0]["interventions"] == 0


def test_an_arm_with_no_human_wired_may_never_record_one(*, tmp_path: Path) -> None:
    """The converse of the test above, and why the two lists are not the same list:
    `two-way-ledge` wires no human at all, so any intervention there is a wiring error
    rather than a finding."""
    root = tmp_path
    _write_all(root=root)
    _write_run(
        directory=root / "two-way-ledge" / "ees",
        seed=2,
        sweeps=[(0, 0, 0), (12, 0, 0)],
        interventions=1,
    )
    directories = {arm: root / arm / _ARM_METHODS[arm] for arm in _ARM_METHODS}
    with pytest.raises(ValueError, match="control"):
        HumanLadderCurves.load_arms(directories=directories)


def test_a_never_practising_arm_loads_with_one_checkpoint(*, tmp_path: Path) -> None:
    """`skill-oracle` has no --num-cycles flag at all, so it evaluates once at zero
    transitions. The loader must accept that rather than assuming every arm shares the
    learners' checkpoint grid."""
    arms = HumanLadderCurves.load_arms(directories=_write_all(root=tmp_path))
    assert HumanLadderCurves.transitions(arm=arms["skill-oracle"]) == [0]
    assert HumanLadderCurves.pooled_curve(arm=arms["skill-oracle"], family=None) == [(28, 60)]


def test_final_per_seed_is_in_seed_order(*, tmp_path: Path) -> None:
    arms = HumanLadderCurves.load_arms(directories=_write_all(root=tmp_path))
    assert HumanLadderCurves.final_per_seed(arm=arms["stuck-random"], family=None) == [6, 6]


def test_format_count_is_x_over_y() -> None:
    assert HumanLadderCurves.format_count(solved=17, total=20) == "17/20"


def test_each_family_figure_uses_its_own_denominator(*, tmp_path: Path) -> None:
    """The trap the three-way split creates: carrying the overall figure's level onto a
    family panel. `skill-oracle` is 28/60 overall in this fixture but 28/28 on TRASH, and
    a reference line drawn at the overall number would be wrong on both axes.

    Asserted on the arithmetic the renderer reads, not on pixels — a figure test cannot
    tell a correct line from a plausible one."""
    arms = HumanLadderCurves.load_arms(directories=_write_all(root=tmp_path))
    # Two seeds x 14 TRASH = 28; two seeds x 30 = 60 overall. Different denominators,
    # which is the whole point.
    assert HumanLadderCurves.pooled_curve(arm=arms["skill-oracle"], family=None)[-1] == (28, 60)
    assert HumanLadderCurves.pooled_curve(arm=arms["skill-oracle"], family="TRASH")[-1] == (28, 28)
    assert HumanLadderCurves.pooled_curve(arm=arms["skill-oracle"], family="RECYCLING")[-1] == (
        0,
        28,
    )


def test_empty_is_saturated_in_every_arm_so_it_gets_no_figure(*, tmp_path: Path) -> None:
    """EMPTY is 20/20 in all seven real arms — 2 tasks per seed, at ceiling before any
    manipulation — so there is nothing for a curve to show and `main` renders none. It
    stays in `print_report`, where its denominator is visible.

    This pins the *reason*: EMPTY's denominator is 2 per seed, an order of magnitude below
    the throw families' 14. If that ever stops being true the omission needs revisiting."""
    arms = HumanLadderCurves.load_arms(directories=_write_all(root=tmp_path))
    for arm_name in ("no-human", "stuck-initial", "skill-oracle"):
        _, total = HumanLadderCurves.pooled_curve(arm=arms[arm_name], family="EMPTY")[-1]
        assert total == 2 * 2  # 2 EMPTY tasks per seed, 2 seeds in this fixture


def test_render_writes_every_figure(*, tmp_path: Path) -> None:
    arms = HumanLadderCurves.load_arms(directories=_write_all(root=tmp_path))
    output = tmp_path / "figures"
    output.mkdir()
    # skill-oracle included deliberately: it is the reference-line path AND the
    # single-checkpoint path, which is where a curve-shaped renderer would raise.
    for arm in ("no-human", "stuck-initial", "skill-oracle", "random-skills"):
        HumanLadderCurves.render_arm(
            arms=arms, arm_name=arm, output=output / f"{arm}.png", title="t"
        )
        assert (output / f"{arm}.png").stat().st_size > 0
    # Three cross-arm figures, one per goal family, and none for EMPTY.
    for family in (None, "TRASH", "RECYCLING"):
        name = "overall" if family is None else family.lower()
        HumanLadderCurves.render_family(
            arms=arms, family=family, output=output / f"{name}.png", title="t"
        )
        assert (output / f"{name}.png").stat().st_size > 0
    HumanLadderCurves.render_interventions(
        arms=arms, output=output / "interventions.png", title="t"
    )
    assert (output / "interventions.png").stat().st_size > 0


def test_print_report_quotes_counts_never_bare_percentages(*, tmp_path: Path, capsys) -> None:
    arms = HumanLadderCurves.load_arms(directories=_write_all(root=tmp_path))
    HumanLadderCurves.print_report(arms=arms)
    printed = capsys.readouterr().out
    assert "16/60" in printed
    assert "%" not in printed
