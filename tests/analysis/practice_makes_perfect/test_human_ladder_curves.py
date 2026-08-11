"""`HumanLadderCurves` turns three fixed arms plus a rescue-rate sweep into five figures
and one report, so what is pinned here is the arithmetic that does that -- not matplotlib.

Ten things can go wrong silently, and each has a test:

- **a family figure inherits the overall denominator.** The training-curve figures are
  split three ways (overall x/30, TRASH x/14, RECYCLING x/14 per seed), and every level --
  reference lines most of all -- must be that family's own;
- **a fixed arm goes missing.** Three arms cannot express this comparison with one absent,
  so a missing one must raise rather than draw whichever it still can;
- **the rate sweep has too few points to be a sweep.** One N is a single arm, not a
  dose-response, so fewer than two must raise;
- **a family's denominator drifts.** TRASH and RECYCLING are 14 each per seed and EMPTY is
  2, so a goal misfiled between families moves tasks between denominators invisibly;
- **the reset-free manipulation stops holding.** Every run here is
  `--practice-reset-policy never`, so any `num_practice_resets` above zero means the run was
  quietly reset for free;
- **the human accounting stops adding up.** Cost must stay the v0 oracle's flat 1.0 per
  intervention;
- **the two intervention-count rules get confused, in BOTH directions.** A fixed arm with no
  reachable human must record zero interventions; a rate-sweep point, which always has a
  reachable human firing at at least 1/20 per call over 1500 calls, must record a NONZERO
  count -- a true zero there means the trigger never wired, unlike `on-stuck` in #151's
  eight-arm module, which was allowed to report zero as a genuine finding;
- **a never-practising arm is assumed to have a learner's checkpoint grid.**
  `skill-oracle` has no `--num-cycles` flag at all and evaluates once, at zero transitions,
  so the loader must cope with a single checkpoint;
- **the pairing breaks.** Every rate-sweep point shares fixed seeds with `no-human`, so
  per-seed differences must stay aligned rather than pairing seed 3 of one against seed 7
  of the other;
- **solves-per-rescue divides by zero.** A treatment that spent no rescues must report
  `None`, never `inf` or `0`, both of which would read as findings.

Expected values are derived on paper, not recorded from a run of the code.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from analysis.practice_makes_perfect.human_ladder_curves import HumanLadderCurves

_TRASH = "TrashInBin(trash, trash_bin)"
_RECYCLING = "RecyclingInBin(recycling, recycling_bin)"
_EMPTY = "RecyclingBinEmpty(recycling_bin) & TrashBinEmpty(trash_bin)"

_ARM_METHODS = {
    "no-human": "ees",
    "two-way-ledge": "ees",
    "skill-oracle": "skill-oracle",
}


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
            directory=directory, seed=seed, sweeps=sweeps, interventions=interventions, **kwargs
        )
    return directory


def _write_rate_point(
    *, root: Path, n: int, sweeps: list[tuple[int, int, int]], interventions: int, **kwargs
) -> Path:
    directory = root / f"rate-sweep-N{n}" / "ees"
    for seed in (0, 1):
        _write_run(
            directory=directory, seed=seed, sweeps=sweeps, interventions=interventions, **kwargs
        )
    return directory


def _write_fixed_arms(*, root: Path) -> dict[str, Path]:
    """Three arms, two seeds each: `no-human` ends 4 TRASH, `two-way-ledge` 12,
    `skill-oracle` 14 (its one and only checkpoint). Distinct so a pairing bug cannot hide
    behind a tie."""
    return {
        "no-human": _write_arm(root=root, arm="no-human", sweeps=[(0, 0, 0), (4, 0, 0)]),
        "two-way-ledge": _write_arm(root=root, arm="two-way-ledge", sweeps=[(0, 0, 0), (12, 0, 0)]),
        "skill-oracle": _write_arm(root=root, arm="skill-oracle", sweeps=[(14, 0, 0)]),
    }


def _write_rate_sweep(*, root: Path) -> dict[int, Path]:
    """Two N points, each with a nonzero, distinct intervention count and a distinct final
    score -- N=1 rescues heavily and does better (8 TRASH), N=20 rescues lightly and does
    only a little better than the control (5 TRASH)."""
    return {
        1: _write_rate_point(root=root, n=1, sweeps=[(0, 0, 0), (8, 0, 0)], interventions=40),
        20: _write_rate_point(root=root, n=20, sweeps=[(0, 0, 0), (5, 0, 0)], interventions=3),
    }


def test_load_arms_requires_every_fixed_arm(*, tmp_path: Path) -> None:
    directories = _write_fixed_arms(root=tmp_path)
    del directories["two-way-ledge"]
    with pytest.raises(ValueError, match="missing arm"):
        HumanLadderCurves.load_arms(directories=directories)


def test_load_arms_reads_all_three(*, tmp_path: Path) -> None:
    directories = _write_fixed_arms(root=tmp_path)
    arms = HumanLadderCurves.load_arms(directories=directories)
    assert set(arms) == {"no-human", "two-way-ledge", "skill-oracle"}
    # skill-oracle never practises: one checkpoint only.
    assert len(arms["skill-oracle"][0]["transitions"]) == 1
    assert len(arms["no-human"][0]["transitions"]) == 2


def test_rate_sweep_needs_at_least_two_points(*, tmp_path: Path) -> None:
    directories = _write_rate_sweep(root=tmp_path)
    del directories[20]
    with pytest.raises(ValueError, match="at least two"):
        HumanLadderCurves.load_rate_sweep(directories=directories)


def test_rate_sweep_reads_both_points(*, tmp_path: Path) -> None:
    directories = _write_rate_sweep(root=tmp_path)
    rate_sweep = HumanLadderCurves.load_rate_sweep(directories=directories)
    assert set(rate_sweep) == {1, 20}


def test_a_free_practice_reset_anywhere_is_refused(*, tmp_path: Path) -> None:
    directory = tmp_path / "no-human" / "ees"
    _write_run(directory=directory, seed=0, sweeps=[(0, 0, 0)], num_practice_resets=1)
    with pytest.raises(ValueError, match="num_practice_resets"):
        HumanLadderCurves.load_run(directory=directory, label="no-human", expect_no_human=True)


def test_a_fixed_arm_with_an_intervention_is_refused(*, tmp_path: Path) -> None:
    directory = tmp_path / "no-human" / "ees"
    _write_run(directory=directory, seed=0, sweeps=[(0, 0, 0)], interventions=1)
    with pytest.raises(ValueError, match="no reachable human"):
        HumanLadderCurves.load_run(directory=directory, label="no-human", expect_no_human=True)


def test_a_rate_sweep_point_with_zero_interventions_is_refused(*, tmp_path: Path) -> None:
    directory = tmp_path / "rate-sweep-N20" / "ees"
    _write_run(directory=directory, seed=0, sweeps=[(0, 0, 0)], interventions=0)
    with pytest.raises(ValueError, match="zero human interventions"):
        HumanLadderCurves.load_run(directory=directory, label="N=20", expect_no_human=False)


def test_a_cost_that_is_not_the_v0_flat_rate_is_refused(*, tmp_path: Path) -> None:
    directory = tmp_path / "rate-sweep-N1" / "ees"
    _write_run(directory=directory, seed=0, sweeps=[(0, 0, 0)], interventions=5, human_cost=3.0)
    with pytest.raises(ValueError, match="summed_human_cost_recorded"):
        HumanLadderCurves.load_run(directory=directory, label="N=1", expect_no_human=False)


def test_overall_and_family_denominators_are_independent(*, tmp_path: Path) -> None:
    directories = _write_fixed_arms(root=tmp_path)
    arms = HumanLadderCurves.load_arms(directories=directories)
    overall_final = HumanLadderCurves.pooled_curve(run=arms["skill-oracle"], family=None)[-1]
    trash_final = HumanLadderCurves.pooled_curve(run=arms["skill-oracle"], family="TRASH")[-1]
    # skill-oracle: 14 TRASH, 0 RECYCLING, 0 EMPTY per seed, 2 seeds -> 28/60 overall,
    # 28/28 on TRASH alone.
    assert overall_final == (28, 60)
    assert trash_final == (28, 28)


def test_a_drifted_family_composition_is_refused(*, tmp_path: Path) -> None:
    directory = tmp_path / "no-human" / "ees"
    run = directory / "0"
    run.mkdir(parents=True)
    outcomes = _sweep(trash_solved=0, recycling_solved=0, empty_solved=0)
    # Corrupt one TRASH goal into something GoalFamilies does not recognise.
    outcomes[0]["goal"] = "SomeUnrelatedPredicate(x)"
    run.joinpath("stats.json").write_text(
        json.dumps({
            "breakdowns": [{"num_online_transitions": 0, "outcomes": outcomes}],
            "num_practice_resets": 0,
            "num_human_interventions_recorded": 0,
            "summed_human_cost_recorded": 0.0,
            "task_name": "default",
        })
    )
    with pytest.raises(ValueError, match="unrecognised goal"):
        HumanLadderCurves.load_run(directory=directory, label="no-human", expect_no_human=True)


def test_a_differing_number_of_checkpoints_is_refused(*, tmp_path: Path) -> None:
    directory = tmp_path / "no-human" / "ees"
    _write_run(directory=directory, seed=0, sweeps=[(0, 0, 0), (1, 0, 0)])
    _write_run(directory=directory, seed=1, sweeps=[(0, 0, 0)])
    run = HumanLadderCurves.load_run(directory=directory, label="no-human", expect_no_human=True)
    with pytest.raises(ValueError, match="disagree on the number"):
        HumanLadderCurves.transitions(run=run)


def test_rate_sweep_point_is_paired_against_no_human_within_a_seed(*, tmp_path: Path) -> None:
    arms = HumanLadderCurves.load_arms(directories=_write_fixed_arms(root=tmp_path))
    rate_sweep = HumanLadderCurves.load_rate_sweep(directories=_write_rate_sweep(root=tmp_path))

    # no-human ends at 4 TRASH per seed; N=1 ends at 8 -> +4 per seed, both seeds.
    differences = HumanLadderCurves.paired_final_differences(
        treatment=rate_sweep[1], control=arms["no-human"], family=None
    )
    assert differences == [4.0, 4.0]


def test_solves_per_rescue_prices_a_gap_by_what_it_cost(*, tmp_path: Path) -> None:
    arms = HumanLadderCurves.load_arms(directories=_write_fixed_arms(root=tmp_path))
    rate_sweep = HumanLadderCurves.load_rate_sweep(directories=_write_rate_sweep(root=tmp_path))

    # N=1: gap is +4 per seed (2 seeds) = +8 total, over 40 rescues per seed x 2 seeds =
    # 80 pooled rescues -> 0.1.
    ratio = HumanLadderCurves.solves_per_rescue(treatment=rate_sweep[1], control=arms["no-human"])
    assert ratio == pytest.approx(8 / 80)


def test_solves_per_rescue_is_none_when_never_rescued(*, tmp_path: Path) -> None:
    arms = HumanLadderCurves.load_arms(directories=_write_fixed_arms(root=tmp_path))
    # two-way-ledge is a fixed, no-human arm: it never rescues.
    ratio = HumanLadderCurves.solves_per_rescue(
        treatment=arms["two-way-ledge"], control=arms["no-human"]
    )
    assert ratio is None


def test_convergence_summary_needs_at_least_20_checkpoints(*, tmp_path: Path) -> None:
    directory = tmp_path / "no-human" / "ees"
    # 19 checkpoints: one short of the 10-vs-10 window the rule needs.
    _write_run(directory=directory, seed=0, sweeps=[(0, 0, 0)] * 19)
    run = HumanLadderCurves.load_run(directory=directory, label="no-human", expect_no_human=True)
    with pytest.raises(ValueError, match=">= 20 checkpoints"):
        HumanLadderCurves.convergence_summary(run=run)


def test_convergence_summary_reports_zero_delta_on_a_flat_curve(*, tmp_path: Path) -> None:
    directory = tmp_path / "no-human" / "ees"
    # 20 identical checkpoints (7 TRASH each, both seeds): last 10 vs. previous 10 must
    # agree exactly.
    _write_run(directory=directory, seed=0, sweeps=[(7, 0, 0)] * 20)
    _write_run(directory=directory, seed=1, sweeps=[(7, 0, 0)] * 20)
    run = HumanLadderCurves.load_run(directory=directory, label="no-human", expect_no_human=True)
    summary = HumanLadderCurves.convergence_summary(run=run)
    assert summary["prev10_fraction"] == pytest.approx(7 / 30)
    assert summary["last10_fraction"] == pytest.approx(7 / 30)
    assert summary["delta"] == pytest.approx(0.0)
    assert summary["final"] == (14, 60)


def test_convergence_summary_reports_positive_delta_on_a_climbing_curve(*, tmp_path: Path) -> None:
    directory = tmp_path / "no-human" / "ees"
    # 20 checkpoints, TRASH solved = index capped at 14 (both seeds identical): prev10
    # (indices 0-9) sums to 45 pooled-per-seed, last10 (indices 10-19, capped at 14 from
    # index 14 on) sums to 130 -- derived on paper, not recorded from a run of the code.
    trash_by_checkpoint = [min(index, 14) for index in range(20)]
    sweeps = [(trash, 0, 0) for trash in trash_by_checkpoint]
    _write_run(directory=directory, seed=0, sweeps=sweeps)
    _write_run(directory=directory, seed=1, sweeps=sweeps)
    run = HumanLadderCurves.load_run(directory=directory, label="no-human", expect_no_human=True)
    summary = HumanLadderCurves.convergence_summary(run=run)
    assert summary["prev10_fraction"] == pytest.approx(2 * 45 / 600)
    assert summary["last10_fraction"] == pytest.approx(2 * 130 / 600)
    assert summary["delta"] == pytest.approx(2 * 130 / 600 - 2 * 45 / 600)
    assert summary["delta"] > 0


def test_render_family_and_rate_sweep_write_files(*, tmp_path: Path) -> None:
    """Rendering just has to not crash and has to write a file -- the arithmetic feeding
    it is what the tests above pin."""
    arms = HumanLadderCurves.load_arms(directories=_write_fixed_arms(root=tmp_path))
    rate_sweep = HumanLadderCurves.load_rate_sweep(directories=_write_rate_sweep(root=tmp_path))

    output_dir = tmp_path / "figures"
    output_dir.mkdir()
    for family, name in ((None, "overall"), ("TRASH", "trash"), ("RECYCLING", "recycling")):
        output = output_dir / f"human-ladder-{name}.png"
        HumanLadderCurves.render_family(
            arms=arms, rate_sweep=rate_sweep, family=family, output=output, title="test"
        )
        assert output.exists()
    rate_output = output_dir / "human-ladder-rate-sweep.png"
    HumanLadderCurves.render_rate_sweep(
        arms=arms, rate_sweep=rate_sweep, output=rate_output, title="test"
    )
    assert rate_output.exists()
    trajectories_output = output_dir / "human-ladder-rate-sweep-trajectories.png"
    HumanLadderCurves.render_rate_sweep_trajectories(
        arms=arms, rate_sweep=rate_sweep, output=trajectories_output, title="test"
    )
    assert trajectories_output.exists()
    plt.close("all")
