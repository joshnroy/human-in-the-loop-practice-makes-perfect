"""Covers the two-arm reset-free comparison rule.

Built from synthetic curves with a *known* shape, because the property under test is
"does this rule reach the right verdict", and only a fabricated curve has a verdict known
independently of the code. A test that read the real sweep would assert whatever the run
happened to do.

The two that matter are `test_two_arms_that_differ_are_not_called_a_null_result` and
`test_two_arms_that_do_not_differ_are_called_a_null_result`: a rule that cannot tell
those apart is the same class of instrument failure this whole PR stack exists to fix --
#178's arms were identical and nothing noticed. Everything else here is plumbing.
"""

import json
import statistics
from pathlib import Path

import pytest

from analysis.practice_makes_perfect.paired_tests import PairedTests
from analysis.practice_makes_perfect.tossing3d_reset_free_arms import (
    NEVER,
    SCHEDULED,
    WINDOW,
    Tossing3DResetFree,
)
from hitl_pmp.sampler_draws import SAMPLER_DRAWS_FILENAME

NUM_SWEEPS = 101
NUM_TASKS = 10


def _write_run(
    *,
    run_dir: Path,
    solved_per_sweep: list[int],
    transitions_per_cycle: list[int] | None = None,
    practice_outcomes_per_cycle: list[dict[str, dict[str, int]]] | None = None,
) -> None:
    """One run's `stats.json`. `transitions_per_cycle` defaults to a steady 4 per cycle,
    i.e. a robot that never strands; pass zeros to model one that does.
    `practice_outcomes_per_cycle` defaults to empty per-cycle dicts -- fine for tests that
    never call `toss_transition_index`, which is the only reader of this field."""
    run_dir.mkdir(parents=True, exist_ok=True)
    steps = transitions_per_cycle or [4] * (len(solved_per_sweep) - 1)
    cumulative = [0]
    for step in steps:
        cumulative.append(cumulative[-1] + step)
    evaluations = [[cumulative[i], solved, NUM_TASKS] for i, solved in enumerate(solved_per_sweep)]
    outcomes = practice_outcomes_per_cycle or [{} for _ in steps]
    (run_dir / "stats.json").write_text(
        json.dumps({"evaluations": evaluations, "practice_outcomes_per_cycle": outcomes})
    )


def _outcomes_cycle(
    *, pick_attempts: int = 0, move_attempts: int = 0, toss_attempts: int = 0
) -> dict[str, dict[str, int]]:
    """One cycle's `practice_outcomes_per_cycle` entry, real `stats.json` shape, with only
    the fields `toss_transition_index` reads (`num_attempts`) populated for real -- the
    rest of the real schema (`num_random_attempts` etc.) is irrelevant to that derivation."""
    return {
        "Pick": {"num_attempts": pick_attempts, "num_successes": min(pick_attempts, 1)},
        "MoveToThrowPose": {"num_attempts": move_attempts, "num_successes": min(move_attempts, 1)},
        "Toss": {"num_attempts": toss_attempts, "num_successes": min(toss_attempts, 1)},
    }


def _write_draws(*, run_dir: Path, draws: list[dict[str, object]]) -> None:
    """One run's `sampler_draws.jsonl`, real shape but only the fields the stranding
    readers touch. Every line stands alone, exactly as the recorder flushes them."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / SAMPLER_DRAWS_FILENAME).write_text(
        "".join(
            json.dumps({
                "cycle": 0,
                "consultation": "uninformative",
                "success": False,
                "params": [],
                **draw,
            })
            + "\n"
            for draw in draws
        )
    )


def _features(*, gripper: float, cube_x: float, cube_z: float) -> dict[str, float]:
    """A whole Tossing3D-o1 feature vector, varying only the three numbers every test on
    this page turns on. The bin box is upstream's own inflated `blocks_goal_region`; the
    barrier sits at the sampled 1.30 every seed of the real sweep recorded."""
    return {
        "robot.pos_base_x": 0.0,
        "robot.pos_base_y": 0.0,
        "robot.pos_base_rot": 0.0,
        "robot.pos_gripper": gripper,
        "cube_0.x": cube_x,
        "cube_0.y": 0.0,
        "cube_0.z": cube_z,
        "cube_0.qx": 0.0,
        "cube_0.qy": 0.0,
        "cube_0.bb_z": 0.05,
        "cuboid_barrier.x": 1.3,
        "cuboid_barrier.y": 0.0,
        "cuboid_barrier.z": 0.1,
        "bin_0.x": 2.0,
        "bin_0.y": 0.0,
        "bin_0.z": 0.0,
        "bin_0.x_min": 1.85,
        "bin_0.y_min": -0.15,
        "bin_0.z_min": 0.0,
        "bin_0.x_max": 2.15,
        "bin_0.y_max": 0.15,
        "bin_0.z_max": 0.15,
    }


def _settled(*, level: int, jitter: list[int] | None = None) -> list[int]:
    """A curve that climbs early then sits at `level`, with optional per-sweep noise.

    The noise is not decoration: Tossing3D's real per-seed score moves by several tasks
    between adjacent sweeps with no learning event, and a rule that mistook that for an
    arm difference would be useless."""
    curve = [min(level, i) for i in range(15)] + [level] * (NUM_SWEEPS - 15)
    for offset, delta in enumerate(jitter or []):
        curve[-(offset + 1)] = max(0, min(NUM_TASKS, curve[-(offset + 1)] + delta))
    return curve


def _write_arms(*, root: Path, scheduled_levels: list[int], never_levels: list[int]) -> None:
    for seed, level in enumerate(scheduled_levels):
        _write_run(
            run_dir=root / SCHEDULED / "ees" / str(seed), solved_per_sweep=_settled(level=level)
        )
    for seed, level in enumerate(never_levels):
        _write_run(run_dir=root / NEVER / "ees" / str(seed), solved_per_sweep=_settled(level=level))


def test_load_arms_reads_both_arms_keyed_by_seed(*, tmp_path: Path) -> None:
    _write_arms(root=tmp_path, scheduled_levels=[8, 7], never_levels=[8, 7])
    arms = Tossing3DResetFree.load_arms(results_root=tmp_path)
    assert sorted(arms) == sorted([NEVER, SCHEDULED])
    assert sorted(arms[SCHEDULED]) == [0, 1]
    assert len(arms[SCHEDULED][0]) == NUM_SWEEPS


def test_late_scores_average_the_last_window_not_the_final_sweep(*, tmp_path: Path) -> None:
    """The whole rule rests on scoring a window rather than a sweep -- #178 measured a
    final sweep 5 tasks below its own late window on the same data."""
    curve = _settled(level=8, jitter=[-8])
    _write_run(run_dir=tmp_path / SCHEDULED / "ees" / "0", solved_per_sweep=curve)
    _write_run(run_dir=tmp_path / NEVER / "ees" / "0", solved_per_sweep=curve)
    arms = Tossing3DResetFree.load_arms(results_root=tmp_path)
    scores = Tossing3DResetFree.late_scores(curves=arms[SCHEDULED])
    assert curve[-1] == 0
    # Nine sweeps at 8 and one at 0.
    assert scores[0] == statistics.fmean([8] * (WINDOW - 1) + [0])


def test_two_arms_that_do_not_differ_are_called_a_null_result(*, tmp_path: Path) -> None:
    """The honest outcome when the manipulation does nothing -- and the one that must
    still come with an MDE, or it cannot be told apart from no power."""
    levels = [8, 7, 9, 8, 7, 8, 9, 7, 8, 8]
    _write_arms(root=tmp_path, scheduled_levels=levels, never_levels=levels)
    arms = Tossing3DResetFree.load_arms(results_root=tmp_path)
    late = {arm: Tossing3DResetFree.late_scores(curves=arms[arm]) for arm in arms}
    seeds = Tossing3DResetFree.shared_seeds(arms=arms)
    differences = [late[NEVER][s] - late[SCHEDULED][s] for s in seeds]

    assert differences == [0.0] * len(seeds)
    assert PairedTests.sign_flip(differences=differences).p_value == 1.0


def test_two_arms_that_differ_are_not_called_a_null_result(*, tmp_path: Path) -> None:
    """The failure this PR stack exists to prevent, in reverse: a real gap on every seed
    must come out significant, or the instrument cannot see the thing it is for."""
    scheduled = [8, 7, 9, 8, 7, 8, 9, 7, 8, 8]
    _write_arms(
        root=tmp_path,
        scheduled_levels=scheduled,
        never_levels=[level - 3 for level in scheduled],
    )
    arms = Tossing3DResetFree.load_arms(results_root=tmp_path)
    late = {arm: Tossing3DResetFree.late_scores(curves=arms[arm]) for arm in arms}
    seeds = Tossing3DResetFree.shared_seeds(arms=arms)
    differences = [late[NEVER][s] - late[SCHEDULED][s] for s in seeds]

    assert differences == [-3.0] * len(seeds)
    assert PairedTests.sign_flip(differences=differences).p_value < 0.05


def test_pairing_uses_only_seeds_present_in_both_arms(*, tmp_path: Path) -> None:
    """A lost run must shrink the pairing, not silently pair a seed against nothing."""
    _write_arms(root=tmp_path, scheduled_levels=[8, 7, 9], never_levels=[8, 7])
    arms = Tossing3DResetFree.load_arms(results_root=tmp_path)
    assert Tossing3DResetFree.shared_seeds(arms=arms) == [0, 1]


def test_pooled_reports_a_count_over_the_seeds_denominator(*, tmp_path: Path) -> None:
    """`x/y`, never a bare percentage -- and `x` stays a float because each seed's
    contribution is a window mean, so rounding here would hide that."""
    _write_arms(root=tmp_path, scheduled_levels=[8, 6], never_levels=[8, 6])
    arms = Tossing3DResetFree.load_arms(results_root=tmp_path)
    x, y = Tossing3DResetFree.pooled(
        scores=Tossing3DResetFree.late_scores(curves=arms[SCHEDULED]), num_total=NUM_TASKS
    )
    assert (x, y) == (14.0, 20)


def test_report_says_so_rather_than_raising_when_an_arm_is_missing(
    *, tmp_path: Path, capsys
) -> None:
    """A half-finished sweep must not be reported as a result."""
    _write_run(run_dir=tmp_path / SCHEDULED / "ees" / "0", solved_per_sweep=_settled(level=8))
    Tossing3DResetFree.report(results_root=tmp_path)
    assert "No completed runs" in capsys.readouterr().out


def test_every_figure_is_written(*, tmp_path: Path) -> None:
    """A quantitative result needs a figure, so the figures are part of the deliverable
    rather than a manual step."""
    _write_arms(root=tmp_path, scheduled_levels=[8, 7, 9], never_levels=[6, 5, 7])
    arms = Tossing3DResetFree.load_arms(results_root=tmp_path)
    outputs = [tmp_path / f"{name}.png" for name in ("curves", "paired", "practice")]

    Tossing3DResetFree.render_curves(arms=arms, output=outputs[0])
    Tossing3DResetFree.render_paired(arms=arms, output=outputs[1])
    Tossing3DResetFree.render_practice(arms=arms, output=outputs[2])

    assert all(output.stat().st_size > 0 for output in outputs)


def test_the_committed_experiment_log_layout_loads_too(*, tmp_path: Path) -> None:
    """`run_sweep` writes `<arm>/<method>/<seed>/`; a committed `docs/experiment-logs/`
    tree is `<arm>/<seed>/`. A loader fixed to one depth finds nothing under the other,
    and `report` then prints "No completed runs" and exits 0 -- a silent wrong answer
    rather than a failure, which is the same shape of defect this stack exists to fix."""
    for arm in (SCHEDULED, NEVER):
        for seed in (0, 1):
            _write_run(run_dir=tmp_path / arm / str(seed), solved_per_sweep=_settled(level=8))

    arms = Tossing3DResetFree.load_arms(results_root=tmp_path)

    assert sorted(arms) == sorted([NEVER, SCHEDULED])
    assert Tossing3DResetFree.shared_seeds(arms=arms) == [0, 1]


def test_seeds_are_not_collided_across_arms(*, tmp_path: Path) -> None:
    """Keying on the containing directory alone would fold `scheduled/0` and `never/0`
    into one entry. That collision is invisible for exactly as long as the two arms
    agree -- the condition that produced this experiment."""
    _write_arms(root=tmp_path, scheduled_levels=[9, 9], never_levels=[3, 3])
    arms = Tossing3DResetFree.load_arms(results_root=tmp_path)
    late = {arm: Tossing3DResetFree.late_scores(curves=arms[arm]) for arm in arms}

    assert late[SCHEDULED][0] == 9.0
    assert late[NEVER][0] == 3.0


def test_transitions_per_cycle_is_the_gap_between_consecutive_sweeps(*, tmp_path: Path) -> None:
    """Sweep 0 happens before any practice, so cycle i's cost is the difference between
    consecutive sweeps' `num_online_transitions`."""
    steps = [5, 3, 0, 0]
    _write_run(
        run_dir=tmp_path / SCHEDULED / "ees" / "0",
        solved_per_sweep=[0] * (len(steps) + 1),
        transitions_per_cycle=steps,
    )
    arms = Tossing3DResetFree.load_arms(results_root=tmp_path)

    assert Tossing3DResetFree.transitions_per_cycle(evaluations=arms[SCHEDULED][0]) == steps


def test_a_robot_that_stops_acting_is_reported_as_stranded() -> None:
    """The measurement that separates "learned less" from "practised less". Without it,
    an arm that stopped acting after cycle 2 and an arm that practised badly for 100
    cycles produce the same score gap and the same conclusion."""
    assert Tossing3DResetFree.stranding_onset(transitions=[4, 4, 0, 0, 0]) == 2
    assert Tossing3DResetFree.stranding_onset(transitions=[0, 0, 0]) == 0


def test_a_single_idle_cycle_is_not_stranding() -> None:
    """Terminal-from-here, not "the first gap" -- the same definition
    `pickup_weight_stranding.py` uses, so the two experiments read side by side. A run
    that pauses and resumes was never stranded, and calling it stranded would promote
    ordinary exploration noise into the effect being claimed."""
    assert Tossing3DResetFree.stranding_onset(transitions=[4, 0, 4, 4]) is None
    assert Tossing3DResetFree.stranding_onset(transitions=[4, 0, 0, 4]) is None


# --- toss_transition_index: the figure annotation's whole load-bearing claim ---
#
# Josh's constraint: derive the toss and the stranding point from the per-cycle
# transition/skill-attempt record, never from where the score curve visually goes flat.
# These tests exercise exactly that derivation against a fabricated run whose "real"
# answer is known by construction (7 -- one Pick, five MoveToThrowPose attempts, one
# Toss, matching what seed 0 of the real never-arm sweep actually recorded).


def test_toss_transition_index_reads_the_last_active_cycles_attempt_counts(
    *, tmp_path: Path
) -> None:
    """The number the whole annotation rests on. `stats.json` here is built to mirror
    seed 0 of the real never-arm sweep exactly: Pick (1 attempt) + MoveToThrowPose
    (5 attempts) + Toss (1 attempt) = 7 transitions in cycle 0, then silence."""
    run_dir = tmp_path / NEVER / "ees" / "0"
    _write_run(
        run_dir=run_dir,
        solved_per_sweep=[0, 0, 0],
        transitions_per_cycle=[7, 0],
        practice_outcomes_per_cycle=[
            _outcomes_cycle(pick_attempts=1, move_attempts=5, toss_attempts=1),
            _outcomes_cycle(),
        ],
    )
    arms = Tossing3DResetFree.load_arms(results_root=tmp_path)
    outcomes = Tossing3DResetFree.load_practice_outcomes(results_root=tmp_path)

    index = Tossing3DResetFree.toss_transition_index(
        evaluations=arms[NEVER][0], outcomes=outcomes[NEVER][0]
    )

    assert index == 7


def test_toss_transition_index_raises_if_the_run_never_stranded(*, tmp_path: Path) -> None:
    """A run that kept practising has no terminal toss to locate -- annotating one would
    invent a stranding event that the record does not contain."""
    run_dir = tmp_path / NEVER / "ees" / "0"
    _write_run(
        run_dir=run_dir,
        solved_per_sweep=[0, 0, 0, 0],
        transitions_per_cycle=[7, 5, 3],
        practice_outcomes_per_cycle=[_outcomes_cycle(toss_attempts=1)] * 3,
    )
    arms = Tossing3DResetFree.load_arms(results_root=tmp_path)
    outcomes = Tossing3DResetFree.load_practice_outcomes(results_root=tmp_path)

    with pytest.raises(ValueError, match="never stranded"):
        Tossing3DResetFree.toss_transition_index(
            evaluations=arms[NEVER][0], outcomes=outcomes[NEVER][0]
        )


def test_toss_transition_index_raises_if_the_last_active_cycle_never_attempted_a_toss(
    *, tmp_path: Path
) -> None:
    """Guards against misattributing a different kind of stall -- e.g. MoveToThrowPose
    exhausting its budget without ever succeeding -- to a toss that never happened."""
    run_dir = tmp_path / NEVER / "ees" / "0"
    _write_run(
        run_dir=run_dir,
        solved_per_sweep=[0, 0, 0],
        transitions_per_cycle=[5, 0],
        practice_outcomes_per_cycle=[
            _outcomes_cycle(pick_attempts=1, move_attempts=4, toss_attempts=0),
            _outcomes_cycle(),
        ],
    )
    arms = Tossing3DResetFree.load_arms(results_root=tmp_path)
    outcomes = Tossing3DResetFree.load_practice_outcomes(results_root=tmp_path)

    with pytest.raises(ValueError, match="no Toss attempt"):
        Tossing3DResetFree.toss_transition_index(
            evaluations=arms[NEVER][0], outcomes=outcomes[NEVER][0]
        )


def test_load_practice_outcomes_reads_both_directory_layouts(*, tmp_path: Path) -> None:
    """Same two-layout hazard as `load_arms` -- a loader fixed to one depth would find
    nothing under the committed `docs/experiment-logs/` tree."""
    _write_run(
        run_dir=tmp_path / NEVER / "0",
        solved_per_sweep=[0, 0],
        transitions_per_cycle=[3],
        practice_outcomes_per_cycle=[_outcomes_cycle(toss_attempts=1)],
    )
    outcomes = Tossing3DResetFree.load_practice_outcomes(results_root=tmp_path)
    assert outcomes[NEVER][0][0]["Toss"]["num_attempts"] == 1


# --- the stranding *route*, and the abstract state it leaves behind ---
#
# `toss_transition_index` above assumes the last active cycle contained a Toss. At the
# 2026-08-08 pins that held on 10/10 never-arm seeds, so the assumption was invisible.
# It is not a safe assumption in general, and the guard raising is how that was found --
# so the route is measured rather than assumed, and the resulting abstract state is read
# off the run's own recorded features through the domain's own classifiers.


def test_a_last_active_cycle_without_a_toss_is_not_called_a_toss(*, tmp_path: Path) -> None:
    """The distinction `toss_transition_index` can only express by raising. A seed whose
    only practice period ended on a failed `Pick` stranded for a different reason than one
    that threw the cube past the barrier, and collapsing the two would restate a
    mechanism the record does not support."""
    _write_run(
        run_dir=tmp_path / NEVER / "ees" / "0",
        solved_per_sweep=[0, 0, 0],
        transitions_per_cycle=[7, 0],
        practice_outcomes_per_cycle=[
            _outcomes_cycle(pick_attempts=1, move_attempts=5, toss_attempts=1),
            _outcomes_cycle(),
        ],
    )
    _write_run(
        run_dir=tmp_path / NEVER / "ees" / "1",
        solved_per_sweep=[0, 0, 0],
        transitions_per_cycle=[1, 0],
        practice_outcomes_per_cycle=[_outcomes_cycle(pick_attempts=1), _outcomes_cycle()],
    )
    arms = Tossing3DResetFree.load_arms(results_root=tmp_path)
    outcomes = Tossing3DResetFree.load_practice_outcomes(results_root=tmp_path)

    assert Tossing3DResetFree.ended_on_a_toss(
        evaluations=arms[NEVER][0], outcomes=outcomes[NEVER][0]
    )
    assert not Tossing3DResetFree.ended_on_a_toss(
        evaluations=arms[NEVER][1], outcomes=outcomes[NEVER][1]
    )


def test_last_practice_action_reports_the_skills_of_the_last_active_cycle(
    *, tmp_path: Path
) -> None:
    """`render_practice`'s annotation, generalised off `Toss`. The transition index is the
    same cumulative count `toss_transition_index` returns; the skill set is whatever that
    cycle actually attempted, so a seed that never threw is labelled for what it did do."""
    _write_run(
        run_dir=tmp_path / NEVER / "ees" / "0",
        solved_per_sweep=[0, 0, 0],
        transitions_per_cycle=[2, 0],
        practice_outcomes_per_cycle=[
            _outcomes_cycle(pick_attempts=1, move_attempts=1),
            _outcomes_cycle(),
        ],
    )
    arms = Tossing3DResetFree.load_arms(results_root=tmp_path)
    outcomes = Tossing3DResetFree.load_practice_outcomes(results_root=tmp_path)

    skills, index = Tossing3DResetFree.last_practice_action(
        evaluations=arms[NEVER][0], outcomes=outcomes[NEVER][0]
    )

    assert index == 2
    assert skills == ("MoveToThrowPose", "Pick")


def test_load_final_practice_features_keeps_the_most_recent_value_of_each_feature(
    *, tmp_path: Path
) -> None:
    """A draw records only the objects its own ground skill binds, so `MoveToThrowPose`
    carries no barrier. Merging forward is exact here rather than convenient: the barrier
    and the bin are immovable within an episode, and a reset-free run is one episode, so
    the newest recorded value of every feature *is* the final state."""
    _write_draws(
        run_dir=tmp_path / NEVER / "ees" / "0",
        draws=[
            {"skill": "Pick", "achieved": {"cuboid_barrier.x": 1.3, "cube_0.x": 0.4}},
            {"skill": "MoveToThrowPose", "achieved": {"cube_0.x": 0.9}},
        ],
    )
    features = Tossing3DResetFree.load_final_practice_features(results_root=tmp_path)

    assert features[NEVER][0] == {"cuboid_barrier.x": 1.3, "cube_0.x": 0.9}


def test_a_state_the_robot_can_still_pick_from_is_not_called_stranded() -> None:
    """The positive control. Without it a classifier that returned "nothing applicable"
    unconditionally would pass every other test on this page."""
    applicable = Tossing3DResetFree.applicable_skills(
        features=_features(gripper=0.0, cube_x=0.4, cube_z=0.025)
    )
    assert applicable["Pick"]


def test_a_cube_thrown_past_the_barrier_leaves_no_skill_applicable() -> None:
    """The published mechanism: `Toss` puts the cube past a wall the base cannot cross,
    so `Reachable` is gone and `Pick` -- the only skill that starts from an empty hand --
    can never fire again."""
    applicable = Tossing3DResetFree.applicable_skills(
        features=_features(gripper=0.0, cube_x=1.8, cube_z=0.025)
    )
    assert applicable == {"Pick": False, "MoveToThrowPose": False, "Toss": False}


def test_a_closed_gripper_holding_nothing_leaves_no_skill_applicable() -> None:
    """The *other* way to strand this domain, and the one the published mechanism does not
    cover: the cube is back on the floor on the robot's own side of the barrier, so it is
    `Reachable`, but the gripper stayed shut. `HandEmpty` is false, so `Pick` cannot fire;
    `Holding` is false, so neither can `MoveToThrowPose` or `Toss`."""
    features = _features(gripper=1.0, cube_x=0.73, cube_z=0.025)
    held = Tossing3DResetFree.held_predicates(features=features)

    assert held["Reachable"]
    assert not held["HandEmpty"]
    assert not held["Holding"]
    assert Tossing3DResetFree.applicable_skills(features=features) == {
        "Pick": False,
        "MoveToThrowPose": False,
        "Toss": False,
    }


def test_render_stranding_is_written(*, tmp_path: Path) -> None:
    """The figure carrying the route split, so it is part of the deliverable rather than
    a manual step."""
    for seed, toss_attempts in enumerate([1, 0]):
        _write_run(
            run_dir=tmp_path / NEVER / "ees" / str(seed),
            solved_per_sweep=[0, 0, 0],
            transitions_per_cycle=[3, 0],
            practice_outcomes_per_cycle=[
                _outcomes_cycle(pick_attempts=1, move_attempts=1, toss_attempts=toss_attempts),
                _outcomes_cycle(),
            ],
        )
        _write_draws(
            run_dir=tmp_path / NEVER / "ees" / str(seed),
            draws=[{"skill": "Pick", "achieved": _features(gripper=1.0, cube_x=0.7, cube_z=0.025)}],
        )
        _write_run(
            run_dir=tmp_path / SCHEDULED / "ees" / str(seed),
            solved_per_sweep=[0, 0, 0],
            transitions_per_cycle=[3, 3],
        )
    arms = Tossing3DResetFree.load_arms(results_root=tmp_path)
    outcomes = Tossing3DResetFree.load_practice_outcomes(results_root=tmp_path)
    features = Tossing3DResetFree.load_final_practice_features(results_root=tmp_path)
    output = tmp_path / "stranding.png"

    Tossing3DResetFree.render_stranding(
        arms=arms, outcomes=outcomes, features=features, output=output
    )

    assert output.stat().st_size > 0


def test_render_practice_annotates_a_seed_that_never_threw(*, tmp_path: Path) -> None:
    """The regression this whole section exists for: at the new KINDER pins 4/10 never-arm
    seeds stranded without ever attempting a `Toss`, and the figure must describe them
    rather than raise."""
    for seed, toss_attempts in enumerate([1, 0]):
        _write_run(
            run_dir=tmp_path / NEVER / "ees" / str(seed),
            solved_per_sweep=[0, 0, 0],
            transitions_per_cycle=[3, 0],
            practice_outcomes_per_cycle=[
                _outcomes_cycle(pick_attempts=1, move_attempts=1, toss_attempts=toss_attempts),
                _outcomes_cycle(),
            ],
        )
        _write_run(
            run_dir=tmp_path / SCHEDULED / "ees" / str(seed),
            solved_per_sweep=[0, 0, 0],
            transitions_per_cycle=[3, 3],
        )
    arms = Tossing3DResetFree.load_arms(results_root=tmp_path)
    outcomes = Tossing3DResetFree.load_practice_outcomes(results_root=tmp_path)
    output = tmp_path / "practice.png"

    Tossing3DResetFree.render_practice(arms=arms, output=output, outcomes=outcomes, annotate_seed=1)

    assert output.stat().st_size > 0


def test_render_practice_with_outcomes_annotates_without_raising(*, tmp_path: Path) -> None:
    """The figure Josh asked for: an annotated `render_practice` must still just work,
    given real-shaped `outcomes`, and still write a figure."""
    for seed, (transitions, _toss_index) in enumerate([([7, 0, 0], 7), ([3, 0, 0], 3)]):
        move_attempts = transitions[0] - 2  # 1 pick + 1 toss + the rest MoveToThrowPose
        _write_run(
            run_dir=tmp_path / NEVER / "ees" / str(seed),
            solved_per_sweep=[0] * len(transitions) + [0],
            transitions_per_cycle=transitions,
            practice_outcomes_per_cycle=[
                _outcomes_cycle(pick_attempts=1, move_attempts=move_attempts, toss_attempts=1),
                *[_outcomes_cycle() for _ in transitions[1:]],
            ],
        )
        _write_run(
            run_dir=tmp_path / SCHEDULED / "ees" / str(seed),
            solved_per_sweep=[0] * len(transitions) + [0],
            transitions_per_cycle=[4] * len(transitions),
        )
    arms = Tossing3DResetFree.load_arms(results_root=tmp_path)
    outcomes = Tossing3DResetFree.load_practice_outcomes(results_root=tmp_path)
    output = tmp_path / "practice.png"

    Tossing3DResetFree.render_practice(arms=arms, output=output, outcomes=outcomes, annotate_seed=0)

    assert output.stat().st_size > 0
    # The figure's annotation must match what the fixture actually encodes, not merely
    # fail to raise.
    for seed, (_transitions, expected_index) in enumerate([([7, 0, 0], 7), ([3, 0, 0], 3)]):
        assert (
            Tossing3DResetFree.toss_transition_index(
                evaluations=arms[NEVER][seed], outcomes=outcomes[NEVER][seed]
            )
            == expected_index
        )
