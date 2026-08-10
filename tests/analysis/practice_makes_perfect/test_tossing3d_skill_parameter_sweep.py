"""Tests for the Tossing3D skill-parameter-sweep analysis.

The module under test reads a flat JSON grid of `(param..., seed)` cells and aggregates
it into success rates and physical-diagnostic curves. What is worth pinning here is the
aggregation arithmetic and, especially, `measured_collision_boundary` -- the one function
that turns a column of numbers into a single claimed number ("the sweep no longer
collides past this standoff"), which is exactly the kind of thing that is easy to get
backwards (scanning the wrong direction, or accepting a lucky low reading inside an
otherwise-noisy region).

No test here builds a real 1872- or 732-row grid; small hand-built fixtures with a mix
of all-success, all-failure and mixed cells exercise every branch without needing the
real sweep output on disk.
"""

from pathlib import Path

import pytest

from analysis.practice_makes_perfect.tossing3d_skill_parameter_sweep import (
    DISPLACEMENT_NOISE_FLOOR_M,
    Tossing3DSkillParameterSweep,
)


def _pick_row(
    *,
    distance: float,
    rotation: float,
    seed: int,
    success: bool,
    pick_error: str | None = None,
) -> dict:
    return {
        "distance": distance,
        "rotation": rotation,
        "seed": seed,
        "success": success,
        "pick_error": pick_error,
        "pick_terminated": pick_error is None,
        "pick_steps": 60,
        "robot_base_x": 0.3,
        "robot_base_y": 0.1,
        "cube_x": 0.4,
        "cube_y": 0.05,
        "cube_z": 0.585,
    }


def _move_row(
    *,
    standoff: float,
    seed: int,
    success: bool,
    pick_success: bool = True,
    bin_x: float = 2.0,
    bin_x_initial: float = 2.0,
    bin_y: float = 0.0,
    bin_y_initial: float = 0.0,
    barrier_x: float = 1.3,
    barrier_x_initial: float = 1.3,
) -> dict:
    return {
        "standoff": standoff,
        "seed": seed,
        "success": success,
        "pick_success": pick_success,
        "pick_error": None if pick_success else "grasp failed",
        "move_error": None,
        "move_terminated": True,
        "robot_base_x": bin_x - standoff,
        "robot_base_y": 0.0,
        "lateral_offset": 0.0,
        "bin_x_initial": bin_x_initial,
        "bin_y_initial": bin_y_initial,
        "bin_x": bin_x,
        "bin_y": bin_y,
        "barrier_x_initial": barrier_x_initial,
        "barrier_x": barrier_x,
    }


# ---- Pick aggregation ----------------------------------------------------------------


def test_pick_grid_axes_are_sorted_and_deduplicated() -> None:
    results = [
        _pick_row(distance=0.6, rotation=0.5, seed=0, success=True),
        _pick_row(distance=0.5, rotation=-0.5, seed=0, success=True),
        _pick_row(distance=0.5, rotation=0.5, seed=1, success=True),
    ]
    distances, rotations = Tossing3DSkillParameterSweep.pick_grid_axes(results=results)
    assert distances == [0.5, 0.6]
    assert rotations == [-0.5, 0.5]


def test_pick_cell_counts_pools_only_matching_seeds() -> None:
    results = [
        _pick_row(distance=0.5, rotation=0.0, seed=0, success=True),
        _pick_row(distance=0.5, rotation=0.0, seed=1, success=False),
        _pick_row(distance=0.6, rotation=0.0, seed=0, success=True),
    ]
    counts = Tossing3DSkillParameterSweep.pick_cell_counts(results=results)
    assert counts[(0.5, 0.0)] == (1, 2)
    assert counts[(0.6, 0.0)] == (1, 1)


def test_pick_overall_sums_every_row() -> None:
    results = [
        _pick_row(distance=0.5, rotation=0.0, seed=0, success=True),
        _pick_row(distance=0.5, rotation=0.0, seed=1, success=False),
        _pick_row(distance=0.5, rotation=0.0, seed=2, success=True),
    ]
    assert Tossing3DSkillParameterSweep.pick_overall(results=results) == (2, 3)


def test_pick_seed_breakdown_isolates_a_pathological_seed() -> None:
    """A single seed failing every cell must be visible as one bad seed, not smeared
    into an overall rate that looks like a uniformly mediocre parameter space."""
    results = [
        _pick_row(distance=0.5, rotation=0.0, seed=0, success=True),
        _pick_row(distance=0.6, rotation=0.5, seed=0, success=True),
        _pick_row(distance=0.5, rotation=0.0, seed=1, success=False),
        _pick_row(distance=0.6, rotation=0.5, seed=1, success=False),
    ]
    breakdown = Tossing3DSkillParameterSweep.pick_seed_breakdown(results=results)
    assert breakdown[0] == (2, 2)
    assert breakdown[1] == (0, 2)


def test_pick_rotation_marginal_pools_over_distance_and_seed() -> None:
    results = [
        _pick_row(distance=0.5, rotation=0.7, seed=0, success=False),
        _pick_row(distance=0.6, rotation=0.7, seed=0, success=False),
        _pick_row(distance=0.5, rotation=0.0, seed=0, success=True),
    ]
    marginal = Tossing3DSkillParameterSweep.pick_rotation_marginal(results=results)
    assert marginal[0.7] == (0, 2)
    assert marginal[0.0] == (1, 1)


# ---- MoveToThrowPose aggregation ------------------------------------------------------


def test_move_cell_counts_and_overall() -> None:
    results = [
        _move_row(standoff=1.2, seed=0, success=True),
        _move_row(standoff=1.2, seed=1, success=True),
        _move_row(standoff=0.6, seed=0, success=False),
    ]
    counts = Tossing3DSkillParameterSweep.move_cell_counts(results=results)
    assert counts[1.2] == (2, 2)
    assert counts[0.6] == (0, 1)
    assert Tossing3DSkillParameterSweep.move_overall(results=results) == (2, 3)


def test_move_pick_confound_count_flags_failed_oracle_picks() -> None:
    results = [
        _move_row(standoff=1.2, seed=0, success=True, pick_success=True),
        _move_row(standoff=1.2, seed=1, success=False, pick_success=False),
    ]
    failed, total = Tossing3DSkillParameterSweep.move_pick_confound_count(results=results)
    assert (failed, total) == (1, 2)


def test_move_mean_bin_displacement_is_zero_when_bin_never_moves() -> None:
    results = [
        _move_row(standoff=1.2, seed=0, success=True, bin_x=2.0, bin_x_initial=2.0),
        _move_row(standoff=1.2, seed=1, success=True, bin_x=2.0, bin_x_initial=2.0),
    ]
    displacement = Tossing3DSkillParameterSweep.move_mean_bin_displacement(results=results)
    assert displacement[1.2] == 0.0


def test_move_mean_bin_displacement_averages_a_genuine_shove() -> None:
    results = [
        _move_row(standoff=0.35, seed=0, success=False, bin_x=2.05, bin_x_initial=2.0),
        _move_row(standoff=0.35, seed=1, success=False, bin_x=2.03, bin_x_initial=2.0),
    ]
    displacement = Tossing3DSkillParameterSweep.move_mean_bin_displacement(results=results)
    assert displacement[0.35] == pytest.approx(0.04, abs=1e-9)


def test_move_mean_barrier_displacement_is_one_dimensional() -> None:
    """Unlike the bin, only `barrier.x` ever moves in this domain -- there is no
    `barrier_y`/`barrier_y_initial` field to hypot against."""
    results = [
        _move_row(standoff=0.35, seed=0, success=False, barrier_x=1.5, barrier_x_initial=1.3),
    ]
    displacement = Tossing3DSkillParameterSweep.move_mean_barrier_displacement(results=results)
    assert displacement[0.35] == pytest.approx(0.2, abs=1e-9)


# ---- measured_collision_boundary -------------------------------------------------------


def test_measured_collision_boundary_finds_the_clean_transition() -> None:
    displacement = {0.35: 0.20, 0.65: 0.10, 0.95: 0.01, 1.05: 0.001, 1.35: 0.0}
    boundary = Tossing3DSkillParameterSweep.measured_collision_boundary(
        displacement_by_standoff=displacement
    )
    assert boundary == 1.05


def test_measured_collision_boundary_none_when_nothing_is_ever_clean() -> None:
    displacement = {0.35: 0.20, 0.65: 0.10, 0.95: 0.05}
    assert (
        Tossing3DSkillParameterSweep.measured_collision_boundary(
            displacement_by_standoff=displacement
        )
        is None
    )


def test_measured_collision_boundary_none_when_a_larger_standoff_recollides() -> None:
    """A lucky low reading inside an otherwise-noisy region must not be mistaken for the
    boundary -- something larger than the naive first-clean-reading has to re-cross the
    noise floor to prove that reading was not representative."""
    displacement = {
        0.35: 0.20,
        0.65: 0.001,  # a lucky clean reading, surrounded by collisions
        0.95: 0.15,
        1.25: 0.0,
    }
    assert (
        Tossing3DSkillParameterSweep.measured_collision_boundary(
            displacement_by_standoff=displacement
        )
        is None
    )


def test_measured_collision_boundary_empty_input_is_none() -> None:
    assert (
        Tossing3DSkillParameterSweep.measured_collision_boundary(displacement_by_standoff={})
        is None
    )


def test_displacement_noise_floor_is_below_measured_collision_shoves_above_typical_noise() -> None:
    """A sanity bound on the constant itself: it must sit strictly between float/episode
    noise (millimetre scale) and the shoves this sweep actually measures (centimetre
    scale), or the boundary function above would mislabel one as the other."""
    assert 0.0 < DISPLACEMENT_NOISE_FLOOR_M < 0.02


# ---- figures render without raising ----------------------------------------------------


def test_figure_pick_renders_from_a_minimal_grid() -> None:
    results = [
        _pick_row(distance=d, rotation=r, seed=s, success=(s % 2 == 0))
        for d in (0.5, 0.55, 0.6)
        for r in (-0.5, 0.0, 0.5)
        for s in range(3)
    ]
    figure = Tossing3DSkillParameterSweep.figure_pick(results=results)
    assert figure is not None


def test_figure_move_renders_from_a_minimal_grid() -> None:
    results = [
        _move_row(standoff=s, seed=seed, success=(1.15 <= s <= 1.375))
        for s in (0.6, 1.0, 1.2, 1.6)
        for seed in range(3)
    ]
    figure = Tossing3DSkillParameterSweep.figure_move(results=results)
    assert figure is not None


def test_plot_writes_both_png_files(*, tmp_path: Path) -> None:
    pick_results = [
        _pick_row(distance=d, rotation=r, seed=s, success=True)
        for d in (0.5, 0.6)
        for r in (-0.5, 0.5)
        for s in range(2)
    ]
    move_results = [
        _move_row(standoff=s, seed=seed, success=(s == 1.2))
        for s in (0.6, 1.2)
        for seed in range(2)
    ]
    pick_output = tmp_path / "pick.png"
    move_output = tmp_path / "move.png"
    Tossing3DSkillParameterSweep.plot(
        pick_results=pick_results,
        move_results=move_results,
        pick_output=pick_output,
        move_output=move_output,
    )
    assert pick_output.is_file()
    assert move_output.is_file()


# ---- print_report is total over a small realistic grid --------------------------------


def test_print_report_does_not_raise(*, capsys: pytest.CaptureFixture[str]) -> None:
    pick_results = [
        _pick_row(distance=d, rotation=r, seed=s, success=(s == 0))
        for d in (0.5, 0.6)
        for r in (-0.5, 0.5)
        for s in range(2)
    ]
    move_results = [
        _move_row(standoff=s, seed=seed, success=(s == 1.2), bin_x=2.0 if s >= 1.0 else 2.05)
        for s in (0.6, 1.2)
        for seed in range(2)
    ]
    Tossing3DSkillParameterSweep.print_report(pick_results=pick_results, move_results=move_results)
    captured = capsys.readouterr()
    assert "Pick success box" in captured.out
    assert "MoveToThrowPose success box" in captured.out
