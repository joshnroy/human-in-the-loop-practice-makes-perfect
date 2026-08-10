"""Everything in `scripts/tossing3d_skill_parameter_sweep.py` that can be pinned without
a simulator: argument parsing, the grid construction, the result models, and checkpoint
writing.

**No test here drives KINDER, and that is the point rather than a gap** -- see
`test_tossing3d_oracle_demo.py`'s own docstring for the same reasoning. Unlike that
script, this one *does* import `hitl_pmp` (the module docstring explains why), but
`Tossing3DEnvironment`/`predicates`/`skills`/`skill_oracle_policy` are all pure
pydantic/numpy at import time -- `KinderBackend` is the only module in `hitl_pmp` that
imports KINDER, and only lazily, on first `reset()` -- so this whole script, and this
whole test file, import and run on a machine with no MuJoCo. `run_pick_grid`/
`run_move_grid` themselves are not covered here for exactly that reason: they call
`env.reset_to_seed`, which does need a live simulator, and are exercised only by
actually running the sweep (recorded in the PR body and the experiment log, not by
pytest).
"""

import json
from pathlib import Path

import numpy as np
import pytest

from hitl_pmp.environments.tossing3d.skills import PICK_DISTANCE_BOUNDS, PICK_ROTATION_BOUNDS
from scripts.tossing3d_skill_parameter_sweep import (
    MOVE_STANDOFF_MAX,
    MOVE_STANDOFF_MIN,
    MOVE_STANDOFF_STEP,
    PICK_DISTANCE_STEPS,
    PICK_ROTATION_STEPS,
    MoveCellResult,
    PickCellResult,
    _parse_args,
    _rss_mb,
    _write_checkpoint,
)


def test_pick_grid_spans_exactly_the_sampling_bounds() -> None:
    """`Pick` is gridded over its own sampling bounds and nothing wider -- there is
    nothing outside them that `Pick.sample_params` would ever draw."""
    distances = np.linspace(*PICK_DISTANCE_BOUNDS, PICK_DISTANCE_STEPS)
    rotations = np.linspace(*PICK_ROTATION_BOUNDS, PICK_ROTATION_STEPS)
    assert distances[0] == pytest.approx(PICK_DISTANCE_BOUNDS[0])
    assert distances[-1] == pytest.approx(PICK_DISTANCE_BOUNDS[1])
    assert rotations[0] == pytest.approx(PICK_ROTATION_BOUNDS[0])
    assert rotations[-1] == pytest.approx(PICK_ROTATION_BOUNDS[1])


def test_pick_rotation_grid_lands_exactly_on_zero() -> None:
    """`PICK_ROTATION_STEPS` is odd so rotation=0.0 sits on the grid -- the module
    docstring's own claim, pinned here so a future resolution change cannot silently
    lose it."""
    rotations = np.linspace(*PICK_ROTATION_BOUNDS, PICK_ROTATION_STEPS)
    assert any(rotation == pytest.approx(0.0, abs=1e-9) for rotation in rotations)


def test_move_grid_extends_past_both_the_old_and_the_narrower_sampler_bounds() -> None:
    """The default sweep range is wider than any `THROW_STANDOFF_BOUNDS` this domain
    has used -- old `(0.45, 1.75)` and the in-flight proposed `(1.10, 1.75)` -- on both
    ends, per the module docstring."""
    assert MOVE_STANDOFF_MIN < 0.45
    assert MOVE_STANDOFF_MAX > 1.75


def test_move_grid_resolves_the_classifier_band_to_several_points() -> None:
    """The tightened `RobotAtSuccessfulThrowPoseClassifier` band is `[1.150, 1.375]`,
    0.225 m wide -- the step size must be fine enough to place more than one grid point
    strictly inside it, or the sweep could not resolve the band's own edges."""
    num_steps = round((MOVE_STANDOFF_MAX - MOVE_STANDOFF_MIN) / MOVE_STANDOFF_STEP) + 1
    standoffs = np.linspace(MOVE_STANDOFF_MIN, MOVE_STANDOFF_MAX, num_steps)
    inside_band = [s for s in standoffs if 1.150 <= s <= 1.375]
    assert len(inside_band) >= 5


def test_parse_args_defaults_match_the_module_constants() -> None:
    import sys

    argv = ["prog", "--output-dir", "/tmp/wherever"]
    old_argv = sys.argv
    sys.argv = argv
    try:
        args = _parse_args()
    finally:
        sys.argv = old_argv
    assert args.which == "both"
    assert args.pick_distance_steps == PICK_DISTANCE_STEPS
    assert args.pick_rotation_steps == PICK_ROTATION_STEPS
    assert args.move_min == pytest.approx(MOVE_STANDOFF_MIN)
    assert args.move_max == pytest.approx(MOVE_STANDOFF_MAX)
    assert args.move_step == pytest.approx(MOVE_STANDOFF_STEP)


def test_pick_cell_result_round_trips_through_json() -> None:
    record = PickCellResult(
        distance=0.55,
        rotation=0.1,
        seed=0,
        success=True,
        pick_error=None,
        pick_terminated=True,
        pick_steps=61,
        robot_base_x=0.3,
        robot_base_y=0.1,
        cube_x=0.4,
        cube_y=0.05,
        cube_z=0.585,
    )
    restored = PickCellResult.model_validate(json.loads(json.dumps(record.model_dump())))
    assert restored == record


def test_move_cell_result_round_trips_through_json() -> None:
    record = MoveCellResult(
        standoff=1.2,
        seed=0,
        success=True,
        pick_success=True,
        pick_error=None,
        move_error=None,
        move_terminated=True,
        robot_base_x=0.8,
        robot_base_y=0.0,
        lateral_offset=0.0,
        bin_x_initial=2.0,
        bin_y_initial=0.0,
        bin_x=2.0,
        bin_y=0.0,
        barrier_x_initial=1.3,
        barrier_x=1.3,
    )
    restored = MoveCellResult.model_validate(json.loads(json.dumps(record.model_dump())))
    assert restored == record


def test_write_checkpoint_is_atomic_and_leaves_no_tmp_file(*, tmp_path: Path) -> None:
    """A reader must never see a truncated file, and a finished run must not leave a
    stray `.tmp` beside the real one."""
    path = tmp_path / "checkpoint.json"
    records = [
        PickCellResult(
            distance=0.5,
            rotation=0.0,
            seed=seed,
            success=True,
            pick_error=None,
            pick_terminated=True,
            pick_steps=60,
            robot_base_x=0.3,
            robot_base_y=0.1,
            cube_x=0.4,
            cube_y=0.05,
            cube_z=0.585,
        )
        for seed in range(3)
    ]
    _write_checkpoint(path=path, records=list(records))
    assert path.is_file()
    assert not path.with_suffix(path.suffix + ".tmp").exists()
    loaded = json.loads(path.read_text())
    assert len(loaded) == 3
    assert {row["seed"] for row in loaded} == {0, 1, 2}


def test_write_checkpoint_overwrites_a_prior_partial_write(*, tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    _write_checkpoint(
        path=path,
        records=[
            PickCellResult(
                distance=0.5,
                rotation=0.0,
                seed=0,
                success=True,
                pick_error=None,
                pick_terminated=True,
                pick_steps=60,
                robot_base_x=0.3,
                robot_base_y=0.1,
                cube_x=0.4,
                cube_y=0.05,
                cube_z=0.585,
            )
        ],
    )
    _write_checkpoint(path=path, records=[])
    assert json.loads(path.read_text()) == []


def test_rss_mb_is_positive_for_a_running_process() -> None:
    """A loose sanity check on the units conversion (`ru_maxrss` is KiB on Linux) --
    a process that has run at all reports a positive, plausible RSS rather than zero
    or something absurd."""
    rss = _rss_mb()
    assert 1.0 < rss < 100_000.0
