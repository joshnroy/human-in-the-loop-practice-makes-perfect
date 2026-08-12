"""What `Tossing3D-o1-coincident.json` has to say for this domain's arithmetic to hold.

The scene JSON is not inert data: `kindergarden`'s Dynamic3D registration is
**filesystem-derived**, so a task JSON is executable configuration that changes what every
run does, with no code change anywhere to show for it. This domain then does real
arithmetic on top of that geometry, and two of its constants are only correct under
assumptions the JSON happens to satisfy today.

Offline by construction: this reads the file, and never builds a scene.
"""

import json
from pathlib import Path

import pytest

from hitl_pmp.environments.tossing3d.kinder_backend import Tossing3DSceneFiles


def _config() -> dict:
    path = Tossing3DSceneFiles().coincident_task_config()
    return json.loads(Path(path).read_text())


def test_the_bin_yaw_range_is_pinned_to_zero() -> None:
    """**The whole `base_x = bin_x - standoff` arithmetic assumes it.**

    `RobotAtSuccessfulThrowPoseClassifier` predicts where a throw lands as
    `base_x + THROW_RANGE` -- a displacement along **+x only**. That is sound because
    `MoveToThrowPose` pins `rot = 0` *and* the bin's yaw range is `[[0, 0]]`, so a base on
    the bin's axis faces `+x` and the throw's displacement is collinear with the axis the
    goal box is tested on.

    Give the bin a nonzero yaw and `move_to_target` puts the base on a rotated approach
    axis, at which point the throw's displacement acquires a y component that
    `THROW_RANGE` does not model and the predicted landing point is wrong by
    `THROW_RANGE * (1 - cos(yaw))` in x plus `THROW_RANGE * sin(yaw)` in y. Nothing would
    raise. The predicate would simply start accepting poses the throw misses from, and the
    error grows with the standoff.

    The classifier's docstring already *states* this assumption; until this test existed
    nothing checked it. It is worth pinning specifically because the movable-bin work is
    live -- `kindergarden` PR #126 moves the bin, and this domain's design deliberately
    derives the success band from live scene geometry so a *translated* bin needs no
    change here. A **rotated** one is the case that derivation does not cover, so the
    config must keep promising it will not happen.
    """
    yaw_ranges = _config()["regions"]["bin_init_region"]["yaw_ranges"]
    assert yaw_ranges == [[0, 0]], (
        "the bin's yaw range is no longer pinned to zero, so a sampled bin can be "
        "rotated -- RobotAtSuccessfulThrowPoseClassifier's `base_x + THROW_RANGE` "
        "landing prediction is a pure +x displacement and is wrong for a rotated bin"
    )


def test_the_bin_init_region_is_a_point_not_a_range() -> None:
    """The bin is placed at one spot, not sampled over an interval.

    `2.0 <= x <= 2.001` and `|y| <= 0.0005` is a 1 mm x 1 mm box -- a point with slack for
    floating point, not a distribution. Every committed Tossing3D number is therefore about
    one bin position, and `THROW_RANGE`'s calibration in particular assumed the bin sits
    where the goal region is. If this ever becomes a real range, the constant needs
    re-deriving rather than reusing.
    """
    (x_min, y_min, x_max, y_max) = _config()["regions"]["bin_init_region"]["ranges"][0]
    assert x_max - x_min == pytest.approx(0.001, abs=1e-9)
    assert y_max - y_min == pytest.approx(0.001, abs=1e-9)


def test_the_goal_region_still_brackets_the_bin() -> None:
    """The defect this whole task config exists to fix, guarded so it cannot come back.

    Under upstream's stock `o1` the bin sits at x = 2.23 while `blocks_goal_region`
    inflates to x in [1.85, 2.15], so a cube that lands **in the bin scores a failure** and
    only a cube that misses it scores a success -- training against that would reward
    missing. The coincident config puts the bin back at x = 2.0, inside the goal box.

    Asserted as containment rather than as two literals, so moving both together stays
    legal and moving one apart from the other does not.
    """
    config = _config()
    bin_x_min, _, bin_x_max, _ = config["regions"]["bin_init_region"]["ranges"][0]
    goal = config["regions"]["blocks_goal_region"]["ranges"][0]
    goal_x_min, goal_x_max = goal[0], goal[3]
    assert goal_x_min < bin_x_min <= bin_x_max < goal_x_max
