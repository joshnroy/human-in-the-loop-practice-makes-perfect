"""`OverheadFootprints`: upstream's overhead footprints, memoized on exact state bytes.

Toss sampling asks the base planner about up to a hundred stands per state, and each
ask rebuilt every robot and obstacle footprint from the same snapshot. The memo must
change nothing but the time: the same verdicts, the same paths, and never a footprint
from a state other than the one asked about.
"""

import contextlib
import importlib.util

import numpy as np
import pytest

from hitl_pmp.environments.tossing3d.kinder_backend import KinderBackend, OverheadFootprints

pytestmark = [
    pytest.mark.skipif(
        importlib.util.find_spec("kinder") is None
        or importlib.util.find_spec("kinder_models") is None,
        reason="needs KINDER",
    ),
    pytest.mark.xdist_group("tossing3d_overhead_footprints"),
]

STANDS = [(distance, rotation) for distance in (1.3, 1.8, 2.3) for rotation in (0.0, np.pi / 2)]


@pytest.fixture(scope="module")
def snapshot():
    backend = KinderBackend()
    try:
        backend.reset(seed=0)
        yield backend.live_state().copy()
    finally:
        backend.close()


def uncached(*, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(OverheadFootprints, "installed", contextlib.nullcontext)
    monkeypatch.setattr(OverheadFootprints, "of", OverheadFootprints.upstream)


class Paths:
    @staticmethod
    def plan(*, snapshot, distance: float, rotation: float):
        from kinder_models.dynamic3d.utils import (
            WORLD_X_BOUNDS,
            WORLD_Y_BOUNDS,
            get_overhead_object_se2_pose,
            get_target_robot_pose_from_parameters,
            run_base_motion_planning,
        )

        target = get_target_robot_pose_from_parameters(
            get_overhead_object_se2_pose(
                snapshot, snapshot.get_object_from_name(KinderBackend.bin_name)
            ),
            np.float32(distance),
            np.float32(rotation),
        )
        with OverheadFootprints.installed():
            plan = run_base_motion_planning(
                state=snapshot,
                target_base_pose=target,
                x_bounds=WORLD_X_BOUNDS,
                y_bounds=WORLD_Y_BOUNDS,
                seed=0,
                disable_collision_objects=[KinderBackend.cube_name],
            )
        return None if plan is None else [(p.x, p.y, p.theta()) for p in plan]


def test_the_memo_gives_the_verdicts_paths_and_geometry_the_planner_gives(*, snapshot, monkeypatch):
    def run():
        OverheadFootprints.clear()
        return (
            [
                KinderBackend.toss_base_plan_failure(
                    snapshot=snapshot, distance=distance, rotation=rotation
                )
                for distance, rotation in STANDS
            ],
            [Paths.plan(snapshot=snapshot, distance=d, rotation=r) for d, r in STANDS],
            KinderBackend.toss_feasibility_geometry(snapshot=snapshot),
        )

    cached = run()
    assert OverheadFootprints.hits > 0
    with monkeypatch.context() as patch:
        uncached(monkeypatch=patch)
        reference = run()
    assert cached == reference
    assert any(verdict is None for verdict in cached[0])
    assert any(path is not None for path in cached[1])


def test_one_state_builds_its_footprints_once(*, snapshot):
    OverheadFootprints.clear()
    for distance, rotation in STANDS:
        KinderBackend.toss_base_plan_failure(
            snapshot=snapshot, distance=distance, rotation=rotation
        )
    KinderBackend.toss_feasibility_geometry(snapshot=snapshot)
    assert (OverheadFootprints.misses, OverheadFootprints.hits) == (1, len(STANDS))


def test_a_changed_state_never_gets_the_old_footprints(*, snapshot):
    from kinder_models.dynamic3d.utils import get_overhead_kinematic2ds

    OverheadFootprints.clear()
    state = snapshot.copy()
    robot = state.get_object_from_name("robot")
    before = OverheadFootprints.of(state)
    # The same object, edited in place, and a copy: both are new states.
    state.set(robot, "pos_base_x", state.get(robot, "pos_base_x") + 0.25)
    moved = OverheadFootprints.of(state)
    assert OverheadFootprints.misses == 2
    assert moved == get_overhead_kinematic2ds(state)
    assert moved["robot"] != before["robot"]
    widened = state.copy()
    widened.set(robot, "pos_base_y", widened.get(robot, "pos_base_y") - 0.1)
    assert OverheadFootprints.of(widened) == get_overhead_kinematic2ds(widened)
    boxes = {KinderBackend.bin_name: (1.0, 1.0, 1.0)}
    assert OverheadFootprints.of(widened, boxes) == get_overhead_kinematic2ds(widened, boxes)
    assert OverheadFootprints.misses == 4
    # A caller editing what it was handed cannot reach the memo.
    moved.clear()
    assert OverheadFootprints.of(state)["robot"] == get_overhead_kinematic2ds(state)["robot"]


def test_the_memo_is_only_in_place_for_the_planners_own_call(*, snapshot):
    from kinder_models.dynamic3d import utils

    original = utils.get_overhead_kinematic2ds
    with OverheadFootprints.installed():
        assert utils.get_overhead_kinematic2ds is not original
    assert utils.get_overhead_kinematic2ds is original
    with pytest.raises(RuntimeError), OverheadFootprints.installed():
        raise RuntimeError
    assert utils.get_overhead_kinematic2ds is original
