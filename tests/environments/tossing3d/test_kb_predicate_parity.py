"""The five Tossing3D predicates are kinder-baselines', not ours.

This file is the contract of the swap: each hitl `Predicate` must agree with the
kinder-baselines classifier it is now backed by, *by construction* rather than by
a threshold copied across and kept in step by hand.

Everything here is offline. `ObjectCentricState` is constructible with no MuJoCo,
and three of kinder-baselines' five classifiers are `@staticmethod`s, so they are
callable with no abstractor instance and hence no simulator process. The other two
(`Holding`'s forward-kinematics conjunct, `MovableInGoalRegion`'s ground fixture)
genuinely need a simulator and are covered in `test_kinder_fidelity.py`.

## The throw-pose band is gone from this file, and it was most of it

Seven tests here probed `RobotAtThrowPose` -- the accepted standoff band, its upper edge,
the lateral tolerance, the heading conjunct, and the three post-`Pick` hazard cases. That
classifier is **deleted upstream** at the pin this branch carries: composing the base move
into the toss removed the pose between the two skills, so `Tossing3DStateAbstractor` no
longer declares `RobotAtThrowPose`, `THROW_STANDOFF_BOUNDS` or `THROW_POSE_TOLERANCE`, and
there is nothing left for these to call.

They are deleted rather than ported because the property they protected moved rather than
vanished: the standoff is now the composed toss's first continuous parameter, drawn from
upstream's own `TARGET_DISTANCE_BOUNDS`, and `test_kinder_pin.py` holds this package's copy
of those bounds to upstream's. What is genuinely lost is the *symbolic* rejection of a bad
pose before a throw happens -- upstream's own stated trade, and stated again in `skills.py`.
"""

import importlib.util

import pytest

from .object_centric import BARRIER_X, object_centric_state

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None, reason="requires the tossing3d extra"
)


def _kb():
    """Upstream's abstractor, imported on use.

    Lazy for the same reason `object_centric.py`'s imports are: pytest imports every test
    module at collection time, and a module-scope `import kinder_models` here would put
    `mujoco` in `sys.modules` before the several tests asserting the *absence* of a
    simulator import ever run.
    """
    from kinder_models.dynamic3d.tossing.state_abstractions import Tossing3DStateAbstractor

    return Tossing3DStateAbstractor


def _abstractor_static(*, name: str):
    return getattr(_kb(), name)


def test_hand_empty_holds_only_at_an_open_gripper() -> None:
    """Upstream reads the gripper *command*, not finger pose, so this and `Holding` are
    deliberately not complementary."""
    check = _abstractor_static(name="_check_gripper_open")
    for gripper, expected in ((0.0, True), (0.0005, True), (0.05, False), (0.9, False)):
        state, objects = object_centric_state(gripper=gripper)
        assert check(state, objects["robot"]) is expected, gripper


def test_on_ground_holds_for_a_cube_resting_flat_on_the_floor() -> None:
    check = _abstractor_static(name="_check_on_ground")
    state, objects = object_centric_state(cube_z=0.025)
    assert check(state, objects["cube_0"])


def test_on_ground_rejects_a_cube_in_the_air() -> None:
    check = _abstractor_static(name="_check_on_ground")
    state, objects = object_centric_state(cube_z=0.4)
    assert not check(state, objects["cube_0"])


def test_on_ground_rejects_a_cube_resting_on_a_corner() -> None:
    """Flatness is upstream's own conjunct and is load-bearing rather than decorative:
    `pick_shelf` builds its grasp pose from the object's orientation, so a cube that came
    to rest on a corner is not a cube this grasp is modelled on."""
    check = _abstractor_static(name="_check_on_ground")
    state, objects = object_centric_state(cube_z=0.025, cube_qx=0.5)
    assert not check(state, objects["cube_0"])
    state, objects = object_centric_state(cube_z=0.025, cube_qy=0.5)
    assert not check(state, objects["cube_0"])


def test_movable_is_down_x_is_a_one_way_door_across_the_barrier() -> None:
    """The irreversibility this whole domain exists to exhibit: the barrier is a 5 m
    immovable wall, so a cube past it can never be picked up again."""
    check = _abstractor_static(name="_check_is_down_x")
    near, objects = object_centric_state(cube_x=BARRIER_X - 0.1)
    assert check(near, objects["cube_0"], objects["cuboid_barrier"])
    far, objects = object_centric_state(cube_x=BARRIER_X + 0.1)
    assert not check(far, objects["cube_0"], objects["cuboid_barrier"])


def test_movable_is_down_x_reads_the_barriers_live_x_rather_than_a_constant() -> None:
    """The barrier's pose is sampled from `barrier_init_region` per episode, so a literal
    would be right for one seed and quietly wrong for the next."""
    check = _abstractor_static(name="_check_is_down_x")
    state, objects = object_centric_state(cube_x=1.5, barrier_x=1.8)
    assert check(state, objects["cube_0"], objects["cuboid_barrier"])
    state, objects = object_centric_state(cube_x=1.5, barrier_x=1.2)
    assert not check(state, objects["cube_0"], objects["cuboid_barrier"])
