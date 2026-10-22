"""SweepOffShelf3D's skills on KINDER's controllers, run through `take_action`."""

import importlib.util

import numpy as np
import pytest

from hitl_pmp.environments.sweepoffshelf3d.environment import (
    SKILL_IDS,
    SweepOffShelf3DEnvironment,
)
from hitl_pmp.environments.sweepoffshelf3d.kinder_skills import HOME_ARM, RETRACT_TOLERANCE
from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import CARRY_Z, RESEAT_TURN_DEG

# Module-scoped fixtures: keep them on one xdist worker (see tests/test_xdist_groups.py).
pytestmark = [
    pytest.mark.skipif(
        importlib.util.find_spec("kinder") is None
        or importlib.util.find_spec("kinder_models") is None,
        reason="needs KINDER",
    ),
    pytest.mark.xdist_group("sweepoffshelf3d_test_kinder_skills"),
]


@pytest.fixture(name="env", scope="module")
def _env():
    env = SweepOffShelf3DEnvironment(scene_bg=False)
    try:
        yield env
    finally:
        env.close()


def _act(*, env, skill, cube=0, height=0.005, speed=0.05):
    env.take_action(action=np.array([float(SKILL_IDS[skill]), float(cube), height, speed]))
    return env.get_current_state().abstract_atoms


def _arm_from_home(*, env):
    backend = env.backend()
    model, data = backend.model(), backend.data()
    joints = [model.joint(f"robot_joint_{i}").id for i in range(1, 8)]
    arm = data.qpos[model.jnt_qposadr[joints]]
    return float(np.max(np.abs((arm - HOME_ARM + np.pi) % (2 * np.pi) - np.pi)))


def _wiper_carried_clear(*, env):
    """`KinderSkills.retract`'s end state holding the wiper: the hand at the carry height
    and the whole wiper outside the cupboard's open front."""
    skills = env.controllers()._skills  # noqa: SLF001
    far = env.controllers()._kinder.wiper_far_x()  # noqa: SLF001
    return skills.site_pose()[2, 3] > CARRY_Z - 0.02 and far < skills.front_x


def _teleport(*, env, name, position):
    obj = env.backend().core().get_object(name)
    obj.set_pose(list(position), [1.0, 0.0, 0.0, 0.0])
    obj.set_velocity([0.0] * 3, [0.0] * 3)
    env.backend().sim().forward()
    for _ in range(20):
        env.backend().env().step(np.zeros(11))


def test_pick_wiper_takes_it_off_its_shelf_and_carries_it_clear(*, env):
    env.reset_to_seed(seed=0)
    atoms = _act(env=env, skill="PickWiper")
    assert ("HoldingWiper", ("robot", "wiper_0")) in atoms, env.last_skill_error()
    assert _wiper_carried_clear(env=env)


def _turn_wiper_in_its_slot(*, env, degrees):
    backend = env.backend()
    obj = backend.core().get_object("wiper_0")
    position, quaternion = (np.asarray(v, dtype=float) for v in obj.get_pose())
    w, x, y, z = quaternion
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)) + np.radians(degrees)
    obj.set_pose(list(position), [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    obj.set_velocity([0.0] * 3, [0.0] * 3)
    backend.sim().forward()
    for _ in range(30):
        backend.env().step(np.zeros(11))
    env.controllers().on_objects_moved()
    env.current_state = env._observed_state(seed=0, steps_taken=0)  # noqa: SLF001
    assert ("WiperHome", ("wiper_0",)) in env.backend().abstract_atoms()


@pytest.mark.parametrize("degrees", [-0.2, 0.15, 0.3])
def test_pick_wiper_takes_a_wiper_sitting_a_fraction_of_a_degree_off_square(*, env, degrees):
    """The EES p0 stage-5 tally (56 failures of "the wiper did not come with the hand"):
    the wiper home at 90.05 deg instead of 90.00. Upstream's retract, planned blind to
    the held wiper, then took the hand 9 cm deeper into the cupboard, the wiper struck
    the back panel and was torn out of the pads, left turned ~25 deg on its shelf --
    where every later attempt failed too."""
    env.reset_to_seed(seed=0)
    _turn_wiper_in_its_slot(env=env, degrees=degrees)
    atoms = _act(env=env, skill="PickWiper")
    assert ("HoldingWiper", ("robot", "wiper_0")) in atoms, env.last_skill_error()
    assert _wiper_carried_clear(env=env)


def test_pick_wiper_from_an_arm_left_over_the_box(*, env):
    """PR #445's one PickWiper failure (Lane B's E2, seed 3): the arm where a
    PlaceCubeInBox leaves it. With the base square on to the wiper no candidate planned
    (17 tried); swung a little to one side, one does."""
    from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import Frames

    env.reset_to_seed(seed=3)
    skills = env.controllers()._skills  # noqa: SLF001
    rotation = Frames.leaned(
        closing=np.array([0.0, 1.0, 0.0]), lean_toward=np.array([1.0, 0.0, 0.0]), lean_deg=30.0
    )
    base = np.array([0.75, 0.0, 0.0])
    arm, _ = skills.ik(position=[1.235, 0.0, 0.36], rotation=rotation, base=base)
    _pose_robot(env=env, base=base, arm=arm)
    env.current_state = env._observed_state(seed=3, steps_taken=0)  # noqa: SLF001
    atoms = _act(env=env, skill="PickWiper")
    assert ("HoldingWiper", ("robot", "wiper_0")) in atoms, env.last_skill_error()


def test_pick_wiper_takes_a_wiper_left_turned_on_its_shelf(*, env):
    """Where a torn-out pick left it in the stage-5 rerun: 25 deg off square, 5 cm in."""
    env.reset_to_seed(seed=0)
    _turn_wiper_in_its_slot(env=env, degrees=-26.6)
    atoms = _act(env=env, skill="PickWiper")
    assert ("HoldingWiper", ("robot", "wiper_0")) in atoms, env.last_skill_error()


def test_a_cube_on_the_open_floor_is_picked_up(*, env):
    # Behind the robot's start, where the scripted floor pick found no IK solution.
    env.reset_to_seed(seed=0)
    _teleport(env=env, name="cube_1", position=[0.60, -0.55, 0.011])
    assert ("OnFloor", ("cube_1",)) in env.backend().abstract_atoms()
    atoms = _act(env=env, skill="PickCubeFromFloor", cube=1)
    assert ("HoldingCube", ("robot", "cube_1")) in atoms, env.last_skill_error()
    assert _arm_from_home(env=env) < RETRACT_TOLERANCE


def test_a_cube_picked_off_the_floor_is_placed_in_the_box(*, env):
    env.reset_to_seed(seed=0)
    _teleport(env=env, name="cube_2", position=[0.95, 0.45, 0.011])
    atoms = _act(env=env, skill="PickCubeFromFloor", cube=2)
    assert ("HoldingCube", ("robot", "cube_2")) in atoms, env.last_skill_error()
    atoms = _act(env=env, skill="PlaceCubeInBox", cube=2)
    assert {("InBox", ("cube_2", "box_0")), ("HandEmpty", ("robot",))} <= atoms, (
        env.last_skill_error()
    )
    assert _arm_from_home(env=env) < RETRACT_TOLERANCE


@pytest.mark.parametrize("cube", [0, 1])
def test_pick_cube_from_shelf_takes_a_cube_out_of_the_cupboard(*, env, cube):
    # cube_1 rests on the board so exactly that upstream's retract plan, holding it,
    # started in collision and never planned.
    env.reset_to_seed(seed=0)
    atoms = _act(env=env, skill="PickCubeFromShelf", cube=cube)
    assert ("HoldingCube", ("robot", f"cube_{cube}")) in atoms, env.last_skill_error()
    assert _arm_from_home(env=env) < RETRACT_TOLERANCE


# Cube layouts a human `reset_cubes` left (`_cube_placements`, human seeds 0, 61 and
# 77), as fixed poses: position, then MuJoCo's (w, x, y, z) quaternion. Each failed
# the shelf pick before it screened its arm configurations in MuJoCo:
# - seed 0, cube_0 7 cm from the -y side panel and cube_2 at the back: no grasp planned;
# - seed 61, cube_1 in the back corner: no grasp planned, and then no retract or
#   approach plan from the pre-grasp until both went through the open front;
# - seed 77, cube_4 with cube_1 and cube_3 under 4 cm away: the close shoved it 9 cm
#   along the shelf onto its neighbour, and every retry missed the same way.
RESET_LAYOUT = {
    "cube_0": ([1.4678, -0.2203, 0.2899], [-0.8328, 0.0, 0.0, 0.5536]),
    "cube_1": ([1.5578, 0.0512, 0.2899], [-0.1366, 0.0, 0.0, 0.9906]),
    "cube_2": ([1.5609, 0.1516, 0.2899], [-0.9013, 0.0, 0.0, 0.4331]),
    "cube_3": ([1.4347, 0.1102, 0.2899], [-0.9090, 0.0, 0.0, 0.4167]),
    "cube_4": ([1.5058, -0.0961, 0.2899], [0.9960, 0.0, 0.0, 0.0889]),
}
BACK_CORNER_LAYOUT = {
    "cube_0": ([1.5381, 0.0821, 0.2899], [0.2066, 0.0, 0.0, 0.9784]),
    "cube_1": ([1.5696, 0.2391, 0.2899], [0.8230, 0.0, 0.0, 0.5680]),
    "cube_2": ([1.5218, -0.1253, 0.2899], [0.9703, 0.0, 0.0, 0.2420]),
    "cube_3": ([1.5475, -0.2305, 0.2899], [-0.9802, 0.0, 0.0, 0.1982]),
    "cube_4": ([1.5658, 0.1722, 0.2899], [0.8802, 0.0, 0.0, 0.4745]),
}
CLUSTER_LAYOUT = {
    "cube_0": ([1.5072, -0.1229, 0.2899], [0.5393, 0.0, 0.0, 0.8421]),
    "cube_1": ([1.4846, 0.1446, 0.2899], [0.3867, 0.0, 0.0, 0.9222]),
    "cube_2": ([1.5408, 0.1242, 0.2899], [0.9168, 0.0, 0.0, 0.3993]),
    "cube_3": ([1.4804, 0.1780, 0.2899], [-0.6059, 0.0, 0.0, 0.7956]),
    "cube_4": ([1.4525, 0.1817, 0.2899], [0.4834, 0.0, 0.0, 0.8754]),
}


def _pour(*, env, layout):
    backend = env.backend()
    for name, (position, quaternion) in layout.items():
        obj = backend.core().get_object(name)
        obj.set_pose(position, list(np.asarray(quaternion) / np.linalg.norm(quaternion)))
        obj.set_velocity([0.0] * 3, [0.0] * 3)
    backend.sim().forward()
    env.controllers().on_objects_moved()
    env.current_state = env._observed_state(seed=0, steps_taken=0)  # noqa: SLF001


@pytest.mark.parametrize("yaw_deg", [0.0, 20.0, 45.0, 70.0])
def test_every_shelf_grasp_closes_the_pads_on_opposite_faces(*, env, yaw_deg):
    """Josh, 2026-10-15: no grasp closes the pads across the cube off its faces."""
    from scipy.spatial.transform import Rotation

    env.reset_to_seed(seed=0)
    position, _ = env.backend().core().get_object("cube_1").get_pose()
    yaw = np.deg2rad(yaw_deg)
    _pour(
        env=env, layout={"cube_1": (list(position), [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])}
    )
    kinder = env.controllers()._kinder  # noqa: SLF001
    centre = kinder.shelf.cube_position(cube="cube_1")
    quaternion = env.backend().core().get_object("cube_1").get_pose()[1]
    faces = Rotation.from_quat(np.asarray(quaternion)[[1, 2, 3, 0]]).as_matrix()[:, :2]
    hands = kinder.shelf_hands(cube="cube_1", centre=centre)
    assert hands
    for hand, _ in hands:
        closing = hand[:3, 0]
        assert np.max(np.abs(faces.T @ closing)) > np.cos(np.deg2rad(1.0)), closing


def _picks_from(*, env, layout, cube):
    env.reset_to_seed(seed=0)
    _pour(env=env, layout=layout)
    assert ("OnShelf", (f"cube_{cube}",)) in env.get_current_state().abstract_atoms
    atoms = _act(env=env, skill="PickCubeFromShelf", cube=cube)
    assert ("HoldingCube", ("robot", f"cube_{cube}")) in atoms, env.last_skill_error()
    assert _arm_from_home(env=env) < RETRACT_TOLERANCE


@pytest.mark.parametrize("cube", [0, 2], ids=["by-side-panel", "back-of-shelf"])
def test_pick_cube_from_shelf_takes_a_cube_from_a_human_reset_layout(*, env, cube):
    _picks_from(env=env, layout=RESET_LAYOUT, cube=cube)


@pytest.mark.main_only
@pytest.mark.parametrize(
    ("layout", "cube"),
    [
        pytest.param(
            BACK_CORNER_LAYOUT,
            1,
            marks=pytest.mark.xfail(
                strict=True,
                reason="only a grasp with the pads turned off the cube's faces reached it, "
                "and Josh ruled those out (2026-10-15)",
            ),
            id="back-corner",
        ),
        pytest.param(
            CLUSTER_LAYOUT,
            4,
            marks=pytest.mark.xfail(
                strict=False,
                reason="marginal without turned grasps: picked in the lab2 benchmark, but "
                "closes on nothing in this plain scene -- the remaining-miss class",
            ),
            id="cluster",
        ),
    ],
)
def test_pick_cube_from_shelf_takes_a_cornered_or_crowded_cube(*, env, layout, cube):
    _picks_from(env=env, layout=layout, cube=cube)


def test_a_wiper_lost_from_the_hand_is_not_still_believed_held(*, env):
    """The absorbing failure behind CCCC's PickWiper tally: a skill that loses the
    wiper used to leave the controllers believing it was in hand, so every later
    hand-empty skill refused without moving."""
    env.reset_to_seed(seed=0)
    atoms = _act(env=env, skill="PickWiper")
    assert ("HoldingWiper", ("robot", "wiper_0")) in atoms, env.last_skill_error()
    # The wiper slips out of the hand onto the floor, and nobody puts it back.
    _teleport(env=env, name="wiper_0", position=[0.55, -0.6, 0.02])
    env.current_state = env._observed_state(seed=0, steps_taken=1)  # noqa: SLF001
    assert ("HandEmpty", ("robot",)) in env.get_current_state().abstract_atoms
    _act(env=env, skill="PickCubeFromShelf", cube=1)
    assert "not empty" not in (env.last_skill_error() or "")
    assert env.last_controller_steps() > 0


@pytest.mark.main_only
def test_the_wiper_sweeps_the_shelf_and_goes_back(*, env):
    env.reset_to_seed(seed=0)
    assert ("HoldingWiper", ("robot", "wiper_0")) in _act(env=env, skill="PickWiper")
    atoms = _act(env=env, skill="SweepShelf", height=0.005, speed=0.05)
    # Whether every cube lands in the box is the sweep's competence; the wiper staying
    # in hand through the retract that follows is not.
    assert not any(name == "OnShelf" for name, _ in atoms), env.last_skill_error()
    assert ("HoldingWiper", ("robot", "wiper_0")) in atoms
    atoms = _act(env=env, skill="ReturnWiper")
    assert {("WiperHome", ("wiper_0",)), ("HandEmpty", ("robot",))} <= atoms, env.last_skill_error()
    assert _arm_from_home(env=env) < RETRACT_TOLERANCE


@pytest.mark.parametrize(
    ("skill", "where"),
    [("ReturnCubeToShelfFromBox", "box"), ("ReturnCubeToShelfFromFloor", "floor")],
)
def test_a_returned_cube_rests_on_the_shelf_or_the_skill_says_it_failed(*, env, skill, where):
    env.reset_to_seed(seed=3)
    if where == "box":
        box = env.backend().core().get_object("box_0").get_pose()[0]
        _teleport(env=env, name="cube_3", position=[box[0], box[1], 0.06])
        assert ("InBox", ("cube_3", "box_0")) in env.backend().abstract_atoms()
    else:
        _teleport(env=env, name="cube_3", position=[0.95, 0.45, 0.011])
        assert ("OnFloor", ("cube_3",)) in env.backend().abstract_atoms()
    atoms = _act(env=env, skill=skill, cube=3)
    assert env.last_controller_steps() > 0
    returned = ("OnShelf", ("cube_3",)) in atoms
    assert returned == (env.last_skill_error() is None), env.last_skill_error()
    if returned:
        assert ("HandEmpty", ("robot",)) in atoms
    assert _arm_from_home(env=env) < RETRACT_TOLERANCE


# Where a SweepShelf that lost the wiper left the arm, reproducing cccc-determinized-p1
# on 2026-10-14: the hand just inside the cupboard's open front, clear of everything in
# MuJoCo, but a finger pad inside KINDER's PyBullet cupboard model.
STRANDED_BASE = (0.523, -0.174, 0.0)
STRANDED_ARM = (0.333, 1.217, 2.526, -1.478, 2.23, -0.759, 2.628)


def _pose_robot(*, env, base, arm):
    backend = env.backend()
    model, data = backend.model(), backend.data()
    for name, value in zip(("x", "y", "th"), base, strict=True):
        data.qpos[model.jnt_qposadr[model.joint(f"robot_joint_{name}").id]] = value
    for i, value in enumerate(arm, start=1):
        data.qpos[model.jnt_qposadr[model.joint(f"robot_joint_{i}").id]] = value
    data.qvel[:] = 0.0
    backend.sim().forward()
    env.controllers().on_objects_moved()


def test_an_arm_stranded_in_the_cupboard_front_is_retracted_after_the_next_skill(*, env):
    """The absorbing state behind CCCC's PickCubeFromShelf/PickWiper plan-time tally:
    the retract planned from a start PyBullet calls a collision, never planned, and left
    the arm out; every later skill planned its approach from there and failed with 0
    control steps, and so did its own retract."""
    env.reset_to_seed(seed=0)
    _pose_robot(env=env, base=STRANDED_BASE, arm=STRANDED_ARM)
    assert _arm_from_home(env=env) > RETRACT_TOLERANCE
    _act(env=env, skill="PickWiper")
    assert _arm_from_home(env=env) < RETRACT_TOLERANCE, env.last_skill_error()
    atoms = _act(env=env, skill="PickWiper")
    assert ("HoldingWiper", ("robot", "wiper_0")) in atoms, env.last_skill_error()


def test_a_sweep_that_loses_the_wiper_stops_at_once_and_says_so(*, env, monkeypatch):
    """`HoldingWiper` is re-evaluated at every control step of a sweep. The stage-5
    rerun's wiper turned 93 deg in the grasp and fell into the box; the sweep carried on
    aiming the hand from the fallen wiper ("no base pose reaches the hand target: no IK")
    and stranded the hand inside the cupboard front."""
    from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import ShelfSkills

    pull = ShelfSkills.pull

    def pull_then_drop(self, **kwargs):  # noqa: PLR0917 -- stands in for a method
        pull(self, **kwargs)
        _teleport(env=env, name="wiper_0", position=[1.235, 0.0, 0.05])  # into the box

    env.reset_to_seed(seed=0)
    assert ("HoldingWiper", ("robot", "wiper_0")) in _act(env=env, skill="PickWiper")
    monkeypatch.setattr(ShelfSkills, "pull", pull_then_drop)
    atoms = _act(env=env, skill="SweepShelf", height=0.005, speed=0.05)
    error = env.last_skill_error() or ""
    assert "wiper left the hand" in error, error
    assert "no base pose" not in error, error
    assert ("HandEmpty", ("robot",)) in atoms
    assert _arm_from_home(env=env) < RETRACT_TOLERANCE


def test_a_wiper_lost_inside_a_tolerant_motion_still_fails_the_sweep_cleanly(*, env, monkeypatch):
    """`raise_hand` catches `SkillFailed` from each 3 cm lift and stops there, so the
    per-step `HoldingWiper` check's failure was swallowed: the stroke went on with no
    wiper in hand and died on `recalibrate_grasp`'s assert. That crashed the stage-5
    CCCC determinized p1 relaunch (job 15616001_1, 2026-10-15)."""
    from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import ShelfSkills

    raise_hand = ShelfSkills.raise_hand
    dropped = []

    def drop_then_raise(self, **kwargs):  # noqa: PLR0917 -- stands in for a method
        if self.watch_wiper and not dropped:  # inside a stroke, with the check on
            dropped.append(True)
            _teleport(env=env, name="wiper_0", position=[1.235, 0.0, 0.05])  # into the box
        raise_hand(self, **kwargs)

    env.reset_to_seed(seed=0)
    assert ("HoldingWiper", ("robot", "wiper_0")) in _act(env=env, skill="PickWiper")
    monkeypatch.setattr(ShelfSkills, "raise_hand", drop_then_raise)
    atoms = _act(env=env, skill="SweepShelf", height=0.005, speed=0.05)
    error = env.last_skill_error() or ""
    assert "wiper left the hand" in error, error
    assert ("HandEmpty", ("robot",)) in atoms


# The turn a grasp that let go left in the slip counter: the 2b37e1e5b harvest's
# diagnosis found this same 126.2 deg in four attempts with fresh picks between them.
STALE_TURN_DEG = 126.2


def _turned(*, in_hand, degrees):
    """`in_hand` as if the wiper had since turned `degrees` in the pads."""
    from scipy.spatial.transform import Rotation

    turned = in_hand.copy()
    turned[:3, :3] = Rotation.from_euler("z", degrees, degrees=True).as_matrix() @ in_hand[:3, :3]
    return turned


def _left_turned_by_an_earlier_grasp(*, env):
    """Seed 0 with an empty hand, and the slip counter as a grasp that turned the wiper
    and then let go leaves it."""
    env.reset_to_seed(seed=0)
    skills = env.controllers()._skills  # noqa: SLF001
    assert skills.in_hand is None
    skills.slip, skills.slip_deg = 0.05, STALE_TURN_DEG


def test_a_fresh_pick_starts_the_slip_counter_from_zero(*, env):
    """SweepShelf bug B: the slip counter was zeroed only by `ShelfSkills.take_wiper`,
    which the practice PickWiper never calls, so a fresh grasp inherited the turn of
    the last one."""
    _left_turned_by_an_earlier_grasp(env=env)
    atoms = _act(env=env, skill="PickWiper")
    assert ("HoldingWiper", ("robot", "wiper_0")) in atoms, env.last_skill_error()
    skills = env.controllers()._skills  # noqa: SLF001
    assert skills.slip_deg < 2.0
    assert skills.slip is not None and skills.slip < 0.02


def test_the_slip_counter_keeps_counting_within_one_grasp(*, env):
    """Only taking the wiper zeroes the counter: the bookkeeping every skill re-derives
    (`KinderSkills.sync`) and re-measuring the held grasp keep the grasp's turn so far."""
    env.reset_to_seed(seed=0)
    assert ("HoldingWiper", ("robot", "wiper_0")) in _act(env=env, skill="PickWiper")
    skills = env.controllers()._skills  # noqa: SLF001
    skills.in_hand = _turned(in_hand=skills.in_hand, degrees=10.0)
    skills.track_slip()
    assert skills.slip_deg > RESEAT_TURN_DEG
    env.controllers()._kinder.sync()  # noqa: SLF001
    skills.recalibrate_grasp()
    skills.track_slip()
    assert skills.slip_deg > RESEAT_TURN_DEG


@pytest.mark.main_only
def test_a_sweep_right_after_a_fresh_pick_does_not_reseat(*, env):
    """Bug B's consequence: with the counter inherited from an earlier grasp above
    RESEAT_TURN_DEG, every stroke ended in a reseat. (On this branch's two-grasp wiper
    the sweep's own regrasp zeroes the counter before any stroke, so the reseat half of
    this is a guard for the one-grasp wiper; the regrasp's own note is not.)"""
    _left_turned_by_an_earlier_grasp(env=env)
    assert ("HoldingWiper", ("robot", "wiper_0")) in _act(env=env, skill="PickWiper")
    _act(env=env, skill="SweepShelf", height=0.015, speed=0.15)
    notes = env.controllers()._skills.notes  # noqa: SLF001
    assert not [n for n in notes if "set down after turning" in n], notes
    assert not [n for n in notes if f"{STALE_TURN_DEG:.1f} deg" in n], notes


@pytest.mark.main_only
def test_a_turn_in_this_grasp_still_reseats(*, env, monkeypatch):
    """The other side of bug B's fix: a wiper that really turns in the grasp it sweeps
    with is still set down and taken again after the stroke."""
    from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import ShelfSkills, SkillFailed

    pull = ShelfSkills.pull
    reseats = []

    def pull_and_turn(self, **kwargs):  # noqa: PLR0917 -- stands in for a method
        pull(self, **kwargs)
        self.in_hand = _turned(in_hand=self.in_hand, degrees=10.0)
        self.track_slip()

    def reseat(self, *, label):  # noqa: PLR0917 -- stands in for a method
        reseats.append((label, self.slip_deg))
        raise SkillFailed("reseat reached")

    env.reset_to_seed(seed=0)
    assert ("HoldingWiper", ("robot", "wiper_0")) in _act(env=env, skill="PickWiper")
    monkeypatch.setattr(ShelfSkills, "pull", pull_and_turn)
    monkeypatch.setattr(ShelfSkills, "reseat_wiper", reseat)
    _act(env=env, skill="SweepShelf", height=0.015, speed=0.15)
    assert len(reseats) == 1, env.last_skill_error()
    assert reseats[0][1] > RESEAT_TURN_DEG


def test_every_attempt_leaves_a_trace_naming_what_it_tried(*, env):
    from hitl_pmp.environments.sweepoffshelf3d.skill_trace import SkillTraces

    env.reset_to_seed(seed=0)
    _act(env=env, skill="SweepShelf")
    trace = SkillTraces.last(env=env)
    assert trace is not None and trace.outcome == "plan_failure"
    assert "wiper is not in hand" in (trace.error or "")
    _act(env=env, skill="PickWiper")
    trace = SkillTraces.last(env=env)
    assert trace is not None and trace.outcome == "success", trace
    assert [n.kind for n in trace.notes if n.stage == "pick_wiper"][-1] == "executed"
    assert "HoldingWiper(robot,wiper_0)" in trace.added
