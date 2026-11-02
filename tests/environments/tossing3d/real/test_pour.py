"""`PourCubeOutOfBin` on the real robot: the skill server's `dump_bin`, offered where the
sim offers it (the robot and the bin on the same side of the tape, the cube in the bin)."""

import math

import numpy as np

from hitl_pmp.environments.tossing3d.predicates import IN_BIN, ON_GROUND
from hitl_pmp.environments.tossing3d.real.backend import PlacementRequest, RealSkillStepCosts
from hitl_pmp.environments.tossing3d.real.contract import DumpReply
from hitl_pmp.environments.tossing3d.real.environment import RealTossing3DEnvironment
from hitl_pmp.environments.tossing3d.real.fake_robot import FakePerson, FakeTossingRobot
from hitl_pmp.environments.tossing3d.real.frame import BIN_SPOTS, ROOM_FRAME
from hitl_pmp.environments.tossing3d.sides import Tossing3DSide
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.planning.grounding import SkillGrounder

ROBOT_SPOT = BIN_SPOTS[Tossing3DSide.ROBOT]


class _PutsTheBottleInTheBin(FakePerson):
    """Puts the bin where asked, and the bottle in it rather than on the floor."""

    def placed(self, *, request: PlacementRequest) -> None:
        center = request.bin.center
        self.robot.put(cube=center, bin_center=center, bin_yaw=request.bin.yaw)


def _env() -> tuple[RealTossing3DEnvironment, FakeTossingRobot]:
    robot = FakeTossingRobot()
    person = _PutsTheBottleInTheBin(robot=robot, write=lambda _line: None)
    env = RealTossing3DEnvironment(tossing_robot=robot, person=person)
    env.hard_reset()
    return env, robot


def _bottle_in_the_robot_side_bin(
    *, env: RealTossing3DEnvironment, robot: FakeTossingRobot
) -> None:
    env.reset_movables(destination=Tossing3DSide.ROBOT.value)
    assert robot.bin is not None and robot.bin[:2] == ROBOT_SPOT.center


def _pour(*, env: RealTossing3DEnvironment) -> None:
    env.take_action(action=np.array([env.pour_bin_id, 0, 0, 0, 0], dtype=float))


def test_the_real_env_offers_the_pour_where_the_sim_does() -> None:
    env, robot = _env()
    provider = Tossing3DSkillProvider(env=env)
    assert "PourCubeOutOfBin" in {s.name for s in provider.skills()}
    _bottle_in_the_robot_side_bin(env=env, robot=robot)
    atoms = SkillGrounder.abstract_state(
        state=env.get_current_state(), objects=provider.objects(), predicates=provider.predicates()
    )
    applicable = SkillGrounder.applicable_ground_skills(
        skills=provider.skills(), objects=provider.objects(), true_atoms=atoms
    )
    assert "PourCubeOutOfBin" in {s.skill.name for s in applicable}


def test_a_pour_is_one_dump_bin_call_charged_its_sim_mean_ticks() -> None:
    env, robot = _env()
    _bottle_in_the_robot_side_bin(env=env, robot=robot)
    counts = {"after": 0}

    def after() -> None:
        counts["after"] += 1

    env.set_control_step_observers(before=None, after=after)
    robot.calls.clear()
    _pour(env=env)
    # The robot calls besides the looks (`observe`, and the `can_pick` dry run each one
    # makes for `PickPlannable`).
    assert [name for name, _ in robot.calls if name not in {"observe", "can_pick"}] == ["dump_bin"]
    assert env.last_skill_error() is None
    assert counts["after"] == RealSkillStepCosts().pour_bin == 680
    state = env.get_current_state()
    assert not IN_BIN.holds(state=state, objects=(env.cube, env.bin))
    assert ON_GROUND.holds(state=state, objects=(env.cube,))


def test_a_refused_dump_moved_nothing_and_is_not_charged() -> None:
    env, robot = _env()
    _bottle_in_the_robot_side_bin(env=env, robot=robot)
    robot.holding = True  # the robot side refuses to dump while holding
    _pour(env=env)
    assert env.last_controller_steps() == (0,)
    assert "refused" in (env.last_skill_error() or "")


def test_a_dump_that_moved_but_did_not_empty_the_bin_is_a_charged_failure() -> None:
    env, robot = _env()
    _bottle_in_the_robot_side_bin(env=env, robot=robot)
    robot.dump_grips = False  # closed on nothing at the rim: let go and withdrew
    _pour(env=env)
    assert env.last_controller_steps() == (RealSkillStepCosts().pour_bin,)
    assert "did not empty" in (env.last_skill_error() or "")
    assert IN_BIN.holds(state=env.get_current_state(), objects=(env.cube, env.bin))


def test_the_robot_side_bin_spot_is_where_every_real_dump_was_tested() -> None:
    """2026-11-02: every real dump stood the bin at (-0.30, -1.00), so the pour lands clear
    of the ceiling cameras' overlap band (map y -0.15..0.18)."""
    assert ROBOT_SPOT.center == (-0.30, -1.00)
    assert ROOM_FRAME.to_domain(x=ROBOT_SPOT.center[0], y=ROBOT_SPOT.center[1])[0] < 0.0
    assert math.isclose(ROBOT_SPOT.yaw, math.pi / 2)


def test_the_dump_reply_is_the_skill_servers() -> None:
    """The fields of prpl-tidybot's `dump_bin` reply this package reads."""
    reply = DumpReply.model_validate({
        "dumped": True,
        "refused": False,
        "reason": None,
        "note": None,
        "bin": {"x": -0.3, "y": -1.0, "yaw": 1.57, "length": 0.365, "width": 0.26, "height": 0.384},
        "stages": [{"stage": "hover"}],
        "gripper": 0.93,
        "stand": [0.48, -1.03, 3.06],
        "face": 0,
        "turn_deg": -90.0,
        "pour_point": [0.34, -0.26],
    })
    assert reply.dumped and reply.pour_point == (0.34, -0.26)
