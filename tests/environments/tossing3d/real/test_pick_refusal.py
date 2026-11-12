"""`PickPlannable` on the real robot remembers a pick that failed without touching the cube.

The robot side's `can_pick` dry run checks the planned hover and grasp; it cannot see a
pick that its own wrist servo then refuses to finish. On 2026-11-11 that left the cube in
the near bin with `can_pick` feasible and every attempt refused at the first 1 cm servo
step (`fixtures/pick_servo_step_refused_20261111.json`). The memory is an observed fact,
not a mask: the atom goes false while the cube and the bin stay where they were, and
whatever else is applicable -- the pour -- is the planner's to choose.
"""

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import Field

from hitl_pmp.environments.tossing3d.real.backend import RealSceneSnapshot
from hitl_pmp.environments.tossing3d.real.contract import ObserveReply, PickReply
from hitl_pmp.environments.tossing3d.real.environment import RealTossing3DEnvironment
from hitl_pmp.environments.tossing3d.real.fake_robot import (
    FAKE_SKILL_SECONDS,
    FakePerson,
    FakeTossingRobot,
)
from hitl_pmp.environments.tossing3d.real.frame import BIN_SPOTS, CUBE_SPOT
from hitl_pmp.environments.tossing3d.real.pick_refusal import (
    BIN_MOVED_M,
    CUBE_MOVED_M,
    RealPickRefusal,
    RealPickRefusalMemory,
)
from hitl_pmp.environments.tossing3d.sides import Tossing3DSide
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.planning.grounding import SkillGrounder

ROBOT_SPOT = BIN_SPOTS[Tossing3DSide.ROBOT]
INCIDENT = Path(__file__).parent / "fixtures" / "pick_servo_step_refused_20261111.json"
# The gripper's reading in the incident's failed pick: open, never closed.
OPEN_READING = 0.008733651041984558
SERVO_NOTE = "lateral servo step did not arrive (refused or short)"


class _ScriptedRobot(FakeTossingRobot):
    """A fake robot whose next picks and looks can be scripted: `misses` are pick replies
    returned as they are, the scene untouched unless `nudge` moves the cube first;
    `looks` are observe replies returned as they are."""

    misses: list[dict[str, Any]] = Field(default_factory=list)
    nudge: tuple[float, float] | None = None
    looks: list[dict[str, Any]] = Field(default_factory=list)

    def observe(self) -> dict[str, Any]:
        if self.looks:
            self.calls.append(("observe", ()))
            return self.looks.pop(0)
        return super().observe()

    def pick(self) -> dict[str, Any]:
        if not self.misses:
            return super().pick()
        self.seconds += FAKE_SKILL_SECONDS["pick"]
        self.calls.append(("pick", ()))
        if self.nudge is not None and self.cube is not None:
            self.cube = (self.cube[0] + self.nudge[0], self.cube[1] + self.nudge[1], self.cube[2])
        return self.misses.pop(0)


def _never_closed(*, robot: FakeTossingRobot) -> dict[str, Any]:
    assert robot.cube is not None
    return {
        "held": False,
        "gripper": OPEN_READING,
        "note": SERVO_NOTE,
        "refused": False,
        "reason": None,
        "cube": {"x": robot.cube[0], "y": robot.cube[1], "yaw_deg": 0.0, "in_bin": True},
        "stand": [robot.cube[0] + 0.55, robot.cube[1], 3.14],
        "attempts": [],
    }


def _env() -> tuple[RealTossing3DEnvironment, _ScriptedRobot]:
    """The bin at the close spot with the cube in it, as a toss leaves them."""
    robot = _ScriptedRobot()
    person = FakePerson(robot=robot, write=lambda _line: None)
    env = RealTossing3DEnvironment(tossing_robot=robot, person=person, clock=robot.now)
    env.hard_reset()
    env.reset_movables(destination=Tossing3DSide.ROBOT.value)
    robot.put(cube=ROBOT_SPOT.center, bin_center=ROBOT_SPOT.center, bin_yaw=ROBOT_SPOT.yaw)
    _look_again(env=env)
    return env, robot


def _look_again(*, env: RealTossing3DEnvironment) -> None:
    env.real_backend()._seen = None  # noqa: SLF001 -- something moved: look again.
    env.reobserve(state=env.get_current_state())


def _pick(*, env: RealTossing3DEnvironment) -> None:
    env.take_action(action=np.array([env.pick_cube_id, 0, 0, 0, 0], dtype=float))


def _atoms(*, env: RealTossing3DEnvironment) -> set[str]:
    return {name for name, _ in env.get_current_state().abstract_atoms or ()}


def _applicable(*, env: RealTossing3DEnvironment) -> set[str]:
    provider = Tossing3DSkillProvider(env=env)
    atoms = SkillGrounder.abstract_state(
        state=env.get_current_state(), objects=provider.objects(), predicates=provider.predicates()
    )
    return {
        ground.skill.name
        for ground in SkillGrounder.applicable_ground_skills(
            skills=provider.skills(), objects=provider.objects(), true_atoms=atoms
        )
    }


def _failed_untouched(*, env: RealTossing3DEnvironment, robot: _ScriptedRobot) -> None:
    robot.misses.append(_never_closed(robot=robot))
    _pick(env=env)
    assert "did not hold" in (env.last_skill_error() or "")


def test_a_pick_that_never_closed_on_an_unmoved_cube_is_not_plannable_there_again() -> None:
    env, robot = _env()
    assert "PickPlannable" in _atoms(env=env)
    assert "PickCube" in _applicable(env=env)
    _failed_untouched(env=env, robot=robot)
    assert "PickPlannable" not in _atoms(env=env)
    # The robot's own dry run still says feasible: the atom is the memory's doing.
    snapshot = env.get_current_state().object_centric
    assert isinstance(snapshot, RealSceneSnapshot)
    assert snapshot.can_pick is True
    assert snapshot.pick_refusal is not None
    assert snapshot.pick_refusal.note == SERVO_NOTE
    diagnostics = env.real_backend().abstraction_diagnostics()
    assert diagnostics["can_pick"] is True
    assert diagnostics["pick_refusal"]["note"] == SERVO_NOTE
    # Nothing is forced: the pick is gone and the pour is still there to choose.
    applicable = _applicable(env=env)
    assert "PickCube" not in applicable
    assert "PourCubeOutOfBin" in applicable


def test_the_refusal_stands_through_a_look_that_finds_nothing_moved() -> None:
    env, robot = _env()
    _failed_untouched(env=env, robot=robot)
    assert robot.cube is not None
    # Under a millimetre of camera noise, and a yaw the cube's symmetry makes arbitrary.
    robot.cube = (robot.cube[0] + 0.0008, robot.cube[1] - 0.0006, robot.cube[2] + 0.13)
    _look_again(env=env)
    assert "PickPlannable" not in _atoms(env=env)


def test_a_cube_that_moved_is_plannable_again() -> None:
    env, robot = _env()
    _failed_untouched(env=env, robot=robot)
    assert robot.cube is not None
    robot.cube = (robot.cube[0] + 2 * CUBE_MOVED_M, robot.cube[1], robot.cube[2])
    _look_again(env=env)
    assert "PickPlannable" in _atoms(env=env)
    assert env.get_current_state().object_centric.pick_refusal is None
    # Forgotten for good: moving it back is a new scene, not the remembered one.
    robot.cube = (robot.cube[0] - 2 * CUBE_MOVED_M, robot.cube[1], robot.cube[2])
    _look_again(env=env)
    assert "PickPlannable" in _atoms(env=env)


def test_a_bin_that_moved_is_plannable_again() -> None:
    env, robot = _env()
    _failed_untouched(env=env, robot=robot)
    assert robot.bin is not None
    robot.bin = (robot.bin[0], robot.bin[1] + 2 * BIN_MOVED_M, robot.bin[2])
    _look_again(env=env)
    assert "PickPlannable" in _atoms(env=env)


def test_a_pour_that_empties_the_bin_makes_the_pick_plannable_again() -> None:
    env, robot = _env()
    _failed_untouched(env=env, robot=robot)
    env.take_action(action=np.array([env.pour_bin_id, 0, 0, 0, 0], dtype=float))
    assert env.last_skill_error() is None
    assert "PickPlannable" in _atoms(env=env)
    assert "PickCube" in _applicable(env=env)


def _floor_env() -> tuple[RealTossing3DEnvironment, _ScriptedRobot]:
    """The cube on the floor at the reset's own spot, outside the bin at the close spot."""
    env, robot = _env()
    robot.put(cube=CUBE_SPOT, bin_center=ROBOT_SPOT.center, bin_yaw=ROBOT_SPOT.yaw)
    _look_again(env=env)
    return env, robot


def test_a_refused_cube_on_the_floor_leaves_no_robot_skill_and_both_human_resets() -> None:
    env, robot = _floor_env()
    assert _applicable(env=env) == {"PickCube"}
    _failed_untouched(env=env, robot=robot)
    assert "PickPlannable" not in _atoms(env=env)
    # Nothing the robot can do; the person's resets need nothing, so they are still there.
    assert _applicable(env=env) == set()
    resets = Tossing3DSkillProvider(env=env).human_cube_bin_reset_skills()
    assert {reset.skill.name for reset in resets} == {
        "ResetCubeBinSameSide",
        "ResetCubeBinOppositeSide",
    }
    assert all(not reset.skill.preconditions for reset in resets)


def test_a_human_reset_forgets_the_refusal_even_with_the_cube_put_back_on_its_spot() -> None:
    """The reset's operator promises `PickPlannable`, and a person has handled the cube.
    A fake person puts it back to the millimetre: a memory that waited for the cube to
    be seen elsewhere would leave the resets as the only skills, for ever."""
    env, robot = _floor_env()
    _failed_untouched(env=env, robot=robot)
    before = robot.cube
    env.reset_movables(destination=Tossing3DSide.ROBOT.value)
    assert robot.cube == before
    assert env.real_backend().pick_refusal() is None
    assert "PickPlannable" in _atoms(env=env)
    assert "PickCube" in _applicable(env=env)


def test_a_gripper_that_closed_and_missed_stays_retryable() -> None:
    env, robot = _env()
    miss = _never_closed(robot=robot) | {
        "gripper": 0.93,
        "note": "gripper closed on nothing (read 0.930)",
    }
    robot.misses.append(miss)
    _pick(env=env)
    assert "did not hold" in (env.last_skill_error() or "")
    assert "PickPlannable" in _atoms(env=env)
    assert env.get_current_state().object_centric.pick_refusal is None


def test_a_failed_pick_that_moved_the_cube_stays_retryable() -> None:
    env, robot = _env()
    robot.nudge = (0.03, 0.0)
    robot.misses.append(_never_closed(robot=robot))
    _pick(env=env)
    assert "PickPlannable" in _atoms(env=env)


def test_a_pick_refused_before_moving_is_the_dry_runs_business_not_the_memorys() -> None:
    env, robot = _env()
    refused = _never_closed(robot=robot) | {
        "refused": True,
        "reason": "no top-down pick into the bin",
        "note": "no top-down pick into the bin",
    }
    robot.misses.append(refused)
    _pick(env=env)
    assert "refused" in (env.last_skill_error() or "")
    assert env.get_current_state().object_centric.pick_refusal is None


def test_the_refusal_survives_a_resume_and_an_old_checkpoint_resumes_without_one() -> None:
    env, robot = _env()
    _failed_untouched(env=env, robot=robot)
    saved = pickle.loads(pickle.dumps(env.get_current_state()))
    person = FakePerson(robot=robot, write=lambda _line: None)
    resumed = RealTossing3DEnvironment(tossing_robot=robot, person=person, clock=robot.now)
    resumed.reobserve(state=saved)
    assert "PickPlannable" not in _atoms(env=resumed)
    # A snapshot pickled before the field existed: no memory, and it still loads.
    old = env.get_current_state().object_centric
    state = old.__getstate__()
    del state["__dict__"]["pick_refusal"]
    revived = RealSceneSnapshot.__new__(RealSceneSnapshot)
    revived.__setstate__(state)
    assert revived.pick_refusal is None
    assert revived.reply == old.reply


def test_the_evaluation_environment_starts_with_no_memory() -> None:
    env, robot = _env()
    _failed_untouched(env=env, robot=robot)
    person = FakePerson(robot=robot, write=lambda _line: None)
    evaluation = RealTossing3DEnvironment(
        tossing_robot=robot, person=person, clock=robot.now, evaluation=True
    )
    assert evaluation.real_backend().pick_refusal() is None
    assert env.real_backend().pick_refusal() is not None


def test_the_recording_rule_reads_only_the_replies_stable_fields() -> None:
    calls = json.loads(INCIDENT.read_text())["calls"]
    before = ObserveReply.model_validate(calls[0]["reply"])
    pick = PickReply.model_validate(calls[1]["reply"])
    after = ObserveReply.model_validate(calls[2]["reply"])
    refusal = RealPickRefusalMemory.recorded(before=before, pick=pick, after=after)
    assert isinstance(refusal, RealPickRefusal)
    assert refusal.cube == pytest.approx((after.cube[0], after.cube[1]))  # type: ignore[index]
    assert refusal.note == SERVO_NOTE
    assert RealPickRefusalMemory.stands(refusal=refusal, reply=after)
    # The incident's own yaw jitter on the unmoved cube (0.151 -> 0.016 rad) is ignored.
    assert RealPickRefusalMemory.stands(refusal=refusal, reply=before)
    held = PickReply.model_validate(calls[3]["reply"])
    assert RealPickRefusalMemory.recorded(before=after, pick=held, after=after) is None


def test_the_incidents_replies_end_not_plannable_after_the_first_failed_pick() -> None:
    """2026-11-11 20:47 US Eastern, `tidybot-nuc-prpl`: the look, the pick and the look
    after it, as the skill server logged them."""
    calls = json.loads(INCIDENT.read_text())["calls"]
    assert [call["call"] for call in calls] == ["observe", "pick", "observe", "pick"]
    env, robot = _env()
    robot.looks.append(calls[0]["reply"])
    _look_again(env=env)
    before = _atoms(env=env)
    assert {"HandEmpty", "MovableInGoalRegion", "PickPlannable"} <= before
    assert "PickCube" in _applicable(env=env)
    robot.misses.append(calls[1]["reply"])
    robot.looks.append(calls[2]["reply"])
    _pick(env=env)
    after = _atoms(env=env)
    assert {"HandEmpty", "MovableInGoalRegion", "BinOnGround"} <= after
    assert "PickPlannable" not in after
    assert before - after == {"PickPlannable"}
    applicable = _applicable(env=env)
    # The second dispatch the run made at 20:49 is no longer on offer; the pour is.
    assert "PickCube" not in applicable
    assert "PourCubeOutOfBin" in applicable
