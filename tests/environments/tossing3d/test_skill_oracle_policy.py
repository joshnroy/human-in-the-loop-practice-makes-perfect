from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.skill_oracle_policy import SkillOraclePolicy

from .conftest import BARRIER_X, build_state, throw_pose_base

_ENV = Tossing3DEnvironment


def _next_skill(*, state) -> str:
    return SkillOraclePolicy.get_labeled_action(state=state, env=Tossing3DEnvironment()).label


def test_oracle_picks_first_when_the_hand_is_empty() -> None:
    assert _next_skill(state=build_state(holding=0.0)) == "Pick"


def test_oracle_moves_to_the_throw_pose_once_holding() -> None:
    state = build_state(holding=1.0, cube=(0.6, 0.0, 0.587))
    assert _next_skill(state=state) == "MoveToThrowPose"


def test_oracle_tosses_once_holding_and_in_position() -> None:
    state = build_state(holding=1.0, cube=(0.6, 0.0, 0.587), base=throw_pose_base())
    assert _next_skill(state=state) == "Toss"


def test_oracle_emits_its_measured_swing() -> None:
    state = build_state(holding=1.0, cube=(0.6, 0.0, 0.587), base=throw_pose_base())
    action = SkillOraclePolicy.get_labeled_action(state=state, env=Tossing3DEnvironment()).action
    assert action[0] == _ENV.SKILL_TOSS
    assert action[1] == SkillOraclePolicy.ORACLE_SWING


def test_the_oracle_swing_is_inside_the_sampling_prior() -> None:
    """An oracle constant outside the prior would mean a learned sampler literally
    cannot reach the value that works, which would make any shortfall uninterpretable."""
    env = Tossing3DEnvironment()
    assert env.swing_low <= SkillOraclePolicy.ORACLE_SWING <= env.swing_high


def test_the_oracle_pick_pose_is_inside_kinders_own_sampling_bounds() -> None:
    env = Tossing3DEnvironment()
    assert env.pick_distance_low <= SkillOraclePolicy.ORACLE_PICK_DISTANCE <= env.pick_distance_high
    assert env.pick_rot_low <= SkillOraclePolicy.ORACLE_PICK_ROT <= env.pick_rot_high


def test_oracle_keeps_trying_to_pick_a_cube_it_can_never_reach() -> None:
    """There is no recovery skill for a cube past the barrier, and the oracle does not
    pretend otherwise: it re-issues Pick and the episode runs out its horizon. Pinning
    this stops a future 'fix' from quietly inventing a retrieval the domain lacks."""
    state = build_state(cube=(BARRIER_X + 0.5, 0.0, 0.025), holding=0.0)
    assert _next_skill(state=state) == "Pick"
