"""PourCubeOutOfBin's base approach on the kitchen scene.

The fixture is the state Della v5 `ees/2` refused a pour from 7,301 times: the bin
beside the robot after `ResetCubeBinSameSide`, the cube tossed into it, and the robot
parked by that toss at the kitchen's edge, within a robot half-width of the mat line.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from hitl_pmp.environments.kitchen_recycling3d.domain import KitchenConfig
from hitl_pmp.environments.kitchen_recycling3d.drive_zone import KitchenDriveZone
from hitl_pmp.environments.kitchen_recycling3d.environment import KitchenRecycling3DEnvironment
from hitl_pmp.environments.tossing3d.predicates import BIN_ON_GROUND, IN_BIN, ON_GROUND

pytestmark = [
    pytest.mark.skipif(
        importlib.util.find_spec("kinder") is None
        or importlib.util.find_spec("kinder_models") is None,
        reason="needs KINDER",
    ),
    pytest.mark.xdist_group("kitchen_recycling3d_pour"),
]

HERE = Path(__file__).resolve().parent
CONFIG = HERE.parents[2] / "scripts/kitchen_recycling/experiment-config.json"
REFUSED = HERE / "fixtures/ees2_pour_refusal.json"


@pytest.fixture(scope="module")
def env():
    environment = KitchenRecycling3DEnvironment(
        kitchen_config=KitchenConfig.model_validate_json(CONFIG.read_text()), scene_bg=False
    )
    environment.hard_reset()
    yield environment
    environment.close()


def restore_refusal(*, env: KitchenRecycling3DEnvironment):
    """The recorded state, with the cube moved 4.6 cm toward the bin's centre line: it
    rested 9.6 cm off it inside the old, wider bin, which is inside the tapered bin's
    wall."""
    plain = json.loads(REFUSED.read_text())
    plain["cube_0"][1] = plain["bin_0"][1] + 0.05
    return env.restore_plain_snapshot(plain=plain)


def test_the_pour_approach_plans_from_beside_the_mat_line_inside_the_drive_zone(*, env):
    """The approach plans from a robot whose footprint is within the barrier fixture's
    legacy 0.61 x 0.26 m box, and every waypoint stays where the robot may drive."""
    restore_refusal(env=env)
    backend = env.kinder_backend()
    state = backend.live_state()
    robot = state.get_object_from_name("robot")
    assert state.get(robot, "pos_base_x") > 1.15 - 0.305 - 0.275
    assert abs(state.get(robot, "pos_base_y")) < 0.13 + 0.275
    controller = backend._ground_controller(  # noqa: SLF001
        module="tossing",
        key="move_to_target",
        object_names=("robot", "bin_0"),
        state=state,
    )
    controller.reset(state, np.array([0.76, np.pi / 2]))
    plan = controller._current_base_motion_plan  # noqa: SLF001
    assert plan
    for pose in plan[1:]:
        assert KitchenDriveZone.allows(config=env.kitchen_config, position=(pose.x, pose.y))


def test_the_refused_state_now_pours_the_cube_out(*, env):
    before = restore_refusal(env=env)
    assert IN_BIN.holds(before, (env.cube, env.bin))
    after = env.take_action(action=np.array([env.pour_bin_id, 0, 0, 0, 0]))
    assert env.last_skill_error() is None, env.last_skill_error()
    assert sum(env.last_controller_steps()) > 300
    assert not IN_BIN.holds(after, (env.cube, env.bin))
    assert ON_GROUND.holds(after, (env.cube,))
    assert BIN_ON_GROUND.holds(after, (env.bin,))
