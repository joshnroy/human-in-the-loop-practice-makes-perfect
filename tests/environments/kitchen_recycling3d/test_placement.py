"""Human placements never drop the bin on the robot or an item into the bin.

Both scenarios are taken from Della 16640681_0 (EES seed 0), which crashed when a
recovery placed the cube inside a bin that an earlier relocation had dropped onto the
robot.
"""

import numpy as np

from hitl_pmp.environments.kitchen_recycling3d import environment as environment_module
from hitl_pmp.environments.kitchen_recycling3d.domain import KitchenConfig
from hitl_pmp.environments.kitchen_recycling3d.environment import KitchenRecyclingEnvironment
from hitl_pmp.environments.kitchen_recycling3d.placement import KitchenHumanPlacement
from hitl_pmp.environments.kitchen_recycling3d.predicates import bin_position_retrievable
from hitl_pmp.environments.kitchen_recycling3d.types import NEAR_BIN_POSITION

CONFIG = KitchenConfig()
# The scene's static colliders (x, y, size_x, size_y): counters, stove, desk, chairs.
FURNITURE = (
    (-1.6, -0.8, 0.68, 1.14),
    (-1.6, 2.0, 0.68, 1.14),
    (-1.6, 0.6, 0.66, 0.92),
    (2.5, 1.35, 0.704, 1.74),
    (2.15, 0.67, 0.492, 0.53),
)


def _rectangle_distance(*, point, center, half):
    offset = np.maximum(np.abs(np.asarray(point) - np.asarray(center)) - np.asarray(half), 0.0)
    return float(np.linalg.norm(offset))


def test_near_bin_spot_is_unchanged_when_the_robot_is_clear():
    destination = KitchenHumanPlacement.near_bin_destination(
        config=CONFIG, robot=(-0.154, 1.409), items=((0.35, -1.41),), furniture=FURNITURE
    )
    assert destination == NEAR_BIN_POSITION


def test_near_bin_goes_next_to_a_robot_standing_on_its_spot():
    robot = (0.424, 1.248)  # 16640681_0, relocation at motor step 4677
    destination = KitchenHumanPlacement.near_bin_destination(
        config=CONFIG, robot=robot, items=((0.35, -1.41),), furniture=FURNITURE
    )
    half = np.array(CONFIG.bin_outside[:2]) / 2
    assert destination != NEAR_BIN_POSITION
    assert destination[2] == NEAR_BIN_POSITION[2]
    assert _rectangle_distance(point=robot, center=destination[:2], half=half) >= (
        CONFIG.robot_radius
    )
    assert bin_position_retrievable(config=CONFIG, position=destination, tolerance=0.025)
    assert destination[0] + half[0] <= CONFIG.kitchen_bounds[2]
    # Next to the robot: the closest clear spot, not somewhere across the kitchen.
    assert np.linalg.norm(np.array(destination[:2]) - NEAR_BIN_POSITION[:2]) < 0.5


def test_near_bin_is_not_dropped_on_an_item():
    item = (0.65, 1.4)
    destination = KitchenHumanPlacement.near_bin_destination(
        config=CONFIG, robot=(-0.5, -1.0), items=(item,), furniture=FURNITURE
    )
    half = np.array(CONFIG.bin_outside[:2]) / 2
    assert _rectangle_distance(point=item, center=destination[:2], half=half) > 0.05


def test_recovery_rejects_a_spot_inside_the_bin():
    bin_pose = ((-0.514, 1.566, -0.009), (-0.003, -0.007, -0.115, 0.993))  # drifted bin
    assert not KitchenHumanPlacement.recovery_clear_of_bin(
        config=CONFIG, position=(-0.336, 1.462), item_radius=0.0433, bin_pose=bin_pose
    )
    assert KitchenHumanPlacement.recovery_clear_of_bin(
        config=CONFIG, position=(0.531, -0.376), item_radius=0.0433, bin_pose=bin_pose
    )


def test_recovery_ignores_the_bin_at_its_target_on_the_mats():
    bin_pose = (CONFIG.bin_position, (0.0, 0.0, 0.0, 1.0))
    for position in ((-0.9, -2.1), (0.8, 2.15), (0.0, 0.0)):
        assert KitchenHumanPlacement.recovery_clear_of_bin(
            config=CONFIG, position=position, item_radius=0.0433, bin_pose=bin_pose
        )


def _move_robot_base(*, env, x, y):
    simulator = env.simulator()
    state = simulator.backend._require_state().copy()
    robot = state.get_object_from_name(simulator.backend.robot_name)
    state.set(robot, "pos_base_x", x)
    state.set(robot, "pos_base_y", y)
    simulator.scene.set_state(state)
    simulator.backend._state = state


def test_live_relocation_puts_the_bin_beside_a_robot_on_its_spot():
    env = KitchenRecyclingEnvironment(config=KitchenConfig(scatter_objects=False))
    env.hard_reset()
    try:
        _move_robot_base(env=env, x=0.424, y=1.248)
        assert env.reset_movables(destination="relocate_bin_near")
        state = env.get_current_state()
        robot, bin_ = state.data[env.robot][:2], state.data[env.bin][:2]
        half = np.array(env.config.bin_outside[:2]) / 2
        assert _rectangle_distance(point=robot, center=bin_, half=half) >= env.config.robot_radius
        assert state.data[env.bin][7] == 1.0  # near
    finally:
        env.close()


def test_live_recovery_skips_a_seeded_spot_inside_the_bin(*, monkeypatch):
    env = KitchenRecyclingEnvironment(config=KitchenConfig(scatter_objects=False))
    env.hard_reset()
    try:
        env.simulator().human(action="relocate_bin", destination=(-0.3, -1.3, 0.0))
        inside, clear = (-0.3, -1.3), (0.6, -0.4)
        draws = iter([[inside], [clear]])
        monkeypatch.setattr(environment_module, "initial_positions", lambda **_: next(draws))
        assert env.reset_movables(destination="recover:cube_control")
        cube = env.simulator().episode.observations["cube_control"].position
        np.testing.assert_allclose(cube[:2], clear, atol=0.025)
    finally:
        env.close()
