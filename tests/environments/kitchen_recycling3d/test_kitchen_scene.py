"""Tossing3D's skills driving the live kitchen scene."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.kitchen_recycling3d.controllers import RecyclingPickController
from hitl_pmp.environments.kitchen_recycling3d.domain import KitchenConfig
from hitl_pmp.environments.kitchen_recycling3d.environment import KitchenRecycling3DEnvironment
from hitl_pmp.environments.kitchen_recycling3d.placement import NEAR_BIN_POSITION
from hitl_pmp.environments.tossing3d.predicates import IN_BIN, PICK_PLANNABLE
from hitl_pmp.environments.tossing3d.skills import Tossing3DSkills
from hitl_pmp.environments.tossing3d.toss import Tossing3DToss

pytestmark = [
    pytest.mark.skipif(
        importlib.util.find_spec("kinder") is None
        or importlib.util.find_spec("kinder_models") is None,
        reason="needs KINDER",
    ),
    pytest.mark.xdist_group("kitchen_recycling3d_scene"),
]

CONFIG = Path(__file__).resolve().parents[3] / "scripts/kitchen_recycling/experiment-config.json"
SEED = 3


@pytest.fixture(scope="module")
def env():
    environment = KitchenRecycling3DEnvironment(
        kitchen_config=KitchenConfig.model_validate_json(CONFIG.read_text()), scene_bg=False
    )
    yield environment
    environment.close()


def toss_skill(*, env: KitchenRecycling3DEnvironment) -> GroundSkill:
    return GroundSkill(
        skill=Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS,
        objects=(env.robot, env.bin, env.cube, env.barrier),
    )


def test_the_far_bins_standoff_band_starts_at_the_drive_zone_edge(*, env):
    state = env.reset_to_seed(seed=SEED)
    assert state.drive_zone == env.drive_zone()
    low, high = Tossing3DToss.standoff_bounds(ground_skill=toss_skill(env=env), state=state)
    assert low == pytest.approx(2.05 - 0.71, abs=1e-4)
    assert high == 2.6


def test_pick_plannable_is_the_kitchen_pick_controllers_verdict(*, env):
    state = env.reset_to_seed(seed=SEED)
    backend = env.kinder_backend()
    controller = backend._ground_controller(
        module="tossing",
        key="pick_cube",
        object_names=(backend.robot_name, backend.cube_name, backend.barrier_name),
        state=state.object_centric,
    )
    assert isinstance(controller, RecyclingPickController)
    assert PICK_PLANNABLE.holds(state, (env.robot, env.cube))


def test_a_known_throw_lands_in_the_bin_by_kinders_classifier(*, env):
    env.reset_to_seed(seed=SEED)
    held = env.take_action(action=np.array([Tossing3DEnvironmentIds.pick, 0, 0, 0, 0.0]))
    assert env.last_skill_error() is None
    params = np.array([1.6, 180.0, 650.0])
    landed = env.take_action(action=Tossing3DToss.compute_action(params=params, state=held))
    assert IN_BIN.holds(landed, (env.cube, env.bin))
    assert env.is_solved()


def test_same_side_reset_brings_the_bin_beside_the_robot(*, env):
    env.reset_to_seed(seed=SEED)
    env.reset_movables(destination="robot_side")
    state = env.get_current_state()
    assert np.allclose(state.data[env.bin][:2], NEAR_BIN_POSITION[:2], atol=0.02)
    assert state.data[env.cube][0] < 1.15


def test_opposite_side_reset_restores_the_bin_target_on_the_mats(*, env):
    env.reset_to_seed(seed=SEED)
    env.reset_movables(destination="robot_side")
    env.reset_movables(destination="opposite_side")
    state = env.get_current_state()
    assert np.allclose(state.data[env.bin][:2], (2.05, 0.0), atol=0.02)
    assert state.data[env.cube][0] < 1.15


class Tossing3DEnvironmentIds:
    pick = float(KitchenRecycling3DEnvironment.pick_cube_id)
