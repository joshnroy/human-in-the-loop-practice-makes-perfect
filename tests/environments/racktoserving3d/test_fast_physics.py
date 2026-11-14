"""Opt-in simulator speed settings: a coarser physics timestep and a lean control step."""

import importlib.util

import numpy as np
import pytest
from pydantic import ValidationError

from hitl_pmp.environments.racktoserving3d.config import RackServingConfig
from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment

needs_kinder = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None, reason="KINDER is not installed"
)


def test_speed_settings_are_off_by_default():
    config = RackServingConfig()
    assert config.physics_timestep is None
    assert config.lean_control_step is False
    assert config.base_step_limit is None
    assert config.planned_joint_step is None
    assert config.settle_until_still is False


@needs_kinder
@pytest.mark.parametrize(("until_still", "most"), [(False, 40), (True, 3)])
def test_a_stroke_waits_for_food_only_until_it_is_still_when_asked(*, until_still, most):
    env = RackServingEnvironment(config=RackServingConfig(settle_until_still=until_still))
    try:
        env.reset_to_seed(seed=0)
        controllers = env.rack_controllers()
        backend = env.backend()
        # Let the episode's freshly placed food come to rest first.
        controllers.shelf.hold(count=40, label="test_rest")
        before = backend.control_steps()
        controllers.settle_food()
        taken = backend.control_steps() - before
        assert taken == most if not until_still else 1 <= taken <= most
    finally:
        env.close()


@needs_kinder
@pytest.mark.parametrize(("configured", "points"), [(None, 6), (0.1, 3)])
def test_a_planned_arm_path_is_commanded_in_steps_of_the_planned_joint_step(
    *, configured, points, monkeypatch
):
    from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import SkillFailed

    env = RackServingEnvironment(config=RackServingConfig(planned_joint_step=configured))
    try:
        env.reset_to_seed(seed=0)
        tool = env.rack_controllers().tool
        shelf = tool.shelf
        start = shelf.data.qpos[shelf.qa].copy()
        commanded: list[tuple[float, ...]] = []
        # The arm never moves, so each point is commanded until its catch-up runs out.
        monkeypatch.setattr(
            tool, "step", lambda *, arm, label: commanded.append(tuple(np.round(arm, 6)))
        )
        with pytest.raises(SkillFailed):
            tool.follow_arm_path(path=[start, start + np.r_[0.3, np.zeros(6)]], label="test")
        assert len(dict.fromkeys(commanded)) == points
    finally:
        env.close()


@needs_kinder
def test_a_base_step_limit_sets_how_far_each_control_step_moves_the_base():
    env = RackServingEnvironment(config=RackServingConfig(base_step_limit=0.04))
    try:
        env.reset_to_seed(seed=0)
        shelf = env.rack_controllers().shelf
        assert shelf.base_step_limit == 0.04
        start = shelf.data.qpos[shelf.ba].copy()
        target = start + np.array([0.0, 0.0, 0.5])
        shelf.base_target = target
        shelf.hold(count=1, label="test_turn")
        # The commanded heading moves the configured 0.04 rad, not the default 0.015.
        assert shelf.data.qpos[shelf.ba][2] - start[2] > 0.02
    finally:
        env.close()


@pytest.mark.parametrize("timestep", [0.0003, 0.0007, 0.03])
def test_a_physics_timestep_must_tile_the_control_period_and_not_refine_it(*, timestep):
    with pytest.raises(ValidationError):
        RackServingConfig(physics_timestep=timestep)


@pytest.mark.parametrize("timestep", [0.0005, 0.001, 0.002, 0.0025, 0.005])
def test_a_physics_timestep_that_tiles_the_control_period_is_accepted(*, timestep):
    assert RackServingConfig(physics_timestep=timestep).physics_timestep == timestep


def _hold(*, config: RackServingConfig, count: int) -> tuple[np.ndarray, float, int]:
    import mujoco

    env = RackServingEnvironment(config=config)
    calls = [0]
    original = mujoco.mj_step

    def counted(*args, **kwargs):
        calls[0] += 1
        return original(*args, **kwargs)

    try:
        env.reset_to_seed(seed=0)
        backend = env.backend()
        start = float(backend.data().time)
        mujoco.mj_step = counted
        try:
            env.rack_controllers().shelf.hold(count=count, label="test_hold")
        finally:
            mujoco.mj_step = original
        return (
            backend.data().qpos.copy(),
            float(backend.data().time) - start,
            calls[0],
        )
    finally:
        env.close()


@needs_kinder
def test_a_lean_control_step_advances_exactly_the_physics_the_native_step_does():
    native, native_time, native_calls = _hold(config=RackServingConfig(), count=5)
    lean, lean_time, lean_calls = _hold(config=RackServingConfig(lean_control_step=True), count=5)
    assert native_calls == lean_calls == 5 * 200
    assert lean_time == pytest.approx(native_time) == pytest.approx(0.5)
    np.testing.assert_array_equal(lean, native)


@needs_kinder
def test_a_coarser_timestep_covers_each_control_period_in_fewer_physics_steps():
    config = RackServingConfig(physics_timestep=0.002)
    env = RackServingEnvironment(config=config)
    try:
        env.reset_to_seed(seed=0)
        assert env.backend().model().opt.timestep == 0.002
    finally:
        env.close()
    _, elapsed, calls = _hold(config=config, count=5)
    assert calls == 5 * 50
    assert elapsed == pytest.approx(0.5)
