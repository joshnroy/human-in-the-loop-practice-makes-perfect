"""The executor's configuration checks: the same answers, asked once.

`RackScraper.move_base` checked its chosen route's first leg twice before the base
moved (once choosing the route, once again leaving on it), and `segment_samples`
repeats each leg's ends; a configuration checked twice with nothing moved in between
gets the same answer. `tool_clear` recomputed the obstacles' half extents and the
scraper's own geom rotations, constants of the model, on every check.
"""

import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None, reason="KINDER simulator"
)


@pytest.fixture(scope="module")
def held():
    from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment

    env = RackServingEnvironment()
    env.reset_to_seed(seed=0)
    env.take_action(action=np.array([0.0, 0, 0, 0, 0]))
    assert env.last_skill_error() is None
    yield env
    env.close()


pytestmark = [pytestmark, pytest.mark.xdist_group("rack_executor_checks")]


def test_before_the_base_moves_no_configuration_is_checked_twice(*, held, monkeypatch):
    controllers = held.rack_controllers()
    shelf, tool = controllers.shelf, controllers.tool
    checked = []
    check = type(tool).check_configuration

    def recording(self, **kwargs):  # noqa: PLR0917 -- a method
        configuration = np.r_[kwargs["base"], kwargs["arm"]].astype(float).tobytes()
        checked.append((shelf.ticks, configuration))
        return check(self, **kwargs)

    monkeypatch.setattr(type(tool), "check_configuration", recording)
    config = held.backend().config
    with controllers.feedback():
        tool.move_base(
            target=np.array([tool.base[0], config.counter_y_min - 0.30, 0.0]), label="out"
        )
    before = [key for ticks, key in checked if ticks == 0]
    assert before
    assert len(before) == len(set(before))
    assert any(ticks > 0 for ticks, _ in checked)


def test_tool_clear_answers_as_before_with_its_constants_cached(*, held):
    from hitl_pmp.environments.racktoserving3d import tool as tool_module
    from hitl_pmp.environments.racktoserving3d.transit import geoms_half_extents

    controllers = held.rack_controllers()
    shelf, tool = controllers.shelf, controllers.tool
    model = shelf.model
    obstacles = np.array(list(shelf.cupboard_geoms | shelf.box_geoms), dtype=int)
    cached = tool_module.ModelConstants.half_extents(model=model, geoms=obstacles)
    np.testing.assert_array_equal(cached, geoms_half_extents(model=model, geoms=obstacles))
    assert tool_module.ModelConstants.half_extents(model=model, geoms=obstacles) is cached
    assert not cached.flags.writeable
    for geom in (tool.head_geom, tool.handle_geom):
        from scipy.spatial.transform import Rotation

        expected = Rotation.from_quat(model.geom_quat[geom][[1, 2, 3, 0]]).as_matrix()
        np.testing.assert_array_equal(tool.geom_rotation(geom=geom), expected)
