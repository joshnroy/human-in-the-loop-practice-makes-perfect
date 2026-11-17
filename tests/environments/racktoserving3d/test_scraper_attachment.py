"""Attachment transitions must preserve physical contacts and reject failed returns."""

import importlib.util
from unittest.mock import patch

import numpy as np
import pytest

from hitl_pmp.environments.racktoserving3d.config import RackServingConfig
from hitl_pmp.environments.racktoserving3d.scraper_attachment import ScraperAttachment
from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import SkillFailed

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None, reason="KINDER simulator"
)


def test_home_constraint_lifecycle_preserves_collision_masks_and_refuses_failed_return():
    from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment

    env = RackServingEnvironment(config=RackServingConfig(scraper_attachment="home_weld"))
    try:
        env.reset_to_seed(seed=0)
        backend = env.backend()
        tool, shelf = env.rack_controllers().tool, env.rack_controllers().shelf
        model, data = backend.model(), backend.data()
        weld = ScraperAttachment.home_id(model=model)
        before = tool.wiper_pose().copy()
        masks = np.column_stack([model.geom_contype, model.geom_conaffinity]).copy()
        assert data.eq_active[weld]
        shelf.hold(count=100, label="attachment_test_parked")
        np.testing.assert_allclose(tool.wiper_pose(), before, atol=1e-5)
        assert tool.at_home()
        tool.accept_grasp()
        assert not data.eq_active[weld]
        before = data.qpos.copy()
        with (
            patch.object(tool, "table_support", return_value=False),
            pytest.raises(SkillFailed, match="home seat"),
        ):
            tool.park_after_seating()
        np.testing.assert_array_equal(data.qpos, before)
        assert not data.eq_active[weld]
        tool.initialize_rest()
        assert data.eq_active[weld]
        np.testing.assert_array_equal(
            np.column_stack([model.geom_contype, model.geom_conaffinity]), masks
        )
    finally:
        env.close()


@pytest.mark.parametrize("mode", ["home_weld", "grasp_weld"])
def test_weld_pick_and_return_execute(*, mode):
    from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment

    env = RackServingEnvironment(config=RackServingConfig(scraper_attachment=mode))
    try:
        env.reset_to_seed(seed=861111389)
        backend = env.backend()
        weld = ScraperAttachment.home_id(model=backend.model())
        env.take_action(action=np.array([0.0, 0.0, 0.0, 0.0, 0.0]))
        assert env.last_skill_error() is None
        assert not backend.data().eq_active[weld]
        assert backend.held(name="wiper_0")
        if mode == "grasp_weld":
            shelf, tool = env.rack_controllers().shelf, env.rack_controllers().tool
            assert ScraperAttachment.grasped(model=backend.model(), data=backend.data())
            with patch(
                "hitl_pmp.environments.sweepoffshelf3d.scripted_skills.ShelfSkills.pads_hold_wiper",
                return_value=False,
            ):
                assert shelf.pads_hold_wiper()
                tool.require_held(label="test_welded_hold")
            recorded = backend.motion_state()["scraper_attachment_state"]
            ScraperAttachment.release_grasp(model=backend.model(), data=backend.data())
            assert not backend.held(name="wiper_0")
            ScraperAttachment.restore(model=backend.model(), data=backend.data(), state=recorded)
            assert backend.held(name="wiper_0")
            shelf.hold(count=50, label="test_welded_hold")
        env.take_action(action=np.array([2.0, 0.0, 0.0, 0.0, 0.0]))
        assert env.last_skill_error() is None
        assert backend.data().eq_active[weld]
        assert env.rack_controllers().tool.at_home()
        assert not backend.held(name="wiper_0")
    finally:
        env.close()


def test_failed_grasp_keeps_home_anchor_and_does_not_attach_to_hand():
    from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment

    env = RackServingEnvironment(config=RackServingConfig(scraper_attachment="grasp_weld"))
    try:
        env.reset_to_seed(seed=861111389)
        backend, tool = env.backend(), env.rack_controllers().tool
        with patch.object(tool, "pads_hold_grip", return_value=False):
            env.take_action(action=np.array([0.0, 0.0, 0.0, 0.0, 0.0]))
        assert "both pads" in env.last_skill_error()
        assert backend.data().eq_active[ScraperAttachment.home_id(model=backend.model())]
        assert not backend.held(name="wiper_0")
        tool.initialize_rest()
        assert not ScraperAttachment.grasped(model=backend.model(), data=backend.data())
    finally:
        env.close()
