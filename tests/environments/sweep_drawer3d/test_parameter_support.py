"""Approved fixed supports vary without mutating upstream nominal controllers."""

import numpy as np
import pytest


def test_approved_supports_are_bounded_and_nondegenerate():
    pytest.importorskip("shapely")
    from hitl_pmp.core.method.types import GroundSkill
    from hitl_pmp.environments.sweep_drawer3d.environment import SweepDrawerEnvironment
    from hitl_pmp.environments.sweep_drawer3d.skill_provider import SweepDrawerSkillProvider
    from hitl_pmp.environments.sweep_drawer3d.symbolic import SweepSymbols

    provider = SweepDrawerSkillProvider(env=SweepDrawerEnvironment())
    rng = np.random.default_rng(20260928)
    for skill in provider.deployment_skills():
        ground = GroundSkill(skill=skill, objects=(SweepSymbols.SCENE,))
        draws = np.array([provider.sample_params(ground_skill=ground, rng=rng) for _ in range(24)])
        bounds = np.array(provider.stock_parameter_bounds[skill.name])
        assert len(np.unique(draws, axis=0)) == 24
        assert np.all(draws >= bounds[:, 0]) and np.all(draws <= bounds[:, 1])
    provider.stock_parameter_bounds["OpenDrawer"] = ((.8, .8), (-4., -3.))
    with pytest.raises(ValueError, match="degenerate"):
        provider.validate_trainable_support()
