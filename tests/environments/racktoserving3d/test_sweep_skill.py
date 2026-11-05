"""Standalone Sweep(d), on by default, and the lined-up start layouts.

Sweep came from the Sweep vs GatherAndSweep tradeoff probe (2026-11-05) and became
a default robot skill with the gathered-fraction start (Josh, 2026-11-05); the
lined-up layouts stay opt-in.
"""

import importlib.util
import math

import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.racktoserving3d.config import RackServingConfig
from hitl_pmp.environments.racktoserving3d.layout import RackLayout
from hitl_pmp.environments.racktoserving3d.objects import RackServing3DObjects as Objects
from hitl_pmp.environments.racktoserving3d.semantics import SupportRegion
from hitl_pmp.environments.racktoserving3d.skill_provider import RackServingActions
from hitl_pmp.environments.racktoserving3d.skills import SKILL_IDS, SWEEP_BOUNDS
from hitl_pmp.environments.racktoserving3d.skills import RackServing3DSkills as Skills

RACK = SupportRegion(x_min=1.3725, x_max=1.6275, y_min=-0.1875, y_max=0.1875, top_z=0.925)
SWEEP_OBJECTS = (Objects.robot, Objects.scraper, Objects.rack, Objects.tray, *Objects.food)


def test_sweep_is_a_robot_skill_by_default_and_can_be_disabled():
    assert RackServingConfig().sweep_skill is True
    assert Skills.robot_skills()[-1] == Skills.SWEEP
    assert Skills.SWEEP not in Skills.robot_skills(include_sweep=False)


def test_sweep_has_gather_and_sweeps_d_bounds_and_effect_shape():
    assert SWEEP_BOUNDS == (Skills.GATHER_AND_SWEEP_BOUNDS[2],) == ((0.04, 0.12),)
    sweep, gather = Skills.SWEEP, Skills.GATHER_AND_SWEEP
    assert sweep.param_dim == 1
    assert sweep.parameters == gather.parameters
    assert sweep.preconditions == gather.preconditions
    assert sweep.add_effects == gather.add_effects
    assert sweep.delete_effects == gather.delete_effects
    assert sweep.conditional_add_effects == gather.conditional_add_effects


def test_sweep_encodes_d_in_the_sweep_depth_slot():
    ground = GroundSkill(skill=Skills.SWEEP, objects=SWEEP_OBJECTS)
    action = RackServingActions.encode(ground_skill=ground, params=np.array([0.08]))
    np.testing.assert_allclose(action, [SKILL_IDS["Sweep"], 0, 0, 0, 0.08])
    with pytest.raises(ValueError, match="bounds"):
        RackServingActions.encode(ground_skill=ground, params=np.array([0.13]))


def test_the_scattered_layout_stays_the_default():
    assert RackServingConfig().start_layout == "scattered"


@pytest.mark.parametrize("kind", ["pile2col", "stack", "row"])
@pytest.mark.parametrize("seed", range(20))
def test_lined_up_layouts_sit_behind_the_front_edge_on_the_centre_line(*, kind, seed):
    layout = RackLayout.lined_up(
        rng=np.random.default_rng(seed), rack=RACK, center_y=0.0, kind=kind
    )
    assert len(layout) == 5
    xs = [x for x, _, _ in layout]
    ys = [y for _, y, _ in layout]
    assert all(RACK.x_min < x < RACK.x_max for x in xs)
    assert all(RACK.y_min < y < RACK.y_max for y in ys)
    assert abs(np.mean(ys)) < 0.01
    # The nearest piece's centre: 10 mm gap plus half its x extent, plus jitter.
    assert min(xs) - RACK.x_min < 0.045
    width = {"pile2col": 0.04, "stack": 0.012, "row": 0.14}[kind]
    assert max(ys) - min(ys) <= width
    for _, _, yaw in layout:
        nominal = 0.0 if kind == "stack" else math.pi / 2
        assert abs(yaw - nominal) <= 0.15


kinder = pytest.mark.skipif(importlib.util.find_spec("kinder") is None, reason="KINDER simulator")


@kinder
@pytest.mark.parametrize("kind", ["pile2col", "stack"])
def test_a_lined_up_start_places_five_settled_pieces_on_the_rack(*, kind):
    from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment

    env = RackServingEnvironment(config=RackServingConfig(start_layout=kind))
    try:
        env.reset_to_seed(seed=3)
        backend = env.backend()
        atoms = env.get_current_state().abstract_atoms
        for name in backend.cube_names:
            assert ("OnRack", (name, "rack_0")) in atoms
            model, data = backend.model(), backend.data()
            dof = model.jnt_dofadr[model.body_jntadr[model.body(name).id]]
            assert np.linalg.norm(data.qvel[dof : dof + 3]) <= 0.02
        ys = [backend.food_centre(name=name)[1] for name in backend.cube_names]
        assert max(ys) - min(ys) < 0.07
    finally:
        env.close()


@kinder
def test_a_gathered_fraction_mixes_stack_and_scattered_starts():
    from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment

    env = RackServingEnvironment(config=RackServingConfig(gathered_fraction=0.5))
    try:
        kinds = set()
        for seed in range(8):
            env.reset_to_seed(seed=seed)
            kinds.add(env.backend().start_kind())
        assert kinds == {"stack", "scattered"}
    finally:
        env.close()


@kinder
def test_sweep_is_refused_unless_enabled():
    from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment

    env = RackServingEnvironment(config=RackServingConfig(sweep_skill=False))
    try:
        env.reset_to_seed(seed=0)
        env.take_action(action=np.array([SKILL_IDS["Sweep"], 0, 0, 0, 0.08]))
        assert "not enabled" in (env._last_skill_error or "")
    finally:
        env.close()


def test_the_gathered_fraction_defaults_to_a_quarter():
    assert RackServingConfig().gathered_fraction == 0.25


@kinder
def test_the_human_batch_return_draws_from_the_same_start_mixture():
    from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment

    env = RackServingEnvironment(config=RackServingConfig(gathered_fraction=1.0))
    try:
        env.reset_to_seed(seed=0)
        backend = env.backend()
        backend.batch_return()
        assert backend.start_kind() == "stack"
        ys = [backend.food_centre(name=name)[1] for name in backend.cube_names]
        assert max(ys) - min(ys) < 0.02
    finally:
        env.close()


@kinder
def test_a_scattered_only_config_keeps_the_batch_return_scattered():
    from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment

    env = RackServingEnvironment(config=RackServingConfig(gathered_fraction=0.0))
    try:
        env.reset_to_seed(seed=0)
        backend = env.backend()
        backend.batch_return()
        assert backend.start_kind() == "scattered"
    finally:
        env.close()
