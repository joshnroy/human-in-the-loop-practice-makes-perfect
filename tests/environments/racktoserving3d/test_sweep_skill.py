"""The sweep controller, Sweep(d), under the scene's internal driver, and the lined-up
start layouts. sweep3 and sweep5 call it (`RackServingToyEnvironment`)."""

import importlib.util
import math

import numpy as np
import pytest

from hitl_pmp.environments.racktoserving3d.config import RackServingConfig
from hitl_pmp.environments.racktoserving3d.layout import RackLayout
from hitl_pmp.environments.racktoserving3d.objects import RackServing3DObjects as Objects
from hitl_pmp.environments.racktoserving3d.semantics import SupportRegion
from hitl_pmp.environments.racktoserving3d.skill_provider import RackServingActions
from hitl_pmp.environments.racktoserving3d.skills import (
    GATHER_AND_SWEEP_BOUNDS,
    SKILL_IDS,
    SWEEP_BOUNDS,
)
from hitl_pmp.environments.racktoserving3d.toy_interface import SWEEP3, SWEEP5, TOY_SKILL_IDS

RACK = SupportRegion(x_min=1.3725, x_max=1.6275, y_min=-0.1875, y_max=0.1875, top_z=0.925)
SWEEP_OBJECTS = (Objects.robot, Objects.scraper, Objects.rack, Objects.tray, *Objects.food)


def test_sweep_takes_gather_and_sweeps_d_bounds():
    assert SWEEP_BOUNDS == (GATHER_AND_SWEEP_BOUNDS[2],) == ((0.0, 0.12),)


@pytest.mark.parametrize("name", [SWEEP3, SWEEP5])
def test_both_sweeps_encode_d_in_the_sweep_depth_slot(*, name):
    action = RackServingActions.encode(name=name, params=np.array([0.08]), bounds=SWEEP_BOUNDS)
    np.testing.assert_allclose(action, [TOY_SKILL_IDS[name], 0, 0, 0, 0.08])
    with pytest.raises(ValueError, match="bounds"):
        RackServingActions.encode(name=name, params=np.array([0.13]), bounds=SWEEP_BOUNDS)


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
