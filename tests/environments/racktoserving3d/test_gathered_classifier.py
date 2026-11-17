"""The split controllers' gather and the measured Gathered classifier, which decide
whether the toy interface's gather succeeded. Kept from the retired split-skill tests."""

import numpy as np
import pytest

from hitl_pmp.environments.racktoserving3d.controllers import RackControllers
from hitl_pmp.environments.racktoserving3d.predicates import RackServing3DAtoms
from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import SkillFailed


@pytest.mark.parametrize(
    ("ys", "gathered"),
    [([0.0, 0.05, -0.05], True), ([0.0, 0.09], True), ([0.0, 0.11], False), ([], False)],
)
def test_gathered_means_every_food_on_the_rack_lies_inside_the_blade_band(*, ys, gathered):
    corners = [np.array([[1.5, y - 0.03, 0.93], [1.5, y + 0.03, 0.93]]) for y in ys]
    assert (
        RackServing3DAtoms.gathered(on_rack_corners=corners, center_y=0.0, blade_width=0.25)
        is gathered
    )


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda c: c.gather(s=0.2, delta=0.03), "Gather requires"),
        (lambda c: c.gather(s=0.07, delta=0.0), "Gather requires"),
        (lambda c: c.gather(s=float("nan"), delta=0.03), "Gather requires"),
    ],
)
def test_gather_rejects_parameters_outside_bounds_before_touching_the_scene(*, call, match):
    controller = RackControllers.__new__(RackControllers)
    controller.gather_and_sweep_bounds = None
    with pytest.raises(SkillFailed, match=match):
        call(controller)
