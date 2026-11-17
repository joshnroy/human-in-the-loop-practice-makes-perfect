"""`split_gather_and_sweep`: GatherAndSweep as two learned skills, Gather then Sweep5,
beside the plain Sweep, under the two-route rewards."""

from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.environments.racktoserving3d.config import RackServingConfig
from hitl_pmp.environments.racktoserving3d.controllers import RackControllers
from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment
from hitl_pmp.environments.racktoserving3d.objects import RackServing3DObjects as Objects
from hitl_pmp.environments.racktoserving3d.predicates import RackServing3DAtoms
from hitl_pmp.environments.racktoserving3d.skill_provider import RackServingSkillProvider
from hitl_pmp.environments.racktoserving3d.skills import SKILL_IDS
from hitl_pmp.environments.racktoserving3d.skills import RackServing3DSkills as Skills
from hitl_pmp.environments.racktoserving3d.step_protocol import RackServingStepProtocol
from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import SkillFailed

TWO_ROUTE = {"bulk_transfer_only": True, "two_route_rewards": True}


def _provider(*, split: bool, **fields) -> RackServingSkillProvider:
    return RackServingSkillProvider(
        env=RackServingEnvironment(
            config=RackServingConfig(**TWO_ROUTE, split_gather_and_sweep=split, **fields)
        )
    )


def _ground(*, skill: object) -> GroundSkill:
    return GroundSkill(
        skill=skill,
        objects=(Objects.robot, Objects.scraper, Objects.rack, Objects.tray, *Objects.food),
    )


def test_the_split_is_off_by_default_and_needs_the_two_route_rewards():
    assert RackServingConfig().split_gather_and_sweep is False
    with pytest.raises(ValidationError, match="split_gather_and_sweep needs two_route_rewards"):
        RackServingConfig(split_gather_and_sweep=True)


def test_gather_and_sweep5_take_gather_and_sweeps_place_and_sweep_stays():
    names = [skill.name for skill in _provider(split=True).skills()]
    assert "GatherAndSweep" not in names
    assert names[1:3] == ["Gather", "Sweep5"]
    assert "Sweep" in names
    assert [s.name for s in _provider(split=False).skills()] == [
        s.name for s in Skills.robot_skills(include_placement=False, two_route=True)
    ]


def test_sweep5_needs_gathered_food_and_is_the_five_route():
    assert "Gathered" in {a.predicate.name for a in Skills.SWEEP5.preconditions}
    assert {"Served", "ServedFive"} <= {a.predicate.name for a in Skills.SWEEP5.add_effects}
    assert "Gathered" in {a.predicate.name for a in Skills.SWEEP5.delete_effects}
    assert {a.predicate.name for a in Skills.GATHER.add_effects} == {"Gathered"}
    assert "Gathered" not in {a.predicate.name for a in Skills.TWO_ROUTE_SWEEP.preconditions}
    assert (Skills.GATHER.param_dim, Skills.SWEEP5.param_dim) == (2, 1)
    assert "Gathered" in {p.name for p in _provider(split=True).predicates()}
    assert "Gathered" not in {p.name for p in _provider(split=False).predicates()}


def test_split_skills_sample_and_encode_gather_and_sweeps_own_parameters():
    provider = _provider(split=True, gather_delta_high=0.2)
    rng = np.random.default_rng(0)
    (s_bounds, delta_bounds, d_bounds) = provider.env.config.gather_and_sweep_bounds()
    rows = np.array([
        provider.sample_params(ground_skill=_ground(skill=Skills.GATHER), rng=rng)
        for _ in range(200)
    ])
    assert rows.shape == (200, 2)
    assert rows[:, 1].max() > 0.075 and rows[:, 1].max() <= delta_bounds[1]
    assert s_bounds[0] <= rows[:, 0].min() and rows[:, 0].max() <= s_bounds[1]
    (d,) = provider.sample_params(ground_skill=_ground(skill=Skills.SWEEP5), rng=rng)
    assert d_bounds[0] <= d <= d_bounds[1]
    state = State(data={})
    np.testing.assert_array_equal(
        provider.compute_action(
            ground_skill=_ground(skill=Skills.GATHER), params=np.array([0.07, 0.15]), state=state
        ),
        [SKILL_IDS["Gather"], 0, 0.07, 0.15, 0],
    )
    np.testing.assert_array_equal(
        provider.compute_action(
            ground_skill=_ground(skill=Skills.SWEEP5), params=np.array([0.1]), state=state
        ),
        [SKILL_IDS["Sweep5"], 0, 0, 0, 0.1],
    )
    with pytest.raises(ValueError, match="outside"):
        provider.compute_action(
            ground_skill=_ground(skill=Skills.GATHER), params=np.array([0.07, 0.3]), state=state
        )


def test_the_learned_sampler_reads_the_split_skills_parameters():
    provider = _provider(split=True, sampler_features="parameters")
    row = provider.hand_selected_feature_transform(
        ground_skill=_ground(skill=Skills.GATHER), state=State(data={}), params=np.array([1, 2])
    )
    assert row == [1.0, 1.0, 2.0]


def test_sweep5_is_judged_as_gather_and_sweep_was():
    assert RackServing3DAtoms.served_verdict(skill="Sweep5", plate_count=5) == frozenset({
        ("Served", ()),
        ("ServedFive", ()),
    })
    assert RackServing3DAtoms.served_verdict(skill="Sweep5", plate_count=4) == frozenset()


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


def test_a_task_ends_with_sweep5_not_with_gather():
    problem = SimpleNamespace(
        env=RackServingEnvironment(
            config=RackServingConfig(**TWO_ROUTE, split_gather_and_sweep=True)
        )
    )
    for name, reason in (("Sweep5", "bulk_attempt_finished"), ("Gather", None)):
        action = np.array([SKILL_IDS[name], 0, 0, 0, 0], dtype=float)
        assert (
            RackServingStepProtocol.evaluation_terminal_reason(problem=problem, action=action)
            == reason
        )
    gather = np.array([SKILL_IDS["Gather"], 0, 0, 0, 0], dtype=float)
    assert (
        RackServingStepProtocol.evaluation_solved_after_action(problem=problem, action=gather)
        is None
    )


def test_the_human_operations_are_unchanged():
    assert set(RackServingStepProtocol.human_skill_names()) == {
        "human_batch_return",
        "human_tool_recovery",
    }


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
