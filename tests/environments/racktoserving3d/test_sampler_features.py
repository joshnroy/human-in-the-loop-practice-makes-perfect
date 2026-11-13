"""The learned sampler's classifier row for Sweep and GatherAndSweep, per `sampler_features`."""

import argparse

import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.environments.racktoserving3d.cli import RackServingCli
from hitl_pmp.environments.racktoserving3d.config import RackServingConfig
from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment
from hitl_pmp.environments.racktoserving3d.objects import RackServing3DObjects as Objects
from hitl_pmp.environments.racktoserving3d.skill_provider import RackServingSkillProvider
from hitl_pmp.environments.racktoserving3d.skills import RackServing3DSkills as Skills
from hitl_pmp.methods.practice_makes_perfect.wrapped_sampler import LearnedSkillSampler

_BULK_OBJECTS = (Objects.robot, Objects.scraper, Objects.rack, Objects.tray, *Objects.food)
_RACK = np.array([1.3725, -0.1875, 1.6275, 0.1875, 0.925])
# The setup-A start: one row on the rack's x centre line, in slot order across y.
_SLOT_Y = (-0.13, -0.04, 0.0, 0.04, 0.13)
# The robot standing at the rack with the scraper in hand, as practice consults the
# sampler, and the robot just after PickScraper, as every evaluation task does.
_AT_RACK = {"robot": (2.05, -0.15, 2.84, 1.0), "scraper": (1.35, 0.0, 0.99, 1.57)}
_AFTER_PICK = {"robot": (1.10, 0.62, -0.40, 0.0), "scraper": (0.95, 0.70, 1.21, -0.3)}


def _provider(*, features=None):
    extra = {} if features is None else {"sampler_features": features}
    return RackServingSkillProvider(env=RackServingEnvironment(config=RackServingConfig(**extra)))


def _ground(*, skill):
    objects = {
        Skills.SWEEP: _BULK_OBJECTS,
        Skills.GATHER_AND_SWEEP: _BULK_OBJECTS,
        Skills.TWO_ROUTE_SWEEP: _BULK_OBJECTS,
        Skills.TWO_ROUTE_GATHER_AND_SWEEP: _BULK_OBJECTS,
        Skills.PICK_SCRAPER: (Objects.robot, Objects.scraper),
        Skills.RETURN_SCRAPER: (Objects.robot, Objects.scraper),
        Skills.PICK_FOOD: (Objects.robot, Objects.food[2], Objects.rack, Objects.tray),
        Skills.RETURN_FOOD_TO_RACK: (Objects.robot, Objects.food[2], Objects.rack),
    }[skill]
    return GroundSkill(skill=skill, objects=objects)


def _state(*, pose=_AT_RACK, jitter=None):
    jitter = np.zeros((5, 2)) if jitter is None else jitter
    data = {
        Objects.robot: np.array(pose["robot"]),
        Objects.scraper: np.array(pose["scraper"]),
        Objects.rack: _RACK,
        Objects.tray: np.array([1.18, -0.095, 1.37, 0.095, 0.907]),
    }
    for food, y, (dx, dy) in zip(Objects.food, _SLOT_Y, jitter, strict=True):
        data[food] = np.array([1.5 + dx, y + dy, 0.938, 0.707, 0.0, 0.0, 0.707])
    return State(data=data)


def _row(*, provider, skill, params, state):
    return provider.hand_selected_feature_transform(
        ground_skill=_ground(skill=skill), state=state, params=np.array(params)
    )


_PARAMS = {
    Skills.SWEEP: [0.06],
    Skills.GATHER_AND_SWEEP: [0.07, 0.03, 0.10],
    Skills.TWO_ROUTE_SWEEP: [0.06],
    Skills.TWO_ROUTE_GATHER_AND_SWEEP: [0.07, 0.03, 0.10],
}


@pytest.mark.parametrize("skill", list(_PARAMS), ids=lambda skill: skill.name)
def test_the_row_is_bias_then_each_dumplings_offset_from_the_rack_then_the_parameters(*, skill):
    jitter = np.array([[0.004, -0.003], [0.0, 0.002], [-0.005, 0.0], [0.001, 0.001], [0.0, 0.0]])
    state = _state(jitter=jitter)
    params = _PARAMS[skill]
    # The rack's centre is (1.5, 0.0).
    layout = [value for y, (dx, dy) in zip(_SLOT_Y, jitter, strict=True) for value in (dx, y + dy)]
    with_layout = _row(
        provider=_provider(features="parameters_and_layout"),
        skill=skill,
        params=params,
        state=state,
    )
    assert len(with_layout) == 1 + 10 + len(params)
    np.testing.assert_allclose(with_layout, [1.0, *layout, *params], atol=1e-12)
    parameters = [1.0, *params]
    assert (
        _row(provider=_provider(features="parameters"), skill=skill, params=params, state=state)
        == parameters
    )
    # The default (Josh, 2026-11-13): the parameters alone.
    assert _row(provider=_provider(), skill=skill, params=params, state=state) == parameters


@pytest.mark.parametrize("skill", list(_PARAMS), ids=lambda skill: skill.name)
def test_the_full_state_value_leaves_the_row_to_the_method(*, skill):
    row = _row(
        provider=_provider(features="full_state"),
        skill=skill,
        params=_PARAMS[skill],
        state=_state(),
    )
    assert row is None


@pytest.mark.parametrize("features", ["parameters", "parameters_and_layout"])
@pytest.mark.parametrize("skill", list(_PARAMS), ids=lambda skill: skill.name)
def test_the_row_ignores_where_the_robot_and_the_scraper_are(*, skill, features):
    provider = _provider(features=features)
    at_rack = _row(provider=provider, skill=skill, params=_PARAMS[skill], state=_state())
    after_pick = _row(
        provider=provider, skill=skill, params=_PARAMS[skill], state=_state(pose=_AFTER_PICK)
    )
    assert at_rack == after_pick


def test_the_layout_is_measured_from_the_rack_not_the_world():
    provider = _provider(features="parameters_and_layout")
    state = _state()
    moved = State(data={obj: np.array(state[obj], dtype=float) for obj in _BULK_OBJECTS})
    for obj in (Objects.rack, *Objects.food):
        moved[obj][0] += 0.25
    moved[Objects.rack][2] += 0.25
    np.testing.assert_allclose(
        _row(provider=provider, skill=Skills.SWEEP, params=[0.06], state=moved),
        _row(provider=provider, skill=Skills.SWEEP, params=[0.06], state=state),
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "skill",
    [Skills.PICK_SCRAPER, Skills.RETURN_SCRAPER, Skills.PICK_FOOD, Skills.RETURN_FOOD_TO_RACK],
    ids=lambda skill: skill.name,
)
@pytest.mark.parametrize("features", ["parameters", "parameters_and_layout", "full_state"])
def test_skills_without_parameters_keep_the_methods_own_row(*, skill, features):
    row = _provider(features=features).hand_selected_feature_transform(
        ground_skill=_ground(skill=skill), state=_state(), params=np.zeros(0)
    )
    assert row is None


@pytest.mark.parametrize("features", ["parameters", "parameters_and_layout"])
def test_a_sampler_trained_at_the_rack_ranks_the_same_just_after_pick_scraper(*, features):
    """Practice's rows come from the robot at the rack holding the scraper; an
    evaluation task consults right after PickScraper. Synthetic outcomes: a Sweep
    succeeds exactly when d >= 0.04."""
    provider = _provider(features=features)
    rng = np.random.default_rng(0)
    sampler = LearnedSkillSampler(skill_name="Sweep", param_dim=1, max_train_iters=3000)
    for _ in range(60):
        d = rng.uniform(0.0, 0.12)
        state = _state(jitter=rng.uniform(-0.005, 0.005, size=(5, 2)))
        row = _row(provider=provider, skill=Skills.TWO_ROUTE_SWEEP, params=[d], state=state)
        sampler.observe(sampler_input=row, success=d >= 0.04)
    sampler.fit()
    candidates = [np.array([d]) for d in np.linspace(0.0, 0.12, 49)]
    jitter = rng.uniform(-0.005, 0.005, size=(5, 2))

    def scores(*, pose):
        return sampler.score_inputs(
            sampler_inputs=[
                _row(
                    provider=provider,
                    skill=Skills.TWO_ROUTE_SWEEP,
                    params=candidate,
                    state=_state(pose=pose, jitter=jitter),
                )
                for candidate in candidates
            ]
        )

    after_pick = scores(pose=_AFTER_PICK)
    assert after_pick == scores(pose=_AT_RACK)
    assert candidates[int(np.argmax(after_pick))][0] >= 0.04
    # The scores separate the ds that worked from the ones that did not.
    assert min(after_pick[17:]) > max(after_pick[:16])


def test_the_flag_selects_the_sampler_features_and_defaults_to_the_parameters_alone():
    def parse(*argv):
        parser = argparse.ArgumentParser()
        parser.add_argument("--seed", type=int, default=0)
        RackServingCli.add_arguments(parser=parser)
        return parser.parse_args(list(argv))

    assert RackServingConfig().sampler_features == "parameters"
    assert parse().sampler_features == "parameters"
    for value in ("parameters", "parameters_and_layout", "full_state"):
        args = parse("--sampler-features", value)
        for mode in ("training", "deployment"):
            config = RackServingCli.build_problem(args=args, mode=mode).env.config
            assert config.sampler_features == value
    with pytest.raises(SystemExit):
        parse("--sampler-features", "everything")
