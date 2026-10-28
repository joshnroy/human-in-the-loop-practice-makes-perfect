"""The practice API keeps task counts separate from learned motor parameters."""

import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.pouring3d.practice.environment import FrozenPourEnvironment
from hitl_pmp.environments.pouring3d.practice.skill_provider import FrozenPourSkillProvider


def test_pour_has_three_controls_and_count_is_a_grounding_argument():
    env = FrozenPourEnvironment(payload_count=3, requested_count=2)
    provider = FrozenPourSkillProvider(env=env)
    pour = next(s for s in provider.skills() if s.name == "Pour")
    assert pour.param_dim == 3
    assert all(s.param_dim == 0 for s in provider.skills() if s != pour)
    ground = provider.ground(name="Pour", request=2)
    assert isinstance(ground, GroundSkill)
    values = np.array([90.5, 0.008, 30.0])
    action = provider.encode(ground_skill=ground, params=values)
    np.testing.assert_array_equal(action, [1, 0, 2, *values])
    assert provider.sample_params(ground_skill=ground, rng=np.random.default_rng(1)).shape == (3,)


def test_human_reset_is_explicit_and_unavailable_in_evaluation():
    env = FrozenPourEnvironment()
    practice = FrozenPourSkillProvider(env=env, human_reset_practice_cost=7)
    evaluation = FrozenPourSkillProvider(env=env, offer_human_reset=False)
    assert len(practice.human_cube_bin_reset_skills()) == 1
    assert practice.human_cube_bin_reset_skill().skill.practice_cost == 7
    assert evaluation.human_cube_bin_reset_skills() == ()
    assert "reset_shrimp" not in {s.name for s in practice.skills()}


def test_invalid_count_and_parameter_shapes_are_rejected():
    with pytest.raises(ValueError):
        FrozenPourEnvironment(payload_count=3, requested_count=4)
    provider = FrozenPourSkillProvider(env=FrozenPourEnvironment())
    with pytest.raises(ValueError):
        provider.encode(ground_skill=provider.ground(name="Pour", request=2), params=np.zeros(2))
    with pytest.raises(ValueError):
        provider.ground(name="Pour", request=4)


def test_planner_must_pour_and_place_to_satisfy_the_requested_count():
    from hitl_pmp.core.problem.tasks.types import GroundAtom
    from hitl_pmp.environments.pouring3d.practice.predicates import PourPredicates as P
    from hitl_pmp.planning.fast_downward import FastDownwardPlanner

    provider = FrozenPourSkillProvider(env=FrozenPourEnvironment())
    o = provider.env.objects()

    def atom(name, *objects):  # noqa: PLR0917
        return GroundAtom(predicate=P.get(name=name), objects=objects)

    initial = frozenset({
        atom("HandEmpty", o.robot),
        atom("PitcherHome", o.pitcher),
        atom("TrayHome", o.tray),
        atom("ActiveRequest", o.requests()[1]),
        *(atom("InPitcher", s, o.pitcher) for s in o.shrimp()),
        *(atom("Settled", s) for s in o.shrimp()),
        *(atom("PayloadOrder", a, b) for a, b in zip(o.shrimp(), o.shrimp()[1:], strict=False)),
    })
    plan = FastDownwardPlanner.plan(
        skills=provider.skills(),
        objects=provider.objects(),
        predicates=provider.predicates(),
        types=provider.types(),
        init_atoms=initial,
        goal=frozenset({atom("Goal", o.requests()[1])}),
    )
    assert [g.skill.name for g in plan] == ["PickPitcher", "Pour", "PlacePitcherAtHome"]
    assert o.requests()[1] in plan[1].objects


def test_parameter_bounds_are_explicit_and_finite():
    with pytest.raises(ValueError):
        FrozenPourSkillProvider(env=FrozenPourEnvironment(), tilt_bounds=(95, 87))
    with pytest.raises(ValueError):
        FrozenPourSkillProvider(env=FrozenPourEnvironment(), amplitude_bounds=(0, float("inf")))


def test_registered_replay_factory_is_lazy():
    from analysis.render_recording import RecordingVideo
    from hitl_pmp.environments.pouring3d.practice.recording import FrozenPourReplayRenderer
    from hitl_pmp.recording.run_record.types import RecordingHeader

    renderer = RecordingVideo.default_renderer(
        header=RecordingHeader(environment="pouring3d", scene={"motion": True, "fps": 10}),
        realistic_background=True,
    )
    assert isinstance(renderer, FrozenPourReplayRenderer)
    assert renderer.fps() == 10
    assert not renderer.is_storyboard()


def test_every_count_is_a_reachable_practice_target():
    from hitl_pmp.planning.fast_downward import FastDownwardPlanner

    provider = FrozenPourSkillProvider(env=FrozenPourEnvironment(requested_count=2))
    pick = provider.ground(name="PickPitcher")
    initial = pick.preconditions
    for count in (1, 2, 3):
        target = provider.ground(name="Pour", request=count)
        plan = FastDownwardPlanner.plan(
            skills=provider.skills(),
            objects=provider.objects(),
            predicates=provider.predicates(),
            types=provider.types(),
            init_atoms=initial,
            goal=target.preconditions,
        )
        assert [g.skill.name for g in plan] == ["PickPitcher"]
        assert target.preconditions <= (initial - pick.delete_effects) | pick.add_effects
