"""Kitchen policy boundaries and ordinary planner integration (no physics launch)."""

import argparse

import numpy as np
import pytest

from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.environments.pouring3d.kitchen.cli import KitchenCli
from hitl_pmp.environments.pouring3d.kitchen.config import KitchenGeometry
from hitl_pmp.environments.pouring3d.kitchen.environment import HUMAN_RESET, KitchenEnvironment
from hitl_pmp.environments.pouring3d.kitchen.predicates import KitchenPredicates as P
from hitl_pmp.environments.pouring3d.kitchen.skill_provider import KitchenSkillProvider
from hitl_pmp.environments.pouring3d.kitchen.tasks import KitchenTasks
from hitl_pmp.environments.pouring3d.practice.environment import FrozenPourEnvironment
from tests.practice_cost_fixtures import CostFixtures


def test_legacy_benchmark_unchanged_and_kitchen_opt_in():
    from hitl_pmp.cli import ENVIRONMENTS, STEP_PROTOCOL_ENVIRONMENTS
    from hitl_pmp.environments.pouring3d.practice.cli import FrozenPourCli

    assert ENVIRONMENTS["pouring3d"] is FrozenPourCli
    assert ENVIRONMENTS["pouring3d_kitchen"] is KitchenCli
    assert "pouring3d_kitchen" in STEP_PROTOCOL_ENVIRONMENTS
    assert FrozenPourEnvironment.action_space.high[0] == 7
    # The kitchen dispatches only the shared robot skills; no lifecycle action ids.
    assert KitchenEnvironment.action_space.high[0] == 7


def test_missing_measurements_stay_unknown_and_proxies_are_configurable():
    g = KitchenGeometry(counter_height_m=0.9)
    assert g.measured_jug_dimensions_m is None
    assert g.measured_pan_inner_diameter_m is None
    config = g.scene_config()
    assert config["fixtures"]["table"]["prep_table"]["height"] == 0.9
    assert config["objects"]["castironskillet"]["tray"]["mass"] == 2.57
    assert "norpro.com" in g.jug_source
    with pytest.raises(ValueError):
        KitchenGeometry(water_depth_proxy_m=0.2)


def test_pan_and_jug_stand_on_the_same_desk_height():
    geometry = KitchenGeometry()
    assert geometry.hob_height_m == pytest.approx(0.74)
    assert geometry.counter_height_m == pytest.approx(0.74)
    config = geometry.scene_config()
    assert config["fixtures"]["pourstovetop"]["stovetop"]["height"] == pytest.approx(0.74)
    pan_region = config["regions"]["tray_home"]["ranges"][0]
    assert pan_region[2] == pan_region[5] == pytest.approx(0.7405)
    jug_region = config["regions"]["pitcher_home"]["ranges"][0]
    assert jug_region[2] == jug_region[5] == pytest.approx(0.7405)
    assert config["fixtures"]["table"]["prep_table"]["height"] == pytest.approx(0.74)


def test_kitchen_skillet_carries_dough_on_cast_iron_friction_and_dumplings_stay_low():
    from hitl_pmp.environments.pouring3d.practice.cli import FrozenPourCli

    assert KitchenEnvironment().surface_mu == pytest.approx(0.1)
    assert KitchenEnvironment().skillet_mu == pytest.approx(0.4)
    kitchen = argparse.ArgumentParser()
    KitchenCli.add_arguments(parser=kitchen)
    assert kitchen.parse_args([]).surface_mu == pytest.approx(0.1)
    # The legacy pouring benchmark keeps its own assumption.
    assert FrozenPourEnvironment().surface_mu == pytest.approx(0.1)
    legacy = argparse.ArgumentParser()
    FrozenPourCli.add_arguments(parser=legacy)
    assert legacy.parse_args([]).surface_mu == pytest.approx(0.1)


def test_reset_cost_and_deployment_boundary_without_native_dependencies():
    practice = KitchenEnvironment(human_interruption_cost=7)
    provider = KitchenSkillProvider(env=practice)
    human = provider.human_cube_bin_reset_skill()
    assert human.skill.name == HUMAN_RESET
    assert HUMAN_RESET not in {s.name for s in provider.skills()}
    deployment = KitchenEnvironment(deployment=True)
    assert KitchenSkillProvider(env=deployment).human_cube_bin_reset_skills() == ()
    # Enforced before native state construction, including a direct call.
    with pytest.raises(ValueError, match="forbidden"):
        deployment.reset_movables(destination=HUMAN_RESET)
    with pytest.raises(ValueError):
        KitchenEnvironment(human_interruption_cost=0)
    deployment._started = True
    with pytest.raises(ValueError, match="No free"):
        deployment.hard_reset()


def test_count_samplers_and_cold_practice_stay_ordinary():
    e = KitchenEnvironment()
    p = KitchenSkillProvider(env=e)
    assert {s.name for s in p.skills() if s.param_dim} == {"Pour1", "Pour2", "Pour3"}
    assert "ActivateCooking" not in {s.name for s in p.skills()}
    for k in range(1, 4):
        g = p.ground(name=f"Pour{k}")
        action = p.encode(ground_skill=g, params=np.array([1.0]))
        assert action[2] == k
    e._started = True
    with pytest.raises(ValueError, match="no free reset"):
        KitchenTasks(env=e).build_task(seed=0)


REMOVED_SKILLS = {"Wait", "ActivateCooking", "MoveAway"}
REMOVED_PREDICATES = {"HobOn", "HobOff", "ColdIngredients", "RobotAway", "MealStarted"}


@pytest.mark.parametrize("deployment", [False, True])
def test_the_kitchen_offers_no_wait_and_no_cooking_lifecycle_skills(*, deployment):
    provider = KitchenSkillProvider(env=KitchenEnvironment(deployment=deployment))
    names = {s.name for s in provider.skills()}
    assert not names & REMOVED_SKILLS
    assert {"PickPitcher", "Pour1", "Pour2", "Pour3", "PlacePitcherAtHome"} <= names
    assert not {s.name for s in provider.domain(count=3)} & REMOVED_SKILLS
    for name in REMOVED_SKILLS:
        with pytest.raises(StopIteration):
            provider.ground(name=name)


def test_the_legacy_pouring_benchmark_keeps_its_wait_skill():
    from hitl_pmp.environments.pouring3d.practice.skill_provider import FrozenPourSkillProvider

    legacy = FrozenPourSkillProvider(env=FrozenPourEnvironment())
    assert "Wait" in {s.name for s in legacy.skills()}


def test_no_kitchen_predicate_or_precondition_mentions_the_cooking_lifecycle():
    assert not {p.name for p in P.all()} & REMOVED_PREDICATES
    for skill in KitchenSkillProvider.domain(count=3):
        mentioned = {
            a.predicate.name for a in skill.preconditions | skill.add_effects | skill.delete_effects
        }
        assert not mentioned & REMOVED_PREDICATES, skill.name


@pytest.mark.parametrize("deployment", [False, True])
def test_a_test_task_goal_is_the_split_and_jug_home_goal_atom(*, deployment):
    env = KitchenEnvironment(deployment=deployment, requested_count=2)
    tasks = KitchenTasks(env=env, requested_counts=(1, 2, 3))
    requests = env.objects().requests()
    assert tasks.goal().atoms == frozenset({
        GroundAtom(predicate=P.get(name="Goal"), objects=(requests[1],))
    })
    assert [g.atoms for g in tasks.test_task_goals()] == [
        frozenset({GroundAtom(predicate=P.get(name="Goal"), objects=(r,))}) for r in requests
    ]


def test_deployment_plan_is_pick_pour_put_down():
    from hitl_pmp.planning.fast_downward import FastDownwardPlanner

    provider = KitchenSkillProvider(env=KitchenEnvironment(deployment=True, requested_count=3))
    o = provider.env.objects()

    def atom(name, *objects):  # noqa: PLR0917
        return GroundAtom(predicate=P.get(name=name), objects=objects)

    initial = provider.ground(name="PickPitcher").preconditions | frozenset({
        atom("ActiveRequest", o.requests()[2]),
        *(atom(f"Count{n}", obj) for n, obj in enumerate(o.requests(), 1)),
    })
    plan = FastDownwardPlanner.plan(
        skills=provider.skills(),
        objects=provider.objects(),
        predicates=provider.predicates(),
        types=provider.types(),
        init_atoms=initial,
        goal=frozenset({atom("Goal", o.requests()[2])}),
    )
    assert [g.skill.name for g in plan] == ["PickPitcher", "Pour3", "PlacePitcherAtHome"]


def test_cli_builds_separate_training_and_deployment_modes():
    parser = argparse.ArgumentParser()
    KitchenCli.add_arguments(parser=parser)
    args = parser.parse_args([])
    assert str(args.practice_reset_policy.value) == "never"
    args.seed = 0
    args.practice_cost_config = CostFixtures.BY_ENV["pouring3d_kitchen"]
    practice = KitchenCli.build_problem(args=args)
    deploy = KitchenCli.build_problem(args=args, deployment=True)
    assert not practice.env.deployment
    assert deploy.env.deployment
    assert next(iter(deploy.tasks.goal().atoms)).predicate.name == "Goal"
    assert next(iter(practice.tasks.goal().atoms)).predicate.name == "Goal"


def test_kitchen_recording_uses_its_own_lazy_renderer():
    from analysis.render_recording import RecordingVideo
    from hitl_pmp.environments.pouring3d.kitchen.recording import KitchenReplayRenderer
    from hitl_pmp.recording.run_record.types import RecordingHeader

    renderer = RecordingVideo.default_renderer(
        header=RecordingHeader(environment="pouring3d_kitchen", scene={"motion": True}),
        realistic_background=True,
    )
    assert isinstance(renderer, KitchenReplayRenderer)
    assert renderer._controller is None


def test_the_open_loop_parameter_models_keep_their_100_mm_wiggle_limit():
    from pydantic import ValidationError

    from hitl_pmp.environments.pouring3d.feedback_controller import FeedbackPourParameters
    from hitl_pmp.environments.pouring3d.frozen_native import ExtendedFeedbackPourParameters
    from hitl_pmp.environments.pouring3d.potsticker_controller import PotstickerParameters

    for parameters in (FeedbackPourParameters, ExtendedFeedbackPourParameters):
        assert parameters(tilt_degrees=95, amplitude_m=0.1, duration_seconds=10).amplitude_m == 0.1
        with pytest.raises(ValidationError):
            parameters(tilt_degrees=95, amplitude_m=0.101, duration_seconds=10)
    assert PotstickerParameters(amplitude_m=0.1).amplitude_m == 0.1


def test_kitchen_pour_learns_only_its_ramp_rate():
    from pydantic import ValidationError

    from hitl_pmp.environments.pouring3d.practice.skill_provider import FrozenPourSkillProvider

    # Josh 2026-11-04: the closed-loop Pour's one learned parameter, 0.25-5 deg/s.
    env = KitchenEnvironment(requested_count=1)
    provider = KitchenSkillProvider(env=env)
    pours = [s for s in provider.skills() if s.name.startswith("Pour")]
    assert sorted(s.name for s in pours) == ["Pour1", "Pour2", "Pour3"]
    assert {s.param_dim for s in pours} == {1}
    assert provider.ramp_rate_bounds == (0.25, 5.0)
    rng = np.random.default_rng(0)
    for k in (1, 2, 3):
        g = provider.ground(name=f"Pour{k}")
        draws = np.array([provider.sample_params(ground_skill=g, rng=rng) for _ in range(200)])
        assert draws.shape == (200, 1)
        assert draws.min() >= 0.25
        assert draws.max() <= 5.0
        # Native action: [Pour, unused, k, ramp rate, 0, 0].
        np.testing.assert_array_equal(
            provider.encode(ground_skill=g, params=np.array([1.0])), [1, 0, k, 1.0, 0, 0]
        )
        with pytest.raises(ValueError, match="dimension"):
            provider.encode(ground_skill=g, params=np.array([90.0, 0.015, 12.0]))
    np.testing.assert_array_equal(KitchenEnvironment.action_space.high[3:], [5.0, 0, 0])
    for bounds in ((0.1, 5.0), (0.25, 6.0), (2.0, 1.0), (0.25, float("inf"))):
        with pytest.raises(ValidationError):
            KitchenSkillProvider(env=env, ramp_rate_bounds=bounds)
    # The open-loop pour's bounds mean nothing to the kitchen and are refused.
    for field, legal in (
        ("tilt_bounds", (87.0, 95.0)),
        ("amplitude_bounds", (0.0, 0.025)),
        ("duration_bounds", (0.0, 30.0)),
    ):
        with pytest.raises(ValidationError, match="ramp rate"):
            KitchenSkillProvider(env=env, **{field: legal})
    # The legacy pouring3d Pour keeps its three open-loop parameters.
    legacy = FrozenPourSkillProvider(env=FrozenPourEnvironment())
    assert {s.param_dim for s in legacy.skills() if s.name.startswith("Pour")} == {3}
    g = legacy.ground(name="Pour2")
    assert legacy.sample_params(ground_skill=g, rng=rng).shape == (3,)


def test_kitchen_cli_takes_ramp_rate_bounds_and_refuses_open_loop_bounds():
    from hitl_pmp.environments.pouring3d.kitchen.skill_provider import KitchenPourOracle

    parser = argparse.ArgumentParser()
    KitchenCli.add_arguments(parser=parser)
    args = parser.parse_args([])
    assert list(args.pour_ramp_rate_bounds) == [0.25, 5.0]
    args.seed = 0
    args.practice_cost_config = CostFixtures.BY_ENV["pouring3d_kitchen"]
    problem = KitchenCli.build_problem(args=args)
    context = KitchenCli.context(args=args, problem=problem)
    assert context.skill_provider.ramp_rate_bounds == (0.25, 5.0)
    assert isinstance(context.oracle, KitchenPourOracle)
    assert context.oracle.parameters == (1.0,)
    assert isinstance(context.oracle.provider(), KitchenSkillProvider)
    narrow = parser.parse_args(["--pour-ramp-rate-bounds", "0.5", "2"])
    narrow.seed = 0
    provider = KitchenCli.context(args=narrow, problem=problem).skill_provider
    assert provider.ramp_rate_bounds == (0.5, 2.0)
    for flag, values in (
        ("--pour-tilt-bounds", ["75", "95"]),
        ("--pour-amplitude-bounds", ["0", "0.1"]),
        ("--pour-duration-bounds", ["0", "60"]),
    ):
        refused = parser.parse_args([flag, *values])
        refused.seed = 0
        with pytest.raises(ValueError, match="--pour-ramp-rate-bounds"):
            KitchenCli.context(args=refused, problem=problem)
