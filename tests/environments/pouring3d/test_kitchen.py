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


def test_legacy_benchmark_unchanged_and_kitchen_opt_in():
    from hitl_pmp.cli import ENVIRONMENTS, STEP_PROTOCOL_ENVIRONMENTS
    from hitl_pmp.environments.pouring3d.practice.cli import FrozenPourCli

    assert ENVIRONMENTS["pouring3d"] is FrozenPourCli
    assert ENVIRONMENTS["pouring3d_kitchen"] is KitchenCli
    assert "pouring3d_kitchen" in STEP_PROTOCOL_ENVIRONMENTS
    assert FrozenPourEnvironment.action_space.high[0] == 7
    assert KitchenEnvironment.action_space.high[0] == 9


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
    assert human.skill.practice_cost == 7
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
        action = p.encode(ground_skill=g, params=np.array([90, 0.015, 12]))
        assert action[2] == k
        assert any(a.predicate.name == "ColdIngredients" for a in g.preconditions)
    e._started = True
    with pytest.raises(ValueError, match="no free reset"):
        KitchenTasks(env=e).build_task(seed=0)


def test_deployment_plan_requires_pour_put_down_activation_and_retreat():
    from hitl_pmp.planning.fast_downward import FastDownwardPlanner

    provider = KitchenSkillProvider(env=KitchenEnvironment(deployment=True, requested_count=3))
    o = provider.env.objects()

    def atom(name, *objects):  # noqa: PLR0917
        return GroundAtom(predicate=P.get(name=name), objects=objects)

    initial = provider.ground(name="PickPitcher").preconditions | frozenset({
        atom("ActiveRequest", o.requests()[2]),
        atom("HobOff", o.hob),
        *(atom(f"Count{n}", obj) for n, obj in enumerate(o.requests(), 1)),
    })
    plan = FastDownwardPlanner.plan(
        skills=provider.skills(),
        objects=provider.objects(),
        predicates=provider.predicates(),
        types=provider.types(),
        init_atoms=initial,
        goal=frozenset({atom("MealStarted", o.requests()[2])}),
    )
    assert [g.skill.name for g in plan] == [
        "PickPitcher",
        "Pour3",
        "PlacePitcherAtHome",
        "ActivateCooking",
        "MoveAway",
    ]
    assert all(g.skill.name != HUMAN_RESET for g in plan)


def test_cli_builds_separate_training_and_deployment_modes():
    parser = argparse.ArgumentParser()
    KitchenCli.add_arguments(parser=parser)
    args = parser.parse_args([])
    assert str(args.practice_reset_policy.value) == "never"
    args.seed = 0
    practice = KitchenCli.build_problem(args=args)
    deploy = KitchenCli.build_problem(args=args, deployment=True)
    assert not practice.env.deployment
    assert deploy.env.deployment
    assert next(iter(deploy.tasks.goal().atoms)).predicate.name == "MealStarted"
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
