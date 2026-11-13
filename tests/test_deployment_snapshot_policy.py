import argparse
import pickle

import pytest

from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.skill_provider import LightSwitchSkillProvider
from hitl_pmp.methods.deployment import DeploymentPolicy, SnapshotDeployment
from hitl_pmp.methods.practice_makes_perfect.dp_method import DpMethod
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod
from hitl_pmp.step_protocol import DeploymentSnapshot, StepPracticeRunner


@pytest.mark.parametrize("env", ["tossing3d", "kitchen_recycling3d"])
def test_simulation_uses_its_method_policy_by_default(*, env):
    assert StepPracticeRunner.method_specific_evaluation(args=argparse.Namespace(env=env))
    assert not StepPracticeRunner.method_specific_evaluation(
        args=argparse.Namespace(env=env, evaluation_policy="shared-ees")
    )


def test_real_retains_its_evaluation_default():
    assert not StepPracticeRunner.method_specific_evaluation(
        args=argparse.Namespace(env="tossing3d_real")
    )


def test_dp_snapshot_retains_policy_and_unobserved_skill_priors():
    env = LightSwitchEnvironment(grid_size=4)
    provider = LightSwitchSkillProvider(env=env)
    method = DpMethod(
        env=env,
        skill_provider=provider,
        dp_budget=5,
        dp_solver="enumeration",
        dp_initial_competence={provider.skills()[0].name: 0.2},
        dp_max_states=100,
    )
    raw = DeploymentSnapshot.encode(method=method)
    restored = DeploymentSnapshot.restore(raw=raw, env=env, provider=provider)
    assert isinstance(restored, SnapshotDeployment)
    assert restored.deployment_policy.kind == "dp_mdp"
    assert restored.deployment_policy.max_states == 100
    assert restored.deployment_policy.dp_priors == method._dp_skill_priors  # noqa: SLF001
    assert restored._competence_models == {}  # noqa: SLF001 -- encoding never fills training priors
    assert method._competence_models == {}  # noqa: SLF001


def test_legacy_snapshot_still_uses_shared_ees_and_unknown_version_fails():
    env = LightSwitchEnvironment(grid_size=4)
    provider = LightSwitchSkillProvider(env=env)
    raw = DeploymentSnapshot.encode(method=EesMethod(env=env, skill_provider=provider))
    payload = pickle.loads(raw)
    payload.pop("version", None)
    payload.pop("deployment_policy", None)
    restored = DeploymentSnapshot.restore(raw=pickle.dumps(payload), env=env, provider=provider)
    assert type(restored) is EesMethod
    payload["version"] = 9000
    with pytest.raises(ValueError, match="snapshot version"):
        DeploymentSnapshot.restore(raw=pickle.dumps(payload), env=env, provider=provider)


@pytest.mark.parametrize(
    "variant,solver", [("curtis", "determinized_astar"), ("algorithm3", "expectimax")]
)
def test_belief_method_snapshot_records_its_valued_deployment_policy(*, variant, solver):
    from tests.methods.belief_space.test_tossing3d_method import _build

    method = _build(pomdp_determinized_variant=variant, pomdp_solver=solver, pomdp_num_particles=32)
    raw = DeploymentSnapshot.encode(method=method)
    restored = DeploymentSnapshot.restore(raw=raw, env=method.env, provider=method.skill_provider)
    assert restored.deployment_policy.kind == "tossing_canonical"
    assert restored.deployment_policy.source_method == f"{variant}/{solver}"
    assert restored.deployment_policy.horizon == method.pomdp_domain.deployment_horizon
    # Explicit historical EES replay remains possible without changing hardware.
    legacy = DeploymentSnapshot.encode(method=method, method_specific=False)
    historical = DeploymentSnapshot.restore(
        raw=legacy, env=method.env, provider=method.skill_provider
    )
    assert historical.deployment_policy.kind == "shared_ees"


def test_a_rewarded_routes_snapshot_deploys_reward_greedy_from_the_domains_prior():
    from hitl_pmp.environments.tworoutes.environment import TwoRoutesEnvironment
    from hitl_pmp.environments.tworoutes.skill_provider import TwoRoutesSkillProvider
    from hitl_pmp.methods.belief_space.rewarded_routes_domain import RewardedRoutesPracticeDomain
    from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod

    env = TwoRoutesEnvironment()
    provider = TwoRoutesSkillProvider(env=env)
    method = Tossing3DPomdpMethod(
        env=env,
        skill_provider=provider,
        seed=0,
        pomdp_num_particles=32,
        pomdp_domain=RewardedRoutesPracticeDomain.from_provider(provider=provider),
    )
    raw = DeploymentSnapshot.encode(method=method)
    restored = DeploymentSnapshot.restore(
        raw=raw, env=env, provider=TwoRoutesSkillProvider(env=env, offer_human_reset=False)
    )
    policy = restored.deployment_policy
    assert policy.kind == "reward_greedy" and policy.horizon == 1
    assert {name: (prior.alpha, prior.beta) for name, prior in policy.dp_priors.items()} == {
        "RouteA": (1.0, 3.0),
        "RouteB": (1.0, 3.0),
    }
    legacy = DeploymentSnapshot.encode(method=method, method_specific=False)
    assert (
        DeploymentSnapshot.restore(raw=legacy, env=env, provider=provider).deployment_policy.kind
        == "shared_ees"
    )


def test_a_two_route_rack_snapshot_executes_the_value_models_own_plan_for_the_planners_belief():
    from hitl_pmp.environments.racktoserving3d.config import RackServingConfig
    from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment
    from hitl_pmp.environments.racktoserving3d.skill_provider import RackServingSkillProvider
    from hitl_pmp.methods.belief_space.rack_serving_domain import RACK_SERVING_DOMAIN
    from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod

    env = RackServingEnvironment(
        config=RackServingConfig(bulk_transfer_only=True, two_route_rewards=True)
    )
    provider = RackServingSkillProvider(env=env)
    method = Tossing3DPomdpMethod(
        env=env,
        skill_provider=provider,
        seed=0,
        pomdp_num_particles=32,
        pomdp_domain=RACK_SERVING_DOMAIN.with_goal_rewards(provider=provider),
    )
    raw = DeploymentSnapshot.encode(method=method)
    restored = DeploymentSnapshot.restore(
        raw=raw, env=env, provider=RackServingSkillProvider(env=env, offer_human_reset=False)
    )
    policy = restored.deployment_policy
    # What is executed at evaluation is what the value model returns for the planner's
    # own belief, with the value model's own step weight.
    valued = method._pomdp_model.valued_domain()
    internal = valued.deployment_plan(beliefs=method._pomdp_state.skill_beliefs)
    assert policy.kind == "rack_value_model"
    assert policy.horizon == valued.deployment_horizon
    assert policy.rack_plan == {
        DeploymentPolicy.rack_key(state=state, remaining=remaining): (
            None if choice is None else (choice[0], choice[1][0])
        )
        for (state, remaining), choice in internal.items()
    }
    assert internal[(5, "empty", True), valued.deployment_horizon][0] == "PickScraper"
    bulk = internal[(5, "scraper", False), valued.deployment_horizon - 1][0]
    assert bulk in {"Sweep", "GatherAndSweep"}

    import numpy as np

    from hitl_pmp.core.problem.tasks.types import Task
    from hitl_pmp.environments.racktoserving3d.objects import RackServing3DObjects as Objects
    from hitl_pmp.environments.racktoserving3d.tasks import RackServingTasks
    from hitl_pmp.environments.sweepoffshelf3d.types import SweepOffShelf3DState

    def state(*, atoms):
        data = {obj: np.zeros(obj.type.dim) for obj in Objects.all_objects()}
        data[Objects.scene] = np.zeros(4)
        on_rack = {("OnRack", (food.name, "rack_0")) for food in Objects.food}
        return SweepOffShelf3DState(
            data=data,
            abstract_atoms=frozenset(on_rack | {("AnyFoodOnRack", ())} | set(atoms)),
        )

    start = state(atoms={("HandEmpty", ("robot",)), ("ScraperHome", ("wiper_0",))})
    held = state(atoms={("HoldingScraper", ("robot", "wiper_0")), ("ScraperAway", ("wiper_0",))})
    run = restored.get_task_policy(
        task=Task(initial_state=start, goal=RackServingTasks(env=env).goal())
    )
    assert run(start).label.startswith("PickScraper(")
    assert run(held).label.startswith(f"{bulk}(")
