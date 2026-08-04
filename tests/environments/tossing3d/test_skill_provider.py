import numpy as np

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.skill_provider import (
    Tossing3DOracle,
    Tossing3DSkillProvider,
)
from hitl_pmp.environments.tossing3d.skills import Tossing3DSkills

from .conftest import build_state

_ENV = Tossing3DEnvironment


def _provider() -> Tossing3DSkillProvider:
    return Tossing3DSkillProvider(env=Tossing3DEnvironment())


def test_every_type_a_skill_parameter_declares_is_exposed() -> None:
    """A planner grounds skills over `objects()` bucketed by `types()`; a type missing
    from either list silently makes a skill ungroundable."""
    provider = _provider()
    declared = {parameter.type for skill in provider.skills() for parameter in skill.parameters}
    assert declared <= set(provider.types())


def test_every_skill_parameter_type_has_at_least_one_object_to_bind() -> None:
    provider = _provider()
    available = {obj.type for obj in provider.objects()}
    for skill in provider.skills():
        for parameter in skill.parameters:
            assert parameter.type in available, f"{skill.name} cannot bind {parameter.name}"


def test_every_predicate_a_skill_references_is_exposed() -> None:
    provider = _provider()
    referenced = {
        atom.predicate
        for skill in provider.skills()
        for atom in (*skill.preconditions, *skill.add_effects, *skill.delete_effects)
    }
    assert referenced <= set(provider.predicates())


def test_the_scene_seed_object_is_kept_out_of_the_planners_universe() -> None:
    """It carries a KINDER reset seed no predicate reads; grounding over it would only
    widen the planner's search."""
    provider = _provider()
    assert _ENV.scene not in provider.objects()
    assert _ENV.scene_type not in provider.types()


def test_sample_params_forwards_the_environments_own_bounds() -> None:
    provider = Tossing3DSkillProvider(env=Tossing3DEnvironment(swing_low=0.6, swing_high=0.61))
    ground = GroundSkill(
        skill=Tossing3DSkills.TOSS,
        objects=(_ENV.robot, _ENV.cube, _ENV.bin_object, _ENV.goal_region, _ENV.barrier),
    )
    for _ in range(20):
        (swing,) = provider.sample_params(ground_skill=ground, rng=np.random.default_rng(0))
        assert 0.6 <= swing < 0.61


def test_oracle_provider_ignores_the_goal_because_the_domain_has_only_one() -> None:
    oracle = Tossing3DOracle(env=Tossing3DEnvironment())
    state = build_state()
    from hitl_pmp.core.problem.tasks.types import Goal

    assert oracle.get_labeled_action(state=state, goal=Goal(atoms=frozenset())).label == "Pick"
