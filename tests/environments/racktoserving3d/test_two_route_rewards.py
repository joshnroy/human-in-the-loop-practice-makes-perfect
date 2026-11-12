"""The opt-in two-route reward structure: Sweep worth 3 at three served, GatherAndSweep 5."""

import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.environments.racktoserving3d.config import RackServingConfig
from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment
from hitl_pmp.environments.racktoserving3d.objects import RackServing3DObjects as Objects
from hitl_pmp.environments.racktoserving3d.predicates import (
    RackServing3DAtoms,
    RackServing3DPredicates,
)
from hitl_pmp.environments.racktoserving3d.skill_provider import (
    RackServingActions,
    RackServingSkillProvider,
)
from hitl_pmp.environments.racktoserving3d.skills import RackServing3DSkills as Skills
from hitl_pmp.environments.racktoserving3d.tasks import RackServingTasks
from hitl_pmp.methods.practice_makes_perfect.dp_mdp import DpTaskModel

TWO_ROUTE = RackServingConfig(bulk_transfer_only=True, two_route_rewards=True)


def _provider(*, config):
    return RackServingSkillProvider(env=RackServingEnvironment(config=config))


def _atoms(*, names, two_route=True):
    predicates = {p.name: p for p in RackServing3DPredicates.all(two_route=two_route)}
    objects = {obj.name: obj for obj in Objects.all_objects()}
    return (
        frozenset(
            GroundAtom(predicate=predicates[name], objects=tuple(objects[key] for key in keys))
            for name, keys in names
        )
        | RackServing3DPredicates.static_atoms()
    )


def _start():
    names = {("OnRack", (food.name, Objects.rack.name)) for food in Objects.food}
    names |= {("PickPlannable", (Objects.robot.name, food.name)) for food in Objects.food}
    names |= {("HandEmpty", (Objects.robot.name,)), ("ScraperHome", (Objects.scraper.name,))}
    return _atoms(names=RackServing3DAtoms.with_summaries(atoms=frozenset(names)))


def _bulk(*, provider, name):
    skill = next(skill for skill in provider.skills() if skill.name == name)
    return GroundSkill(
        skill=skill,
        objects=(Objects.robot, Objects.scraper, Objects.rack, Objects.tray, *Objects.food),
    )


@pytest.mark.parametrize(
    ("on_tray", "expected"),
    [
        (2, set()),
        (3, {"Served", "ServedThree"}),
        (4, {"Served", "ServedThree"}),
        (5, {"Served", "ServedThree", "ServedFive"}),
    ],
)
def test_the_served_facts_follow_the_plate_count(*, on_tray, expected):
    atoms = frozenset(("OnTray", (food.name, "box_0")) for food in Objects.food[:on_tray])
    served = RackServing3DAtoms.with_served(atoms=atoms)
    assert {name for name, _ in served - atoms} == expected


def test_the_default_serving_domain_is_unchanged():
    provider = _provider(config=RackServingConfig())
    assert provider.goal_rewards() == {}
    assert provider.skills() == Skills.robot_skills()
    assert {p.name for p in provider.predicates()}.isdisjoint({"Served", "ServedThree"})
    goal = RackServingTasks(env=provider.env).goal()
    assert goal.atoms == RackServing3DPredicates.goal_atoms()
    assert all(atom.predicate.name == "OnTray" for atom in goal.atoms)


def test_two_route_rewards_need_the_one_attempt_protocol():
    with pytest.raises(ValueError, match="bulk_transfer_only"):
        RackServingConfig(two_route_rewards=True)


def test_each_bulk_skill_promises_its_own_outcome_and_the_goal_is_shared():
    provider = _provider(config=TWO_ROUTE)
    assert provider.goal_rewards() == {"ServedThree": 3.0, "ServedFive": 5.0}
    start = _start()
    held = DpTaskModel.successor(
        atoms=start,
        ground=GroundSkill(skill=Skills.PICK_SCRAPER, objects=(Objects.robot, Objects.scraper)),
    )
    sweep = _bulk(provider=provider, name="Sweep")
    gather = _bulk(provider=provider, name="GatherAndSweep")

    def names(*, ground):
        return {a.predicate.name for a in ground.expected_add_effects(before_atoms=held)}

    assert {"Served", "ServedThree"} <= names(ground=sweep)
    assert not {"ServedFive", "OnTray"} & names(ground=sweep)
    assert {"Served", "ServedThree", "ServedFive", "OnTray"} <= names(ground=gather)
    goal = RackServingTasks(env=provider.env).goal().atoms
    assert {atom.predicate.name for atom in goal} == {"Served"}
    for ground, reward in ((sweep, 3.0), (gather, 5.0)):
        after = DpTaskModel.successor(atoms=held, ground=ground)
        assert goal <= after
        assert DpTaskModel.goal_reward(atoms=after, goal_rewards=provider.goal_rewards()) == reward
        # Either try clears the rack symbolically, so the person's return comes next.
        assert not any(atom.predicate.name == "OnRack" for atom in after)
    (batch, _) = provider.human_cube_bin_reset_skills()
    assert {"Served", "ServedThree", "ServedFive"} <= {
        atom.predicate.name for atom in batch.delete_effects
    }


def test_two_route_bulk_skills_sample_and_encode_as_the_plain_ones_do():
    provider = _provider(config=TWO_ROUTE)
    for name, size in (("Sweep", 1), ("GatherAndSweep", 3)):
        ground = _bulk(provider=provider, name=name)
        params = provider.sample_params(ground_skill=ground, rng=np.random.default_rng(0))
        assert params.shape == (size,)
        plain = _bulk(provider=_provider(config=RackServingConfig()), name=name)
        assert np.array_equal(
            RackServingActions.encode(ground_skill=ground, params=params),
            RackServingActions.encode(ground_skill=plain, params=params),
        )


def test_the_task_mdp_weighs_each_route_by_reward_times_competence():
    provider = _provider(config=TWO_ROUTE)
    goal = RackServingTasks(env=provider.env).goal().atoms
    mdp = DpTaskModel.build(
        tasks=[(_start(), goal)],
        skills=provider.skills(),
        objects=provider.objects(),
        max_states=4096,
        goal_rewards=provider.goal_rewards(),
    )

    def solve(*, sweep, gather):
        value, policy = mdp.solve(
            competences={
                t.skill: {"Sweep": sweep, "GatherAndSweep": gather}.get(t.skill.skill.name, 1.0)
                for t in mdp.transitions
            }
        )
        return value, policy

    value, _ = solve(sweep=0.9, gather=0.4)
    assert value == pytest.approx(3 * 0.9)
    value, _ = solve(sweep=0.9, gather=0.8)
    assert value == pytest.approx(5 * 0.8)
