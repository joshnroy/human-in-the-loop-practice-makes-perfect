"""Each skill's declared effects against the atoms a successful run changes.

The "after" side is what the backend would report: the measured locations run through
`RackServing3DAtoms.with_summaries`, never the skill's own declaration. No physics.
"""

import itertools

import pytest

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.environments.racktoserving3d.objects import RackServing3DObjects as Objects
from hitl_pmp.environments.racktoserving3d.predicates import (
    ON_RACK,
    RackServing3DAtoms,
    RackServing3DPredicates,
)
from hitl_pmp.environments.racktoserving3d.skills import RackServing3DSkills as Skills
from hitl_pmp.methods.practice_makes_perfect.dp_mdp import DpTaskModel
from hitl_pmp.planning.fast_downward import FastDownwardPlanner, PlanningFailure

OFF_RACK = ("OnTray", "SpilledCounter", "SpilledFloor")
FOOD = tuple(food.name for food in Objects.food)
ALL_ON_RACK = dict.fromkeys(FOOD, "OnRack")


def _measured(*, locations, scraper="home"):
    """The atoms the backend reports for these food locations and this scraper."""
    robot, rack, tray = Objects.robot.name, Objects.rack.name, Objects.tray.name
    names = {("PickPlannable", (robot, food)) for food in FOOD}
    for food, location in locations.items():
        if location == "HoldingFood":
            names.add(("HoldingFood", (robot, food)))
        elif location in {"OnRack", "OnTray"}:
            names.add((location, (food, rack if location == "OnRack" else tray)))
        else:
            names.add((location, (food,)))
    if scraper == "held":
        names.add(("HoldingScraper", (robot, Objects.scraper.name)))
    else:
        names.add(("ScraperHome", (Objects.scraper.name,)))
        if "HoldingFood" not in locations.values():
            names.add(("HandEmpty", (robot,)))
    predicates = {predicate.name: predicate for predicate in RackServing3DPredicates.all()}
    objects = {obj.name: obj for obj in Objects.all_objects()}
    return RackServing3DPredicates.static_atoms() | frozenset(
        GroundAtom(predicate=predicates[name], objects=tuple(objects[key] for key in keys))
        for name, keys in RackServing3DAtoms.with_summaries(atoms=frozenset(names))
    )


def _food(*, name):
    return next(food for food in Objects.food if food.name == name)


def _assert_declared_equals_measured(*, ground, before, after):
    assert ground.preconditions <= before
    declared = DpTaskModel.successor(atoms=before, ground=ground)

    def names(*, atoms):
        return sorted(str(atom) for atom in atoms)

    assert names(atoms=declared - after) == [], "declared true, measured false"
    assert names(atoms=after - declared) == [], "measured true, declared false"


def _return(*, name):
    return GroundSkill(
        skill=Skills.RETURN_FOOD_TO_RACK, objects=(Objects.robot, _food(name=name), Objects.rack)
    )


@pytest.mark.parametrize("name", FOOD)
def test_returning_the_last_food_off_the_rack_turns_any_food_off_rack_off(*, name):
    before = _measured(locations={**ALL_ON_RACK, name: "HoldingFood"})
    _assert_declared_equals_measured(
        ground=_return(name=name), before=before, after=_measured(locations=ALL_ON_RACK)
    )


@pytest.mark.parametrize("elsewhere", OFF_RACK)
@pytest.mark.parametrize(("name", "other"), itertools.permutations(FOOD, 2))
def test_returning_food_while_another_is_off_the_rack_keeps_any_food_off_rack(
    *, name, other, elsewhere
):
    rest = {**ALL_ON_RACK, other: elsewhere}
    _assert_declared_equals_measured(
        ground=_return(name=name),
        before=_measured(locations={**rest, name: "HoldingFood"}),
        after=_measured(locations=rest),
    )


def test_returning_food_after_a_bulk_skill_still_lets_the_person_return_the_rest():
    """The case a plain delete would get wrong: `human_batch_return` needs
    AnyFoodOffRack, and four are still off the rack."""
    rest = {**dict.fromkeys(FOOD, "OnTray"), FOOD[0]: "OnRack"}
    after = DpTaskModel.successor(
        atoms=_measured(locations={**rest, FOOD[0]: "HoldingFood"}, scraper="held"),
        ground=_return(name=FOOD[0]),
    )
    batch = GroundSkill(
        skill=Skills.HUMAN_BATCH_RETURN,
        objects=(Objects.robot, Objects.rack, Objects.tray, *Objects.food),
    )
    assert batch.preconditions <= after


def test_fast_downward_reads_the_same_any_food_off_rack_as_the_declaration():
    """Fast Downward must keep AnyFoodOffRack while other food is off the rack and
    drop it when the returned food was the last one."""
    off = next(
        a
        for a in _measured(locations=ALL_ON_RACK)
        | _measured(locations=dict.fromkeys(FOOD, "OnTray"))
        if a.predicate.name == "AnyFoodOffRack"
    )
    kwargs = {
        "skills": (Skills.RETURN_FOOD_TO_RACK,),
        "predicates": RackServing3DPredicates.all(),
        "types": tuple({obj.type for obj in Objects.all_objects()}),
        "objects": Objects.all_objects(),
        "goal": frozenset({
            off,
            GroundAtom(predicate=ON_RACK, objects=(Objects.food[0], Objects.rack)),
        }),
    }
    others_off = {**dict.fromkeys(FOOD, "OnTray"), FOOD[0]: "HoldingFood"}
    assert FastDownwardPlanner.plan(**kwargs, init_atoms=_measured(locations=others_off)) == [
        _return(name=FOOD[0])
    ]
    with pytest.raises(PlanningFailure):
        FastDownwardPlanner.plan(
            **kwargs, init_atoms=_measured(locations={**ALL_ON_RACK, FOOD[0]: "HoldingFood"})
        )


@pytest.mark.parametrize("name", FOOD)
@pytest.mark.parametrize("source", ["OnRack", *OFF_RACK])
def test_pick_food_declares_what_it_changes_while_other_food_stays_on_the_rack(*, name, source):
    before = {**ALL_ON_RACK, name: source}
    _assert_declared_equals_measured(
        ground=GroundSkill(
            skill=Skills.PICK_FOOD,
            objects=(Objects.robot, _food(name=name), Objects.rack, Objects.tray),
        ),
        before=_measured(locations=before),
        after=_measured(locations={**before, name: "HoldingFood"}),
    )


@pytest.mark.xfail(
    strict=True,
    reason="PickFood of the last food on the rack leaves AnyFoodOnRack declared true; "
    "a guard on 'no other food is on the rack' is not expressible with positive conditions",
)
def test_pick_food_of_the_last_food_on_the_rack_turns_any_food_on_rack_off():
    before = {**dict.fromkeys(FOOD, "OnTray"), FOOD[0]: "OnRack"}
    _assert_declared_equals_measured(
        ground=GroundSkill(
            skill=Skills.PICK_FOOD,
            objects=(Objects.robot, Objects.food[0], Objects.rack, Objects.tray),
        ),
        before=_measured(locations=before),
        after=_measured(locations={**before, FOOD[0]: "HoldingFood"}),
    )


@pytest.mark.parametrize("name", FOOD)
@pytest.mark.parametrize("rest", ["OnRack", *OFF_RACK])
def test_place_food_on_tray_declares_what_it_changes(*, name, rest):
    others = dict.fromkeys(FOOD, rest)
    _assert_declared_equals_measured(
        ground=GroundSkill(
            skill=Skills.PLACE_FOOD_ON_TRAY, objects=(Objects.robot, _food(name=name), Objects.tray)
        ),
        before=_measured(locations={**others, name: "HoldingFood"}),
        after=_measured(locations={**others, name: "OnTray"}),
    )


def test_the_scraper_skills_declare_what_they_change():
    home, held = _measured(locations=ALL_ON_RACK), _measured(locations=ALL_ON_RACK, scraper="held")
    objects = (Objects.robot, Objects.scraper)
    _assert_declared_equals_measured(
        ground=GroundSkill(skill=Skills.PICK_SCRAPER, objects=objects), before=home, after=held
    )
    _assert_declared_equals_measured(
        ground=GroundSkill(skill=Skills.RETURN_SCRAPER, objects=objects), before=held, after=home
    )
    _assert_declared_equals_measured(
        ground=GroundSkill(skill=Skills.HUMAN_TOOL_RECOVERY, objects=objects),
        before=held,
        after=home,
    )


@pytest.mark.parametrize("skill", [Skills.GATHER_AND_SWEEP, Skills.SWEEP])
def test_the_default_bulk_skills_declare_every_food_on_the_tray(*, skill):
    _assert_declared_equals_measured(
        ground=GroundSkill(
            skill=skill,
            objects=(Objects.robot, Objects.scraper, Objects.rack, Objects.tray, *Objects.food),
        ),
        before=_measured(locations=ALL_ON_RACK, scraper="held"),
        after=_measured(locations=dict.fromkeys(FOOD, "OnTray"), scraper="held"),
    )


@pytest.mark.parametrize("scraper", ["home", "held"])
@pytest.mark.parametrize(
    "locations",
    [
        dict.fromkeys(FOOD, "OnTray"),
        dict(
            zip(FOOD, ("OnRack", "OnTray", "SpilledCounter", "SpilledFloor", "OnTray"), strict=True)
        ),
        {**ALL_ON_RACK, FOOD[2]: "HoldingFood"},
    ],
)
def test_the_batch_return_declares_what_it_changes(*, locations, scraper):
    if scraper == "held" and "HoldingFood" in locations.values():
        pytest.skip("one hand")
    _assert_declared_equals_measured(
        ground=GroundSkill(
            skill=Skills.HUMAN_BATCH_RETURN,
            objects=(Objects.robot, Objects.rack, Objects.tray, *Objects.food),
        ),
        before=_measured(locations=locations, scraper=scraper),
        after=_measured(locations=ALL_ON_RACK, scraper=scraper),
    )
