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


def _measured(*, locations, scraper="home", served=()):
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
    predicates = {
        predicate.name: predicate for predicate in RackServing3DPredicates.all(two_route=True)
    }
    objects = {obj.name: obj for obj in Objects.all_objects()}
    return RackServing3DPredicates.static_atoms() | frozenset(
        GroundAtom(predicate=predicates[name], objects=tuple(objects[key] for key in keys))
        for name, keys in (
            *RackServing3DAtoms.with_summaries(atoms=frozenset(names)),
            *((name, ()) for name in served),
        )
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


# Two-route Sweep on a lined-up start: the middle three are promised on the plate and
# nothing is promised about the outer two (Josh, 2026-11-12).
MIDDLE, OUTER = FOOD[1:4], (FOOD[0], FOOD[4])
SERVED_THREE = ("Served", "ServedThree")
TWO_ROUTE_SWEEP = GroundSkill(
    skill=Skills.TWO_ROUTE_SWEEP,
    objects=(Objects.robot, Objects.scraper, Objects.rack, Objects.tray, *Objects.food),
)
BEFORE_SWEEP = _measured(locations=ALL_ON_RACK, scraper="held")
ACCEPTED = {
    "both on the counter": ("SpilledCounter", "SpilledCounter"),
    "both still on the rack": ("OnRack", "OnRack"),
    "one each": ("SpilledCounter", "OnRack"),
    "both on the floor": ("SpilledFloor", "SpilledFloor"),
    "floor and rack": ("OnRack", "SpilledFloor"),
}


def _after_sweep(*, outer, middle=("OnTray", "OnTray", "OnTray"), served=SERVED_THREE):
    locations = {**dict(zip(MIDDLE, middle, strict=True)), **dict(zip(OUTER, outer, strict=True))}
    return _measured(locations=locations, scraper="held", served=served)


def _ees_counts_success(*, ground, before, after):
    return ground.expected_add_effects(before_atoms=before) <= after


def _dp_deployment_accepts(*, ground, before, after):
    """`DeploymentPolicy`'s `required` and `forbidden` for a DP plan step."""
    ignored = ground.expected_ignore_effects(before_atoms=before)
    added = ground.expected_add_effects(before_atoms=before)
    required = frozenset(a for a in before if a.predicate not in ignored) - ground.delete_effects
    return required | added <= after and not (ground.delete_effects - added) & after


@pytest.mark.parametrize("outer", ACCEPTED.values(), ids=ACCEPTED.keys())
def test_a_two_route_sweep_holds_wherever_the_outer_two_end_up(*, outer):
    after = _after_sweep(outer=outer)
    assert _ees_counts_success(ground=TWO_ROUTE_SWEEP, before=BEFORE_SWEEP, after=after)
    assert _dp_deployment_accepts(ground=TWO_ROUTE_SWEEP, before=BEFORE_SWEEP, after=after)
    batch = GroundSkill(
        skill=Skills.TWO_ROUTE_HUMAN_BATCH_RETURN,
        objects=(Objects.robot, Objects.rack, Objects.tray, *Objects.food),
    )
    assert batch.preconditions <= after
    assert batch.preconditions <= DpTaskModel.successor(atoms=BEFORE_SWEEP, ground=TWO_ROUTE_SWEEP)


@pytest.mark.parametrize(
    ("middle", "outer", "served"),
    [
        (("OnTray", "SpilledCounter", "OnTray"), ("SpilledCounter", "SpilledCounter"), ()),
        (("OnTray", "OnRack", "OnTray"), ("OnRack", "OnRack"), ()),
        # Three on the plate, so the environment says Served, but not the middle three.
        (("OnTray", "OnTray", "SpilledCounter"), ("OnTray", "SpilledCounter"), SERVED_THREE),
    ],
)
def test_a_two_route_sweep_does_not_hold_without_the_middle_three_on_the_tray(
    *, middle, outer, served
):
    after = _after_sweep(outer=outer, middle=middle, served=served)
    assert not _ees_counts_success(ground=TWO_ROUTE_SWEEP, before=BEFORE_SWEEP, after=after)
    assert not _dp_deployment_accepts(ground=TWO_ROUTE_SWEEP, before=BEFORE_SWEEP, after=after)


def test_a_two_route_sweep_declares_the_middle_three_and_forgets_the_rest_of_the_rack():
    after = DpTaskModel.successor(atoms=BEFORE_SWEEP, ground=TWO_ROUTE_SWEEP)
    changed = {str(atom) for atom in after ^ BEFORE_SWEEP}
    expected = _after_sweep(outer=("OnRack", "OnRack"))
    outer_on_rack = {
        atom for atom in expected if atom.predicate.name in {"OnRack", "AnyFoodOnRack"}
    }
    assert after == expected - outer_on_rack
    assert len(changed) == 12


def test_fast_downward_plans_a_two_route_sweep_and_the_return_after_it():
    predicates = {p.name: p for p in RackServing3DPredicates.all(two_route=True)}
    kwargs = {
        "skills": (*Skills.robot_skills(two_route=True), *Skills.human_skills(two_route=True)),
        "predicates": tuple(predicates.values()),
        "types": tuple({obj.type for obj in Objects.all_objects()}),
        "objects": Objects.all_objects(),
    }
    plan = FastDownwardPlanner.plan(
        **kwargs,
        init_atoms=BEFORE_SWEEP,
        goal=frozenset({
            GroundAtom(predicate=predicates["ServedThree"], objects=()),
            GroundAtom(predicate=predicates["OnTray"], objects=(Objects.food[2], Objects.tray)),
        }),
    )
    assert plan == [TWO_ROUTE_SWEEP]
    back = FastDownwardPlanner.plan(
        **kwargs,
        init_atoms=DpTaskModel.successor(atoms=BEFORE_SWEEP, ground=TWO_ROUTE_SWEEP),
        goal=frozenset(
            GroundAtom(predicate=ON_RACK, objects=(food, Objects.rack)) for food in Objects.food
        ),
    )
    assert [ground.skill.name for ground in back] == ["human_batch_return"]


# RobotBatchReturn (`RackServingConfig.robot_batch_return`): the robot's own batch
# return. Its declaration is the person's, so what follows it is what follows theirs.
ROBOT_RETURN = GroundSkill(
    skill=Skills.ROBOT_BATCH_RETURN,
    objects=(Objects.robot, Objects.rack, Objects.tray, *Objects.food),
)
TWO_ROUTE_ROBOT_RETURN = GroundSkill(
    skill=Skills.TWO_ROUTE_ROBOT_BATCH_RETURN, objects=ROBOT_RETURN.objects
)
AFTER_A_BULK_TRY = {
    "a Sweep's three": dict(
        zip(FOOD, ("OnRack", "OnTray", "OnTray", "OnTray", "OnRack"), strict=True)
    ),
    "all five": dict.fromkeys(FOOD, "OnTray"),
    "the rack emptied onto the counter and the floor": dict(
        zip(
            FOOD,
            ("SpilledCounter", "OnTray", "SpilledFloor", "OnTray", "SpilledCounter"),
            strict=True,
        )
    ),
}


@pytest.mark.parametrize("scraper", ["home", "held"])
@pytest.mark.parametrize("locations", AFTER_A_BULK_TRY.values(), ids=AFTER_A_BULK_TRY.keys())
def test_the_robot_batch_return_declares_what_it_changes(*, locations, scraper):
    """The last case starts with nothing on the rack: PickFood's AnyFoodOnRack defect
    (the strict xfail above) is not on this skill's declared path."""
    _assert_declared_equals_measured(
        ground=ROBOT_RETURN,
        before=_measured(locations=locations, scraper=scraper),
        after=_measured(locations=ALL_ON_RACK, scraper=scraper),
    )


def test_the_robot_batch_return_sets_a_held_dumpling_down_too():
    _assert_declared_equals_measured(
        ground=ROBOT_RETURN,
        before=_measured(locations={**ALL_ON_RACK, FOOD[2]: "HoldingFood"}),
        after=_measured(locations=ALL_ON_RACK),
    )


def test_the_robot_batch_return_is_declared_as_the_persons_is():
    for robot, human in (
        (Skills.ROBOT_BATCH_RETURN, Skills.HUMAN_BATCH_RETURN),
        (Skills.TWO_ROUTE_ROBOT_BATCH_RETURN, Skills.TWO_ROUTE_HUMAN_BATCH_RETURN),
    ):
        assert robot.name == "RobotBatchReturn"
        assert robot == human.model_copy(update={"name": robot.name})


@pytest.mark.parametrize("outer", ACCEPTED.values(), ids=ACCEPTED.keys())
def test_both_bulk_skills_are_applicable_again_after_a_two_route_robot_return(*, outer):
    after_sweep = _after_sweep(outer=outer)
    assert TWO_ROUTE_ROBOT_RETURN.preconditions <= after_sweep
    restored = DpTaskModel.successor(atoms=after_sweep, ground=TWO_ROUTE_ROBOT_RETURN)
    assert restored == BEFORE_SWEEP
    for skill in (Skills.TWO_ROUTE_SWEEP, Skills.TWO_ROUTE_GATHER_AND_SWEEP):
        assert GroundSkill(skill=skill, objects=TWO_ROUTE_SWEEP.objects).preconditions <= restored


def test_a_partial_robot_return_leaves_both_returns_applicable_and_no_bulk_skill():
    """Three of five returned, as measured: the skill failed and declared nothing."""
    partial = _measured(
        locations=dict(zip(FOOD, ("OnRack", "OnRack", "OnRack", "OnTray", "OnTray"), strict=True))
    )
    human = GroundSkill(skill=Skills.TWO_ROUTE_HUMAN_BATCH_RETURN, objects=ROBOT_RETURN.objects)
    assert TWO_ROUTE_ROBOT_RETURN.preconditions <= partial
    assert human.preconditions <= partial
    assert not _ees_counts_success(
        ground=TWO_ROUTE_ROBOT_RETURN,
        before=_after_sweep(outer=("OnRack", "OnRack")),
        after=partial,
    )
    for skill in (Skills.TWO_ROUTE_SWEEP, Skills.TWO_ROUTE_GATHER_AND_SWEEP):
        ground = GroundSkill(skill=skill, objects=TWO_ROUTE_SWEEP.objects)
        assert not ground.preconditions <= partial


def test_the_robot_batch_return_is_offered_only_when_asked_for():
    names = [skill.name for skill in Skills.robot_skills(two_route=True)]
    assert "RobotBatchReturn" not in names
    offered = Skills.robot_skills(two_route=True, include_robot_batch_return=True)
    assert [skill.name for skill in offered] == [*names, "RobotBatchReturn"]
    assert offered[-1] == Skills.TWO_ROUTE_ROBOT_BATCH_RETURN
    assert Skills.robot_skills(include_robot_batch_return=True)[-1] == Skills.ROBOT_BATCH_RETURN


def test_fast_downward_plans_the_robot_return_when_no_person_is_offered():
    predicates = {p.name: p for p in RackServing3DPredicates.all(two_route=True)}
    back = FastDownwardPlanner.plan(
        skills=Skills.robot_skills(two_route=True, include_robot_batch_return=True),
        predicates=tuple(predicates.values()),
        types=tuple({obj.type for obj in Objects.all_objects()}),
        objects=Objects.all_objects(),
        init_atoms=DpTaskModel.successor(atoms=BEFORE_SWEEP, ground=TWO_ROUTE_SWEEP),
        goal=frozenset(
            GroundAtom(predicate=ON_RACK, objects=(food, Objects.rack)) for food in Objects.food
        ),
    )
    assert back == [TWO_ROUTE_ROBOT_RETURN]
