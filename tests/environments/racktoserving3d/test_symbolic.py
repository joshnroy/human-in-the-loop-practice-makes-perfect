"""Core Skill groundings and alternate delivery/recovery plans, without physics."""

import pytest

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.environments.racktoserving3d.objects import RackServing3DObjects as Objects
from hitl_pmp.environments.racktoserving3d.predicates import (
    ANY_FOOD_OFF_RACK,
    ANY_FOOD_ON_RACK,
    FOOD_AVAILABLE,
    HAND_EMPTY,
    HOLDING_FOOD,
    HOLDING_SCRAPER,
    ON_RACK,
    ON_TRAY,
    SPILLED_COUNTER,
    RackServing3DAtoms,
    RackServing3DPredicates,
)
from hitl_pmp.environments.racktoserving3d.skills import GATHER_AND_SWEEP_BOUNDS, SKILL_IDS
from hitl_pmp.environments.racktoserving3d.skills import RackServing3DSkills as Skills
from hitl_pmp.planning.grounding import SkillGrounder


def _atom(*, predicate, objects=()):
    return GroundAtom(predicate=predicate, objects=objects)


def _ground(*, names):
    predicates = {predicate.name: predicate for predicate in RackServing3DPredicates.all()}
    objects = {obj.name: obj for obj in Objects.all_objects()}
    return (
        frozenset(
            _atom(predicate=predicates[name], objects=tuple(objects[key] for key in keys))
            for name, keys in names
        )
        | RackServing3DPredicates.static_atoms()
    )


def _start(*, rack_count=5, elsewhere="OnTray", held_scraper=False):
    names = {("OnRack", (food.name, Objects.rack.name)) for food in Objects.food[:rack_count]}
    names |= {
        (elsewhere, (food.name, Objects.tray.name) if elsewhere == "OnTray" else (food.name,))
        for food in Objects.food[rack_count:]
    }
    # No pick has refused to plan in a state never tried (`pick_plannable.py`).
    names |= {("PickPlannable", (Objects.robot.name, food.name)) for food in Objects.food}
    if held_scraper:
        names.add(("HoldingScraper", (Objects.robot.name, Objects.scraper.name)))
    else:
        names |= {("HandEmpty", (Objects.robot.name,)), ("ScraperHome", (Objects.scraper.name,))}
    return _ground(names=RackServing3DAtoms.with_summaries(atoms=frozenset(names)))


def _groundings(*, skill, atoms):
    return SkillGrounder.applicable_ground_skills(
        skills=(skill,), objects=Objects.all_objects(), true_atoms=atoms
    )


def _apply(*, ground, atoms):
    assert ground.preconditions <= atoms
    return (atoms - ground.delete_effects) | ground.expected_add_effects(before_atoms=atoms)


def test_object_names_features_and_scene_exclusion():
    assert {obj.name for obj in Objects.all_objects()} == {
        "robot",
        "wiper_0",
        "box_0",
        "rack_0",
        *(f"cube_{index}" for index in range(5)),
    }
    assert Objects.food_type.dim == 7
    assert Objects.robot_type.dim == Objects.scraper_type.dim == 4
    assert Objects.scene_type.feature_names == (
        "seed",
        "steps_taken",
        "elapsed_seconds",
        "human_cost",
    )
    assert Objects.scene not in Objects.all_objects()
    assert Objects.scene_type not in Objects.all_types()


def test_robot_ids_bounds_and_human_separation():
    assert [SKILL_IDS[skill.name] for skill in Skills.robot_skills()] == list(range(6))
    assert GATHER_AND_SWEEP_BOUNDS == ((0.03, 0.14), (0.005, 0.075), (0.04, 0.12))
    assert Skills.GATHER_AND_SWEEP.name == "GatherAndSweep"
    assert Skills.GATHER_AND_SWEEP.param_dim == 3
    assert "SweepRack" not in SKILL_IDS
    assert SKILL_IDS["GatherAndSweep"] == 1
    assert all(
        skill.param_dim == 0 for skill in Skills.robot_skills() if skill != Skills.GATHER_AND_SWEEP
    )
    assert not {skill.name for skill in Skills.human_skills()} & set(SKILL_IDS)
    assert Skills.human_skills(mode="deployment") == ()


def test_goal_is_delivery_not_sweep_history():
    assert RackServing3DPredicates.goal_atoms() == frozenset(
        _atom(predicate=ON_TRAY, objects=(food, Objects.tray)) for food in Objects.food
    )
    assert RackServing3DPredicates.goal_atoms() <= _start(rack_count=0)
    assert not RackServing3DPredicates.goal_atoms() <= _start(rack_count=1)


@pytest.mark.parametrize("source", ["OnTray", "SpilledCounter", "SpilledFloor"])
def test_pick_food_applies_from_every_supported_source_at_once(*, source):
    assert len(_groundings(skill=Skills.PICK_FOOD, atoms=_start())) == len(Objects.food)
    assert len(
        _groundings(skill=Skills.PICK_FOOD, atoms=_start(rack_count=0, elsewhere=source))
    ) == len(Objects.food)


def test_autonomous_pick_place_solves_without_scraper_or_human():
    atoms = _start()
    for food in Objects.food:
        pick = GroundSkill(
            skill=Skills.PICK_FOOD, objects=(Objects.robot, food, Objects.rack, Objects.tray)
        )
        atoms = _apply(ground=pick, atoms=atoms)
        place = GroundSkill(
            skill=Skills.PLACE_FOOD_ON_TRAY, objects=(Objects.robot, food, Objects.tray)
        )
        atoms = _apply(ground=place, atoms=atoms)
    assert RackServing3DPredicates.goal_atoms() <= atoms


@pytest.mark.parametrize("source", ["OnTray", "SpilledCounter", "SpilledFloor"])
def test_robot_can_pick_and_return_from_each_supported_destination(*, source):
    atoms = _start(rack_count=4, elsewhere=source)
    food = Objects.food[-1]
    pick = GroundSkill(
        skill=Skills.PICK_FOOD, objects=(Objects.robot, food, Objects.rack, Objects.tray)
    )
    held = _apply(ground=pick, atoms=atoms)
    assert _atom(predicate=HOLDING_FOOD, objects=(Objects.robot, food)) in held
    assert not any(
        atom.objects and atom.objects[0] == food for atom in held if atom.predicate.name == source
    )
    returned = _apply(
        ground=GroundSkill(
            skill=Skills.RETURN_FOOD_TO_RACK, objects=(Objects.robot, food, Objects.rack)
        ),
        atoms=held,
    )
    assert _atom(predicate=ON_RACK, objects=(food, Objects.rack)) in returned
    assert _atom(predicate=FOOD_AVAILABLE, objects=(food,)) in returned
    assert _atom(predicate=HAND_EMPTY, objects=(Objects.robot,)) in returned


@pytest.mark.parametrize("count", [1, 3, 5])
def test_sweep_promises_only_food_measured_on_rack(*, count):
    atoms = _start(rack_count=count, elsewhere="SpilledCounter", held_scraper=True)
    (sweep,) = _groundings(skill=Skills.GATHER_AND_SWEEP, atoms=atoms)
    promised = sweep.expected_add_effects(before_atoms=atoms)
    assert {atom for atom in promised if atom.predicate == ON_TRAY} == {
        _atom(predicate=ON_TRAY, objects=(food, Objects.tray)) for food in Objects.food[:count]
    }
    after = _apply(ground=sweep, atoms=atoms)
    assert _atom(predicate=ANY_FOOD_ON_RACK) not in after
    assert _atom(predicate=ANY_FOOD_OFF_RACK) in after


def test_gather_and_sweep_succeeds_only_when_every_food_on_the_rack_is_served():
    atoms = _start(rack_count=5, held_scraper=True)
    (sweep,) = _groundings(skill=Skills.GATHER_AND_SWEEP, atoms=atoms)
    promised = sweep.expected_add_effects(before_atoms=atoms)
    served = _apply(ground=sweep, atoms=atoms)
    assert promised <= served
    for missed in Objects.food:
        partial = (served - {_atom(predicate=ON_TRAY, objects=(missed, Objects.tray))}) | {
            _atom(predicate=SPILLED_COUNTER, objects=(missed,))
        }
        assert not promised <= partial


def test_an_empty_rack_cannot_be_swept_and_a_fresh_one_can_at_once():
    assert not _groundings(
        skill=Skills.GATHER_AND_SWEEP, atoms=_start(rack_count=0, held_scraper=True)
    )
    assert _groundings(skill=Skills.GATHER_AND_SWEEP, atoms=_start(held_scraper=True))


def test_batch_return_preserves_the_held_scraper():
    atoms = _start(rack_count=2, held_scraper=True)
    (batch,) = _groundings(skill=Skills.HUMAN_BATCH_RETURN, atoms=atoms)
    after = _apply(ground=batch, atoms=atoms)
    assert _atom(predicate=HOLDING_SCRAPER, objects=(Objects.robot, Objects.scraper)) in after
    assert _atom(predicate=HAND_EMPTY, objects=(Objects.robot,)) not in after
    assert all(
        _atom(predicate=ON_RACK, objects=(food, Objects.rack)) in after for food in Objects.food
    )


def test_human_recovery_does_not_drop_held_food():
    atoms = _start()
    pick = _groundings(skill=Skills.PICK_FOOD, atoms=atoms)[0]
    atoms = _apply(ground=pick, atoms=atoms)
    names = frozenset({
        (atom.predicate.name, tuple(obj.name for obj in atom.objects))
        for atom in atoms
        if atom.predicate.name != "ScraperHome"
    })
    atoms = _ground(names=RackServing3DAtoms.with_summaries(atoms=names))
    (recover,) = _groundings(skill=Skills.HUMAN_TOOL_RECOVERY, atoms=atoms)
    after = _apply(ground=recover, atoms=atoms)
    assert _atom(predicate=HOLDING_FOOD, objects=(Objects.robot, pick.objects[1])) in after
    assert _atom(predicate=HAND_EMPTY, objects=(Objects.robot,)) not in after


def test_human_actions_do_not_apply_when_nothing_needs_returning():
    atoms = _start()
    for skill in Skills.human_skills():
        assert not _groundings(skill=skill, atoms=atoms)


def test_missing_measured_abstraction_is_rejected():
    with pytest.raises(ValueError, match="measured"):
        RackServing3DAtoms.holds(state=State(data={}), name="Ready", objects=())


def test_food_starts_ready_so_there_is_no_wait_skill_or_readiness_atom():
    assert "Wait" not in SKILL_IDS
    assert "Wait" not in {skill.name for skill in Skills.robot_skills()}
    names = {predicate.name for predicate in RackServing3DPredicates.all()}
    assert not names & {"Ready", "Waiting"}
    assert {atom.predicate.name for atom in RackServing3DPredicates.goal_atoms()} == {"OnTray"}
    for skill in (*Skills.robot_skills(), *Skills.human_skills()):
        mentioned = {
            atom.predicate.name
            for atom in (*skill.preconditions, *skill.add_effects, *skill.delete_effects)
        }
        assert not mentioned & {"Ready", "Waiting"}, skill.name
