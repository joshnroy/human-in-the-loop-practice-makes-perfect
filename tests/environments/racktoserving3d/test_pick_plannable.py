"""`PickPlannable` on RackToServing3D: Pouring3D's remembered pick refusal, for food.

`food_in_rack_plate_gap.json` is the state before an evaluation PickFood(cube_3) of the
5088ea2 dry run (Curtis, evaluation 0000, task 00): cube_3 lies on the counter in the
1 cm gap between the plate and the rack's front edge, where every grasp's descent puts
a pad into the rack's wires. PickFood refused there at 0 steps every time it was chosen.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from hitl_pmp.core.method.types import LiftedAtom
from hitl_pmp.environments.pouring3d.kitchen.types import PickRefusal, Pose
from hitl_pmp.environments.racktoserving3d.objects import RackServing3DObjects as Objects
from hitl_pmp.environments.racktoserving3d.pick_plannable import RackPickPlannable
from hitl_pmp.environments.racktoserving3d.predicates import (
    PICK_PLANNABLE,
    RackServing3DPredicates,
)
from hitl_pmp.environments.racktoserving3d.skills import RackServing3DSkills
from hitl_pmp.planning.grounding import SkillGrounder
from tests.environments.racktoserving3d.test_planned_food import restored

GAP = Path(__file__).parent / "fixtures" / "food_in_rack_plate_gap.json"
ATOM = ("PickPlannable", ("robot", "cube_3"))


def test_a_planning_refusal_makes_pick_food_inapplicable_until_the_food_moves(*, tmp_path):
    # Recorded in evaluation (deployment mode, no human); replayed in training mode so
    # the human batch return can move the food.
    plain = json.loads(GAP.read_text())
    plain["config"]["mode"] = "training"
    fixture = tmp_path / GAP.name
    fixture.write_text(json.dumps(plain))
    env = restored(fixture=fixture)
    try:
        assert ATOM in env.get_current_state().abstract_atoms
        env.take_action(action=np.array([3.0, 3, 0, 0, 0]))
        assert env._last_controller_steps == 0
        assert (env.last_skill_error() or "").startswith("No collision-free food grasp")
        atoms = env.get_current_state().abstract_atoms
        assert ATOM not in atoms
        assert ("PickPlannable", ("robot", "cube_0")) in atoms
        # What any planner over these operators can choose now: never PickFood(cube_3);
        # the human batch return still.
        state = env.get_current_state()
        true_atoms = SkillGrounder.abstract_state(
            state=state, objects=Objects.all_objects(), predicates=RackServing3DPredicates.all()
        )
        options = SkillGrounder.applicable_ground_skills(
            skills=(*RackServing3DSkills.robot_skills(), RackServing3DSkills.HUMAN_BATCH_RETURN),
            objects=Objects.all_objects(),
            true_atoms=true_atoms,
        )
        picks = {g.objects[1].name for g in options if g.skill.name == "PickFood"}
        assert "cube_3" not in picks
        assert any(g.skill.name == "human_batch_return" for g in options)
        env.reset_movables(destination="human_batch_return")
        assert ATOM in env.get_current_state().abstract_atoms
    finally:
        env.close()


def test_pick_food_requires_the_atom_and_the_human_reset_does_not():
    plannable = LiftedAtom(
        predicate=PICK_PLANNABLE, variables=RackServing3DSkills.PICK_FOOD.parameters[:2]
    )
    assert plannable in RackServing3DSkills.PICK_FOOD.preconditions
    assert not any(
        atom.predicate == PICK_PLANNABLE
        for atom in RackServing3DSkills.HUMAN_BATCH_RETURN.preconditions
    )
    assert any(
        atom.predicate == PICK_PLANNABLE
        for atom in RackServing3DSkills.HUMAN_BATCH_RETURN.add_effects
    )


def _pose(*, x: float) -> Pose:
    return Pose(position=(x, 0.0, 0.9), quat=(0.0, 0.0, 0.0, 1.0))


def test_only_a_zero_step_planning_refusal_with_the_food_unmoved_records():
    before = PickRefusal(dumpling=_pose(x=1.35), pan=_pose(x=1.23))
    backend = SimpleNamespace(cube_names=("cube_0",), held=lambda *, name: False)

    def refusal(*, error: str | None, steps: int, now: float = 1.35) -> PickRefusal | None:
        return RackPickPlannable.refusal(
            backend=backend,
            name="cube_0",
            before=before,
            error=error,
            steps=steps,
            pose=lambda *, name: _pose(x=now),
        )

    planning = "No collision-free food grasp: No planned reach to the food pre-grasp (x: 3)"
    assert refusal(error=planning, steps=0) is not None
    # A physical failure: the pick moved, whatever its message.
    assert refusal(error=planning, steps=12) is None
    assert refusal(error="Food cube_0 is not held by bilateral pad contact", steps=0) is None
    assert refusal(error="PickFood requires an empty hand", steps=0) is None
    assert refusal(error=None, steps=0) is None
    assert refusal(error=planning, steps=0, now=1.36) is None


def test_a_refusal_is_forgotten_once_the_food_or_the_plate_moves():
    refused = PickRefusal(dumpling=_pose(x=1.35), pan=_pose(x=1.23), error="refused")
    poses = {"cube_0": 1.35, "box_0": 1.23}
    backend = SimpleNamespace(cube_names=("cube_0",))

    def atoms(*, refusals: dict) -> set:
        return RackPickPlannable.atoms(
            backend=backend,
            refusals=refusals,
            pose=lambda *, name: _pose(x=poses[name]),
        )

    refusals = {"cube_0": refused}
    assert atoms(refusals=refusals) == set()
    poses["box_0"] = 1.25
    assert atoms(refusals=refusals) == {("PickPlannable", ("robot", "cube_0"))}
    assert refusals == {}
