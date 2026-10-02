"""The native five-cube task requires individually chosen task and recovery skills."""

from hitl_pmp.environments.sweep_simple3d.symbolic import SIMPLE_PREDICATES, SimpleSymbols
from hitl_pmp.planning.grounding import SkillGrounder


def test_task_skills_ground_once_per_cube() -> None:
    atoms = SkillGrounder.all_possible_ground_atoms(
        objects=SimpleSymbols.objects(), predicates=tuple(SIMPLE_PREDICATES.values())
    )
    grounded = SkillGrounder.applicable_ground_skills(
        skills=SimpleSymbols.skills(), objects=SimpleSymbols.objects(), true_atoms=atoms
    )
    assert sum(s.skill.name == "SweepCubeToGoal" for s in grounded) == 5
    assert sum(s.skill.name == "SweepCubeToStart" for s in grounded) == 5
    assert sum(s.skill.name == "PickFloorWiper" for s in grounded) == 1


def test_human_reset_targets_initial_distribution_not_task_goal() -> None:
    reset = SimpleSymbols.human_reset(cost=1.0)
    assert len([a for a in reset.add_effects if a.predicate.name == "AtStart"]) == 5
    assert all(a.predicate.name != "InGoal" for a in reset.add_effects)
    assert len(SimpleSymbols.goal_atoms()) == 5
