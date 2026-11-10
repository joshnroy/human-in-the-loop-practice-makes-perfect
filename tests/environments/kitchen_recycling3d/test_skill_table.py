"""The kitchen's skill table is Tossing3D's: geometry is the only difference.

Diffed field by field against a Tossing3D provider rather than by identity, so a
kitchen-only skill, a near/far copy, a changed bound or a changed cost fails here with
the field that moved.
"""

from hitl_pmp.core.method.types import Skill
from hitl_pmp.environments.kitchen_recycling3d.domain import KitchenConfig
from hitl_pmp.environments.kitchen_recycling3d.environment import KitchenRecycling3DEnvironment
from hitl_pmp.environments.kitchen_recycling3d.skill_provider import KitchenRecyclingSkillProvider
from hitl_pmp.environments.kitchen_recycling3d.step_protocol import KitchenRecyclingStepProtocol
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.environments.tossing3d.toss import Tossing3DToss
from hitl_pmp.methods.belief_space.practice_domain import TOSSING3D_DOMAIN
from hitl_pmp.methods.practice_makes_perfect.cli import Tossing3DPomdpCli

SINGLE_CUBE = KitchenConfig.model_validate({
    "objects": [
        {"name": "cube_control", "kind": "cube", "dimensions": [0.05] * 3, "simulation_mass": 0.1}
    ],
    "required_objects": ["cube_control"],
})


def row(*, skill: Skill) -> dict:
    def atoms(*, values: frozenset) -> list:
        return sorted(
            (atom.predicate.name, tuple(v.name for v in atom.variables)) for atom in values
        )

    return {
        "name": skill.name,
        "parameters": [(v.name, v.type.name) for v in skill.parameters],
        "param_dim": skill.param_dim,
        "preconditions": atoms(values=skill.preconditions),
        "add_effects": atoms(values=skill.add_effects),
        "delete_effects": atoms(values=skill.delete_effects),
        "ignore_effects": sorted(p.name for p in skill.ignore_effects),
        "conditional_add_effects": sorted(repr(e) for e in skill.conditional_add_effects),
        "conditional_ignore_effects": sorted(repr(e) for e in skill.conditional_ignore_effects),
    }


def providers() -> tuple[Tossing3DSkillProvider, Tossing3DSkillProvider]:
    kitchen = KitchenRecycling3DEnvironment(kitchen_config=SINGLE_CUBE)
    tossing = Tossing3DSkillProvider(env=Tossing3DEnvironment())
    return tossing, KitchenRecyclingSkillProvider(env=kitchen)


def test_robot_skills_match_tossing3d_field_by_field():
    tossing, kitchen = providers()
    assert [row(skill=s) for s in kitchen.skills()] == [row(skill=s) for s in tossing.skills()]


def test_human_resets_match_tossing3d_field_by_field():
    tossing, kitchen = providers()
    pairs = zip(kitchen.movables_reset_skills(), tossing.movables_reset_skills(), strict=True)
    for ours, theirs in pairs:
        assert row(skill=ours.skill) == row(skill=theirs.skill)
        assert [o.name for o in ours.objects] == [o.name for o in theirs.objects]
        assert kitchen.movables_reset_destination(
            ground_skill=ours
        ) == tossing.movables_reset_destination(ground_skill=theirs)


def test_predicates_types_and_objects_match_tossing3d():
    tossing, kitchen = providers()
    assert [p.name for p in kitchen.predicates()] == [p.name for p in tossing.predicates()]
    assert kitchen.types() == tossing.types()
    assert kitchen.objects() == tossing.objects()


def test_speed_and_release_bounds_match_tossing3d_and_the_standoff_is_the_kitchens_own():
    tossing, kitchen = providers()
    ours = Tossing3DToss.parameters_of(env=kitchen.env)
    theirs = Tossing3DToss.parameters_of(env=tossing.env)
    assert (ours.speed_bounds, ours.release_ms_bounds) == (
        theirs.speed_bounds,
        theirs.release_ms_bounds,
    )
    assert ours.standoff_bounds == (1.445, 1.455)


def test_one_belief_per_skill_no_context_copies():
    _, kitchen = providers()
    names = [s.name for s in kitchen.skills()]
    assert len(names) == len(set(names))
    assert not any(token in name.lower() for name in names for token in ("near", "far"))


def test_geometry_is_the_difference():
    tossing, kitchen = providers()
    assert kitchen.env.drive_zone() == (-1.06, -2.06, 0.71, 2.06)
    assert not hasattr(tossing.env, "drive_zone")


def test_belief_space_practice_domain_is_tossing3ds():
    assert Tossing3DPomdpCli.DOMAINS["KitchenRecyclingCli"] is TOSSING3D_DOMAIN


def test_reset_destinations_are_charged_under_the_reset_skill_names():
    _, kitchen = providers()
    names = KitchenRecyclingStepProtocol.human_skill_names()
    assert {
        names[kitchen.movables_reset_destination(ground_skill=reset)]
        for reset in kitchen.movables_reset_skills()
    } == {"ResetCubeBinSameSide", "ResetCubeBinOppositeSide"}
