from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod


def build(**kwargs):
    environment = Tossing3DEnvironment()
    return EesMethod(
        env=environment,
        skill_provider=Tossing3DSkillProvider(env=environment),
        use_ucb_bonus=False,
        reproduce_predicators_double_observe=False,
        **kwargs,
    )


def grounding(*, method, name):
    """The one grounding of a skill: Tossing3D has one object of each type and no skill
    binds a side, so every skill grounds exactly once."""
    skill = next(skill for skill in method.skills() if skill.name == name)
    by_type = {obj.type: obj for obj in method.objects()}
    return GroundSkill(
        skill=skill, objects=tuple(by_type[variable.type] for variable in skill.parameters)
    )


def observe(*, method, ground, outcomes):
    for outcome in outcomes:
        method.observe_outcome(ground_skill=ground, success=outcome)


def test_each_grounding_keeps_its_own_competence_history():
    method = build()
    pick = grounding(method=method, name="PickCube")
    pour = grounding(method=method, name="PourCubeOutOfBin")
    observe(method=method, ground=pick, outcomes=[True] * 8)
    observe(method=method, ground=pour, outcomes=[False] * 4)
    assert method.practice_groundings == "observed"
    assert method.competence_model(ground_skill=pick).get_current_competence() != (
        method.competence_model(ground_skill=pour).get_current_competence()
    )


def test_all_groundings_exposes_unseen_targets_without_changing_skills():
    original = build()
    flexible = build(practice_groundings="all")
    assert flexible.skills() == original.skills()
    assert flexible.skill_provider.human_cube_bin_reset_skills() == (
        original.skill_provider.human_cube_bin_reset_skills()
    )
    assert original.practice_candidates() == []
    # One candidate per skill: with no side object there is no never-runnable
    # opposite-side PickCube or Pour grounding left to enumerate.
    assert flexible.practice_candidates() == [
        grounding(method=flexible, name=skill.name) for skill in flexible.skills()
    ]


def test_all_ees_groundings_keep_paramless_open_gripper():
    method = build(practice_groundings="all")
    candidates = method.practice_candidates()
    grippers = [ground for ground in candidates if ground.skill.name == "OpenGripper"]
    assert grippers
    assert all(ground.skill.param_dim == 0 for ground in grippers)
    assert set(candidates) == set(method.choose_practice_target())
