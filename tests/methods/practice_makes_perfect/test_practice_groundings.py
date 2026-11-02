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


def picks(*, method):
    """The side-bound PickCube gives two distinct groundings of one skill."""
    skill = next(skill for skill in method.skills() if skill.name == "PickCube")
    objects = {obj.name: obj for obj in method.objects()}
    return tuple(
        GroundSkill(
            skill=skill,
            objects=tuple(
                objects[name] for name in ("robot", "cube_0", "cuboid_barrier", side, "bin_0")
            ),
        )
        for side in ("robot_side", "opposite_side")
    )


def observe(*, method, ground, outcomes):
    for outcome in outcomes:
        method.observe_outcome(ground_skill=ground, success=outcome)


def test_each_grounding_keeps_its_own_competence_history():
    method = build()
    near, far = picks(method=method)
    observe(method=method, ground=near, outcomes=[True] * 8)
    observe(method=method, ground=far, outcomes=[False] * 4)
    assert method.practice_groundings == "observed"
    assert method.competence_model(ground_skill=near).get_current_competence() != (
        method.competence_model(ground_skill=far).get_current_competence()
    )


def test_all_groundings_exposes_unseen_targets_without_changing_skills():
    original = build()
    flexible = build(practice_groundings="all")
    near, far = picks(method=flexible)
    assert flexible.skills() == original.skills()
    assert flexible.skill_provider.human_cube_bin_reset_skills() == (
        original.skill_provider.human_cube_bin_reset_skills()
    )
    assert original.practice_candidates() == []
    assert {near, far} <= set(flexible.practice_candidates())


def test_all_ees_groundings_keep_paramless_open_gripper():
    method = build(practice_groundings="all")
    candidates = method.practice_candidates()
    grippers = [ground for ground in candidates if ground.skill.name == "OpenGripper"]
    assert grippers
    assert all(ground.skill.param_dim == 0 for ground in grippers)
    assert set(candidates) == set(method.choose_practice_target())
