"""The dump's promised effects match what it physically does: the cube ends clear."""

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.predicates import GRASP_CLEAR
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.methods.belief_space.tossing3d_transition_model import apply_success_effects


def _grounded(*, env: Tossing3DEnvironment, name: str) -> GroundSkill:
    provider = Tossing3DSkillProvider(env=env)
    skill = next(skill for skill in provider.skills() if skill.name == name)
    return GroundSkill(skill=skill, objects=(env.robot, env.cube, env.bin, env.barrier))


def _forecast(*, ground: GroundSkill, before: frozenset[GroundAtom]) -> frozenset[GroundAtom]:
    return apply_success_effects(
        true_atoms=before,
        ground_skill=ground,
        effects={ground: (ground.add_effects, ground.delete_effects, ground.ignore_effects)},
    )


def test_a_dumped_cube_is_forecast_clear_of_the_bin_walls() -> None:
    env = Tossing3DEnvironment()
    dump = _grounded(env=env, name="PourCubeOutOfBin")
    # An in-bin cube against a wall: GraspClear false before the dump.
    before = dump.preconditions
    clear = GroundAtom(predicate=GRASP_CLEAR, objects=(env.cube, env.bin))
    assert clear not in before
    assert clear in _forecast(ground=dump, before=before)


def test_dump_and_same_side_reset_forecast_the_same_cube_state() -> None:
    env = Tossing3DEnvironment()
    dump = _grounded(env=env, name="PourCubeOutOfBin")
    provider = Tossing3DSkillProvider(env=env)
    reset = next(
        skill
        for skill in provider.human_cube_bin_reset_skills()
        if skill.skill.name == "ResetCubeBinSameSide"
    )
    before = dump.preconditions
    after_dump = _forecast(ground=dump, before=before)
    after_reset = _forecast(ground=reset, before=before)
    assert after_dump == after_reset
