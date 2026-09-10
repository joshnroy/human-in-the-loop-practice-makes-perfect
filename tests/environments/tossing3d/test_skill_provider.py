"""Offline tests for Tossing3D's sampler features and reset skills.

Only reset execution touches MuJoCo; the feature and symbolic contracts are plain Python.
"""

import numpy as np
import pytest

from hitl_pmp.core.method.skill_provider import ASK_FOR_RESET_CUBE_BIN_ONLY_NAME
from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.predicates import IN_BIN, ON_GROUND, REACHABLE
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.environments.tossing3d.skills import Tossing3DSkills

from .observations import state


def _provider() -> Tossing3DSkillProvider:
    return Tossing3DSkillProvider(env=Tossing3DEnvironment())


def _toss(*, env: Tossing3DEnvironment) -> GroundSkill:
    return GroundSkill(
        skill=Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS,
        objects=(env.robot, env.bin, env.cube, env.barrier),
    )


def test_toss_sampler_input_is_robot_frame_bin_displacement_then_params() -> None:
    env = Tossing3DEnvironment()
    params = np.array([1.3, -0.01, 125.0, 760.0])
    scene = state(env=env, base_x=0.2, base_y=-0.4, base_rot=np.pi / 2, bin_x=1.7)
    # observation() fixes bin y at zero: world displacement is (1.5, 0.4), which
    # becomes (forward=0.4, lateral=-1.5) for a robot facing +y.
    assert Tossing3DSkillProvider(env=env).oracle_sampler_input(
        ground_skill=_toss(env=env), state=scene, params=params
    ) == pytest.approx([1.0, 0.4, -1.5, 1.3, -0.01, 125.0, 760.0])


def test_toss_sampler_input_and_relative_move_params_are_invariant_to_a_rigid_half_turn() -> None:
    env = Tossing3DEnvironment()
    provider = Tossing3DSkillProvider(env=env)
    params = np.array([1.35, 0.0, 140.0, 792.0])
    original = state(env=env, base_x=0.15, base_rot=0.2, bin_x=2.0)
    rotated = state(env=env, base_x=-0.15, base_rot=0.2 + np.pi, bin_x=-2.0)
    assert provider.oracle_sampler_input(
        ground_skill=_toss(env=env), state=rotated, params=params
    ) == pytest.approx(
        provider.oracle_sampler_input(ground_skill=_toss(env=env), state=original, params=params)
    )


def test_same_side_toss_uses_the_same_relative_feature_layout() -> None:
    from hitl_pmp.environments.tossing3d.layout import Tossing3DLayout
    from hitl_pmp.environments.tossing3d.recovery_skills import SameSideSkills

    env = Tossing3DEnvironment(layout=Tossing3DLayout.SAME_SIDE)
    toss = GroundSkill(
        skill=SameSideSkills.TOSS,
        objects=(env.robot, env.bin, env.cube, env.barrier),
    )
    row = Tossing3DSkillProvider(env=env).oracle_sampler_input(
        ground_skill=toss,
        state=state(env=env, base_x=0.0, base_y=0.0, base_rot=np.pi, bin_x=-2.0),
        params=np.array([1.35, 0.0, 140.0, 792.0]),
    )
    assert row == pytest.approx([1.0, 2.0, 0.0, 1.35, 0.0, 140.0, 792.0])


def test_non_toss_skills_keep_the_generic_sampler_input_fallback() -> None:
    env = Tossing3DEnvironment()
    pick = GroundSkill(
        skill=Tossing3DSkills.PICK_CUBE,
        objects=(env.robot, env.cube, env.barrier),
    )
    assert (
        _provider().oracle_sampler_input(ground_skill=pick, state=state(), params=np.zeros(0))
        is None
    )


def test_the_ground_skill_is_named_for_ees_to_intercept() -> None:
    """`_EesEpisode.step` checks this exact name to divert the skill away from the
    normal controller/execution path -- the same contract `ASK_FOR_RESET_TASK_
    INITIAL_NAME` is for the other reset skill."""
    ground = _provider().human_cube_bin_reset_skill()
    assert ground.skill.name == ASK_FOR_RESET_CUBE_BIN_ONLY_NAME


def test_it_is_bound_to_all_four_domain_objects() -> None:
    env = Tossing3DEnvironment()
    ground = Tossing3DSkillProvider(env=env).human_cube_bin_reset_skill()
    assert ground.objects == (env.robot, env.cube, env.bin, env.barrier)


def test_the_ground_precondition_is_now_empty_not_hand_empty() -> None:
    """Used to require HandEmpty(robot), on the reasoning that 'nothing about Holding
    changes' is only true while the gripper is empty. That guarded a real correctness
    gap but made the rescue unreachable from the one state it exists to rescue -- a
    near-miss grasp leaves HandEmpty and Holding both false, and nothing in this
    domain's operator model ever restores HandEmpty on its own. The framework has no
    negation, so the closest expressible precondition to the right one (not Holding)
    is none at all. See Tossing3DSkillProvider.human_cube_bin_reset_skill's own
    docstring for the full reasoning."""
    env = Tossing3DEnvironment()
    ground = Tossing3DSkillProvider(env=env).human_cube_bin_reset_skill()
    assert ground.preconditions == frozenset()


def test_add_effects_place_the_cube_on_ground_and_reachable() -> None:
    """A fresh ground placement (blocks_init_region) is known to leave the cube resting
    on the ground and on the near side of the barrier -- see KinderBackend.reset_cube_
    and_bin's own docstring for why this is upstream's own placement guarantee, not an
    assumption made here."""
    env = Tossing3DEnvironment()
    ground = Tossing3DSkillProvider(env=env).human_cube_bin_reset_skill()
    assert ground.add_effects == frozenset({
        GroundAtom(predicate=ON_GROUND, objects=(env.cube,)),
        GroundAtom(predicate=REACHABLE, objects=(env.cube, env.barrier)),
    })


def test_delete_effects_remove_in_bin_since_the_fresh_position_cannot_score() -> None:
    env = Tossing3DEnvironment()
    ground = Tossing3DSkillProvider(env=env).human_cube_bin_reset_skill()
    assert ground.delete_effects == frozenset({
        GroundAtom(predicate=IN_BIN, objects=(env.cube, env.bin)),
    })


def test_add_and_delete_effects_never_overlap() -> None:
    ground = _provider().human_cube_bin_reset_skill()
    assert ground.add_effects.isdisjoint(ground.delete_effects)


def test_the_operator_never_names_hand_empty_or_holding_as_an_effect() -> None:
    """The load-bearing claim this skill makes: since the robot is never touched, no
    atom about it is added or deleted -- only preconditioned on. Checked by name
    rather than by predicate identity so a future refactor that renames Holding/
    HandEmpty cannot silently reintroduce an effect on them without this failing."""
    ground = _provider().human_cube_bin_reset_skill()
    touched_names = {atom.predicate.name for atom in (ground.add_effects | ground.delete_effects)}
    assert "HandEmpty" not in touched_names
    assert "Holding" not in touched_names


def test_param_dim_is_zero_so_it_has_no_sampler() -> None:
    """This "skill" has no continuous parameters and is intercepted before
    execute_ground_skill would ever try to sample any."""
    assert _provider().human_cube_bin_reset_skill().skill.param_dim == 0


def test_human_reset_cost_is_five_robot_action_equivalents() -> None:
    assert _provider().human_cube_bin_reset_skill().evaluate_practice_cost() == 5.0


def test_non_human_reset_duplicates_reset_mechanics_with_independent_identity() -> None:
    provider = _provider()
    human = provider.human_cube_bin_reset_skill()
    automatic = provider.non_human_cube_bin_reset_skill()

    assert automatic.skill.name == "non_human_reset_cube_bin_only"
    assert automatic.evaluate_practice_cost() == 5.0
    assert automatic.objects == human.objects
    assert automatic.preconditions == human.preconditions
    assert automatic.add_effects == human.add_effects
    assert automatic.delete_effects == human.delete_effects
    assert provider.movables_reset_skills() == (human, automatic)


def test_human_reset_cost_is_provider_configuration() -> None:
    provider = Tossing3DSkillProvider(env=Tossing3DEnvironment(), human_reset_practice_cost=0.125)
    assert provider.human_cube_bin_reset_skill().evaluate_practice_cost() == 0.125


@pytest.mark.parametrize("stranded", [False, True])
@pytest.mark.parametrize("closed", [False, True])
def test_same_side_plans_with_optional_reset(*, stranded: bool, closed: bool) -> None:
    """Offering a reset must preserve ordinary plans and rescue stranded cubes."""
    from hitl_pmp.environments.tossing3d.layout import Tossing3DLayout
    from hitl_pmp.environments.tossing3d.predicates import HAND_EMPTY, HOLDING
    from hitl_pmp.environments.tossing3d.recovery_skills import CLOSED_EMPTY, ON_FLOOR
    from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod

    env = Tossing3DEnvironment(layout=Tossing3DLayout.SAME_SIDE)
    provider = Tossing3DSkillProvider(env=env, human_reset_practice_cost=0.001)
    method = EesMethod(env=env, skill_provider=provider, seed=0)
    atoms = {
        GroundAtom(
            predicate=CLOSED_EMPTY if closed else HAND_EMPTY,
            objects=(env.robot, env.cube) if closed else (env.robot,),
        )
    }
    if not stranded:
        atoms |= {
            GroundAtom(predicate=ON_FLOOR, objects=(env.cube, env.bin)),
            GroundAtom(predicate=REACHABLE, objects=(env.cube, env.barrier)),
        }
    plan = method.plan_to(
        init_atoms=frozenset(atoms),
        goal=frozenset({GroundAtom(predicate=HOLDING, objects=(env.robot, env.cube))}),
        costs={},
        practicing=True,
    )
    names = [step.skill.name for step in plan]
    assert names[-1] == "PickCubeFromFloor"
    assert names.count(ASK_FOR_RESET_CUBE_BIN_ONLY_NAME) == int(stranded)
    assert names.count("OpenGripper") == int(closed)


def test_same_side_reset_places_cube_on_floor_in_the_declared_vocabulary() -> None:
    from hitl_pmp.environments.tossing3d.layout import Tossing3DLayout
    from hitl_pmp.environments.tossing3d.recovery_skills import ON_FLOOR

    env = Tossing3DEnvironment(layout=Tossing3DLayout.SAME_SIDE)
    provider = Tossing3DSkillProvider(env=env)
    reset = provider.human_cube_bin_reset_skill()
    assert GroundAtom(predicate=ON_FLOOR, objects=(env.cube, env.bin)) in reset.add_effects
    assert {atom.predicate for atom in reset.add_effects | reset.delete_effects} <= set(
        provider.predicates()
    )
