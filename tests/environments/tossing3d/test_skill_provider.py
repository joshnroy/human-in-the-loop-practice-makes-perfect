"""Offline tests for `Tossing3DSkillProvider.human_cube_bin_reset_skill`: the
`ask_for_reset_cube_bin_only` ground skill Tossing3D offers `EesMethod`'s planner.

No simulator needed -- like `test_skills.py`'s two lifted skills, the operator model is
plain Python objects; only `KinderBackend.reset_cube_and_bin`/`Tossing3DEnvironment.
reset_movables` (the *execution* half) touch MuJoCo, and those are covered separately in
`test_kinder_backend.py`/`test_environment.py`.
"""

from hitl_pmp.core.method.skill_provider import ASK_FOR_RESET_CUBE_BIN_ONLY_NAME
from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.predicates import HAND_EMPTY, IN_BIN, ON_GROUND, REACHABLE
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider


def _provider() -> Tossing3DSkillProvider:
    return Tossing3DSkillProvider(env=Tossing3DEnvironment())


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


def test_the_ground_precondition_is_exactly_hand_empty_of_the_robot() -> None:
    """HandEmpty(robot) is what makes 'nothing about Holding changes' true by
    construction: repositioning the cube out from under a closed gripper is not
    something this operator's effects describe, so it must never be offered while the
    robot could be holding the cube. See Tossing3DSkillProvider.human_cube_bin_reset_
    skill's own docstring."""
    env = Tossing3DEnvironment()
    ground = Tossing3DSkillProvider(env=env).human_cube_bin_reset_skill()
    assert ground.preconditions == frozenset({
        GroundAtom(predicate=HAND_EMPTY, objects=(env.robot,)),
    })


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
    """Like ask_for_reset_task_initial, this "skill" has no continuous parameters and
    is intercepted before execute_ground_skill would ever try to sample any."""
    assert _provider().human_cube_bin_reset_skill().skill.param_dim == 0
