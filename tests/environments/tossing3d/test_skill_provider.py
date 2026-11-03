"""Offline tests for Tossing3D's sampler features and reset skills.

Only reset execution touches MuJoCo; the feature and symbolic contracts are plain Python.
"""

import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.predicates import (
    BIN_ON_OPPOSITE_SIDE,
    BIN_ON_ROBOT_SIDE,
    CUBE_ON_OPPOSITE_SIDE,
    CUBE_ON_ROBOT_SIDE,
    HOLDING,
    IN_BIN,
    NOT_HOLDING,
    ON_GROUND,
)
from hitl_pmp.environments.tossing3d.skill_provider import (
    RESET_CUBE_BIN_OPPOSITE_SIDE_NAME,
    RESET_CUBE_BIN_SAME_SIDE_NAME,
    Tossing3DSkillProvider,
)
from hitl_pmp.environments.tossing3d.skills import Tossing3DSkills

from .observations import state


def _provider() -> Tossing3DSkillProvider:
    return Tossing3DSkillProvider(env=Tossing3DEnvironment())


def _toss(*, env: Tossing3DEnvironment) -> GroundSkill:
    return GroundSkill(
        skill=Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS,
        objects=(env.robot, env.bin, env.cube, env.barrier),
    )


def test_toss_sampler_input_is_bias_then_the_three_throw_params() -> None:
    """The pre-move robot-frame bin displacement is gone from the row: it describes
    where the robot happened to be before driving, not the throw (see `toss.py`)."""
    env = Tossing3DEnvironment()
    params = np.array([1.3, 125.0, 760.0])
    scene = state(env=env, base_x=0.2, base_y=-0.4, base_rot=np.pi / 2, bin_x=1.7)
    assert Tossing3DSkillProvider(env=env).hand_selected_feature_transform(
        ground_skill=_toss(env=env), state=scene, params=params
    ) == pytest.approx([1.0, 1.3, 125.0, 760.0])


def test_toss_sampler_input_does_not_depend_on_the_pre_move_pose() -> None:
    env = Tossing3DEnvironment()
    provider = Tossing3DSkillProvider(env=env)
    params = np.array([1.35, 140.0, 792.0])
    original = state(env=env, base_x=0.15, base_rot=0.2, bin_x=2.0)
    elsewhere = state(env=env, base_x=-1.4, base_y=0.9, base_rot=2.9, bin_x=3.1)
    assert provider.hand_selected_feature_transform(
        ground_skill=_toss(env=env), state=elsewhere, params=params
    ) == provider.hand_selected_feature_transform(
        ground_skill=_toss(env=env), state=original, params=params
    )


def test_same_side_toss_uses_the_same_feature_layout() -> None:
    from hitl_pmp.environments.tossing3d.layout import Tossing3DLayout

    env = Tossing3DEnvironment(layout=Tossing3DLayout.SAME_SIDE)
    toss = GroundSkill(
        skill=Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS,
        objects=(env.robot, env.bin, env.cube, env.barrier),
    )
    row = Tossing3DSkillProvider(env=env).hand_selected_feature_transform(
        ground_skill=toss,
        state=state(env=env, base_x=0.0, base_y=0.0, base_rot=np.pi, bin_x=-2.0),
        params=np.array([1.35, 140.0, 792.0]),
    )
    assert row == pytest.approx([1.0, 1.35, 140.0, 792.0])


def test_non_toss_skills_keep_the_generic_sampler_input_fallback() -> None:
    env = Tossing3DEnvironment()
    pick = GroundSkill(
        skill=Tossing3DSkills.PICK_CUBE,
        objects=(env.robot, env.cube, env.barrier, env.bin),
    )
    assert (
        _provider().hand_selected_feature_transform(
            ground_skill=pick, state=state(), params=np.zeros(0)
        )
        is None
    )


def test_the_singular_reset_is_the_same_side_skill() -> None:
    ground = _provider().human_cube_bin_reset_skill()
    assert ground.skill.name == RESET_CUBE_BIN_SAME_SIDE_NAME


def test_reset_skills_bind_no_side_object() -> None:
    env = Tossing3DEnvironment()
    resets = Tossing3DSkillProvider(env=env).human_cube_bin_reset_skills()
    assert [reset.skill.name for reset in resets] == [
        RESET_CUBE_BIN_SAME_SIDE_NAME,
        RESET_CUBE_BIN_OPPOSITE_SIDE_NAME,
    ]
    assert all(reset.objects == (env.robot, env.cube, env.bin, env.barrier) for reset in resets)


def test_side_predicates_are_robot_relative_not_world_signs() -> None:
    env = Tossing3DEnvironment()
    scene = state(env=env, base_x=0.0, bin_x=-2.0)
    scene.set(obj=env.barrier, feature_name="x", feature_val=1.3)
    scene.set(obj=env.cube, feature_name="x", feature_val=0.5)
    assert BIN_ON_ROBOT_SIDE.holds(scene, (env.bin, env.barrier))
    assert CUBE_ON_ROBOT_SIDE.holds(scene, (env.cube, env.barrier))
    assert not BIN_ON_OPPOSITE_SIDE.holds(scene, (env.bin, env.barrier))


def test_cube_on_or_straddling_barrier_is_on_neither_side() -> None:
    env = Tossing3DEnvironment()
    scene = state(env=env, base_x=0.0)
    scene.set(obj=env.barrier, feature_name="x", feature_val=1.3)
    # Barrier half-width 0.03 + cube half-width 0.025 gives a 0.055 m contact edge.
    for cube_x in (1.3, 1.3 - 0.054, 1.3 + 0.054, 1.3 - 0.055, 1.3 + 0.055):
        scene.set(obj=env.cube, feature_name="x", feature_val=cube_x)
        assert not CUBE_ON_ROBOT_SIDE.holds(scene, (env.cube, env.barrier))
        assert not CUBE_ON_OPPOSITE_SIDE.holds(scene, (env.cube, env.barrier))

    # Just past contact either side holds: footprint contact is the whole test, and
    # near the barrier `PickPlannable` decides the pick (see test_cube_grasp_range.py).
    for cube_x in (1.3 - 0.056, 1.3 + 0.056):
        scene.set(obj=env.cube, feature_name="x", feature_val=cube_x)
        assert any(
            side.holds(scene, (env.cube, env.barrier))
            for side in (CUBE_ON_ROBOT_SIDE, CUBE_ON_OPPOSITE_SIDE)
        )


def test_reset_destination_is_decoded_from_the_reset_skill() -> None:
    provider = _provider()
    assert tuple(
        provider.movables_reset_destination(ground_skill=reset)
        for reset in provider.human_cube_bin_reset_skills()
    ) == ("robot_side", "opposite_side")


@pytest.mark.parametrize("layout", ["barrier", "same-side"])
def test_practice_resets_offer_both_bin_destinations_to_the_planner(*, layout: str) -> None:
    """Which side a practice reset sends the bin to is the planner's choice: the
    human and the automatic reset are each grounded for both destinations, each
    grounding priced at that mechanism's own cost, so neither side is favoured."""
    from hitl_pmp.environments.tossing3d.layout import Tossing3DLayout

    provider = Tossing3DSkillProvider(
        env=Tossing3DEnvironment(layout=Tossing3DLayout(layout)),
        human_reset_practice_cost=3.0,
        non_human_reset_practice_cost=7.0,
        offer_non_human_reset=True,
    )
    for group, cost in (
        (provider.human_cube_bin_reset_skills(), 3.0),
        (provider.non_human_cube_bin_reset_skills(), 7.0),
    ):
        assert [provider.movables_reset_destination(ground_skill=g) for g in group] == [
            "robot_side",
            "opposite_side",
        ]
        assert {g.evaluate_practice_cost() for g in group} == {cost}
    assert {
        provider.movables_reset_destination(ground_skill=g)
        for g in provider.movables_reset_skills()
    } == {"robot_side", "opposite_side"}


def test_the_resets_have_no_preconditions() -> None:
    """The robot is on its own side by definition, so nothing restricts a reset."""
    for ground in _provider().human_cube_bin_reset_skills():
        assert ground.preconditions == frozenset()


def test_add_effects_place_the_cube_and_bin_on_selected_sides() -> None:
    """A fresh ground placement (blocks_init_region) is known to leave the cube resting
    on the ground and on the near side of the barrier -- see KinderBackend.reset_cube_
    and_bin's own docstring for why this is upstream's own placement guarantee, not an
    assumption made here."""
    env = Tossing3DEnvironment()
    ground = Tossing3DSkillProvider(env=env).human_cube_bin_reset_skill()
    assert {
        GroundAtom(predicate=ON_GROUND, objects=(env.cube,)),
        GroundAtom(predicate=CUBE_ON_ROBOT_SIDE, objects=(env.cube, env.barrier)),
        GroundAtom(predicate=BIN_ON_ROBOT_SIDE, objects=(env.bin, env.barrier)),
    } <= ground.add_effects
    opposite = Tossing3DSkillProvider(env=env).human_cube_bin_reset_skills()[1]
    assert {
        GroundAtom(predicate=CUBE_ON_ROBOT_SIDE, objects=(env.cube, env.barrier)),
        GroundAtom(predicate=BIN_ON_OPPOSITE_SIDE, objects=(env.bin, env.barrier)),
    } <= opposite.add_effects


def test_delete_effects_remove_in_bin_and_holding_since_the_cube_is_relocated() -> None:
    env = Tossing3DEnvironment()
    ground = Tossing3DSkillProvider(env=env).human_cube_bin_reset_skill()
    assert {
        GroundAtom(predicate=IN_BIN, objects=(env.cube, env.bin)),
    } <= ground.delete_effects


def test_reset_replaces_existing_side_facts_with_quantified_ignore_effects() -> None:
    for ground in _provider().human_cube_bin_reset_skills():
        assert ground.ignore_effects == frozenset({
            CUBE_ON_ROBOT_SIDE,
            CUBE_ON_OPPOSITE_SIDE,
            BIN_ON_ROBOT_SIDE,
            BIN_ON_OPPOSITE_SIDE,
        })


def test_reset_preserves_the_gripper_command() -> None:
    ground = _provider().human_cube_bin_reset_skill()
    touched_names = {atom.predicate.name for atom in (ground.add_effects | ground.delete_effects)}
    assert "HandEmpty" not in touched_names
    assert "Holding" in touched_names


def test_param_dim_is_zero_so_it_has_no_sampler() -> None:
    """This "skill" has no continuous parameters and is intercepted before
    execute_ground_skill would ever try to sample any."""
    assert _provider().human_cube_bin_reset_skill().skill.param_dim == 0


def test_human_reset_cost_is_five_robot_action_equivalents() -> None:
    assert _provider().human_cube_bin_reset_skill().evaluate_practice_cost() == 5.0


def test_practice_offers_no_non_human_reset_by_default() -> None:
    """The automatic reset is a relabelled human reset -- same exception, same oracle,
    same intervention count -- so offering both makes the choice between them noise."""
    provider = _provider()
    assert provider.movables_reset_skills() == provider.human_cube_bin_reset_skills()
    assert not any(
        "non_human_reset" in reset.skill.name for reset in provider.movables_reset_skills()
    )


def test_the_non_human_reset_is_offered_only_when_asked_for() -> None:
    provider = Tossing3DSkillProvider(env=Tossing3DEnvironment(), offer_non_human_reset=True)
    assert provider.movables_reset_skills() == (
        *provider.human_cube_bin_reset_skills(),
        *provider.non_human_cube_bin_reset_skills(),
    )


def test_human_reset_is_two_skills_each_costing_five() -> None:
    provider = _provider()
    human_destinations = provider.human_cube_bin_reset_skills()
    assert {reset.skill.name for reset in human_destinations} == {
        RESET_CUBE_BIN_SAME_SIDE_NAME,
        RESET_CUBE_BIN_OPPOSITE_SIDE_NAME,
    }
    assert {reset.evaluate_practice_cost() for reset in human_destinations} == {5.0}


def test_human_reset_cost_is_provider_configuration() -> None:
    provider = Tossing3DSkillProvider(env=Tossing3DEnvironment(), human_reset_practice_cost=0.125)
    assert provider.human_cube_bin_reset_skill().evaluate_practice_cost() == 0.125


@pytest.mark.parametrize("stranded", [False, True])
@pytest.mark.parametrize("closed", [False, True])
def test_same_side_plans_with_optional_reset(*, stranded: bool, closed: bool) -> None:
    """Offering a reset must preserve ordinary plans and rescue stranded cubes."""
    from hitl_pmp.environments.tossing3d.layout import Tossing3DLayout
    from hitl_pmp.environments.tossing3d.predicates import (
        BIN_ON_GROUND,
        GRIPPER_CLOSED,
        HAND_EMPTY,
        PICK_PLANNABLE,
    )
    from hitl_pmp.environments.tossing3d.recovery_skills import CLOSED_EMPTY, ON_FLOOR
    from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod

    env = Tossing3DEnvironment(layout=Tossing3DLayout.SAME_SIDE)
    provider = Tossing3DSkillProvider(env=env, human_reset_practice_cost=0.001)
    # Bare-price reset: scored, it also pays -log(competence), and the gate declines it.
    method = EesMethod(env=env, skill_provider=provider, seed=0, score_human_skills=False)
    atoms = {
        GroundAtom(
            predicate=CLOSED_EMPTY if closed else HAND_EMPTY,
            objects=(env.robot, env.cube) if closed else (env.robot,),
        ),
        GroundAtom(predicate=NOT_HOLDING, objects=(env.robot, env.cube)),
        GroundAtom(predicate=BIN_ON_GROUND, objects=(env.bin,)),
    }
    if closed:
        atoms.add(GroundAtom(predicate=GRIPPER_CLOSED, objects=(env.robot,)))
    if not stranded:
        atoms |= {
            GroundAtom(predicate=ON_FLOOR, objects=(env.cube, env.bin)),
            GroundAtom(predicate=CUBE_ON_ROBOT_SIDE, objects=(env.cube, env.barrier)),
            GroundAtom(predicate=PICK_PLANNABLE, objects=(env.robot, env.cube)),
        }
    plan = method.plan_to(
        init_atoms=frozenset(atoms),
        goal=frozenset({GroundAtom(predicate=HOLDING, objects=(env.robot, env.cube))}),
        costs={},
        practicing=True,
    )
    names = [step.skill.name for step in plan]
    assert names[-1] == "PickCubeFromFloor"
    assert names.count(RESET_CUBE_BIN_SAME_SIDE_NAME) + names.count(
        RESET_CUBE_BIN_OPPOSITE_SIDE_NAME
    ) == int(stranded)
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


def test_the_lifted_resets_ground_to_either_destination_in_a_live_abstract_state() -> None:
    """EES hands the planner the LIFTED reset skills, which it grounds itself. Both
    destinations must be applicable from an abstract state, and both must be among the
    groundings the provider prices, so neither falls back to a default cost."""
    from hitl_pmp.planning.grounding import SkillGrounder

    env = Tossing3DEnvironment()
    provider = Tossing3DSkillProvider(env=env)
    resets = provider.human_cube_bin_reset_skills()
    applicable = SkillGrounder.applicable_ground_skills(
        skills=tuple(reset.skill for reset in resets),
        objects=provider.objects(),
        true_atoms=frozenset(),
    )
    assert {provider.movables_reset_destination(ground_skill=g) for g in applicable} == {
        "robot_side",
        "opposite_side",
    }
    assert set(applicable) == set(resets)


@pytest.mark.parametrize("destination", ["robot_side", "opposite_side"])
def test_ees_plans_a_reset_to_either_side_only_while_practicing(*, destination: str) -> None:
    """A practicing EES plan can reach a toss at either bin side through the reset
    grounding for that side; an evaluation plan is never offered a reset at all."""
    from hitl_pmp.environments.tossing3d.predicates import HAND_EMPTY
    from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod
    from hitl_pmp.planning.fast_downward import PlanningFailure

    env = Tossing3DEnvironment()
    provider = Tossing3DSkillProvider(env=env, human_reset_practice_cost=0.001)
    # Bare-price reset: scored, it also pays -log(competence), and the gate declines it.
    method = EesMethod(env=env, skill_provider=provider, seed=0, score_human_skills=False)
    side, other = (
        (BIN_ON_ROBOT_SIDE, BIN_ON_OPPOSITE_SIDE)
        if destination == "robot_side"
        else (BIN_ON_OPPOSITE_SIDE, BIN_ON_ROBOT_SIDE)
    )
    init_atoms = frozenset({
        GroundAtom(predicate=HAND_EMPTY, objects=(env.robot,)),
        GroundAtom(predicate=NOT_HOLDING, objects=(env.robot, env.cube)),
        GroundAtom(predicate=other, objects=(env.bin, env.barrier)),
    })
    goal = frozenset({
        GroundAtom(predicate=HOLDING, objects=(env.robot, env.cube)),
        GroundAtom(predicate=side, objects=(env.bin, env.barrier)),
    })
    plan = method.plan_to(init_atoms=init_atoms, goal=goal, costs={}, practicing=True)
    names = {RESET_CUBE_BIN_SAME_SIDE_NAME, RESET_CUBE_BIN_OPPOSITE_SIDE_NAME}
    resets = [step for step in plan if step.skill.name in names]
    assert len(resets) == 1
    assert resets[0] in provider.human_cube_bin_reset_skills()
    assert provider.movables_reset_destination(ground_skill=resets[0]) == destination
    with pytest.raises(PlanningFailure):
        method.plan_to(init_atoms=init_atoms, goal=goal, costs={}, practicing=False)


def test_practice_offers_the_human_reset_by_default() -> None:
    provider = _provider()
    assert provider.offer_human_reset is True
    assert {reset.skill.name for reset in provider.movables_reset_skills()} == {
        RESET_CUBE_BIN_SAME_SIDE_NAME,
        RESET_CUBE_BIN_OPPOSITE_SIDE_NAME,
    }


def test_a_provider_without_the_human_reset_offers_no_reset_at_all() -> None:
    """The "is the human reset needed?" baseline: the skill is absent, not merely
    expensive, so there is no cost at which a planner could still choose it."""
    provider = Tossing3DSkillProvider(env=Tossing3DEnvironment(), offer_human_reset=False)
    assert provider.human_cube_bin_reset_skill() is None
    assert provider.human_cube_bin_reset_skills() == ()
    assert provider.movables_reset_skills() == ()


def test_each_reset_destination_names_its_human_skill() -> None:
    provider = _provider()
    assert tuple(
        provider.human_skill_name(ground_skill=reset)
        for reset in provider.human_cube_bin_reset_skills()
    ) == ("reset_cube_and_bin_near", "reset_cube_far")
