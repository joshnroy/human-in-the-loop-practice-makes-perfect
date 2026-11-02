"""Robot-only restoration and safe human tool resets, without physics models.

Review TDD evidence: the initial 51 tests here plus the five held-ball backend
guards failed 56/56 before implementation, then passed 56/56 after implementation.
Command: scripts/with_env.sh pytest -q tests/environments/sweeptable3d/test_review_semantics.py
tests/environments/sweeptable3d/test_backend.py -k 'all_32 or planner_can_restore or
completion_context or defers_activation or holding_any_ball or rejects_held_ball'.
Later cases cover each possible last target and safe reset recovery. No physics runs.
"""

import itertools

import numpy as np
import pytest

from hitl_pmp.environments.sweeptable3d.objects import SweepTable3DObjects as O
from hitl_pmp.environments.sweeptable3d.predicates import (
    BALLS_RESTORED,
    HAND_EMPTY,
    HOLDING_WIPER,
    TASK_ACTIVE,
    SweepTable3DPredicates,
)
from hitl_pmp.environments.sweeptable3d.skill_provider import SweepTable3DActions
from hitl_pmp.environments.sweeptable3d.skill_trace import SkillTraces
from hitl_pmp.environments.sweeptable3d.skills import SKILL_IDS
from hitl_pmp.environments.sweeptable3d.skills import SweepTable3DSkills as S
from hitl_pmp.methods.belief_space.tossing3d_transition_model import apply_success_effects
from hitl_pmp.planning.fast_downward import FastDownwardPlanner

from .test_symbolic import SymbolicScene


@pytest.mark.parametrize("in_tray", itertools.product((False, True), repeat=5))
def test_all_32_robot_return_combinations_restore_the_next_sweep(*, in_tray):
    atoms = SymbolicScene.atoms(
        state=SymbolicScene.state(
            on_table=(),
            in_box=tuple(ball for ball, tray in zip(O.balls, in_tray, strict=True) if tray),
            on_floor=tuple(ball for ball, tray in zip(O.balls, in_tray, strict=True) if not tray),
            active=False,
        )
    )
    for index, ball in enumerate(O.balls):
        skill = (
            S.RETURN_CUBE_TO_TABLE_FROM_BOX if in_tray[index] else S.RETURN_CUBE_TO_TABLE_FROM_FLOOR
        )
        ground = next(
            ground
            for ground in SymbolicScene.groundings(skill=skill, atoms=atoms)
            if ground.objects[1] == ball
        )
        predicted = apply_success_effects(
            true_atoms=atoms,
            ground_skill=ground,
            effects={ground: (ground.add_effects, ground.delete_effects, ground.ignore_effects)},
        )
        assert predicted == SymbolicScene.apply(ground=ground, atoms=atoms)
        atoms = predicted
        for predicate in (BALLS_RESTORED, TASK_ACTIVE):
            assert (SymbolicScene.atom(predicate=predicate) in atoms) is (index == 4)
    pick = SymbolicScene.groundings(skill=S.PICK_WIPER, atoms=atoms)
    assert len(pick) == 1
    atoms = SymbolicScene.apply(ground=pick[0], atoms=atoms)
    sweep = SymbolicScene.groundings(skill=S.SWEEP_TABLE, atoms=atoms)
    assert len(sweep) == 1
    atoms = SymbolicScene.apply(ground=sweep[0], atoms=atoms)
    assert SweepTable3DPredicates.goal_atoms() <= atoms


@pytest.mark.parametrize("tray_count", [0, 2, 5])
def test_planner_can_restore_all_balls_without_human_help(*, tray_count):
    plan = FastDownwardPlanner.plan(
        skills=S.robot_skills(),
        objects=O.all_objects(),
        predicates=SweepTable3DPredicates.all(),
        types=O.all_types(),
        init_atoms=SymbolicScene.atoms(
            state=SymbolicScene.state(
                on_table=(),
                in_box=O.balls[:tray_count],
                on_floor=O.balls[tray_count:],
                active=False,
            )
        ),
        goal=SweepTable3DPredicates.goal_atoms(),
    )
    assert len(plan) == 7
    assert all(ground.skill in S.RETURN_CUBE_SKILLS for ground in plan[:5])
    assert [ground.skill.name for ground in plan[5:]] == ["PickWiper", "SweepTable"]


@pytest.mark.parametrize("skill", S.RETURN_CUBE_SKILLS, ids=lambda skill: skill.name)
@pytest.mark.parametrize("target", range(5))
def test_return_encoding_and_trace_preserve_each_target_and_completion_context(*, skill, target):
    atoms = SymbolicScene.atoms(
        state=SymbolicScene.state(
            on_table=(),
            in_box=O.balls if skill == S.RETURN_CUBE_TO_TABLE_FROM_BOX else (),
            on_floor=O.balls if skill == S.RETURN_CUBE_TO_TABLE_FROM_FLOOR else (),
            active=False,
        )
    )
    grounds = SymbolicScene.groundings(skill=skill, atoms=atoms)
    assert len(grounds) == 5
    ground = next(ground for ground in grounds if ground.objects[1] == O.balls[target])
    assert ground.objects[-5:] == O.balls
    action = SweepTable3DActions.encode(ground_skill=ground, params=np.zeros(0))
    assert action.tolist() == [SKILL_IDS[skill.name], target, 0, 0]
    assert SkillTraces.ground(action=action) == ground


def test_final_robot_return_defers_activation_until_tool_is_returned():
    atoms = SymbolicScene.atoms(
        state=SymbolicScene.state(
            on_table=O.balls[1:],
            on_floor=O.balls[:1],
            wiper_home=False,
            active=False,
        )
    )
    ground = SymbolicScene.groundings(skill=S.RETURN_CUBE_TO_TABLE_FROM_FLOOR, atoms=atoms)[0]
    atoms = SymbolicScene.apply(ground=ground, atoms=atoms)
    assert SymbolicScene.atom(predicate=BALLS_RESTORED) in atoms
    assert SymbolicScene.atom(predicate=TASK_ACTIVE) not in atoms
    reset = SymbolicScene.groundings(skill=S.RESET_WIPER, atoms=atoms)[0]
    atoms = SymbolicScene.apply(ground=reset, atoms=atoms)
    assert SymbolicScene.atom(predicate=TASK_ACTIVE) in atoms


@pytest.mark.parametrize("ball", O.balls, ids=lambda ball: ball.name)
def test_human_wiper_reset_is_inapplicable_while_holding_any_ball(*, ball):
    atoms = SymbolicScene.atoms(
        state=SymbolicScene.state(
            holding=ball,
            wiper_home=False,
            active=False,
        )
    )
    assert SymbolicScene.groundings(skill=S.RESET_WIPER, atoms=atoms) == []


@pytest.mark.parametrize("skill", S.RETURN_CUBE_SKILLS, ids=lambda skill: skill.name)
@pytest.mark.parametrize("target", range(5))
@pytest.mark.parametrize("home", [False, True])
def test_each_final_target_restores_balls_and_requires_tool_home_to_activate(
    *, skill, target, home
):
    atoms = SymbolicScene.atoms(
        state=SymbolicScene.state(
            on_table=tuple(ball for ball in O.balls if ball != O.balls[target]),
            in_box=(O.balls[target],) if skill == S.RETURN_CUBE_TO_TABLE_FROM_BOX else (),
            on_floor=(O.balls[target],) if skill == S.RETURN_CUBE_TO_TABLE_FROM_FLOOR else (),
            wiper_home=home,
            active=False,
        )
    )
    grounds = SymbolicScene.groundings(skill=skill, atoms=atoms)
    assert len(grounds) == 1
    after = SymbolicScene.apply(ground=grounds[0], atoms=atoms)
    assert SymbolicScene.atom(predicate=BALLS_RESTORED) in after
    assert (SymbolicScene.atom(predicate=TASK_ACTIVE) in after) is home
    if not home:
        after = after - {SymbolicScene.atom(predicate=HAND_EMPTY, objects=(O.robot,))}
        after |= {SymbolicScene.atom(predicate=HOLDING_WIPER, objects=(O.robot, O.wiper))}
        returned = SymbolicScene.groundings(skill=S.RETURN_WIPER, atoms=after)[0]
        assert SymbolicScene.atom(predicate=TASK_ACTIVE) in SymbolicScene.apply(
            ground=returned, atoms=after
        )


def test_human_ball_reset_makes_tool_reset_safe_again():
    atoms = SymbolicScene.atoms(
        state=SymbolicScene.state(
            holding=O.balls[0],
            wiper_home=False,
            active=False,
        )
    )
    assert SymbolicScene.groundings(skill=S.RESET_WIPER, atoms=atoms) == []
    reset_balls = SymbolicScene.groundings(skill=S.RESET_CUBES, atoms=atoms)[0]
    after = SymbolicScene.apply(ground=reset_balls, atoms=atoms)
    assert len(SymbolicScene.groundings(skill=S.RESET_WIPER, atoms=after)) == 1
