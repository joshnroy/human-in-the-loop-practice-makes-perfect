"""Every test here except the executable-lookup ones is an INTEGRATION test: it
shells out to a real, locally-installed Fast Downward (see planning/README.md for
the install steps). They are deliberately not skipped -- the cost-patching protocol
is the load-bearing EES mechanism and a mock would not exercise it at all."""

from pathlib import Path

import pytest

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.problem.environment.types import Object
from hitl_pmp.core.problem.tasks.types import GroundAtom, Predicate
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.predicates import (
    ADJACENT,
    LIGHT_IN_CELL,
    LIGHT_OFF,
    LIGHT_ON,
    ROBOT_IN_CELL,
)
from hitl_pmp.environments.lightswitch.skills import LightSwitchSkills
from hitl_pmp.planning.fast_downward import FastDownwardPlanner, PlanningFailure
from hitl_pmp.planning.grounding import SkillGrounder

_PREDICATES: tuple[Predicate, ...] = (ADJACENT, LIGHT_IN_CELL, LIGHT_OFF, LIGHT_ON, ROBOT_IN_CELL)
_SKILLS = (
    LightSwitchSkills.MOVE_ROBOT,
    LightSwitchSkills.TURN_ON_LIGHT,
    LightSwitchSkills.TURN_OFF_LIGHT,
    LightSwitchSkills.JUMP_TO_LIGHT,
)


def _setup(*, grid_size: int = 5, light_level: float = 0.0, light_target: float = 0.8) -> dict:
    env = LightSwitchEnvironment(grid_size=grid_size)
    state = env.build_initial_state(light_level=light_level, light_target=light_target)
    objects: tuple[Object, ...] = (env.robot, env.light, *env.get_cells())
    init_atoms = SkillGrounder.abstract_state(state=state, objects=objects, predicates=_PREDICATES)
    return {
        "skills": _SKILLS,
        "predicates": _PREDICATES,
        "types": (env.robot_type, env.light_type, env.cell_type),
        "objects": objects,
        "init_atoms": init_atoms,
        "goal": frozenset({GroundAtom(predicate=LIGHT_ON, objects=(env.light,))}),
    }


def _cell(*, objects: tuple[Object, ...], index: int) -> Object:
    return next(o for o in objects if o.name == f"cell{index}")


def test_integration_plan_solves_a_reachable_lightswitch_goal() -> None:
    setup = _setup()
    plan = FastDownwardPlanner.plan(**setup)
    assert isinstance(plan, list)
    assert plan, "expected a non-empty plan for a reachable goal"
    assert all(isinstance(step, GroundSkill) for step in plan)
    assert all(step.skill in _SKILLS for step in plan)
    assert all(all(obj in setup["objects"] for obj in step.objects) for step in plan)
    assert plan[-1].skill is LightSwitchSkills.TURN_ON_LIGHT
    # Unit costs: the 2-hop JumpToLight beats an all-MoveRobot route across 5 cells.
    assert len(plan) == 4
    assert any(step.skill is LightSwitchSkills.JUMP_TO_LIGHT for step in plan)


def test_integration_ground_skill_costs_change_the_chosen_plan() -> None:
    """The load-bearing EES mechanism: per-ground-skill costs, patched into the SAS
    file, must actually steer optimal search onto a different skeleton."""
    setup = _setup()
    objects = setup["objects"]
    expensive_jump = GroundSkill(
        skill=LightSwitchSkills.JUMP_TO_LIGHT,
        objects=(
            LightSwitchEnvironment.robot,
            _cell(objects=objects, index=2),
            _cell(objects=objects, index=3),
            _cell(objects=objects, index=4),
            LightSwitchEnvironment.light,
        ),
    )
    cheap_plan = FastDownwardPlanner.plan(**setup)
    costly_plan = FastDownwardPlanner.plan(**setup, ground_skill_costs={expensive_jump: 100.0})

    assert any(step.skill is LightSwitchSkills.JUMP_TO_LIGHT for step in cheap_plan)
    assert not any(step.skill is LightSwitchSkills.JUMP_TO_LIGHT for step in costly_plan)
    assert costly_plan != cheap_plan
    # The all-MoveRobot route: 4 moves (cell0 -> cell4) plus TurnOnLight.
    assert len(costly_plan) == 5
    assert sum(step.skill is LightSwitchSkills.MOVE_ROBOT for step in costly_plan) == 4


def test_integration_plan_is_empty_when_the_goal_already_holds() -> None:
    setup = _setup(light_level=0.8, light_target=0.8)
    assert FastDownwardPlanner.plan(**setup) == []


def test_integration_plan_raises_planning_failure_for_an_unreachable_goal() -> None:
    setup = _setup()
    light = LightSwitchEnvironment.light
    setup["goal"] = frozenset({
        GroundAtom(predicate=LIGHT_ON, objects=(light,)),
        GroundAtom(predicate=LIGHT_OFF, objects=(light,)),
    })
    with pytest.raises(PlanningFailure):
        FastDownwardPlanner.plan(**setup)


def test_fd_dir_prefers_the_fd_exec_path_environment_variable(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FD_EXEC_PATH", "/some/where/downward")
    assert FastDownwardPlanner.fd_dir() == "/some/where/downward"


def test_plan_raises_a_clear_error_when_the_fast_downward_script_is_missing(
    *, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FD_EXEC_PATH", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="fast-downward.py"):
        FastDownwardPlanner.plan(**_setup())
