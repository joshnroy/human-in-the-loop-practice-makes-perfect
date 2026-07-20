import shutil
from pathlib import Path

import pytest

from hitl_pmp.core.method.types import GroundSkill, LiftedAtom, Skill, Variable
from hitl_pmp.core.problem.environment.types import Object, State, Type
from hitl_pmp.core.problem.tasks.types import Goal, GroundAtom, Predicate
from hitl_pmp.planning.fast_downward import FastDownwardPlanner, PlanningFailure

_ROBOT_TYPE = Type(name="robot", feature_names=())
_CELL_TYPE = Type(name="cell", feature_names=())
_ROBOT = Object(name="robot", type=_ROBOT_TYPE)
_CELL0 = Object(name="cell0", type=_CELL_TYPE)
_CELL1 = Object(name="cell1", type=_CELL_TYPE)
_CELL2 = Object(name="cell2", type=_CELL_TYPE)
_CELL3 = Object(name="cell3", type=_CELL_TYPE)
_CELL4 = Object(name="cell4", type=_CELL_TYPE)


def _move_skill() -> Skill:
    robot = Variable(name="robot", type=_ROBOT_TYPE)
    current = Variable(name="current_cell", type=_CELL_TYPE)
    target = Variable(name="target_cell", type=_CELL_TYPE)
    at = Predicate(name="At", types=(_ROBOT_TYPE, _CELL_TYPE), holds=lambda state, objects: True)
    return Skill(
        name="MoveRobot",
        parameters=(robot, current, target),
        preconditions=frozenset({LiftedAtom(predicate=at, variables=(robot, current))}),
        add_effects=frozenset({LiftedAtom(predicate=at, variables=(robot, target))}),
        delete_effects=frozenset({LiftedAtom(predicate=at, variables=(robot, current))}),
        param_dim=0,
    )


def _jump_skill() -> Skill:
    robot = Variable(name="robot", type=_ROBOT_TYPE)
    c1 = Variable(name="cell1", type=_CELL_TYPE)
    c2 = Variable(name="cell2", type=_CELL_TYPE)
    c3 = Variable(name="cell3", type=_CELL_TYPE)
    light = Variable(name="light", type=_ROBOT_TYPE)
    at = Predicate(name="At", types=(_ROBOT_TYPE, _CELL_TYPE), holds=lambda state, objects: True)
    return Skill(
        name="JumpToLight",
        parameters=(robot, c1, c2, c3, light),
        preconditions=frozenset({LiftedAtom(predicate=at, variables=(robot, c1))}),
        add_effects=frozenset({LiftedAtom(predicate=at, variables=(robot, c3))}),
        delete_effects=frozenset({LiftedAtom(predicate=at, variables=(robot, c1))}),
        param_dim=1,
    )


def test_sas_operator_name_lowercases_skill_and_object_names() -> None:
    skill = _move_skill()
    ground_skill = GroundSkill(skill=skill, objects=(_ROBOT, _CELL0, _CELL1))
    assert (
        FastDownwardPlanner._sas_operator_name(ground_skill=ground_skill)
        == "moverobot robot cell0 cell1"
    )


def _sample_sas_text() -> str:
    return "\n".join([
        "begin_metric",
        "0",
        "end_metric",
        "begin_operator",
        "moverobot robot cell0 cell1",
        "0",
        "1",
        "end_operator",
        "begin_operator",
        "jumptolight robot cell2 cell3 cell4 light",
        "0",
        "1",
        "end_operator",
    ])


def test_patch_sas_costs_turns_on_the_metric_flag() -> None:
    patched = FastDownwardPlanner._patch_sas_costs(
        sas_text=_sample_sas_text(), ground_skill_costs={}, default_cost=1.0
    )
    assert "begin_metric\n1\nend_metric" in patched


def test_patch_sas_costs_overwrites_matching_operators_and_defaults_the_rest() -> None:
    move_ground_skill = GroundSkill(skill=_move_skill(), objects=(_ROBOT, _CELL0, _CELL1))
    patched = FastDownwardPlanner._patch_sas_costs(
        sas_text=_sample_sas_text(),
        ground_skill_costs={move_ground_skill: 2.5},
        default_cost=1.0,
        cost_precision=3,
    )
    lines = patched.split("\n")
    assert lines[6] == "2500"  # 2.5 * 10**3, the moverobot operator's cost line
    assert lines[11] == "1000"  # default_cost * 10**3, jumptolight wasn't costed


def test_parse_plan_reconstructs_ground_skills_in_order() -> None:
    output = "\n".join([
        "[t=0.004s] Solution found!",
        "moverobot robot cell0 cell1 (1)",
        "jumptolight robot cell2 cell3 cell4 light (1)",
        "[t=0.004s] Plan length: 2 step(s).",
    ])
    move_skill = _move_skill()
    jump_skill = _jump_skill()
    skill_by_name = {"moverobot": move_skill, "jumptolight": jump_skill}
    object_by_name = {obj.name: obj for obj in (_ROBOT, _CELL0, _CELL1, _CELL2, _CELL3, _CELL4)} | {
        "light": Object(name="light", type=_ROBOT_TYPE)
    }

    plan = FastDownwardPlanner._parse_plan(
        output=output, skill_by_name=skill_by_name, object_by_name=object_by_name
    )

    assert plan == [
        GroundSkill(skill=move_skill, objects=(_ROBOT, _CELL0, _CELL1)),
        GroundSkill(
            skill=jump_skill,
            objects=(_ROBOT, _CELL2, _CELL3, _CELL4, object_by_name["light"]),
        ),
    ]


def test_parse_plan_returns_empty_list_for_a_trivially_solved_goal() -> None:
    output = "Solution found!\nPlan length: 0 step(s)."
    plan = FastDownwardPlanner._parse_plan(output=output, skill_by_name={}, object_by_name={})
    assert plan == []


def test_parse_plan_raises_when_fast_downward_reports_no_solution() -> None:
    with pytest.raises(PlanningFailure, match="did not find a plan"):
        FastDownwardPlanner._parse_plan(
            output="Search stopped without finding a solution.",
            skill_by_name={},
            object_by_name={},
        )


def test_parse_plan_raises_on_malformed_success_output_with_no_plan_lines() -> None:
    """Defensive check against Fast Downward's own output format changing or
    being truncated -- "Solution found!" present but no line matches the
    "<action> (<cost>)" pattern the rest of parsing depends on."""
    with pytest.raises(PlanningFailure, match="no plan lines"):
        FastDownwardPlanner._parse_plan(
            output="Solution found!\nsomething unexpected with no cost suffix",
            skill_by_name={},
            object_by_name={},
        )


def test_plan_wires_pddl_generation_cost_injection_and_parsing_together(
    *, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Exercises the full plan() orchestration -- PDDL writing, temp files,
    cost injection, output parsing -- without needing a real Fast Downward
    install: _translate/_search (the only pieces that actually shell out) are
    monkeypatched to a fake translate (writes a minimal SAS file) and a canned
    search output, matching what a real Fast Downward invocation would return
    for this same domain/problem (verified against the real binary separately;
    see test_plan_solves_a_real_light_switch_instance_via_fast_downward below)."""
    move_skill = _move_skill()
    ground_skill = GroundSkill(skill=move_skill, objects=(_ROBOT, _CELL0, _CELL1))

    injected_costs: dict[str, dict] = {}

    def _fake_translate(*, fd_exec_path, domain_file, problem_file, sas_file, timeout) -> None:
        del fd_exec_path, domain_file, problem_file, timeout
        sas_file.write_text(_sample_sas_text())

    def _fake_search(*, fd_exec_path, sas_file, timeout) -> str:
        del fd_exec_path, timeout
        injected_costs["sas_text"] = sas_file.read_text()
        return "Solution found!\nmoverobot robot cell0 cell1 (1)\nPlan length: 1 step(s)."

    monkeypatch.setattr(FastDownwardPlanner, "_translate", staticmethod(_fake_translate))
    monkeypatch.setattr(FastDownwardPlanner, "_search", staticmethod(_fake_search))

    at = list(move_skill.preconditions)[0].predicate
    goal = Goal(atoms=frozenset({GroundAtom(predicate=at, objects=(_ROBOT, _CELL1))}))
    state = State(data={})

    plan = FastDownwardPlanner.plan(
        state=state,
        goal=goal,
        objects=(_ROBOT, _CELL0, _CELL1),
        types=(_ROBOT_TYPE, _CELL_TYPE),
        predicates=(at,),
        skills=(move_skill,),
        fd_exec_path=tmp_path,
        ground_skill_costs={ground_skill: 2.5},
    )

    assert plan == [ground_skill]
    # Confirms cost injection actually ran on the sas file _search saw.
    assert "2500" in injected_costs["sas_text"]


_FD_EXEC_PATH = Path.home() / "fast-downward"


@pytest.mark.skipif(
    not (_FD_EXEC_PATH / "fast-downward.py").exists() or shutil.which("gtimeout") is None,
    reason="requires a real Fast Downward install (see planning/README.md) and gtimeout",
)
def test_plan_solves_a_real_light_switch_instance_via_fast_downward() -> None:
    """A genuine, non-mocked integration test against the actual Fast Downward
    binary -- skipped (not failed) wherever FD isn't installed, since it's an
    external dependency this repo doesn't vendor. Mirrors the manual smoke test
    used to validate the whole planning/ package during development."""
    from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
    from hitl_pmp.environments.lightswitch.predicates import (
        ADJACENT,
        LIGHT_IN_CELL,
        LIGHT_OFF,
        LIGHT_ON,
        ROBOT_IN_CELL,
    )
    from hitl_pmp.environments.lightswitch.skills import LightSwitchSkills

    env = LightSwitchEnvironment
    original_grid_size = env.grid_size
    try:
        env.grid_size = 5
        state = env.build_initial_state(light_level=0.0, light_target=0.7)
        cells = env.get_cells()
        objects = (env.robot, env.light, *cells)
        types = (env.robot_type, env.light_type, env.cell_type)
        predicates = (ADJACENT, LIGHT_IN_CELL, LIGHT_OFF, LIGHT_ON, ROBOT_IN_CELL)
        skills = (
            LightSwitchSkills.MOVE_ROBOT,
            LightSwitchSkills.TURN_ON_LIGHT,
            LightSwitchSkills.TURN_OFF_LIGHT,
            LightSwitchSkills.JUMP_TO_LIGHT,
        )
        goal = Goal(atoms=frozenset({GroundAtom(predicate=LIGHT_ON, objects=(env.light,))}))

        plan = FastDownwardPlanner.plan(
            state=state,
            goal=goal,
            objects=objects,
            types=types,
            predicates=predicates,
            skills=skills,
            fd_exec_path=_FD_EXEC_PATH,
        )

        assert len(plan) > 0
        assert plan[-1].skill.name == "TurnOnLight"
    finally:
        env.grid_size = original_grid_size
