import subprocess
import sys
import tempfile
from pathlib import Path

from hitl_pmp.core.method.types import LiftedAtom, Skill, Variable
from hitl_pmp.core.problem.environment.types import Object, Type
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
from hitl_pmp.planning.fast_downward import FastDownwardPlanner
from hitl_pmp.planning.pddl import PddlWriter

_ROBOT_TYPE = Type(name="robot", feature_names=())
_CELL_TYPE = Type(name="cell", feature_names=())
_ROBOT = Object(name="robby", type=_ROBOT_TYPE)
_CELL0 = Object(name="cell0", type=_CELL_TYPE)
_CELL1 = Object(name="cell1", type=_CELL_TYPE)

_IN_CELL = Predicate(
    name="InCell", types=(_ROBOT_TYPE, _CELL_TYPE), holds=lambda state, objects: True
)
_ADJACENT = Predicate(
    name="Adjacent", types=(_CELL_TYPE, _CELL_TYPE), holds=lambda state, objects: True
)
# A 0-arity predicate: no types, so no variables/objects ever appear with it.
_HANDS_FREE = Predicate(name="HandsFree", types=(), holds=lambda state, objects: True)

_ROBOT_VAR = Variable(name="robot", type=_ROBOT_TYPE)
_CURRENT_VAR = Variable(name="current_cell", type=_CELL_TYPE)
_TARGET_VAR = Variable(name="target_cell", type=_CELL_TYPE)

_MOVE = Skill(
    name="Move",
    parameters=(_ROBOT_VAR, _CURRENT_VAR, _TARGET_VAR),
    preconditions=frozenset({
        LiftedAtom(predicate=_IN_CELL, variables=(_ROBOT_VAR, _CURRENT_VAR)),
        LiftedAtom(predicate=_ADJACENT, variables=(_CURRENT_VAR, _TARGET_VAR)),
        LiftedAtom(predicate=_HANDS_FREE, variables=()),
    }),
    add_effects=frozenset({LiftedAtom(predicate=_IN_CELL, variables=(_ROBOT_VAR, _TARGET_VAR))}),
    delete_effects=frozenset({
        LiftedAtom(predicate=_IN_CELL, variables=(_ROBOT_VAR, _CURRENT_VAR))
    }),
    param_dim=0,
)

# Non-monotone: teleporting leaves the robot in exactly one cell, and no `(not
# (InCell ?robot ?c))` over this skill's own parameters can say "and not in any of
# the others" -- that is what ignore_effects is for.
_TELEPORT = Skill(
    name="Teleport",
    parameters=(_ROBOT_VAR, _TARGET_VAR),
    preconditions=frozenset(),
    add_effects=frozenset({LiftedAtom(predicate=_IN_CELL, variables=(_ROBOT_VAR, _TARGET_VAR))}),
    delete_effects=frozenset(),
    ignore_effects=frozenset({_IN_CELL, _HANDS_FREE}),
    param_dim=0,
)

_X0_VAR = Variable(name="x0", type=_ROBOT_TYPE)
_X1_VAR = Variable(name="x1", type=_CELL_TYPE)

# Parameters named exactly what predicators hardcodes for its `forall` quantifiers,
# so the collision-avoidance branch is the one exercised.
_COLLIDE = Skill(
    name="Collide",
    parameters=(_X0_VAR, _X1_VAR),
    preconditions=frozenset(),
    add_effects=frozenset({LiftedAtom(predicate=_IN_CELL, variables=(_X0_VAR, _X1_VAR))}),
    delete_effects=frozenset(),
    ignore_effects=frozenset({_IN_CELL}),
    param_dim=0,
)

_WAIT = Skill(
    name="Wait",
    parameters=(_ROBOT_VAR,),
    preconditions=frozenset(),
    add_effects=frozenset({LiftedAtom(predicate=_HANDS_FREE, variables=())}),
    delete_effects=frozenset(),
    param_dim=0,
)


def _domain() -> str:
    return PddlWriter.domain_str(
        skills=(_MOVE, _WAIT),
        predicates=(_IN_CELL, _ADJACENT, _HANDS_FREE),
        types=(_ROBOT_TYPE, _CELL_TYPE),
    )


def _problem() -> str:
    return PddlWriter.problem_str(
        objects=(_ROBOT, _CELL0, _CELL1),
        init_atoms=frozenset({
            GroundAtom(predicate=_IN_CELL, objects=(_ROBOT, _CELL0)),
            GroundAtom(predicate=_ADJACENT, objects=(_CELL0, _CELL1)),
            GroundAtom(predicate=_HANDS_FREE, objects=()),
        }),
        goal=frozenset({GroundAtom(predicate=_IN_CELL, objects=(_ROBOT, _CELL1))}),
    )


def test_domain_str_emits_question_mark_prefixed_variables_everywhere() -> None:
    """Our Variables are named "robot"/"current_cell" (no "?"), unlike predicators',
    which requires the "?" to already be part of the name. The writer must add it,
    in :parameters and in every atom that references a variable."""
    domain = _domain()
    assert ":parameters (?robot - robot ?current_cell - cell ?target_cell - cell)" in domain
    assert "(InCell ?robot ?current_cell)" in domain
    assert "(Adjacent ?current_cell ?target_cell)" in domain
    assert "(not (InCell ?robot ?current_cell))" in domain
    # No bare (unprefixed) variable reference should survive anywhere.
    assert "(InCell robot current_cell)" not in domain


def test_domain_str_structure_matches_predicators_format() -> None:
    domain = _domain()
    assert domain.startswith("(define (domain hitlpmp)")
    assert "(:requirements :typing)" in domain
    assert "(:types cell robot)" in domain  # sorted by name
    assert "(:predicates" in domain
    assert "(:action Move" in domain
    assert "(:action Wait" in domain
    assert domain.rstrip().endswith(")")


def test_domain_str_predicate_declarations_use_indexed_typed_slots() -> None:
    domain = _domain()
    assert "(InCell ?x0 - robot ?x1 - cell)" in domain
    assert "(Adjacent ?x0 - cell ?x1 - cell)" in domain
    # A 0-arity predicate declares no slots at all.
    assert "(HandsFree)" in domain


def test_domain_str_emits_delete_effects_as_negations_after_add_effects() -> None:
    domain = _domain()
    effect = domain.split(":effect (and ")[1]
    assert "(InCell ?robot ?target_cell)" in effect
    assert effect.index("(InCell ?robot ?target_cell)") < effect.index(
        "(not (InCell ?robot ?current_cell))"
    )


def test_domain_str_omits_negation_block_when_a_skill_has_no_delete_effects() -> None:
    domain = _domain()
    wait_action = domain.split("(:action Wait")[1]
    assert "(not " not in wait_action


def test_domain_str_emits_ignore_effects_as_universally_quantified_negations() -> None:
    """An `ignore_effects` predicate is "false for *every* object tuple afterwards",
    which only a `forall`-negation can say -- a plain `(not ...)` would have to name
    objects the action's parameters do not mention. Mirrors predicators'
    `STRIPSOperator.pddl_str`."""
    domain = PddlWriter.domain_str(
        skills=(_TELEPORT,), predicates=(_IN_CELL, _HANDS_FREE), types=(_ROBOT_TYPE, _CELL_TYPE)
    )
    effect = domain.split(":effect (and ")[1]
    assert "(forall (?x0 - robot ?x1 - cell) (not (InCell ?x0 ?x1)))" in effect
    # Ignore effects come after the add/delete blocks, as in predicators.
    assert effect.index("(InCell ?robot ?target_cell)") < effect.index("(forall ")


def test_domain_str_writes_a_zero_arity_ignore_effect_as_a_plain_negation() -> None:
    """`(forall () ...)` is not legal PDDL, and a 0-arity predicate has exactly one
    ground atom anyway, so the universal delete degenerates to `(not (P))`."""
    domain = PddlWriter.domain_str(
        skills=(_TELEPORT,), predicates=(_IN_CELL, _HANDS_FREE), types=(_ROBOT_TYPE, _CELL_TYPE)
    )
    assert "(not (HandsFree))" in domain
    assert "(forall ()" not in domain


def test_domain_str_ignore_effect_quantifiers_cannot_shadow_a_parameter_name() -> None:
    """predicators hardcodes `?x0`/`?x1` for the quantified variables; an action whose
    own parameter is named `?x0` would have it silently shadowed by the `forall`,
    wiping every atom of that predicate. Here the prefix grows until it is unique --
    by repeating the letter, since a PDDL name must begin with one."""
    domain = PddlWriter.domain_str(
        skills=(_COLLIDE,), predicates=(_IN_CELL,), types=(_ROBOT_TYPE, _CELL_TYPE)
    )
    assert ":parameters (?x0 - robot ?x1 - cell)" in domain
    assert "(forall (?xx0 - robot ?xx1 - cell) (not (InCell ?xx0 ?xx1)))" in domain


def test_domain_str_omits_the_forall_block_when_a_skill_has_no_ignore_effects() -> None:
    assert "forall" not in _domain()


def test_domain_str_is_deterministic_regardless_of_input_order() -> None:
    shuffled = PddlWriter.domain_str(
        skills=(_WAIT, _MOVE),
        predicates=(_HANDS_FREE, _ADJACENT, _IN_CELL),
        types=(_CELL_TYPE, _ROBOT_TYPE),
    )
    assert shuffled == _domain()


def test_domain_str_honors_a_custom_domain_name() -> None:
    domain = PddlWriter.domain_str(
        skills=(_MOVE,), predicates=(_IN_CELL,), types=(_ROBOT_TYPE,), domain_name="othername"
    )
    assert domain.startswith("(define (domain othername)")


def test_problem_str_structure_matches_predicators_format() -> None:
    problem = _problem()
    assert problem.startswith("(define (problem hitlpmpproblem) (:domain hitlpmp)")
    assert "cell0 - cell" in problem
    assert "cell1 - cell" in problem
    assert "robby - robot" in problem
    assert "(InCell robby cell0)" in problem
    assert "(HandsFree)" in problem
    assert "(:goal (and (InCell robby cell1)))" in problem


def test_problem_str_sorts_objects_and_atoms_by_name_for_determinism() -> None:
    problem = PddlWriter.problem_str(
        objects=(_CELL1, _ROBOT, _CELL0),
        init_atoms=frozenset({
            GroundAtom(predicate=_HANDS_FREE, objects=()),
            GroundAtom(predicate=_ADJACENT, objects=(_CELL0, _CELL1)),
            GroundAtom(predicate=_IN_CELL, objects=(_ROBOT, _CELL0)),
        }),
        goal=frozenset({GroundAtom(predicate=_IN_CELL, objects=(_ROBOT, _CELL1))}),
    )
    assert problem == _problem()
    objects_block = problem.split("(:objects")[1].split(")")[0]
    assert objects_block.index("cell0") < objects_block.index("cell1")
    assert objects_block.index("cell1") < objects_block.index("robby")


def test_problem_str_honors_custom_domain_and_problem_names() -> None:
    problem = PddlWriter.problem_str(
        objects=(_ROBOT,),
        init_atoms=frozenset(),
        goal=frozenset({GroundAtom(predicate=_IN_CELL, objects=(_ROBOT, _CELL0))}),
        domain_name="othername",
        problem_name="otherproblem",
    )
    assert problem.startswith("(define (problem otherproblem) (:domain othername)")


def _lightswitch_pddl(*, grid_size: int) -> tuple[str, str, tuple[Object, ...]]:
    env = LightSwitchEnvironment(grid_size=grid_size)
    state = env.build_initial_state(light_level=0.0, light_target=0.8)
    objects = (env.robot, env.light, *env.get_cells())
    predicates = (ADJACENT, LIGHT_IN_CELL, LIGHT_OFF, LIGHT_ON, ROBOT_IN_CELL)
    from hitl_pmp.planning.grounding import SkillGrounder

    init_atoms = SkillGrounder.abstract_state(state=state, objects=objects, predicates=predicates)
    skills = (
        LightSwitchSkills.MOVE_ROBOT,
        LightSwitchSkills.TURN_ON_LIGHT,
        LightSwitchSkills.TURN_OFF_LIGHT,
        LightSwitchSkills.JUMP_TO_LIGHT,
    )
    types = (env.robot_type, env.light_type, env.cell_type)
    domain = PddlWriter.domain_str(skills=skills, predicates=predicates, types=types)
    problem = PddlWriter.problem_str(
        objects=objects,
        init_atoms=init_atoms,
        goal=frozenset({GroundAtom(predicate=LIGHT_ON, objects=(env.light,))}),
    )
    return domain, problem, objects


def test_lightswitch_declares_no_ignore_effects_so_its_pddl_is_unchanged() -> None:
    """All four `grid_row` NSRTs pass an empty `ignore_effects` in predicators, so
    adding the field must leave Light Switch's emitted domain byte-identical -- no
    `forall` anywhere."""
    assert all(
        skill.ignore_effects == frozenset()
        for skill in (
            LightSwitchSkills.MOVE_ROBOT,
            LightSwitchSkills.TURN_ON_LIGHT,
            LightSwitchSkills.TURN_OFF_LIGHT,
            LightSwitchSkills.JUMP_TO_LIGHT,
        )
    )
    domain, _, _ = _lightswitch_pddl(grid_size=4)
    assert "forall" not in domain


def _translate(*, domain: str, problem: str) -> str:
    """Run a real Fast Downward translator over the emitted PDDL and return the SAS
    text -- proof that it actually parses, not merely that it looks right."""
    with tempfile.TemporaryDirectory() as tmp:
        dom_file = Path(tmp) / "domain.pddl"
        prob_file = Path(tmp) / "problem.pddl"
        sas_file = Path(tmp) / "output.sas"
        dom_file.write_text(domain, encoding="utf-8")
        prob_file.write_text(problem, encoding="utf-8")
        result = subprocess.run(
            [
                sys.executable,
                str(Path(FastDownwardPlanner.fd_dir()) / "fast-downward.py"),
                "--alias",
                "seq-opt-lmcut",
                "--sas-file",
                str(sas_file),
                "--translate",
                str(dom_file),
                str(prob_file),
            ],
            capture_output=True,
            text=True,
            cwd=tmp,
            check=False,
        )
        output = result.stdout + result.stderr
        assert "Driver aborting" not in output, output
        assert sas_file.exists(), output
        return sas_file.read_text(encoding="utf-8")


def test_integration_fd_translator_accepts_the_emitted_lightswitch_pddl() -> None:
    """INTEGRATION (shells out to a real Fast Downward)."""
    domain, problem, _ = _lightswitch_pddl(grid_size=4)
    assert "begin_operator" in _translate(domain=domain, problem=problem)


def test_integration_fd_translator_accepts_ignore_effect_forall_syntax() -> None:
    """INTEGRATION (shells out to a real Fast Downward). Both quantifier spellings
    must parse: the plain `?x0` predicators emits, and the escalated `?xx0` this
    writer falls back to when the action's own parameters would shadow it (a PDDL
    name has to start with a letter, so the fallback repeats one rather than
    prefixing punctuation). The escalated form is otherwise unreachable in this repo,
    which is exactly why no parser would ever see it without this test."""
    for skill in (_TELEPORT, _COLLIDE):
        domain = PddlWriter.domain_str(
            skills=(skill,),
            predicates=(_IN_CELL, _HANDS_FREE),
            types=(_ROBOT_TYPE, _CELL_TYPE),
        )
        problem = PddlWriter.problem_str(
            objects=(_ROBOT, _CELL0, _CELL1),
            init_atoms=frozenset({GroundAtom(predicate=_IN_CELL, objects=(_ROBOT, _CELL0))}),
            goal=frozenset({GroundAtom(predicate=_IN_CELL, objects=(_ROBOT, _CELL1))}),
        )
        assert "forall" in domain
        assert "begin_operator" in _translate(domain=domain, problem=problem)
