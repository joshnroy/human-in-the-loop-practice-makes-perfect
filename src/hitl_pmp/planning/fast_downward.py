import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import ClassVar

from hitl_pmp.core.method.types import GroundSkill, Skill
from hitl_pmp.core.problem.environment.types import Object, State, Type
from hitl_pmp.core.problem.tasks.types import Goal, Predicate

from .pddl import PddlWriter


class PlanningFailure(Exception):
    """Raised when Fast Downward can't find a plan, or the underlying subprocess
    itself fails (translation error, timeout, missing FD install)."""


class FastDownwardPlanner:
    """Shells out to an external Fast Downward install to find a minimum-cost
    plan over this codebase's Skills -- matching predicators'
    _sesame_plan_with_fast_downward exactly: a two-stage translate-then-search
    protocol (generate_sas_file_for_fd / _update_sas_file_with_costs /
    fd_plan_from_sas_file in hitl-practice/predicators/planning.py), so that
    per-GroundSkill costs (EES's -log(competence)) can be injected into the
    translated SAS file before the actual search runs -- the built-in astar
    planner predicators itself defaults to elsewhere doesn't support this at
    all (grepped: ground_op_costs is only ever consumed by the FD code path).
    A static-method container, never instantiated, same as every other
    business-logic class in this project.

    Setup: build Fast Downward (git clone https://github.com/aibasel/downward.git
    && cd downward && ./build.py) and pass its directory as fd_exec_path. On
    macOS, `brew install coreutils` for gtimeout."""

    alias: ClassVar[str] = "seq-opt-lmcut"  # optimal search, matches the paper's fdopt-costs

    @staticmethod
    def plan(
        *,
        state: State,
        goal: Goal,
        objects: tuple[Object, ...],
        types: tuple[Type, ...],
        predicates: tuple[Predicate, ...],
        skills: tuple[Skill, ...],
        fd_exec_path: Path,
        ground_skill_costs: dict[GroundSkill, float] | None = None,
        default_cost: float = 1.0,
        timeout: float = 10.0,
    ) -> list[GroundSkill]:
        init_atoms = PddlWriter.abstract_state(state=state, objects=objects, predicates=predicates)
        domain_str = PddlWriter.write_domain(
            domain_name="domain", types=types, predicates=predicates, skills=skills
        )
        problem_str = PddlWriter.write_problem(
            problem_name="problem",
            domain_name="domain",
            objects=objects,
            init_atoms=init_atoms,
            goal_atoms=goal.atoms,
        )

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            domain_file = tmp_dir / "domain.pddl"
            problem_file = tmp_dir / "problem.pddl"
            sas_file = tmp_dir / "output.sas"
            domain_file.write_text(domain_str)
            problem_file.write_text(problem_str)

            FastDownwardPlanner._translate(
                fd_exec_path=fd_exec_path,
                domain_file=domain_file,
                problem_file=problem_file,
                sas_file=sas_file,
                timeout=timeout,
            )
            if ground_skill_costs:
                FastDownwardPlanner._inject_costs(
                    sas_file=sas_file,
                    ground_skill_costs=ground_skill_costs,
                    default_cost=default_cost,
                )
            output = FastDownwardPlanner._search(
                fd_exec_path=fd_exec_path, sas_file=sas_file, timeout=timeout
            )

        skill_by_name = {skill.name.lower(): skill for skill in skills}
        object_by_name = {obj.name.lower(): obj for obj in objects}
        return FastDownwardPlanner._parse_plan(
            output=output, skill_by_name=skill_by_name, object_by_name=object_by_name
        )

    @staticmethod
    def _timeout_command() -> str:
        # gtimeout on macOS (coreutils), timeout everywhere else -- matches
        # predicators' own platform check exactly.
        return "gtimeout" if sys.platform == "darwin" else "timeout"

    @staticmethod
    def _translate(
        *, fd_exec_path: Path, domain_file: Path, problem_file: Path, sas_file: Path, timeout: float
    ) -> None:  # pragma: no cover -- requires a real Fast Downward install
        exec_str = str(fd_exec_path / "fast-downward.py")
        cmd = [
            FastDownwardPlanner._timeout_command(),
            str(timeout),
            exec_str,
            "--alias",
            FastDownwardPlanner.alias,
            "--sas-file",
            str(sas_file),
            str(domain_file),
            str(problem_file),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        output = result.stdout + result.stderr
        if "Driver aborting" in output or not sas_file.exists():
            raise PlanningFailure(f"Fast Downward failed to translate PDDL to SAS: {output}")

    @staticmethod
    def _search(
        *, fd_exec_path: Path, sas_file: Path, timeout: float
    ) -> str:  # pragma: no cover -- requires a real Fast Downward install
        exec_str = str(fd_exec_path / "fast-downward.py")
        cmd = [
            FastDownwardPlanner._timeout_command(),
            str(timeout),
            exec_str,
            "--alias",
            FastDownwardPlanner.alias,
            str(sas_file),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        # Fast Downward leaves per-run artifacts (e.g. plan files) in the cwd;
        # matches predicators' own cleanup call after every search.
        subprocess.run([exec_str, "--cleanup"], capture_output=True, check=False)
        return result.stdout + result.stderr

    @staticmethod
    def _sas_operator_name(*, ground_skill: GroundSkill) -> str:
        tokens = [
            ground_skill.skill.name.lower(),
            *(obj.name.lower() for obj in ground_skill.objects),
        ]
        return " ".join(tokens)

    @staticmethod
    def _patch_sas_costs(
        *,
        sas_text: str,
        ground_skill_costs: dict[GroundSkill, float],
        default_cost: float,
        cost_precision: int = 3,
    ) -> str:
        """Pure text transform (no file I/O, no FD process) -- see
        https://www.fast-downward.org/TranslatorOutputFormat for the SAS format.
        Turns on the 'metric' flag (FD ignores unit action costs otherwise) and
        overwrites each begin_operator/end_operator block's cost line, in SAS's
        own integer-cost units (cost * 10**cost_precision, matching predicators'
        _update_sas_file_with_costs exactly)."""
        sas_text = sas_text.replace("begin_metric\n0\nend_metric", "begin_metric\n1\nend_metric")
        remaining = {
            FastDownwardPlanner._sas_operator_name(ground_skill=ground_skill): cost
            for ground_skill, cost in ground_skill_costs.items()
        }
        lines = sas_text.split("\n")
        num_lines = len(lines)
        for idx in range(num_lines):
            if lines[idx] != "begin_operator":
                continue
            name_idx = idx + 1
            end_idx = next(i for i in range(idx + 1, num_lines) if lines[i] == "end_operator")
            cost_idx = end_idx - 1
            op_name = lines[name_idx]
            cost = remaining.pop(op_name, default_cost)
            lines[cost_idx] = str(int(cost * (10**cost_precision)))
        return "\n".join(lines)

    @staticmethod
    def _inject_costs(
        *,
        sas_file: Path,
        ground_skill_costs: dict[GroundSkill, float],
        default_cost: float,
        cost_precision: int = 3,
    ) -> None:
        sas_file.write_text(
            FastDownwardPlanner._patch_sas_costs(
                sas_text=sas_file.read_text(),
                ground_skill_costs=ground_skill_costs,
                default_cost=default_cost,
                cost_precision=cost_precision,
            )
        )

    @staticmethod
    def _parse_plan(
        *, output: str, skill_by_name: dict[str, Skill], object_by_name: dict[str, Object]
    ) -> list[GroundSkill]:
        if "Solution found!" not in output:
            raise PlanningFailure(f"Fast Downward did not find a plan: {output}")
        if "Plan length: 0 step" in output:
            return []
        plan_lines = re.findall(r"(.+) \(\d+?\)", output)
        if not plan_lines:
            raise PlanningFailure(f"Fast Downward reported success but no plan lines: {output}")
        plan: list[GroundSkill] = []
        for line in plan_lines:
            tokens = line.split()
            skill = skill_by_name[tokens[0]]
            plan_objects = tuple(object_by_name[name] for name in tokens[1:])
            plan.append(GroundSkill(skill=skill, objects=plan_objects))
        return plan
