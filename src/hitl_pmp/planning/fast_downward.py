import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from hitl_pmp.core.method.types import GroundSkill, Skill
from hitl_pmp.core.problem.environment.types import Object, Type
from hitl_pmp.core.problem.tasks.types import GroundAtom, Predicate

from .pddl import PddlWriter


class PlanningFailure(Exception):
    """No plan could be produced -- either Fast Downward's translator aborted (an
    unreachable/contradictory goal is usually detected here, before search even
    starts) or search terminated without "Solution found". Mirrors predicators'
    exception of the same name; a `Method` catches this rather than a bare
    subprocess error, so "this goal is unachievable" stays distinguishable from
    "the planner is broken/missing"."""


class FastDownwardPlanner:
    """Cost-aware optimal task planning over `GroundSkill`s, by shelling out to a
    real Fast Downward. A static-method container, never instantiated, same as every
    other business-logic class in this project.

    This is the piece EES genuinely needs: it plans with a *per-ground-skill* cost
    (EES uses -log(competence)), so the minimum-cost plan is the
    maximum-likelihood-of-success plan. No built-in planner substitutes -- see
    README.md ("Why Fast Downward, not a hand-rolled planner").

    Ported faithfully from the sibling hitl-practice repo's
    `predicators/planning.py` (`generate_sas_file_for_fd`, `_ground_op_to_sas_op`,
    `_update_sas_file_with_costs`, `fd_plan_from_sas_file`), which splits one plan
    call into three stages:

    1. **Translate.** Write the PDDL domain/problem (PddlWriter) to temp files and
       run FD with `--sas-file`, producing a SAS file but no plan yet.
    2. **Patch costs.** Rewrite that SAS file in place: flip its metric on and
       replace each operator's cost line with `int(10**cost_precision * cost)`.
       Costs are injected here, *not* in the PDDL, precisely because SAS operators
       are already ground -- PDDL `(:functions)` could only express one cost per
       lifted action, which is not what EES needs.
    3. **Search.** Run FD again on the patched SAS file, parse the plan lines back
       into `GroundSkill`s by lowercased skill name + lowercased object names.

    Deviations from predicators, all deliberate:

    - `subprocess.run` with an argument list and `cwd=` a temp directory, instead of
      `subprocess.getoutput` with an interpolated shell string. Same commands, but
      no shell quoting hazards, and FD's `sas_plan`/`output.sas` scratch files land
      in the temp directory rather than the caller's working directory.
    - No `Metrics`/`max_horizon`/`atoms_sequence` returned -- this returns just the
      skeleton (`list[GroundSkill]`). The extra bookkeeping predicators threads
      through belongs to a `Method`/`Metrics` here, not the planner.
    - `PlanningTimeout` is not a separate exception: a timed-out run simply produces
      no "Solution found" and so raises `PlanningFailure`."""

    # FD_EXEC_PATH is predicators' own convention for locating the FD checkout, so
    # the same env var works for both repos. The fallback is a `downward/` checkout
    # sitting beside this repo -- the same sibling-repo convention CLAUDE.md already
    # documents for `../hitl-practice` -- rather than an absolute path, so this is
    # portable across machines. See README.md's install steps.
    _FD_EXEC_PATH_ENV_VAR = "FD_EXEC_PATH"

    @staticmethod
    def fd_dir() -> str:
        """The Fast Downward checkout directory (the one containing
        `fast-downward.py`)."""
        sibling_checkout = Path(__file__).resolve().parents[4] / "downward"
        return os.environ.get(FastDownwardPlanner._FD_EXEC_PATH_ENV_VAR, str(sibling_checkout))

    @staticmethod
    def plan(
        *,
        skills: tuple[Skill, ...],
        predicates: tuple[Predicate, ...],
        types: tuple[Type, ...],
        objects: tuple[Object, ...],
        init_atoms: frozenset[GroundAtom],
        goal: frozenset[GroundAtom],
        ground_skill_costs: dict[GroundSkill, float] | None = None,
        default_cost: float = 1.0,
        cost_precision: int = 3,
        timeout: float = 10.0,
        alias: str = "seq-opt-lmcut",
    ) -> list[GroundSkill]:
        """The plan (a skeleton of `GroundSkill`s) that reaches `goal` from
        `init_atoms` at minimum total cost. Every ground skill not named in
        `ground_skill_costs` costs `default_cost`. Raises `PlanningFailure` if no
        plan exists (or FD aborts); raises `FileNotFoundError` if Fast Downward
        isn't installed where `fd_dir()` points."""
        script = FastDownwardPlanner._fast_downward_script()
        with tempfile.TemporaryDirectory() as tmp:
            sas_file = Path(tmp) / "output.sas"
            FastDownwardPlanner._translate(
                skills=skills,
                predicates=predicates,
                types=types,
                objects=objects,
                init_atoms=init_atoms,
                goal=goal,
                sas_file=sas_file,
                script=script,
                tmp_dir=tmp,
                timeout=timeout,
                alias=alias,
            )
            if ground_skill_costs is not None:
                FastDownwardPlanner._update_sas_file_with_costs(
                    sas_file=sas_file,
                    ground_skill_costs=ground_skill_costs,
                    default_cost=default_cost,
                    cost_precision=cost_precision,
                )
            output = FastDownwardPlanner._run(
                args=[
                    *FastDownwardPlanner._timeout_prefix(timeout=timeout),
                    sys.executable,
                    str(script),
                    "--alias",
                    alias,
                    str(sas_file),
                ],
                cwd=tmp,
            )
            FastDownwardPlanner._run(args=[sys.executable, str(script), "--cleanup"], cwd=tmp)
        return FastDownwardPlanner._parse_plan(output=output, skills=skills, objects=objects)

    @staticmethod
    def _fast_downward_script() -> Path:
        script = Path(FastDownwardPlanner.fd_dir()) / "fast-downward.py"
        if not script.is_file():
            raise FileNotFoundError(
                f"No fast-downward.py at {script}. Fast Downward is not vendored with "
                f"this project (see planning/README.md); install it and set the "
                f"{FastDownwardPlanner._FD_EXEC_PATH_ENV_VAR} environment variable to "
                "its checkout directory."
            )
        return script

    @staticmethod
    def _timeout_prefix(*, timeout: float) -> list[str]:
        """GNU coreutils' `timeout`, which is `gtimeout` on macOS (homebrew's
        coreutils) -- same platform split predicators makes."""
        command = "gtimeout" if sys.platform == "darwin" else "timeout"
        return [command, str(timeout)]

    @staticmethod
    def _run(*, args: list[str], cwd: str) -> str:
        completed = subprocess.run(  # noqa: S603
            args, capture_output=True, text=True, cwd=cwd, check=False
        )
        return completed.stdout + completed.stderr

    @staticmethod
    def _translate(
        *,
        skills: tuple[Skill, ...],
        predicates: tuple[Predicate, ...],
        types: tuple[Type, ...],
        objects: tuple[Object, ...],
        init_atoms: frozenset[GroundAtom],
        goal: frozenset[GroundAtom],
        sas_file: Path,
        script: Path,
        tmp_dir: str,
        timeout: float,
        alias: str,
    ) -> None:
        """Stage 1: PDDL text -> SAS file (no search yet). Mirrors predicators'
        `generate_sas_file_for_fd`."""
        domain_file = Path(tmp_dir) / "domain.pddl"
        problem_file = Path(tmp_dir) / "problem.pddl"
        domain_file.write_text(
            PddlWriter.domain_str(skills=skills, predicates=predicates, types=types),
            encoding="utf-8",
        )
        problem_file.write_text(
            PddlWriter.problem_str(objects=objects, init_atoms=init_atoms, goal=goal),
            encoding="utf-8",
        )
        output = FastDownwardPlanner._run(
            args=[
                *FastDownwardPlanner._timeout_prefix(timeout=timeout),
                sys.executable,
                str(script),
                "--alias",
                alias,
                "--sas-file",
                str(sas_file),
                str(domain_file),
                str(problem_file),
            ],
            cwd=tmp_dir,
        )
        if "Driver aborting" in output or not sas_file.is_file():
            raise PlanningFailure(
                "Fast Downward failed to translate the PDDL to SAS -- the goal is "
                f"likely unreachable (a dr-reachability issue). FD output:\n{output}"
            )

    @staticmethod
    def _update_sas_file_with_costs(
        *,
        sas_file: Path,
        ground_skill_costs: dict[GroundSkill, float],
        default_cost: float,
        cost_precision: int,
    ) -> None:
        """Stage 2: rewrite the SAS file in place so each ground operator carries its
        own cost. A direct port of predicators' `_update_sas_file_with_costs`; see
        https://www.fast-downward.org/TranslatorOutputFormat for the SAS format.

        Within a `begin_operator`/`end_operator` block, the first line is the ground
        operator's name and the line immediately before `end_operator` is its cost
        (always "1" as translated, since the PDDL carries no costs)."""
        sas_str = sas_file.read_text(encoding="utf-8")
        # Turn the metric on, or FD ignores operator costs entirely.
        sas_str = sas_str.replace("begin_metric\n0\nend_metric", "begin_metric\n1\nend_metric")
        remaining = {
            FastDownwardPlanner._ground_skill_to_sas_name(ground_skill=ground_skill): cost
            for ground_skill, cost in ground_skill_costs.items()
        }
        lines = sas_str.split("\n")
        # Indexing rather than enumerate(): the loop rewrites `lines` as it goes
        # (each block's cost line), and iterating a list being mutated is banned.
        for idx in range(len(lines)):
            if lines[idx] != "begin_operator":
                continue
            end_idx = next(i for i in range(idx + 1, len(lines)) if lines[i] == "end_operator")
            cost_idx = end_idx - 1
            cost = remaining.pop(lines[idx + 1], default_cost)
            lines[cost_idx] = str(int((10**cost_precision) * cost))
        # Leftover entries are normal, not an error: FD's translator drops operators
        # it proves irrelevant to the goal (predicators only logs a warning too).
        sas_file.write_text("\n".join(lines), encoding="utf-8")

    @staticmethod
    def _ground_skill_to_sas_name(*, ground_skill: GroundSkill) -> str:
        """SAS operator names are lowercased "<skill> <obj> <obj> ..." -- the same
        form `_parse_plan` reads back out of FD's plan output."""
        name = ground_skill.skill.name.lower()
        objects_str = " ".join(obj.name.lower() for obj in ground_skill.objects)
        return f"{name} {objects_str}".strip()

    @staticmethod
    def _parse_plan(
        *, output: str, skills: tuple[Skill, ...], objects: tuple[Object, ...]
    ) -> list[GroundSkill]:
        """Stage 3's tail: FD's search output -> `GroundSkill`s. Mirrors
        `fd_plan_from_sas_file`'s parsing, including its two special cases (no
        solution at all; a trivially-empty plan)."""
        if "Solution found" not in output:
            raise PlanningFailure(f"Fast Downward found no plan. FD output:\n{output}")
        if "Plan length: 0 step" in output:
            return []
        plan_lines = re.findall(r"(.+) \(\d+?\)", output)
        if not plan_lines:
            raise PlanningFailure(
                f"Fast Downward reported a solution but emitted no plan steps:\n{output}"
            )
        skill_by_name = {skill.name.lower(): skill for skill in skills}
        object_by_name = {obj.name.lower(): obj for obj in objects}
        plan: list[GroundSkill] = []
        for plan_line in plan_lines:
            tokens = plan_line.split()
            plan.append(
                GroundSkill(
                    skill=skill_by_name[tokens[0]],
                    objects=tuple(object_by_name[token] for token in tokens[1:]),
                )
            )
        return plan
