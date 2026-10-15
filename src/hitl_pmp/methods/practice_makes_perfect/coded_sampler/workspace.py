"""The coding agent's persistent workspace: docs generated from code, data, versions."""

import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel

from hitl_pmp.core.method.skill_provider import SkillProvider
from hitl_pmp.core.method.types import GroundSkill, ParameterSpec, Skill
from hitl_pmp.core.problem.environment.types import Object

from .types import CodedSamplerMode, CodeVersion


class SamplerInputLayout:
    """The classifier input row of each learned skill, described field by field from
    the same code that builds the row. A static-method container, never instantiated."""

    @staticmethod
    def placeholder_grounding(*, skill: Skill) -> GroundSkill:
        """A grounding of `skill` whose objects are named after its variables. Enough to
        ask a provider which layout a skill uses, which depends on the skill only."""
        return GroundSkill(
            skill=skill,
            objects=tuple(
                Object(name=variable.name, type=variable.type) for variable in skill.parameters
            ),
        )

    @staticmethod
    def parameters(*, provider: SkillProvider, skill: Skill) -> list[dict[str, Any]]:
        specs = provider.parameter_specs(skill=skill)
        if specs is None:
            return [
                {"name": f"param_{i}", "unit": None, "meaning": None, "lower": None, "upper": None}
                for i in range(skill.param_dim)
            ]
        if len(specs) != skill.param_dim:
            raise ValueError(f"{skill.name}: {len(specs)} parameter specs for {skill.param_dim}")
        return [spec.model_dump() for spec in specs]

    @staticmethod
    def describe(*, provider: SkillProvider, skill: Skill) -> dict[str, Any]:
        feature_specs = provider.feature_specs()
        parameters = SamplerInputLayout.parameters(provider=provider, skill=skill)
        grounding = SamplerInputLayout.placeholder_grounding(skill=skill)
        hand_selected = provider.hand_selected_feature_names(ground_skill=grounding)
        fields: list[dict[str, Any]] = []
        if hand_selected is not None:
            by_name = {parameter["name"]: parameter for parameter in parameters}
            for name in hand_selected:
                if name == "bias":
                    fields.append({"name": "bias", "unit": None, "meaning": "Always 1.0."})
                elif name in by_name:
                    parameter = by_name[name]
                    fields.append({
                        "name": f"param.{name}",
                        "unit": parameter["unit"],
                        "meaning": parameter["meaning"],
                    })
                else:
                    raise ValueError(f"{skill.name}: hand-selected field {name!r} is undescribed")
            layout = "hand_selected"
        else:
            fields.append({"name": "bias", "unit": None, "meaning": "Always 1.0."})
            for variable in skill.parameters:
                specs = {spec.name: spec for spec in feature_specs.get(variable.type.name, ())}
                for feature in variable.type.feature_names:
                    spec = specs.get(feature)
                    fields.append({
                        "name": f"{variable.name}.{feature}",
                        "unit": None if spec is None else spec.unit,
                        "meaning": None if spec is None else spec.meaning,
                    })
            fields.extend(
                {
                    "name": f"param.{parameter['name']}",
                    "unit": parameter["unit"],
                    "meaning": parameter["meaning"],
                }
                for parameter in parameters
            )
            layout = "default"
        return {
            "skill": skill.name,
            "param_dim": skill.param_dim,
            "layout": layout,
            "grounding": [
                {"variable": variable.name, "type": variable.type.name}
                for variable in skill.parameters
            ],
            "sampler_input": [{"index": i, **field} for i, field in enumerate(fields)],
            "parameters": parameters,
        }

    @staticmethod
    def describe_all(*, provider: SkillProvider) -> dict[str, Any]:
        learned = [skill for skill in provider.skills() if skill.param_dim > 0]
        return {
            "skills": [SamplerInputLayout.describe(provider=provider, skill=s) for s in learned],
            "object_types": {
                type_.name: [
                    {"name": spec.name, "unit": spec.unit, "meaning": spec.meaning}
                    for spec in provider.feature_specs().get(type_.name, ())
                ]
                or [{"name": name, "unit": None, "meaning": None} for name in type_.feature_names]
                for type_ in provider.types()
            },
        }

    @staticmethod
    def markdown(*, description: dict[str, Any]) -> str:
        lines = ["# Sampler input vectors and parameters", ""]
        lines.append("Generated from the code that builds the rows; do not edit.")
        for skill in description["skills"]:
            lines += ["", f"## `{skill['skill']}` ({skill['layout']} layout)", ""]
            grounding = ", ".join(f"{g['variable']}: {g['type']}" for g in skill["grounding"])
            lines += [f"Grounding (object order): {grounding}", "", "Sampler input row:", ""]
            lines += ["| index | field | unit | meaning |", "| --- | --- | --- | --- |"]
            lines += [
                f"| {f['index']} | `{f['name']}` | {f['unit'] or ''} | {f['meaning'] or ''} |"
                for f in skill["sampler_input"]
            ]
            lines += ["", "Parameters (the vector a sampler returns, in order):", ""]
            lines += [
                "| name | unit | lower | upper | meaning |",
                "| --- | --- | --- | --- | --- |",
            ]
            lines += [
                f"| `{p['name']}` | {p['unit'] or ''} | {p['lower']} | {p['upper']} | "
                f"{p['meaning'] or ''} |"
                for p in skill["parameters"]
            ]
        lines += ["", "## State features per object type", ""]
        for type_name, features in description["object_types"].items():
            lines += [f"### `{type_name}`", ""]
            if not features:
                lines += ["No features.", ""]
                continue
            lines += ["| feature | unit | meaning |", "| --- | --- | --- |"]
            lines += [
                f"| `{f['name']}` | {f['unit'] or ''} | {f['meaning'] or ''} |" for f in features
            ]
            lines.append("")
        return "\n".join(lines) + "\n"


class SkillCatalog:
    """Every skill the planner can choose, rendered from its operator."""

    @staticmethod
    def atoms(*, collection: Any) -> str:
        rendered = sorted(
            f"{atom.predicate.name}({', '.join(v.name for v in atom.variables)})"
            for atom in collection
        )
        return ", ".join(rendered) or "none"

    @staticmethod
    def markdown(*, provider: SkillProvider) -> str:
        skills = [*provider.skills(), *(g.skill for g in provider.human_cube_bin_reset_skills())]
        seen: set[str] = set()
        lines = ["# Skill operators", "", "Generated from the operator definitions.", ""]
        for skill in skills:
            if skill.name in seen:
                continue
            seen.add(skill.name)
            arguments = ", ".join(f"{v.name}: {v.type.name}" for v in skill.parameters)
            atoms = SkillCatalog.atoms
            lines += [
                f"## `{skill.name}({arguments})`",
                "",
                f"- continuous parameters: {skill.param_dim}",
                f"- practice cost: {skill.practice_cost}",
                f"- preconditions: {atoms(collection=skill.preconditions)}",
                f"- add effects: {atoms(collection=skill.add_effects)}",
                f"- delete effects: {atoms(collection=skill.delete_effects)}",
                "- re-read from observation afterwards: "
                + (", ".join(sorted(p.name for p in skill.ignore_effects)) or "none"),
                "",
            ]
        return "\n".join(lines)


class GeneratedImports:
    """What a generated module may import. A static-method container."""

    ALLOWED: ClassVar[frozenset[str]] = frozenset({"numpy", "scipy", "sklearn"})

    @staticmethod
    def violations(*, source: str) -> list[str]:
        """Top-level imports outside numpy, scipy, sklearn, the standard library and the
        module's own directory (`local` names other files beside it)."""
        tree = ast.parse(source)
        names: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names += [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.append(node.module)
        return sorted({
            name
            for name in names
            if name.split(".")[0] not in GeneratedImports.ALLOWED
            and name.split(".")[0] not in sys.stdlib_module_names
        })


class CodingWorkspace(BaseModel):
    """`root` is the coding agent's sandbox directory; `versions` sits beside it."""

    root: Path
    mode: CodedSamplerMode

    MODULE_DIR: ClassVar[str] = "samplers"

    @property
    def versions_dir(self) -> Path:
        return self.root.parent / "versions"

    def module_path(self, *, skill_name: str) -> str:
        return f"{self.MODULE_DIR}/{skill_name}.py"

    def initialise(self, *, provider: SkillProvider, instructions: str) -> None:
        """Write the docs once, before the first refit. Docs are regenerated from code
        each run, never edited by hand."""
        (self.root / "docs").mkdir(parents=True, exist_ok=True)
        (self.root / "data").mkdir(parents=True, exist_ok=True)
        (self.root / self.MODULE_DIR).mkdir(parents=True, exist_ok=True)
        description = SamplerInputLayout.describe_all(provider=provider)
        (self.root / "docs" / "sampler_inputs.json").write_text(
            json.dumps(description, indent=2) + "\n"
        )
        (self.root / "docs" / "sampler_inputs.md").write_text(
            SamplerInputLayout.markdown(description=description)
        )
        (self.root / "docs" / "skills.md").write_text(SkillCatalog.markdown(provider=provider))
        task = provider.task_description()
        if task is not None:
            (self.root / "docs" / "task.md").write_text(task)
        (self.root / "CLAUDE.md").write_text(instructions)

    def snapshot(self, *, index: int, cycle: int, skills: list[str]) -> CodeVersion:
        """Freeze `samplers/` as version `index`. Calls only ever read a snapshot, so
        an evaluation runs exactly the code the refit left, whatever happens later."""
        destination = self.versions_dir / f"v{index:03d}"
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(self.root / self.MODULE_DIR, destination / self.MODULE_DIR)
        modules = {
            skill: self.module_path(skill_name=skill)
            for skill in skills
            if (destination / self.module_path(skill_name=skill)).is_file()
        }
        return CodeVersion(
            index=index,
            cycle=cycle,
            snapshot_dir=destination,
            git_commit=self.git_head(),
            modules=modules,
        )

    def git_head(self) -> str | None:
        if not (self.root / ".git").exists():
            return None
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None


class ParameterCheck:
    """Shape, finiteness and bounds of a generated parameter vector. Never clips."""

    @staticmethod
    def violation(*, value: Any, specs: list[ParameterSpec]) -> str | None:
        if not isinstance(value, list | tuple) or len(value) != len(specs):
            return f"expected a list of {len(specs)} numbers, got {value!r}"[:500]
        for spec, entry in zip(specs, value, strict=True):
            if isinstance(entry, bool) or not isinstance(entry, int | float):
                return f"{spec.name}: not a number: {entry!r}"[:500]
            number = float(entry)
            if number != number or number in (float("inf"), float("-inf")):
                return f"{spec.name}: not finite: {entry!r}"
            if not spec.lower <= number <= spec.upper:
                return f"{spec.name}={number} outside [{spec.lower}, {spec.upper}]"
        return None
