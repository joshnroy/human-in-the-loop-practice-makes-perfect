"""`--preset NAME`: a committed, named set of flag values for the global CLI.

A preset is a JSON file in this directory, `<name>.json`. It never adds behaviour: it
only changes argparse *defaults*. That gives three guarantees:

* an explicit flag always wins, wherever it sits on the command line;
* every value it sets is an ordinary resolved flag, so `config_snapshot.json` records
  it like any other (and records `preset` itself);
* a run through a preset is reproducible without the preset, from its snapshot.

The file names the one `--method` it is for, the values every environment shares
(`shared`), and per-environment values (`environments`). An environment it does not
list is refused rather than half-configured; a key that names no flag is an error, so a
typo cannot silently fall back to a default. Values for an `environments` entry
override `shared` ones. A string `practice_cost_config` resolves against this directory.
A preset runs only under the step protocol (`--practice-step-budget`), the one mode that
reads its cost file.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

PRESET_DIRECTORY = Path(__file__).parent

# Keys whose values are files shipped beside the preset.
PATH_KEYS = ("practice_cost_config",)


class CliPreset(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    description: str
    method: str
    shared: dict[str, Any] = Field(default_factory=dict)
    environments: dict[str, dict[str, Any]]
    directory: Path = PRESET_DIRECTORY

    @staticmethod
    def names(*, directory: Path = PRESET_DIRECTORY) -> list[str]:
        return sorted(
            path.stem for path in directory.glob("*.json") if not path.stem.endswith("_costs")
        )

    @staticmethod
    def load(*, name: str, directory: Path = PRESET_DIRECTORY) -> "CliPreset":
        path = directory / f"{name}.json"
        if not path.is_file():
            raise ValueError(
                f"no preset {name!r} in {directory}; known: {CliPreset.names(directory=directory)}"
            )
        return CliPreset.model_validate({
            **json.loads(path.read_text()),
            "name": name,
            "directory": directory,
        })

    def values(self, *, env: str, method: str) -> dict[str, Any]:
        if method != self.method:
            raise ValueError(
                f"preset {self.name!r} is for --method {self.method}, not --method {method}"
            )
        if env not in self.environments:
            raise ValueError(
                f"preset {self.name!r} defines no values for --env {env}; it names "
                f"{sorted(self.environments)}"
            )
        resolved = {**self.shared, **self.environments[env]}
        for key in PATH_KEYS:
            if isinstance(resolved.get(key), str):
                resolved[key] = (self.directory / resolved[key]).resolve()
        return resolved

    def apply(self, *, parser: argparse.ArgumentParser, env: str, method: str) -> None:
        """Make the preset's values `parser`'s defaults (string values are converted by
        each flag's own `type`, exactly as typed-in values are)."""
        values = self.values(env=env, method=method)
        known = {action.dest for action in parser._actions}  # noqa: SLF001
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(
                f"preset {self.name!r} sets {unknown}, which name no flag of --env {env} "
                f"--method {method}"
            )
        parser.set_defaults(**values)
