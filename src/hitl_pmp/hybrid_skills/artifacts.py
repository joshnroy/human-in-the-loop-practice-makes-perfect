"""Immutable generated source and cumulative persistent-session accounting."""

import ast
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict, model_validator


class SkillBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")
    files: dict[str, str]

    @model_validator(mode="after")
    def validate_source(self) -> "SkillBundle":
        if "skills.py" not in self.files:
            raise ValueError("Submission must contain skills.py")
        for name, source in self.files.items():
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or path.suffix != ".py":
                raise ValueError("Only relative Python submission files are accepted")
            ast.parse(source, filename=name)
        return self

    @classmethod
    def read(cls, *, directory: Path) -> "SkillBundle":
        files = {}
        for path in directory.rglob("*.py"):
            if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
                raise ValueError("Submission files must stay inside the submission directory")
            files[path.relative_to(directory).as_posix()] = path.read_text()
        return cls(files=files)

    def write(self, *, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for name, source in self.files.items():
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source)

    @property
    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.files, sort_keys=True).encode()).hexdigest()


class SessionBudget(BaseModel):
    limit: float
    spent: float = 0

    def record(self, *, cumulative: float | None) -> None:
        if cumulative is None or not math.isfinite(cumulative) or cumulative < self.spent:
            raise ValueError("Missing or decreasing persistent-session model cost")
        self.spent = cumulative

    @property
    def remaining(self) -> float:
        return max(0, self.limit - self.spent)

    @property
    def exhausted(self) -> bool:
        return self.remaining <= 0


class ModelBudgetExhausted(Exception):
    """A reported endpoint; it is not a robot execution failure."""
