"""Validated, source-bound skill priors for the DP allocator, not EES defaults."""

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hitl_pmp.config_snapshot import ConfigSnapshot
from hitl_pmp.core.method.types import Skill


class DpSkillPrior(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    alpha: float = Field(gt=0.0, allow_inf_nan=False)
    beta: float = Field(gt=0.0, allow_inf_nan=False)
    initial_learning_rate: float = Field(ge=0.0, allow_inf_nan=False)


class DpPriorCalibration(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    schema_version: Literal[1]
    environment: str
    layout: str
    variant: str
    skills: dict[str, DpSkillPrior] = Field(min_length=1)
    validation_passed: Literal[True]
    uses_experiment_evaluations: Literal[False]
    fit_seeds: list[int] = Field(min_length=1)
    validation_seeds: list[int] = Field(min_length=1)
    source_sha256: dict[str, str] = Field(min_length=1)
    simulator_pins: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _independent_splits(self) -> "DpPriorCalibration":
        if set(self.fit_seeds) & set(self.validation_seeds):
            raise ValueError("prior fitting and validation seeds must be disjoint")
        if set(self.fit_seeds + self.validation_seeds) & set(range(6)):
            raise ValueError("paired experiment seeds cannot calibrate priors")
        return self

    def verify_simulators(self) -> None:
        for package, expected in self.simulator_pins.items():
            actual, dirty = ConfigSnapshot._repo_state(package=package)
            if dirty or actual != expected:
                raise ValueError(f"prior calibration simulator mismatch: {package}")

    def verify_robot_skills(self, *, skills: tuple[Skill, ...]) -> None:
        names = {skill.name for skill in skills}
        unknown = set(self.skills) - names
        if unknown:
            raise ValueError(f"calibrated skills absent from this domain: {sorted(unknown)}")
        missing = names - set(self.skills)
        if missing:
            raise ValueError(f"robot skills lack calibrated DP priors: {sorted(missing)}")

    @staticmethod
    def load(*, path: Path) -> "DpPriorCalibration":
        calibration = DpPriorCalibration.model_validate_json(path.read_text())
        root = Path(__file__).resolve().parents[4]
        for name, expected in calibration.source_sha256.items():
            source = (root / name).resolve()
            if not source.is_relative_to(root) or not source.is_file():
                raise ValueError(f"invalid calibration source path: {name}")
            if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                raise ValueError(f"stale prior calibration source: {name}")
        return calibration
