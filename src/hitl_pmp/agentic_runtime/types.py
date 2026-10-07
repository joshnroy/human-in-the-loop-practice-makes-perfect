"""Validated language abstractions and generated-policy artifacts."""

import math
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RuntimeManifest(BaseModel):
    """Descriptions and conditional topology frozen for one library revision."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[2] = 2
    revision: str = Field(min_length=1)
    clusters: tuple["StateCluster", ...]
    options: tuple["OptionContract", ...]
    edges: tuple["SkillChainEdge", ...]

    @model_validator(mode="after")
    def validate_topology(self) -> "RuntimeManifest":
        clusters = {item.cluster_id for item in self.clusters}
        options = {item.option_id for item in self.options}
        if not clusters or len(clusters) != len(self.clusters):
            raise ValueError("Cluster IDs must be nonempty and unique")
        if not options or len(options) != len(self.options):
            raise ValueError("Option IDs must be nonempty and unique")
        grouped: dict[tuple[str, str, str], float] = {}
        seen: set[tuple[str, str, str, str]] = set()
        for edge in self.edges:
            if edge.source not in clusters or edge.destination not in clusters:
                raise ValueError("Every edge must reference known clusters")
            if edge.option_id not in options:
                raise ValueError("Every edge must reference a known option")
            key = edge.source, edge.option_id, edge.outcome
            full_key = (*key, edge.destination)
            if full_key in seen:
                raise ValueError("Duplicate conditional destination")
            seen.add(full_key)
            grouped[key] = grouped.get(key, 0.0) + edge.probability
        if any(not math.isclose(total, 1.0, abs_tol=1e-8) for total in grouped.values()):
            raise ValueError("Conditional destination probabilities must sum to one")
        by_id = {option.option_id: option for option in self.options}
        for source, option_id, _ in grouped:
            option = by_id[option_id]
            if option.human_destination is None:
                if (source, option_id, "success") not in grouped or (
                    source,
                    option_id,
                    "failure",
                ) not in grouped:
                    raise ValueError("Robot options require both success and failure destinations")
            else:
                matching = [
                    edge
                    for edge in self.edges
                    if edge.source == source and edge.option_id == option_id
                ]
                if any(
                    edge.outcome == "success" and edge.destination != option.human_destination
                    for edge in matching
                ):
                    raise ValueError(
                        "Human edges with success must reach their declared destination"
                    )
        for option in self.options:
            if option.human_destination is not None and option.human_destination not in clusters:
                raise ValueError("Human destination must be a known cluster")
        return self

    def validate_abstract_state_contract(self) -> "RuntimeManifest":
        """One observed state must identify a unique outcome for a source/skill.

        Edge sources are initiation references; success destinations are successful
        endpoints. Other confidently classified endpoints are failures. Legacy
        textual contracts document intent but are not independently judged.
        """
        success: dict[tuple[str, str], set[str]] = {}
        failure: dict[tuple[str, str], set[str]] = {}
        for edge in self.edges:
            table = success if edge.outcome == "success" else failure
            table.setdefault((edge.source, edge.option_id), set()).add(edge.destination)
        for key in success:
            if success[key] & failure.get(key, set()):
                raise ValueError("Abstract-state success/failure endpoints must be disjoint")
        return self


class StateCluster(BaseModel):
    """A whole natural-language region, rather than a predicate truth vector."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    cluster_id: str = Field(min_length=1)
    description: str = Field(min_length=1)


class OptionContract(BaseModel):
    """Policy binding and three language-defined regions, with external accounting.

    A robot policy binds to ``controller``; a human policy binds through the
    configured option ID to a trusted executor. ``human_destination`` is its
    abstract graph connection, never executable code or a simulator target.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    option_id: str = Field(min_length=1)
    belief_skill: str = Field(min_length=1)
    initiation: str = Field(min_length=1)
    success_termination: str = Field(min_length=1)
    failure_termination: str = Field(min_length=1)
    controller: str | None = None
    human_destination: str | None = None
    cost: float = Field(ge=0, allow_inf_nan=False)
    max_steps: int = Field(default=1000, gt=0)
    parameters: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_controller(self) -> "OptionContract":
        if (self.controller is None) == (self.human_destination is None):
            raise ValueError("An option has either Python code or a human destination")
        if self.controller is not None:
            ArtifactPaths.validate(path=self.controller)
        if any(not math.isfinite(value) for value in self.parameters.values()):
            raise ValueError("Controller parameters must be finite")
        return self


class SkillChainEdge(BaseModel):
    """Destination probability conditioned on a separately modeled skill outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    source: str
    option_id: str
    outcome: Literal["success", "failure"]
    destination: str
    probability: float = Field(gt=0, le=1, allow_inf_nan=False)


class GeneratedLibrary(BaseModel):
    """Generation result; source remains data until the disconnected runner loads it."""

    model_config = ConfigDict(extra="forbid")
    manifest: RuntimeManifest
    controllers: dict[str, str]

    @model_validator(mode="after")
    def validate_files(self) -> "GeneratedLibrary":
        expected = {item.controller for item in self.manifest.options if item.controller}
        if set(self.controllers) != expected:
            raise ValueError("Generated controllers must exactly match the manifest")
        for path, source in self.controllers.items():
            ArtifactPaths.validate(path=path)
            if not source.strip():
                raise ValueError("Controller source must not be empty")
        return self


class RevisionProposal(BaseModel):
    """Between-session code/abstraction repair preserving success and accounting."""

    model_config = ConfigDict(extra="forbid")
    controllers: dict[str, str]
    manifest: RuntimeManifest | None = None

    def apply_to(self, *, library: GeneratedLibrary) -> GeneratedLibrary:
        """Build a complete candidate before any live library is replaced."""
        if not set(self.controllers).issubset(library.controllers):
            raise ValueError("A revision cannot introduce controller slots")
        manifest = self.manifest if self.manifest is not None else library.manifest
        previous = {option.option_id: option for option in library.manifest.options}
        revised = {option.option_id: option for option in manifest.options}
        if previous.keys() != revised.keys():
            raise ValueError("A revision must preserve option identities")
        fixed = (
            "belief_skill",
            "success_termination",
            "controller",
            "cost",
            "max_steps",
            "parameters",
        )
        for option_id, before in previous.items():
            after = revised[option_id]
            if any(getattr(before, key) != getattr(after, key) for key in fixed):
                raise ValueError("A revision must preserve skill success, roles, and accounting")
            if (before.human_destination is None) != (after.human_destination is None):
                raise ValueError("A revision must preserve robot and human option identities")
        # Round-trip the full candidate to recheck graph and file invariants even
        # for locally constructed Pydantic objects; model_copy can bypass validation.
        return GeneratedLibrary.model_validate({
            "manifest": manifest.model_dump(mode="json"),
            "controllers": {**library.controllers, **self.controllers},
        })

    @model_validator(mode="after")
    def validate_files(self) -> "RevisionProposal":
        for path, source in self.controllers.items():
            ArtifactPaths.validate(path=path)
            if not source.strip():
                raise ValueError("Controller source must not be empty")
        return self


class ArtifactPaths:
    """Constrain generated artifacts to ordinary relative Python paths."""

    @staticmethod
    def validate(*, path: str) -> None:
        parsed = PurePosixPath(path)
        if (
            parsed.is_absolute()
            or ".." in parsed.parts
            or "\\" in path
            or parsed.suffix != ".py"
            or str(parsed) != path
            or any(part.startswith(".") for part in parsed.parts)
        ):
            raise ValueError("Controller must be a safe relative Python path")


class ClusterJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    cluster_id: str | None
    reason: str


class ContractJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    value: bool | None
    reason: str


class TerminalJudgment(BaseModel):
    """Resolve explicit terminal regions without complementing missing success."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    success_termination: ContractJudgment
    failure_termination: ContractJudgment

    @property
    def outcome(self) -> Literal["success", "failure"] | None:
        success = self.success_termination.value
        failure = self.failure_termination.value
        if success is True and failure is False:
            return "success"
        if failure is True and success is False:
            return "failure"
        return None

    def audit(self) -> dict[str, Any]:
        """Keep existing monitoring summaries, derived from both region checks."""
        success = self.success_termination.value
        failure = self.failure_termination.value
        ended = (
            True
            if True in (success, failure)
            else (False if success is False and failure is False else None)
        )
        reason = (
            f"Terminal-region membership: success={success}, failure={failure}. "
            "An outcome requires one true and one false judgment."
        )
        return {
            **self.model_dump(mode="json"),
            "outcome": self.outcome,
            "termination": ContractJudgment(value=ended, reason=reason).model_dump(),
            "success": ContractJudgment(
                value=None if self.outcome is None else self.outcome == "success", reason=reason
            ).model_dump(),
        }
