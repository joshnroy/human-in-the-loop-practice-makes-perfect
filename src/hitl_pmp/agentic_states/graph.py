"""Language-defined action applicability and observable outcome regions."""

from collections import deque
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hitl_pmp.agentic_runtime.types import StateCluster

RobotAction = Literal["PickCube", "MoveToTossLocationAndToss", "OpenGripper"]
HumanAction = Literal["reset_cube_far", "reset_cube_and_bin_near"]
Action = Literal[RobotAction, HumanAction]
HUMANS = {"reset_cube_far", "reset_cube_and_bin_near"}


class Edge(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source: str
    action: Action
    success: str | dict[str, float]
    failure: str | dict[str, float]

    def destinations(self, *, outcome: bool) -> dict[str, float]:
        raw = self.success if outcome else self.failure
        return {raw: 1.0} if isinstance(raw, str) else raw


class RobotEdge(Edge):
    action: RobotAction


class LanguageManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    # A run that offers the human reset must represent both of its groundings; a
    # run without it must never name them (see `RobotOnlyManifest`).
    human_reset_offered: ClassVar[bool] = True
    revision: str = Field(min_length=1)
    clusters: tuple[StateCluster, ...]
    goal_clusters: tuple[str, ...]
    edges: tuple[Edge, ...]

    @classmethod
    def for_run(cls, *, human_reset_offered: bool) -> type["LanguageManifest"]:
        return LanguageManifest if human_reset_offered else RobotOnlyManifest

    @model_validator(mode="after")
    def validate_graph(self):
        import math

        ids = {c.cluster_id for c in self.clusters}
        if not ids or len(ids) != len(self.clusters):
            raise ValueError("Unique nonempty language clusters required")
        if not self.goal_clusters or not set(self.goal_clusters) <= ids:
            raise ValueError("Goal clusters must identify the physical task goal")
        seen = set()
        for e in self.edges:
            key = e.source, e.action
            if e.source not in ids or key in seen:
                raise ValueError("One edge per known source/action")
            seen.add(key)
            good, bad = e.destinations(outcome=True), e.destinations(outcome=False)
            if set(good) & set(bad):
                raise ValueError("Success/failure destinations must be distinguishable")
            for distribution in (good, bad):
                if not distribution or not set(distribution) <= ids:
                    raise ValueError("Destinations must be known language clusters")
                if any(
                    not math.isfinite(v) or v <= 0 for v in distribution.values()
                ) or not math.isclose(sum(distribution.values()), 1):
                    raise ValueError("Conditional probabilities must sum to one")
        if self.human_reset_offered and not {e.action for e in self.edges} >= HUMANS:
            raise ValueError("Both human reset skills must remain represented")
        return self


# The manifest of a `--no-human-reset` run. The edge type rejects the reset actions by
# construction, so the schema the coding model reads never names them. No docstring:
# pydantic would publish it as the schema's description.
class RobotOnlyManifest(LanguageManifest):
    human_reset_offered = False
    edges: tuple[RobotEdge, ...]


class StateGraph:
    def __init__(self, *, manifest: LanguageManifest):
        self.manifest = manifest
        self.edges = {(e.source, e.action): e for e in manifest.edges}

    def actions(self, *, cluster: str | None) -> list[str]:
        return [e.action for e in self.manifest.edges if e.source == cluster]

    def outcome(self, *, source: str, action: str, destination: str | None) -> bool | None:
        if destination is None:
            return None
        return destination in self.edges[source, action].destinations(outcome=True)

    def deployment_action(self, *, cluster: str | None) -> str | None:
        """Classical shortest successful path in the frozen generated graph.

        Replan after every observed endpoint. No human actions at deployment.
        No predicates, generated code, or model calls are used by this search.
        """
        if cluster is None or cluster in self.manifest.goal_clusters:
            return None
        queue = deque([(cluster, None)])
        seen = {cluster}
        while queue:
            state, first = queue.popleft()
            for action in self.actions(cluster=state):
                if action in HUMANS:
                    continue
                for target in self.edges[state, action].destinations(outcome=True):
                    selected = first or action
                    if target in self.manifest.goal_clusters:
                        return selected
                    if target not in seen:
                        seen.add(target)
                        queue.append((target, selected))
        return None
