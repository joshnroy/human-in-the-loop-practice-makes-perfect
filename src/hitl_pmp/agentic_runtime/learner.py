"""Session-boundary learner interface independent of parameterization."""

import hashlib
import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from .types import GeneratedLibrary, RevisionProposal


@runtime_checkable
class PolicyLearner(Protocol):
    """A backend stores real trajectories outside hypothetical search states."""

    @property
    def library(self) -> GeneratedLibrary: ...

    def begin_session(self) -> None: ...

    def record(self, *, evidence: dict[str, Any]) -> None: ...

    def end_session(self) -> "PolicyRevision": ...


class CodingAgent(Protocol):
    """Produce code as data inside a disconnected sandbox."""

    def revise(
        self, *, library: GeneratedLibrary, evidence: list[dict[str, Any]], prompt: str
    ) -> RevisionProposal: ...


class PolicyRevision(BaseModel):
    """Revision metadata used to invalidate execution and search caches."""

    model_config = ConfigDict(frozen=True)
    previous_digest: str
    digest: str
    changed_files: tuple[str, ...]
    evidence_count: int = Field(ge=0)


class CodePolicyLearner:
    """Freeze contracts and code within a session; revise whole policies afterwards."""

    def __init__(
        self,
        *,
        library: GeneratedLibrary,
        agent: CodingAgent,
        improvement_prompt: str,
        artifact_dir: Path,
    ) -> None:
        self._library = library.model_copy(deep=True)
        self._agent = agent
        self._prompt = improvement_prompt
        self._artifacts = artifact_dir
        self._active = False
        self._evidence: list[dict[str, Any]] = []
        self._session_start = 0
        self._session = 0

    @property
    def library(self) -> GeneratedLibrary:
        return self._library.model_copy(deep=True)

    @property
    def digest(self) -> str:
        encoded = json.dumps(self.library.model_dump(mode="json"), sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest()

    def begin_session(self) -> None:
        if self._active:
            raise RuntimeError("A practice session is already active")
        self._active = True
        self._session_start = len(self._evidence)

    def record(self, *, evidence: dict[str, Any]) -> None:
        if not self._active:
            raise RuntimeError("Evidence must belong to a real active practice session")
        self._evidence.append(json.loads(json.dumps(evidence, allow_nan=False)))

    def end_session(self) -> PolicyRevision:
        if not self._active:
            raise RuntimeError("No active session to improve")
        previous = self.digest
        evidence_count = len(self._evidence) - self._session_start
        session_dir = self._artifacts / f"session-{self._session:04d}"
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "before.json").write_text(self.library.model_dump_json(indent=2))
        (session_dir / "evidence.json").write_text(json.dumps(self._evidence, indent=2))
        # A failed invocation leaves the active session and previous code intact.
        proposal = (
            self._agent.revise(
                library=self.library.model_copy(deep=True),
                evidence=self._evidence,
                prompt=self._prompt,
            )
            if evidence_count
            else RevisionProposal(controllers={})
        )
        if not set(proposal.controllers).issubset(self.library.controllers):
            raise ValueError("A code revision cannot introduce options or change frozen contracts")
        changed = tuple(
            sorted(
                name
                for name, code in proposal.controllers.items()
                if self.library.controllers[name] != code
            )
        )
        updated = {**self.library.controllers, **proposal.controllers}
        candidate = GeneratedLibrary(manifest=self.library.manifest, controllers=updated)
        (session_dir / "after.json").write_text(candidate.model_dump_json(indent=2))
        self._library = candidate
        revision = PolicyRevision(
            previous_digest=previous,
            digest=self.digest,
            changed_files=changed,
            evidence_count=evidence_count,
        )
        (session_dir / "revision.json").write_text(revision.model_dump_json(indent=2))
        self._active = False
        self._session += 1
        return revision
