"""Wall-clock time inside one robot skill or one human reset, split by phase.

A step-protocol run's event log times a skill only as a whole, so a slow execution
cannot be told apart into motion planning (`controller.reset`), MuJoCo stepping, and
the `PickPlannable` dry run that `KinderBackend.abstract_atoms` makes on the state the
skill leaves. `KinderBackend` and `Tossing3DEnvironment` time those phases into one
`SkillTiming` per execution, and the step protocol writes each beside the run as
`skill_timing.jsonl`. It is never part of `stats.json` or `step_events.jsonl`: those
must stay byte-stable across runs, and wall-clock never is.

Phases are exclusive: the seconds of a phase exclude every phase timed inside it, so
they sum to at most `total_seconds`, and the rest is `untimed_seconds`.
"""

import time
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr


class SkillTiming(BaseModel):
    """One execution's phases: seconds and number of entries per phase name, and
    entry-only counts (cache hits, skipped dry runs) under `counts` alone."""

    kind: str
    seconds: dict[str, float] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)
    total_seconds: float = 0.0

    _started: float = PrivateAttr(default_factory=time.perf_counter)
    # Seconds spent in timed phases nested inside each open phase, innermost last.
    _nested: list[float] = PrivateAttr(default_factory=list)

    @staticmethod
    def phase_of(*, timing: "SkillTiming | None", name: str) -> AbstractContextManager[None]:
        """`timing.phase(name=name)`, or nothing at all while no execution is timed."""
        return nullcontext() if timing is None else timing.phase(name=name)

    @contextmanager
    def phase(self, *, name: str) -> Generator[None, None, None]:
        self._nested.append(0.0)
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            inner = self._nested.pop()
            self.seconds[name] = self.seconds.get(name, 0.0) + elapsed - inner
            self.counts[name] = self.counts.get(name, 0) + 1
            if self._nested:
                self._nested[-1] += elapsed

    def count(self, *, name: str) -> None:
        self.counts[name] = self.counts.get(name, 0) + 1

    def finish(self) -> None:
        self.total_seconds = time.perf_counter() - self._started

    def record(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "total_seconds": self.total_seconds,
            "untimed_seconds": self.total_seconds - sum(self.seconds.values()),
            "seconds": dict(self.seconds),
            "counts": dict(self.counts),
        }
