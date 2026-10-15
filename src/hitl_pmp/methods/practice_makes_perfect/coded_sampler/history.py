"""The run's practice history, exactly as the method saw it, for the coding agent.

Practice only. An evaluation episode records nothing here, the same as it records
nothing into the MLP's training rows (`_EesEpisode.observe_pending` gates both on
`practicing`), so evaluation outcomes can never become training data.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, PrivateAttr

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.problem.environment.types import State

from .types import TrajectoryEvent


class StateCodec:
    """A static-method container, never instantiated."""

    @staticmethod
    def encode(*, state: State) -> dict[str, dict[str, float | None]]:
        """`{object name: {feature name: value}}`, every object the state carries; a
        non-finite value is `None`, since JSON has no NaN."""
        return {
            obj.name: {
                name: float(value) if np.isfinite(value) else None
                for name, value in zip(obj.type.feature_names, np.asarray(features), strict=True)
            }
            for obj, features in state.data.items()
        }

    @staticmethod
    def grounding(*, ground_skill: GroundSkill) -> list[str]:
        return [obj.name for obj in ground_skill.objects]


class TrajectoryHistory(BaseModel):
    """Append-only, ordered. Each consumer keeps its own cursor into it."""

    _events: list[TrajectoryEvent] = PrivateAttr(default_factory=list)
    _cycle: int = PrivateAttr(default=0)

    @property
    def cycle(self) -> int:
        return self._cycle

    def __len__(self) -> int:
        return len(self._events)

    def append(self, *, kind: str, **data: Any) -> TrajectoryEvent:
        event = TrajectoryEvent(index=len(self._events), kind=kind, cycle=self._cycle, data=data)
        self._events.append(event)
        return event

    def end_cycle(self) -> None:
        self.append(kind="cycle_end")
        self._cycle += 1

    def events(self, *, start: int = 0) -> list[TrajectoryEvent]:
        return list(self._events[start:])

    def write(self, *, path: Path, start: int = 0) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as stream:
            for event in self._events[start:]:
                stream.write(json.dumps(event.model_dump(mode="json"), allow_nan=False) + "\n")
