"""Observation contracts shared by execution and state-grounded judgments."""

import json
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

ObservationMode = Literal["rgb", "object_state"]


@runtime_checkable
class ObservationProvider(Protocol):
    """Read the current world without advancing or resetting it."""

    def observe(self) -> dict[str, Any]: ...


class ObservationEvidence:
    """Attach bounded physical history from a host-owned execution trajectory."""

    MAX_HISTORY = 8

    @staticmethod
    def with_history(
        *,
        observation: dict[str, Any],
        trajectory_path: str | None = None,
        before: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = dict(observation)
        if observation.get("observation_mode") != "object_state":
            return result
        result.pop("recent_observations", None)
        initial = (
            ObservationEvidence._physical(observation=before)
            if before is not None and before.get("observation_mode") == "object_state"
            else None
        )
        # Two streaming passes retain at most eight frames rather than an entire
        # trajectory. Endpoints alone cannot establish lifting or persistent grasp.
        count = int(initial is not None)
        if trajectory_path is not None:
            count += sum(1 for _ in ObservationEvidence._trajectory(path=trajectory_path))
        if not count:
            return result
        indices = ObservationEvidence._indices(count=count)
        history = [initial] if initial is not None else []
        if trajectory_path is not None:
            for index, sample in enumerate(
                ObservationEvidence._trajectory(path=trajectory_path),
                start=int(initial is not None),
            ):
                if index in indices:
                    history.append(sample)
        result["recent_observations"] = history
        return result

    @staticmethod
    def _physical(*, observation: dict[str, Any]) -> dict[str, Any]:
        return deepcopy({
            key: value
            for key, value in observation.items()
            if key not in {"execution", "execution_seed", "recent_observations"}
        })

    @staticmethod
    def _trajectory(*, path: str) -> Iterator[dict[str, Any]]:
        with Path(path).open() as stream:
            for line in stream:
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError("Trajectory records must be JSON objects")
                observation = event.get("observation")
                if (
                    isinstance(observation, dict)
                    and observation.get("observation_mode") == "object_state"
                ):
                    yield ObservationEvidence._physical(observation=observation)

    @staticmethod
    def _indices(*, count: int) -> set[int]:
        if count <= ObservationEvidence.MAX_HISTORY:
            return set(range(count))
        early_count = ObservationEvidence.MAX_HISTORY - 3
        return {
            *(round(index * (count - 4) / (early_count - 1)) for index in range(early_count)),
            count - 3,
            count - 2,
            count - 1,
        }
