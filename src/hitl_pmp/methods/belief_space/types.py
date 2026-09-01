"""The state, action, and parameter types in the belief-space pseudocode."""

import gzip
import json
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Protocol, TextIO

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from hitl_pmp.core.log_timing import LogTiming


@contextmanager
def open_trace(*, path: Path) -> Iterator[TextIO]:
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8", compresslevel=1) as stream:
            yield stream
    else:
        with path.open("w", encoding="utf-8") as stream:
            yield stream


class SearchTrace(BaseModel):
    """Actual search events, recorded without additional model calls or RNG draws."""

    events: list[dict[str, Any]] = Field(default_factory=list)
    path: Path | None = Field(default=None, exclude=True)
    retain_events: bool = Field(default=True, exclude=True)
    _stream: TextIO | None = PrivateAttr(default=None)
    _stream_stack: ExitStack = PrivateAttr(default_factory=ExitStack)
    _interned: dict[tuple[str, str], int] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = self._stream_stack.enter_context(open_trace(path=self.path))

    def record(self, *, event: str, **fields: Any) -> None:
        record = {"event": event, **fields, **LogTiming.fields()}
        if self.retain_events or fields.get("node") == 0:
            self.events.append(record)
        if self._stream is not None:
            streamed = dict(record)
            for field, kind in {
                "environment_state": "state",
                "successor": "state",
                "belief_state": "belief",
                "action": "action",
            }.items():
                value = streamed.get(field)
                if not isinstance(value, (dict, list)):
                    continue
                encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
                key = (kind, encoded)
                identifier = self._interned.get(key)
                if identifier is None:
                    identifier = len(self._interned)
                    self._interned[key] = identifier
                    self._stream.write(
                        LogTiming.encode(
                            record={
                                "event": "intern",
                                "kind": kind,
                                "id": identifier,
                                "value": value,
                            }
                        )
                    )
                streamed[field] = {"$ref": identifier, "$kind": kind}
            self._stream.write(LogTiming.encode(record=streamed))

    def close(self) -> None:
        if self._stream is not None:
            self._stream_stack.close()
            self._stream = None


class EnvironmentState(BaseModel):
    """Environment MDP state; domain subclasses supply hashable fields."""

    model_config = ConfigDict(frozen=True)


class BeliefState(BaseModel):
    """Belief over theta; domain subclasses supply hashable fields."""

    model_config = ConfigDict(frozen=True)


class POMDPAction(BaseModel):
    """Practice action; domain subclasses supply the action's parameters."""

    model_config = ConfigDict(frozen=True)


class Theta(BaseModel):
    """Sampled skill parameters used to construct and evaluate a policy."""

    model_config = ConfigDict(frozen=True)


STOP_ACTION = POMDPAction()
NUM_SAMPLES = 1


class BeliefSpaceModel(Protocol):
    """Domain implementations of the pseudocode's seven model functions."""

    def sample_theta_from_belief(self, *, belief_state: BeliefState) -> Theta: ...

    def evaluate_policy(self, *, sampled_theta: Theta) -> float: ...

    def score_pomdp_value_from_policy_value_and_cost(
        self, *, policy_value: float, summed_cost: float
    ) -> float: ...

    def get_valid_actions(self, *, environment_state: EnvironmentState) -> list[POMDPAction]: ...

    def sample_next_states(
        self,
        *,
        environment_state: EnvironmentState,
        practice_action: POMDPAction,
        belief_state: BeliefState,
    ) -> list[tuple[EnvironmentState, float]]:
        """Return potential next environment states and their sampled costs."""
        ...

    def update_belief_state(
        self,
        *,
        belief_state: BeliefState,
        environment_state: EnvironmentState,
        potential_next_environment_state: EnvironmentState,
        practice_action: POMDPAction,
    ) -> BeliefState: ...

    def transition_probability(
        self,
        *,
        potential_next_environment_state: EnvironmentState,
        sampled_cost: float,
        environment_state: EnvironmentState,
        practice_action: POMDPAction,
        belief_state: BeliefState,
    ) -> float: ...


class BatchedBeliefSpaceModel(Protocol):
    """Optional hot-path extensions; the seven pseudocode methods remain canonical."""

    def sample_thetas_from_belief(
        self, *, belief_state: BeliefState, num_samples: int
    ) -> list[Theta]: ...

    def transition_outcomes(
        self,
        *,
        environment_state: EnvironmentState,
        practice_action: POMDPAction,
        belief_state: BeliefState,
    ) -> list[tuple[EnvironmentState, float, float]]: ...

    def search_cache_key(
        self,
        *,
        environment_state: EnvironmentState,
        summed_cost: float,
        belief_state: BeliefState,
        horizon: int,
    ) -> object: ...
