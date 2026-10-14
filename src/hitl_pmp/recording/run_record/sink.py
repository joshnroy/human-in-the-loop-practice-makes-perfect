import abc
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel


class SkillTickSink(BaseModel, abc.ABC):
    """Where an environment reports the skill it is executing and every physics tick
    that skill produced. The environment knows the ticks; only the caller that
    dispatched the action knows its outcome, so the two arrive separately."""

    @abc.abstractmethod
    def record_skill(
        self,
        *,
        name: str,
        objects: Sequence[str],
        params: Sequence[float],
        controller: Mapping[str, Any],
    ) -> None: ...

    @abc.abstractmethod
    def record_tick(
        self, *, state: Mapping[str, Sequence[float]], diagnostics: dict[str, Any] | None = None
    ) -> None: ...
