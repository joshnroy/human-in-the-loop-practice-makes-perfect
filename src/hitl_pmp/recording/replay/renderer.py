import abc
from typing import ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict

from hitl_pmp.recording.run_record.types import PlainState


class ReplayRenderer(BaseModel, abc.ABC):
    """An environment's drawing of a recorded state.

    `show` must put the drawing scene into exactly `state` and draw it -- it may not
    step physics -- so playback depends on nothing but the recording."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # A storyboard draws one frame per action from a recording that holds only each
    # action's end state: an action that changed nothing (and so recorded no state)
    # still gets a frame, on the scene as it stands.
    storyboard: ClassVar[bool] = False

    def is_storyboard(self) -> bool:
        """Whether this recording is played as a storyboard; a renderer whose recordings
        may or may not carry motion answers per recording."""
        return self.storyboard

    @abc.abstractmethod
    def fps(self) -> int: ...

    @abc.abstractmethod
    def show(self, *, state: PlainState) -> np.ndarray: ...

    @abc.abstractmethod
    def current_state(self) -> PlainState:
        """The drawing scene's state, read back -- what a round-trip check compares."""

    def close(self) -> None:
        return None
