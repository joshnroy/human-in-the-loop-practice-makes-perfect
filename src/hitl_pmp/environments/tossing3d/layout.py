"""Named scenes; the original benchmark always remains the default."""

from enum import Enum
from pathlib import Path


class Tossing3DLayout(str, Enum):
    BARRIER = "barrier"
    SAME_SIDE = "same-side"

    def task_config_path(self) -> Path | None:
        if self is Tossing3DLayout.BARRIER:
            return None
        return Path(__file__).parent / "scenes" / "same_side.json"
