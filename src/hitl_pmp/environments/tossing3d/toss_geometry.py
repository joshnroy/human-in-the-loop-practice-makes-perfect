"""Where the toss pipeline's geometry comes from: the simulator, or the real room.

`TossDirectionSelector` and `TossStandoffBand` are written against one
`TossFeasibilityGeometry` plus one "is this stand reachable" check. Which backend a
state came from is told by the snapshot it carries (`Tossing3DState.object_centric`): a
`RealSceneSnapshot` is the real room, anything else is KINDER's. This is the one place
that distinction is made, so the selection rule itself is shared, not forked.
"""

from typing import Any

from .kinder_backend import KinderBackend
from .real.geometry import RealTossGeometry
from .real.types import RealSceneSnapshot
from .types import TossFeasibilityGeometry


class TossGeometrySource:
    """A static-method container, never instantiated."""

    @staticmethod
    def is_real(*, snapshot: Any) -> bool:
        return isinstance(snapshot, RealSceneSnapshot)

    @staticmethod
    def geometry(*, snapshot: Any) -> TossFeasibilityGeometry | None:
        if isinstance(snapshot, RealSceneSnapshot):
            return RealTossGeometry.of(snapshot=snapshot)
        return KinderBackend.toss_feasibility_geometry(snapshot=snapshot)

    @staticmethod
    def base_plan_failure(*, snapshot: Any, distance: float, rotation: float) -> str | None:
        """The sim asks its base motion planner. The real room's gate already holds
        everything a stand can fail on -- room, obstacles, the uncrossable barrier --
        so there is nothing further to ask (see `real/geometry.py`)."""
        if isinstance(snapshot, RealSceneSnapshot):
            return None
        return KinderBackend.toss_base_plan_failure(
            snapshot=snapshot, distance=distance, rotation=rotation
        )
