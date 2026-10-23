"""The one tie rule both belief-space planners compare values and path costs by."""

import math
from typing import ClassVar


class ValueResolution:
    """A static-method container, never instantiated.

    Objective values J and determinized path costs g are compared on a grid of
    `RESOLUTION`. One control step's cost in the objective is lambda * price = 3e-6,
    three orders of magnitude above it; roundoff from summing the same costs in another
    order (or at another, mathematically equivalent, cost scale) is near 1e-16, seven
    below. So two quantities equal in exact arithmetic compare equal, and the planner's
    own tie rule -- STOP first, then the first action or path found -- decides, never
    their last bits. A grid rather than a pairwise tolerance keeps a total order, which
    a heap needs."""

    RESOLUTION: ClassVar[float] = 1e-9

    @staticmethod
    def rank(*, value: float) -> float:
        """`value` on the grid; an infinity stays itself and so ranks outside it."""
        if not math.isfinite(value):
            return value
        return float(round(value / ValueResolution.RESOLUTION))
