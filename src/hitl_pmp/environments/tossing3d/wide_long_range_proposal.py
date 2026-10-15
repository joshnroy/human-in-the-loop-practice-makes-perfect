"""The toss's only proposal: one draw for every receiver side and both layouts.

All three learned parameters -- standoff, speed, release -- are drawn jointly and
independently: speed and release over the controller's full ranges, standoff over
the derived shared band (see the bounds' comment; under the graded receiver region
it equals the controller's full range). There is no yaw: where the robot stands
about the bin is chosen per state by `TossDirectionSelector`, not drawn here. The
interface is deliberately identical for both bin sides: the fixed far-receiver
launch pose (standoff 2.5 m, yaw 0) that survived
from the calibrated proposal's development ridge was a point mass that (a) made
far evaluation a 2-parameter problem while robot-side practice explored four,
(b) starved refits of far-side variety, and (c) kept every "far" number tied to
one geometry. Its removal intentionally supersedes the far-receivers-byte-
identical contract two earlier PRs pinned.

History of the support: the calibrated predecessor coupled release to speed
along its development ridge, so an untrained learner started at 10/10 and
practice had nothing to teach it (the calibration survives as `release_ridge`).
The 2026-09-22 far coverage sweep and the 2026-09-23 controller-wide
short-standoff probe together show scoring support across the standoff range on
anti-correlated (speed, release) ridges, alongside a large region of completed
physical misses -- which is what makes learning real on both sides.
"""

import numpy as np

from .sides import BIN_RESET_REGION_BY_SIDE, Tossing3DSide
from .skills import (
    TOSS_DISTANCE_BOUNDS,
    TOSS_RELEASE_MS_BOUNDS,
    TOSS_SPEED_BOUNDS,
)
from .types import TossParameterSet

# The easternmost legal stand x, measured 2026-09-23 through the geometry gate
# itself (0.55 m robot footprint against the barrier at x = 1.3: the shortest
# gate-accepted standoff at the natural far bin put the stand at x = 0.994; see
# the standoff-floor PR). Not derivable from constants in this package -- the
# barrier pose and robot footprint live in the scene -- so it is pinned here
# and a change to either upstream quantity must move it. The 2026-09-24
# feasibility map confirmed it with the real base planner: a far bin is
# plannable exactly when standoff >= bin_x - 0.994 (away from the barrier-edge
# corners). 0.99 keeps every per-bin band 4 mm inside that line.
FAR_STAND_X_LIMIT_M = 0.99

# Speed and release are the controller's own ranges. Standoff's floor is
# DERIVED: max(the controller's own floor, the nearest far bin
# (`BIN_RESET_REGION_BY_SIDE[OPPOSITE]`'s x-min) minus the legal standing line
# above) -- so every draw's standoff is one a far receiver could also demand,
# and the clamp keeps the derivation meaningful whatever the receiver region.
# Under the far-only region (x-min 2.6) this floor was 1.61 and cut the
# regime mismatch that held far evals at 2/10; under the graded receiver
# region (x-min 1.48, the restored near-barrier support) the far term is 0.49
# and the CONTROLLER floor binds, so the same-side and far-side feasible
# standoff bands coincide at [1.25, 2.6] exactly.
WIDE_TOSS_STANDOFF_BOUNDS = (
    max(
        TOSS_DISTANCE_BOUNDS[0],
        BIN_RESET_REGION_BY_SIDE[Tossing3DSide.OPPOSITE].ranges[0][0] - FAR_STAND_X_LIMIT_M,
    ),
    TOSS_DISTANCE_BOUNDS[1],
)
WIDE_TOSS_SPEED_BOUNDS = TOSS_SPEED_BOUNDS
WIDE_TOSS_RELEASE_MS_BOUNDS = TOSS_RELEASE_MS_BOUNDS

# The simulator's parameter set: exactly the three constants above, so a sim run draws
# what it always drew.
SIM_TOSS_PARAMETERS = TossParameterSet(
    standoff_bounds=WIDE_TOSS_STANDOFF_BOUNDS,
    speed_bounds=WIDE_TOSS_SPEED_BOUNDS,
    release_ms_bounds=WIDE_TOSS_RELEASE_MS_BOUNDS,
)

# The real TidyBot's set, derived from the real swing (2026-10-14): what the real
# robot's `toss` accepts, and nothing a real run draws may fall outside it. It differs
# from the sim's on all three axes; the sim's bounds are untouched by it.
REAL_TOSS_PARAMETERS = TossParameterSet(
    standoff_bounds=(0.9, 1.45),
    speed_bounds=(100.0, 140.0),
    release_ms_bounds=(500.0, 850.0),
)


class WideLongRangeTossProposal:
    """A state-independent proposal; it neither simulates nor supplies labels."""

    @staticmethod
    def release_ridge(*, speed: float) -> float:
        """The calibrated proposal's release center for a speed, in milliseconds."""
        return 450.0 + (420.0 - speed) / 3.0

    @staticmethod
    def far_standoff_bounds(
        *, bin_x: float, parameters: TossParameterSet = SIM_TOSS_PARAMETERS
    ) -> tuple[float, float]:
        """The standoff band a bin across the barrier can actually be thrown at from.

        The robot must stand west of the barrier (x <= `FAR_STAND_X_LIMIT_M`) and west
        of the bin, `standoff` from it, so any standoff below `bin_x - limit` has no
        stand at all. Computed from the bin pose before drawing, so nothing is
        rejected and redrawn -- a filtered draw would silently reshape the pool's
        standoff distribution. For a near-barrier bin the controller floor binds.

        Sim geometry only: `FAR_STAND_X_LIMIT_M` was measured in the sim's room. A real
        bin's band comes from the four-direction scan instead (see `toss.py`)."""
        return (
            max(parameters.standoff_bounds[0], bin_x - FAR_STAND_X_LIMIT_M),
            parameters.standoff_bounds[1],
        )

    @staticmethod
    def sample(
        *,
        rng: np.random.Generator,
        standoff_bounds: tuple[float, float] | None = None,
        parameters: TossParameterSet = SIM_TOSS_PARAMETERS,
    ) -> np.ndarray:
        """Return metres, degrees/second, and milliseconds -- all three drawn
        jointly and uniformly from `parameters`, the standoff over `standoff_bounds`
        (the set's full band unless a per-bin band is supplied)."""
        standoff = float(rng.uniform(*(standoff_bounds or parameters.standoff_bounds)))
        speed = float(rng.uniform(*parameters.speed_bounds))
        release = float(rng.uniform(*parameters.release_ms_bounds))
        return np.array([standoff, speed, release])

    @staticmethod
    def contains(*, params: np.ndarray, parameters: TossParameterSet = SIM_TOSS_PARAMETERS) -> bool:
        """Whether a `(standoff, speed, release)` row lies in the closed support."""
        standoff, speed, release = (float(value) for value in params)
        return parameters.violation(params=(standoff, speed, release)) is None
