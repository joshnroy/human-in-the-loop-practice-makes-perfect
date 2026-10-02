"""Whether the bin rests upright on the floor: the classifier behind `BinOnGround`.

Evaluated once per state at the boundary, by `KinderBackend.abstract_atoms`, because it
reads the bin's orientation, which the flat `core.State` does not carry (the bin's
state row is its position plus the scored box). `predicates.BIN_ON_GROUND` looks the
resulting atom up, the same way the five upstream-backed predicates do.

## Why not upstream's `OnGround`

Upstream's `Tossing3DStateAbstractor._check_on_ground` has a strict branch for a
non-cube movable (`|qx|, |qy| < ON_GROUND_TOLERANCE`), but upstream only evaluates it
for cubes, and its height test (`z - bb_z / 2` within 0.05 m of 0) assumes a
centre-origin body. The bin's origin is at its base -- an upright bin at rest reads
z ~ 0 with bb_z = 0.2 -- so that test is false for every upright bin, and its 0.05 m
tolerance would also accept a bin resting flat on top of the 0.05 m cube. So it is not
reusable here; this is the local classifier, and moving it upstream is follow-up work.

## The two thresholds, measured

Resting bin states (linear speed < 1e-3 m/s, angular < 1e-2 rad/s) across all 18 EXP-21
`tossing3d_state_log.jsonl` files (six arms x seeds 0-2), tilt = acos(1 - 2 (qx^2 + qy^2)).
"Poses" are physically distinct resting poses, deduplicated across arms that share a
seed's prefix:

| population | count | tilt (deg) | z (m) |
| --- | --- | --- | --- |
| upright on the floor | 2,315,817 ticks | 0.0 to 0.2 | -0.001 to 0.000 |
| on top of the cube, flat or nearly | 7 poses (+186 ticks < 1 deg) | 0 to 7.1 | 0.047 to 0.049 |
| propped on the cube by one edge | 18 poses | 2.7 to 7.3 | 0.007 to 0.019 |
| leaning on the cube | 132 poses | 11.3 to 21.3 | 0.029 to 0.054 |
| on its side | 89 poses | 90.0 | 0.150 |

Of 2,317,679 resting ticks, the 1,862 that are not upright all have z >= 0.007 m and none
has 0.001 < z < 0.007, so height alone separates every observed population;
`BIN_ON_GROUND_MAX_HEIGHT_M` sits in that gap. Tilt alone separates all but a bin lying
flat on the cube; `BIN_ON_GROUND_MAX_TILT_DEG` is 1 degree, five times the largest upright
resting tilt and below the smallest tilted resting pose (1.23 deg, on the cube). Both are
required, so each guards the other's blind spot: a bin balanced flat on the cube fails
height, and a bin rotated off upright where it stands would fail tilt.
"""

import math
from collections.abc import Mapping

# The atom name `KinderBackend.abstract_atoms` emits and `BIN_ON_GROUND` looks up.
KB_BIN_ON_GROUND = "BinOnGround"

# See the module docstring's table: upright |z| <= 0.001 m; nothing resting on anything
# else is below z = 0.007 m.
BIN_ON_GROUND_MAX_HEIGHT_M = 0.004

# See the module docstring's table: upright tilt <= 0.2 deg; the smallest tilted resting
# pose is 1.23 deg.
BIN_ON_GROUND_MAX_TILT_DEG = 1.0


class BinOnGroundGeometry:
    """A static-method container, never instantiated."""

    @staticmethod
    def tilt_deg(*, bin_: Mapping[str, float]) -> float:
        """The angle between the bin's up axis and the world's, in degrees."""
        cos_tilt = 1.0 - 2.0 * (bin_["qx"] ** 2 + bin_["qy"] ** 2)
        return math.degrees(math.acos(max(-1.0, min(1.0, cos_tilt))))

    @staticmethod
    def holds(*, bin_: Mapping[str, float]) -> bool:
        """Upright (face up) and resting on the floor rather than on anything else."""
        return (
            abs(bin_["z"]) <= BIN_ON_GROUND_MAX_HEIGHT_M
            and BinOnGroundGeometry.tilt_deg(bin_=bin_) <= BIN_ON_GROUND_MAX_TILT_DEG
        )
