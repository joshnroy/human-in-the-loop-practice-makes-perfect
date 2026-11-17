"""The rack's controller-level action IDs and parameter bounds.

`RackServingEnvironment.take_action` dispatches these IDs to the controllers; the
validation and probe scripts drive the scene through them. They are not planner
actions: the planner's skills are the chained toy's (`toy_interface.py`), which call
these controllers (`RackServingToyEnvironment`).
"""

SKILL_IDS = {
    "PickScraper": 0,
    "GatherAndSweep": 1,
    "ReturnScraper": 2,
    "PickFood": 3,
    "PlaceFoodOnTray": 4,
    "ReturnFoodToRack": 5,
    "Sweep": 6,
    "RobotBatchReturn": 7,
    # `split_gather_and_sweep` only.
    "Gather": 8,
    "Sweep5": 9,
}
# GatherAndSweep(s, delta, d): s (m) is how far behind the rack's front edge the two
# x-squeeze passes stop; delta (m) is how far from the line y = c (the plate centre's
# y) each y gather stops, an absolute distance whatever the food; d (m) is how far
# before the plate's outer near edge the final sweep's blade edge ends. Starting
# bounds from the driven-blade grid (2026-11-03), not calibrated values. Speed is a
# constant, not a parameter.
# delta floor 0.005 (Josh, 2026-11-03): from the +-0.85 side stances the deepest
# stop that lifts cleanly on the arm is 0.5 cm from c; 0 cm failed. The range is the
# 2026-11-03 calibration's stops (then 0.045 m minus delta, delta in [-0.03, 0.04]),
# made absolute (Josh, 2026-11-04) so a food's size no longer moves them.
# d floor 0 (Josh, 2026-11-12), the planner's own limit, so that wrong values of d
# exist once the plate is tucked under the rack (`RackServingConfig.plate_tuck`). The
# blade stops at rack front + plate_tuck - d: with the well under the rack's front edge
# every stroke that reaches that edge delivers, and only one that stops short of it
# fails. Measured on middle_three (sides 130 mm), tuck 40 mm, 20 env seeds per d:
# Sweep lands exactly three on 0/20 at d = 0, 0.01, 0.02, 0.03 and 0.035 (the pieces
# stay on the rack or overhang its edge), and on 20/20 at every d from 0.04 to 0.20;
# 0.24 is refused by the fixture-collision check. No refusal and no scraper-rack
# contact anywhere in [0, 0.20]. About 0.69 of [0, 0.12] works.
GATHER_AND_SWEEP_BOUNDS = ((0.03, 0.14), (0.005, 0.075), (0.0, 0.12))
# Sweep(d): GatherAndSweep's final stroke alone, over the same d bounds; offered unless
# `RackServingConfig.sweep_skill` is False.
SWEEP_BOUNDS = (GATHER_AND_SWEEP_BOUNDS[2],)
SWEEP_SPEED_M_S = 0.06
