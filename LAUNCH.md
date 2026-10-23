# SweepOffShelf3D launch gates, 2026-10-23

Deliberate gate lowering: #539's neighbour refusals leave ~1/4 of shelf picks refused;
fix (b) in progress; Josh 2026-10-23.

- `PickCubeFromShelf`: at least 72/100. The 6 mm wiper-margin probe measured 75/100;
  the gate is that measurement minus a small allowance.
- `ReturnCubeToShelfFromBox`: unchanged, at least 5/8 of the harvested family.
- `PickCubeFromFloor`, `ReturnCubeToShelfFromFloor`: unchanged, 11/11.
- Every other skill: unchanged.

The gates live in `skill_benchmark/verdicts.py` (`THRESHOLDS`).
