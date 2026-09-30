# Simple pickup observation/model parity correction

The production seed-2 pickup `production-pick-seed2-v244` completed its physical lift at tick 268 with bilateral pad contacts. Ordinary action dispatch nevertheless reported failure with an empty error because the operator asserted `RobotAway`. The native observed state was `HoldingWiper=True`, `RobotHome=True`, `RobotAway=False`. Diagnosis is preserved in `scratchpad/sweepsimple3d/production-pick-seed2-v244-postcondition-audit.json` (read-only recorded-state restoration, rounded qpos and zero velocities; no physics steps).

The corrected pickup operator guarantees HoldingWiper and leaves RobotHome/RobotAway unknown in symbolic prediction through the existing ignore_effects mechanism. Actual execution continues to observe both native base predicates unchanged. Neither predicates, native region checks, physical grasp requirements, task goal, solver nor costs are relaxed. The dispatch regression covers held and unheld outcomes with home and away bases, and verifies observed base facts remain intact.

The same authorized correction applies to SweepCubeToGoal and SweepCubeToStart: their geometry-dependent stance cannot guarantee RobotAway. Both now ignore predicted base predicates while requiring their existing cube add effects. Regression tests cover attained and unattained target conditions while the base remains observed home. ReturnRobotToStart and the human reset retain their explicit home guarantees.

Fresh native proofs remain required after the model correction. This is an integration bug fix, not scientific success evidence.

## Loaded native joint reserve correction

Authorized before implementation: production seed1 v246 reached 5/5 goals, then reverse contact motion exhausted joint_2 reserve. Final observed 2.2401583195 exceeds native 2.24 by .0001583195 rad. Last selected approach was valid with .020914 rad reserve; loaded corrections subsequently consumed it. Introduce a .01 rad loaded-only reserve guard at both base drive and corrective descent, ending the stroke into the existing checked retreat. Keep strict native limits and observed-state validation; no clipping or limit/physics changes. Any recorded-state v249 replay is counterfactual diagnostics, with nine prior actions retained and one retry counted as action ten, not readiness evidence.
