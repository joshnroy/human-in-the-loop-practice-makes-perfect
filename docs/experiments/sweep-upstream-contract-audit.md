# Pinned Sweep contract audit — September29

Josh's standing requirement: directly reuse pinned upstream predicates, state abstractions, skill/controller implementations, parameter semantics, preconditions/effects, termination and success behavior whenever possible. Exact clones require a concrete technical reason and parity evidence. Keep explicitly approved experiment deviations named; never silently strengthen an upstream predicate or redesign its controller to compensate for adapter mistakes.

Source pin: kinder-baselines427ad6cdfcc79efb489ebabc8fd39d77e104bb4e. No upstream checkout was modified.

| Contract | Upstream implementation | Previous adapter | Disposition |
|---|---|---|---|
|DrawerOpen|state_abstractions.py75: pos>0.04|pos>=0.15|Unintended divergence introduced da719cbe; restore direct reuse|
|DrawerClosed|else branch complement of Open|pos<0.01|Restore exact complement; precise reset closure remains distinctly named physical diagnostic|
|HandEmpty|np.isclose(gripper,0,atol=.001)|gripper<.2 plus geometric no-holding checks|Restore upstream; physical recovery hand state must be distinct|
|Holding(robot,wiper)|gripper>.1, no geometric check|gripper>.2 and EE/wiper geometry|Restore upstream alias HoldingWiper; geometric hold needs separate name|
|OnTable(movable)|z>.45; wiper excluded when gripper>.1|substituted declared region/yaw InPile or WiperHome|Add exact mapped OnTable facts; retain region facts solely for approved recovery target|
|InDrawer(movable,drawer)|DrawerOpen and movablez<.4|exact declared physical goal containment|Restore exact upstream name; floor can satisfy original heuristic. Primary symbolic vs physical-goal reporting conflict sent to parent|
|OpenDrawer preconditions|HandEmpty,OnTable(wiper),all5OnTable,DrawerClosed|HandEmpty,DrawerNotOpen|Restore original stock operator|
|PickWiper preconditions|HandEmpty,OnTable(wiper),all5OnTable,DrawerOpen|HandEmpty,WiperHome|Restore original stock operator|
|Sweep preconditions|Holding(wiper),all5OnTable,DrawerOpen|HoldingWiper,DrawerOpen,AnyCubeInPile,all5SweepReachable|Restore original stock operator; do not silently retain alternative partial-progress policy|
|Stock effects|Open addsOpen/deletesClosed;Pick addsHolding/deletesHandEmpty+OnTable(wiper);Sweep adds5InDrawer/deletes5OnTable|different derived recovery/helper effects|Restore shared effects; reconcile distinctly named helper consequences separately|
|Original abstract goal|Holding(wiper),DrawerOpen,5InDrawer|5physicalcontainment facts only|Conflict surfaced; no silent choice of scientific primary objective|
|Stock controllers|actual imported create_lifted_controllers,reset/step/observe/terminated|directly reused|Preserve; NO geometry/stroke repair implemented|
|Stock planning wrapper|native collision sets|inherited prototype filters movable base obstacles outsidez[.05,.15],removes island slab in armchecks|Deviation predatesintegration; explicit disposition requested|
|Parameter supports|all3singleton distance/rotation|approved nondegenerate supports|Explicit Josh-approved deviation retained|
|After Sweep termination|upstreamcontroller terminates directly|adapter15settleticks inherited fromprototype attempt|Audit requires explicit disposition; not silently part of stock termination|
|Execution cap|controller has no intrinsic task loopcap; upstreamtests300/300/200ticks|wrapper600ticks|Bounded failure guard documented; no success asserted at timeout|

## Evidence and preservation

Approved-range calibration72/72 contexts and both1cycle×20action smokes are INVALID as faithful-upstream readiness evidence because predicates/operators differed. Preserve all raw records under scratchpad/sweep-integration/approved-calibration-* and approved-{ees,pomdp}-smoke. Do not use their priors for experiment launch. Recalculate original Open predicate from final raw states gives16/24 (seed0:0/8,seed1:8/8,seed2:8/8); this is a diagnostic rescore, not an overwrite. The11cm stockstroke is consistent with upstream4cm threshold; prior assertion that it must reach15cm was based on our adapter error.

## Direct reuse implementation underway

upstream.py uses the actual Sweep3DStateAbstractor for every observed fact and goal. Its constructor normally resets a simulator to obtain object metadata; a read-only snapshot facade returns a copy of the current raw state, so no live reset occurs. Native private PyBullet simulator is explicitly closed.25/25 drawer/gripper boundary combinations agree with original methods, and live MuJoCo qpos is unchanged before/after. Boundary tests compare storedfloat32 state values, preserving original rounding behavior at .001/.04/.1.

The stock operators are embedded inside upstream create_bilevel_planning_models, whose factory creates/resets another live environment and constructs an entire planner. Our distinct planning framework requires conversion to its own Skill types. Prefer extracting actual native GroundOperators once from an isolated model factory if lifecycle can be closed cleanly; otherwise exact declarations need explicit parity against that native factory, with no scientific reinterpretation.
