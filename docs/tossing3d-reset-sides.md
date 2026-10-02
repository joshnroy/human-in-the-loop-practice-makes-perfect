# Tossing3D reset destinations

Human and automatic movable resets each offer two groundings of the same lifted
operator: put the bin on the robot side or the opposite side. The cube always
returns to its initial region. `RobotAtSide`, `CubeAtSide`, and `BinAtSide` bind
these choices without adding physical objects to the simulator. The robot pose
and gripper command remain unchanged. Resetting a held cube removes `Holding`
and adds `ClosedEmpty`, so the planner opens the gripper before picking again.

The two destinations share the belief and configured cost of their reset
mechanism. Human and automatic resets retain separate belief identities and
cost configuration. Same-destination symbolic self-loops are still pruned;
changing the destination is a distinct state transition.

Opposite-side resets preserve the current far-bin center range x=[2.60, 3.42],
y=[-2.3, 2.3]. Robot-side resets use x=[-0.9, 0.2], y=[-1.0, 1.5] with the receiver
reversed -- the measured maximal grasp-safe rectangle, east of the grasp-planner
refusal zone at x <= -1.0 (boundary probes in sides.py).
The reset adapter preserves center-region support while retaining the simulator's
room and obstacle checks. Omitting the destination still uses the scene's initial
placement region. Every barrier toss draws the wide long-range proposal
regardless of the grounded target side: the per-side calibrated proposal
selection was removed together with the proposal choice itself.

This integration reapplies the committed reset-side work from `b75a5709` onto
merged `755eeeae`. It retains the newer simulator/controller pins, empirical
failed-action model, S/F-only inference, smoothing, learner refit bookkeeping,
calibrated throw witnesses, and conditional gripper reset effects. The older
branch's short-range geometry, unconditional release-on-failure assumption,
and removal of automatic resets are superseded. Adding destination choices
changes the practice action space; matched experiments without this factor
must run on the merged baseline instead.
