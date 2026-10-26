# Charged post-toss settling

A toss can terminate while the released cube is still moving. If `OnGround` is
false, the practice policy can ask for a human reset and evaluation can report
`no_plan` before the cube lands. The controller now holds the final robot pose
for up to two simulated seconds, ending after three consecutive ticks with cube
linear speed at most 0.01 m/s and angular speed at most 0.1 rad/s.

Settling uses the existing controller execution loop. Its ticks are recorded,
charged, and interrupted by practice/evaluation budgets. The ordinary toss
controller limit also bounds settling. Upstream failure and timeout hooks remain
visible, so an aborted toss is still reported as a failure.

The regression fixtures preserve three distinct pre-reset states from pilot
`t3-pour-pilot-20261025`, revision
`0fa7c204209ad2f32102cd5b40b37cd785398c6c`. The tests start at the recorded
end of the upstream toss, then execute settling and picking with native physics.
The fixture scene explicitly preserves its movable 1 kg bin and 100 g cube;
this change does not alter the production scene or dependency revisions.

Picking a cube supported by the robot's base remains excluded by the existing
skill preconditions. That is intentional. Picking cubes inside the bin remains
supported.
