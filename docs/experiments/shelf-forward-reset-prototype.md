# Shelf-o8 forward and reverse prototype

## Scope

Josh authorized developing a physical forward/reset demonstration in parallel with
the native bilevel benchmark. Neither track authorizes a learning launch. This is
an isolated diagnostic prototype, not a production Shelf adapter.

Native KINDER is pinned at `8f600231`; native controllers at `427ad6c`.
Forward scripts directly invoke those unchanged controllers, samplers, predicates,
and physical goal checks. The scripted object order and optional selection among
native placement samples are demonstration policies, not bilevel benchmark scores.

## Results at the current checkpoint

- A fixed seed-1 scripted forward run executed 16/16 native Pick/Place actions with
  their symbolic effects. Only 5/8 cubes satisfied the physical task goal: two
  were stacked and one tilted. The 8/8 OnFixture atoms do not establish success.
- A second script choosing clearance among 20 native placement samples reached
  6/8 physical goals, then dropped cube 7. Fixed seeds 2 and 3 reached 1/8 and 2/8
  physical goals before planning failed. All outcomes remain recorded.
- The new single-cube retrieval controller completed in 1640 control ticks from
  the actual recorded first-placement state. Both native Holding and controller
  termination passed. This is an isolated replay diagnostic, not an eight-cube
  reset or a continuous full-task proof.
- Ground return completes with native OnGround, HandEmpty, the initial position
  region, and arm home. Its latest measured-object feedback version took 2496
  control ticks. Final yaw error is 1.45 degrees from the declared zero-yaw start;
  full start-distribution restoration is therefore not claimed.
- Packed eight-cube retrieval remains blocked: the tested horizontal grasp and
  preshaping scans found 0/1152 collision-clear candidates each.

## New reverse skills

`shelf_reverse_controllers.py` defines individually counted retrieval and ground
return skills. It reuses native base and arm motion planning. It does not modify
the live state except by robot actions, relax task predicates, change forward
parameter ranges, or remove collision objects.

Shelf retrieval needs explicit support separation: a settled cube has 0.212 mm
initial contact with its supporting shelf. The new skill permits only that
existing upward support contact to separate monotonically; all robot and other
object collisions remain checked. The full native held-object checker passes
after 0.25 mm of upward motion. It then uses the native planner again.

Reverse candidates are additionally checked against the actual MuJoCo joint
limits. PyBullet IK produced an unreachable -2.29256-radian joint-2 target in an
earlier failed candidate. No joint limit was widened. Checked-waypoint execution
maintains base feedback and waits for convergence.

Ground return targets the original valid sampled cube pose, aligns the measured
held cube above the floor, lowers it, and requires real cube/ground support inside
the native region before opening and retracting. Remaining yaw drift is retained
as a reset-definition failure, not hidden behind the broad OnGround predicate.

## Evidence and limitations

Raw states, real videos, candidate rejection logs, and per-action counts live in
the worktree's `scratchpad/shelf-forward-reset/`. `CHECKPOINT.md` names each
version. Native forward controller tests use a 400-tick diagnostic cap; reverse
prototypes use a separately reported 3000-tick cap. These are not the 100-tick
native bilevel benchmark and are not approved Shelf learning horizons.

No continuous eight-cube forward-plus-reset success video exists. No robot-home
skill or production adapter is complete. A corrected native baseline, reliable
physical forward behavior, full reverse proof, and explicitly frozen learning
protocol are all required before Ours versus EES.
