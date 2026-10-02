# Sweep recovery integration repair

This revision fixes the planner interface to the existing robot recovery controllers.
It does not make the native task controllers experimentally ready. Keep the original
restricted-recovery runs separate from any replacement experiment.

## Recovery contracts

- Cube picks require an observed plan from the same fixed pick controller used at execution.
- Parking requires physically holding the wiper; the upstream closed-gripper classifier
  is unchanged.
- Recovery opening uses the existing drawer controller's target and tolerance.
- Explicit release is available for a closed hand, including when holding cubes. It
  guarantees no destination for released objects.
- Each drawer-wiggle action targets one cube becoming physically pickable. The shared
  existing wiggle controller executes unchanged; observed target feasibility determines
  success. No action asserts that every cube becomes pickable.
- Floor-wiper pickup reuses the physical recovery controller. Success includes its
  collision-checked return to the arm home pose. The existing stow tracker gets a larger
  bounded tracking allowance while retaining its endpoint tolerance and collision path.
- Pair and three-cube row pick/place actions are distinct fixed skills. Exact row membership
  must match the candidate grasp. Every held cube has its own observed transform; a partial
  release cannot count as successful placement of the whole group.

The three trainable task skills retain their exact upstream operators, predicates,
controllers, success criteria, and approved parameter supports. All recovery actions are
fixed, separately selected and charged at the existing default robot skill cost. No scripted
full-reset policy is exposed as one action. Native configured cost is per skill rather than
elapsed simulation ticks; event logs retain actual ticks and measured cost observations
continue through the unchanged runner.

## Verification and remaining launch gate

Run `scripts/sweep_repair_env.sh pytest tests/environments/sweep_drawer3d` with a memory cap.
The wrapper pins imports to this checkout's populated reference trees. Physical adapter
regressions record state and replay logs and reject any controller error, even if some
symbolic goal facts happened to be achieved before the error.

`analysis/sweep_recovery_readiness/native_check.py` checks the unchanged nominal native
sequence on the accepted practice and evaluation seeds. Those diagnostics and the existing
native collision investigation remain a separate readiness gate. They do not establish
that every parameter in the approved ranges is infeasible. No launch should silently
replace native controllers, exclude valid starts, or restore collision overrides.

One inherited regression fixture assumes native pickup succeeded on seed eleven before
asking the wiper-parking controller to stow the tool. That assumption is false under the
required native collision model. Preserve this failure as a documented fixture limitation;
it is not evidence that an unheld tool can be parked.

Recorded diagnostic clips can be rendered with
`analysis/sweep_recovery_readiness/render_diagnostics.py`. These are explicitly labeled
constructed diagnostic fixtures, not episodes from the learning comparison.
