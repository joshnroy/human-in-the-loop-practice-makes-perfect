# Independent practice measurements

The optional Tossing3D step protocol counts each executed controller step once and
each human invocation by `--human-skill-steps` (default 1). Reset implementation
physics and held-out evaluation do not advance the practice counter. Planner costs
are unchanged.

`--practice-step-budget 85000 --measurement-interval-steps 1700` replaces the total
learning-cycle count with a physical interaction budget. Existing session boundaries
and `--max-steps-per-interaction 20` still govern learning. At each measurement
threshold, including one reached inside a controller, the runner serializes the
current deployable policy and queues evaluation in a spawned process. It does not
refit, end the practice session, or reset the practice environment. A final partial
skill at the total budget is recorded as interrupted (`interrupted`, `interruption`,
and a non-null `skill_error`), without labeling its outcome as a completed training
example; the environment adopts the simulator's actual state at the interruption. A
human request that no longer fits in the remaining budget ends practice the same way
(`human_refused_at_budget`), and the final measurement still runs. One increment that
crosses several measurement thresholds -- a human invocation longer than the interval
-- measures once per threshold, each at its threshold's practice step.

The structured arms both inherit EES deployment. Snapshots contain their sampler
classifiers, competence estimates and RNG state, with symbolic skills rebuilt by
identity against the evaluation environment. The evaluator has its own policy and
environment and returns results only to logging. It uses the native goal criterion,
no human assistance, and `--evaluation-control-steps 500`. A no-plan or zero-physics
controller failure ends an evaluation rather than spinning without consuming time.

`step_events.jsonl` records actual robot execution ranges, human reset destinations,
measurement points and learning boundaries. `measurements.json` attributes each
snapshot to its practice-step count; `evaluations/<index>/results.json` becomes
available as that asynchronous sweep finishes. `step_evaluations.json` joins them at
run completion. `stats.json` records each evaluation at its measurement's
`practice_steps` (robot control steps plus human-skill steps), so its x values are
unique and increasing; `robot_invocations` stays in the measurement records. Runs made
before this change recorded `stats.json` on skill invocations, where a mid-skill and a
final measurement could share an x. State logs support later video replay.

Each evaluation runs in its own freshly spawned process, one at a time, so simulator
memory is released between evaluations. If practice fails or is interrupted, queued
evaluations are cancelled and the running one is terminated; `step_evaluations.json`
marks them `"cancelled": true`. A completed run still waits for every evaluation.

The evaluator latches success on the first control step at which the native goal
holds and stops the controller there, so a cube passing through the goal region
mid-flight counts as solved. The incumbent evaluation scores the state after the
whole skill. The two success rates are therefore not the same measurement.

Use `scripts/with_step_env.sh` to resolve the runner and both simulator packages
inside the isolated checkout. Copy or populate that checkout's dependencies before
running; the wrapper does not install packages. Verify package import paths before launching; they must not
resolve to a shared checkout. `--defer-rendering` and
`--practice-reset-policy never` are required for this protocol.

Validation covers mid-skill boundaries, human accounting, step limits, immutable
sampler snapshots and RNG isolation, plus actual simulator smoke checks. The
protocol is opt-in; existing experiment commands keep their original loop.
