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
skill at the total budget is recorded as interrupted, without labeling its outcome
as a completed training example.

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
run completion. Existing `stats.json` retains its skill-invocation axis; use the new
step records for the normalized comparison. State logs support later video replay.

Use `scripts/with_step_env.sh` to resolve the runner and both simulator packages
inside the isolated checkout. Copy or populate that checkout's dependencies before
running; the wrapper does not install packages. Verify package import paths before launching; they must not
resolve to a shared checkout. `--defer-rendering` and
`--practice-reset-policy never` are required for this protocol.

Validation covers mid-skill boundaries, human accounting, step limits, immutable
sampler snapshots and RNG isolation, plus actual simulator smoke checks. The
protocol is opt-in; existing experiment commands keep their original loop.
