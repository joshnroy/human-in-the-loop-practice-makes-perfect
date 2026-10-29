# SweepTable3D production preflight

The [dated integration log](experiment-logs/2026-10-28-sweeptable3d-production-preflight.md)
records actual test results and remaining launch limits. A runnable environment
or passing unit suite alone is not an overnight-experiment clearance.

## Contract and evidence boundary

The production entrypoint is `python -m hitl_pmp.cli --env sweeptable3d`.
The table environment uses five 20 mm diameter balls, a round 20 cm capsule wiper
with a commanded 45 degree handle lean, and a short tray. Its rim is 5 mm **above
the 10 mm bottom panel**, not 5 mm total height. Legacy object/skill identifiers
still say `cube`; these names do not imply cube geometry.

`SweepTable` takes `[angle_deg, speed_m_s]`, bounded by `[-45, 45]` degrees and
`[0.05, 0.30]` m/s. The default fixed bin offset is `-0.18` m. The harness checks
the production schema, scene configuration, live ball/head dimensions, commanded
lean and live bin offset. It pins the contact controller's rev3 source SHA-256
(`ddf53e8ccaa6267649e50ccec73348182ef26c88dc99458969cd8b1f49b03fee`). It does not
select an unload counterfactual, change containment, patch controllers, or import
investigation scripts.

`--scene-bg` is enabled by default, matching the production environment and the
previous successful prototype sweeps. `--no-scene-bg` explicitly disables it.
This changes static colliders and the floor, **not merely rendering**. The chosen
value is included in the configuration hash, environment construction, recording
header and generated learner commands. Default learner commands preserve the
production background; only an explicit opt-out adds `--no-scene-bg`.
Earlier harness runs forced the background off and are evidence for that different
physical scene, not the default production scene. The parent reports an exact
initial-qpos hash match to the successful prototype (`46adcaae...`) with background
enabled after object ordering and rolling-configuration timing fixes; this setup
match does not itself establish sweep or full-five recovery success.

Delivery is binary. `binary_delivery_success` requires all five `InBox` atoms,
the production `backend.task_succeeded()` verdict, no task failure, no controller
error and no budget interruption. Controller completion alone is not delivery.
`task_failed()` includes consumed/aborted unsuccessful strokes as well as ground
contact: **it is not evidence of floor contact**. Ground coverage requires a
nonempty `ground_cubes()` observation. Tray footprint containment remains strict;
near-rim or airborne balls do not earn relaxed success.

Older exploratory outcomes do not transfer to this checkout's dependency pins.
The parent reported a first full5 initial-sweep failure despite controller
completion: a ball slightly exceeded the tray footprint. It subsequently fixed
the production scene's object ordering to preserve the prototype ordering.
That first failure remains evidence about its original source snapshot, not the
revised scene. A parent-run human case passed 1/1 while sources changed;
`source_unchanged=false` correctly prevents treating it as a stable gate pass.
Full5-v2 is diagnostic; no launch readiness is established by these observations.

## Small run plan

1. **Plan, no simulator:** choose bounded cases/seeds and inspect source hashes.
2. **Physics, parent-controlled slots:** run reference strokes, full-five robot
   recovery, and failure/human recovery against frozen sources. A single successful
   ball return is not evidence that all five can be returned.
3. **Learner CLI, separate gate:** only after physics acceptance, validate short EES
   and POMDP runs through the production CLI. Physics actions are not training.

`scripts.sweeptable3d_preflight` defaults to `--mode plan`. `--mode learner-plan`
also only writes a plan: its `learner_commands` are bounded production CLI command
arrays, never automatically executed. An exit code of zero from either planning
mode means only that a plan was written.

### Learner configuration is not an algorithm fix

The generated commands are deliberately small integration smokes, not calibrated
overnight experiments. Their EES command explicitly sets goal-pursuit horizon to
zero, matching the earlier shelf launcher but overriding the EES CLI default
(`None`, goal pursuit enabled). EES practice candidates come from previously
executed skills, so this setting can leave SweepTable outside its candidate set.
A separately labeled default-EES diagnostic can omit that override; setting a
new horizon of two is not a restoration of the existing default. The generated
command remains unchanged to avoid silently changing the experimental treatment.

POMDP's zero goal-pursuit horizon is intentional: its planner chooses practice
actions directly. Its smoke uses four-action sessions and 20 search iterations,
not the historical shelf launcher's 20 actions and 100 iterations. STOP decisions
and sampled failure-dependent detours must be reported, not overridden to force
sweep practice. Objectives, priors, learning rules and sampling remain unchanged.

### Bookkeeping and diagnostic format

The provider declares the complete, invariant `CubeOrder` facts. POMDP grounds
only bindings compatible with those facts: 30 actions with human resets, rather
than 37,518 combinations, most of which were never applicable. Other providers
retain their previous behavior unless they explicitly declare invariant facts.
Regression tests compare feasible action order and both planners' decisions
against the unfiltered table model.

Execution-cost diagnostics use `Skill(object, ...)` labels instead of repeating
the complete skill definition in every key. Belief snapshots omit the duplicate
`expected_execution_costs` table; the same values are logged under
`estimated_costs`. Internal forecast keys, values and persisted belief models
are unchanged. This is a diagnostic-format change, not new cost estimation.

```bash
scripts/with_env.sh python -m scripts.sweeptable3d_preflight \
    --output-dir /tmp/table-preflight-plan --seeds 0 1 2

scripts/with_env.sh python -m scripts.sweeptable3d_preflight \
    --mode learner-plan --output-dir /tmp/table-learner-plan \
    --practice-steps 2000 --evaluation-steps 12000
```

The parent coordinates all heavy slots. Do not launch duplicate simulations.
After source freeze, a bounded full-five example is:

```bash
run-heavy systemd-run --user --scope \
    -p MemoryMax=8G -p MemorySwapMax=0 -p OOMPolicy=continue \
    timeout --signal=TERM --kill-after=30s 1100s \
    scripts/with_env.sh python -m scripts.sweeptable3d_preflight \
    --mode physics --subset full5 --seeds 0 --case-seconds 900 \
    --max-cases 1 --max-actions 16 --max-control-steps 60000 \
    --skill-step-limit 12000 --record-motion \
    --output-dir /tmp/table-full5-frozen
```

For the separate failed-attempt/human-reset/next-attempt gate, use
`--subset human --case-seconds 600` and a new output directory. Test centered/left
references with `--subset centered left --seeds 0 1 2`. Defaults are 18 cases,
16 actions per case, 60,000 control steps per case, 12,000 per action and 600 wall
seconds per case. All caps must be positive; oversized case selections and
duplicate seeds/cases are rejected. Memory and hard process time limits belong to
the external launcher; the harness additionally interrupts at action/control-step
boundaries and uses a per-case wall timer. No parallel workers are spawned.

## Physics cases

| Case | Stroke and purpose |
| --- | --- |
| `centered` | 0 degrees, 0.05 m/s, centered bin; all-five capture then physical tool return |
| `left` | -27 degrees, 0.15 m/s, bin -0.18 m; all-five capture then tool return |
| `fixed` | -27 degrees, 0.15 m/s, requested fixed bin offset |
| `full5` | Successful initial sweep, tool return, five actual bin-to-table returns, another pickup/sweep and tool return |
| `failure` | 45 degrees, 0.30 m/s candidate; requires actual ground evidence, blocked repeated sweep, held-tool return, physical ball recovery, next successful sweep |
| `human` | Same candidate; requires an observed unsuccessful attempt, blocked repetition, held-tool return, labeled human ball reset and next successful sweep |

Each case starts with exactly one permitted setup reset. No saved checkpoint is
loaded to fabricate recovery eligibility. Full5 refuses to begin retrieval unless
the initial sweep actually delivers all five. No human fallback is permitted in
full5. Recovery checks that failures stay latched until all five balls are on the
table. Human reset is explicitly `actor="human"`, `privileged_pose_reset=true`,
with `cost_control_steps=null`: it is not a robot reset or free robot work.

Failure candidates are not guaranteed to produce the desired outcome. Missing
floor contact, an inaccessible grasp, lost tool, partial delivery, failed next
sweep or exhausted cap is a finding and a failed gate, never a premise to repair
by moving objects. The human case can cover a non-ground aborted attempt, but does
not claim the `failure` case's ground coverage.

## Reports, cost and replay

The harness writes `report.json` incrementally and `<case>-seed-<seed>/result.json`.
Each case records resolved configuration and its hash, the scene hash, controller
hash/revision, and a reference to the full source manifest with its aggregate hash.
The manifest includes working-tree source content, dependency source origins,
repository HEAD and submodule pins. KINDER packages must resolve inside this
worktree's `reference/` trees, not another checkout. Source changes during the run
invalidate the aggregate gate; untracked output artifacts alone do not.

Every attempted robot action records actual control steps, actual native physics
calls, controller-reported steps, errors and binary delivery. Robot cost is the
observed control-step count, including failed work; it is not a guessed constant.
Per-action and final observations include atoms, all-ball XYZ/quaternions, each
ball's strict tray membership and free-joint velocities, full qpos/qvel, production
success/failure verdicts, ground-contact ball names and contact-controller metrics.
Initial/final compressed state arrays and qpos hashes are saved even without video.

A read-only audit compares live qpos before native physics calls and after actions.
Net changes outside those calls fail the gate, except explicitly labeled human
resets. Setup reset precedes the audit. Transient writes that restore the exact
previous value between observations are not detectable; the report states this
limitation rather than claiming a write-interception proof.

`--record-motion` saves the shared production `recording.jsonl` format, every control
step by default (`--record-stride` changes this). `--video` implies recording and
launches a separately time-limited replay **after closing the execution scene**.
It writes `episode.mp4` and `replay.log`; replay poses a separate scene from saved
states and does not rerun the robot controller. A requested video failure fails
the case. Recordings can instead be rendered later:

```bash
scripts/with_env.sh python -m analysis.render_recording \
    --run /tmp/table-full5-frozen/full5-seed-0 \
    --output /tmp/table-full5-frozen/full5-seed-0/episode.mp4 --plain-background
```

Rendering is still heavy work and needs the same parent-managed memory/time limits.

`passed`/exit zero in physics mode applies only to the selected cases, unchanged
sources and requested artifacts. `coverage_complete` distinguishes subsets from
all cases; `physics_gate` cannot pass with missing case categories. The harness
never executes learners: `learner_gate="not_run"` and `launch_ready=false` remain
explicit even when physics passes. Read the checks and source flags, not just the
process exit status. Stable physics evidence plus separately validated learner
CLI behavior are prerequisites for a later launch decision.

## Harness unit verification

Only fake-environment tests were run by the harness agent; no simulation or
learner was launched. TDD RED was observed first as the absent harness module,
then as the missing ground-contact distinction and diagnostics. GREEN is 28/28
targeted tests before the background follow-up. That follow-up observed RED with
5 failing tests, then GREEN at 33/33, covering the default, both CLI overrides,
environment construction, recording metadata and learner command propagation.
Tests cover bounds, subsets, caps, full-five versus one-ball reset,
next sweep, human labeling, step costs, qpos mutation, hashes, source-aware status
and saved-state replay routing. The parent owns all live validation.

```bash
scripts/with_env.sh pytest -q tests/scripts/test_sweeptable3d_preflight.py
scripts/with_env.sh ruff check scripts/sweeptable3d_preflight.py \
    tests/scripts/test_sweeptable3d_preflight.py
scripts/with_env.sh ruff format --check scripts/sweeptable3d_preflight.py \
    tests/scripts/test_sweeptable3d_preflight.py
```
