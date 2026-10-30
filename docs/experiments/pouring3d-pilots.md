# Three-dumpling pouring pilots

The primary comparison is EES, Curtis and expectimax on the same native environment,
with separate Pour1/Pour2/Pour3 samplers and competences. This launcher freezes the
configuration and verifies the source and dependency revisions before execution.
The launch branch integrates the shared planner/skill-identity stack through
`d079f94b92b442d3eb5473081fae275ce163ac35`. The native pouring dependency remains
pinned; the runtime now selects the three-potsticker scene and timed joint-feedback
controller. Pilot results are pending physical recovery validation and a fresh source
freeze; historical shrimp results are not evidence for this configuration.

The first batch at `cd40854a0` was stopped after it exposed unintended pitcher contact
when starting another pickup after food recovery. Its partial results and abort reason
remain in `artifacts/potsticker-pilots-20261030-cd40854/`. The repaired controller must
use a new manifest/output root; these partial results are not a completed pilot.

## Question / goal

Does practice improve exact-count pouring, and what robot effort and human assistance
does each method use? Full-task success after recovery is a separate outcome.

## Background

Follow the [tossing pilot stages](tossing3d-pour-launch.md) and the
[shelf launch lessons](../experiment-logs/2026-10-23-sweepoffshelf3d-launch-20261023b.md).
The shelf's high initial success and early STOPs make an initial measurement essential.
The [pouring integration smoke](../experiment-logs/2026-10-28-pouring-separate-skills.md)
had three successful full tasks, but count one needed recovery after an incorrect pour.
That single-seed smoke does not establish learning gains.

## Hypothesis

Independent parameter learning can improve exact-count success. The pilots establish
whether there is measurable headroom, enough completed attempts, and usable runtime;
they do not support a ranking of methods from two seeds.

## Guidance given

Use the same methods as the other environments, with separate Pour1/Pour2/Pour3 rather
than a pouring-specific algorithm. Prepare draft PRs and the launch branch first, then
report blockers. The human skill respawns payload only; it never tips the tray or
restores the robot and vessels. Preparation does not submit jobs.

## Methods

| Stage | Methods | Seeds | Runs | Counted practice ticks per run |
| --- | --- | --- | ---: | ---: |
| pilot | EES, Curtis, expectimax | 0, 2 | 6 | 6,400 |
| long-pilot | EES, Curtis, expectimax | 0, 2 | 6 | 25,600 |

The longer stage restarts from the same seeds with its own output root. It is not a
continuation of pilot checkpoints. A full comparison will use six seeds, 0 through 5;
its budget is intentionally not implemented until pilot learning/runtime are reviewed.

- Three rigid frozen-potsticker proxies, contact assumption mu=0.1, normal lab background.
  Tilt 87–95 degrees, horizontal sideways peak amplitude 0–25 mm, feed duration 0–30 s.
- Initial evaluation and every 3,200 counted ticks: nine held-out tasks, three per count,
  identical across methods for each seed/checkpoint; 7,500 control ticks per task.
  Evaluation uses the ordinary robot-only deployment policy, including recovery.
  The cap allows an initial pick/pour/place (~2,100 ticks), three physical recoveries
  (~900 ticks each in selected pan witnesses), and another pour; the old 2,500-tick
  cap left almost no room for recovery. Skill calls retain their 2,500-tick limit.
  Training budgets and costs are unchanged. These are caps, not guaranteed runtime.
- Pour parameters are tilt, horizontal sideways amplitude and feeding duration. Joint
  feedback tracks the commanded movement; there is no food-count cutoff or simulated
  lookahead. Three distinct skills share this implementation. Their competences and
  samplers are separate, using the unchanged shared method implementations.
- Twenty skill slots per interaction, no scheduled resets, goal pursuit off, no
  perfect-score early termination in pilots. Normal STOP/safety termination remains.
- Shared learner: epsilon 0.5, sampler refit limit 10,000, competence window/recency 5/5,
  planning timeout 10 s, target-only exploration, double observation off, target history
  and seen-task order on, random-when-stranded on, skip-perfect off.
- Expectimax: Model B/local_trend, depth 6, grid inference, 1,024 cost particles,
  observation weight 0. Curtis: same model, determinized_astar/curtis, 100 queue
  iterations, node draw scope, all outcomes, observation weight 0.01.
- EES: inherited priors, human scoring on, grounded competence statistics and observed
  practice groundings (the shared implementation's defaults). Record EES decision logs.
  This is one fixed baseline, not a selected winner from the tossing factorial study.
- Prices match the earlier tossing comparison: robot tick 0.01, human payload reset 5,
  lambda 0.0003. Each reset consumes one counted budget tick. This is an experiment
  accounting convention, not a measured human duration. EES uses the human price in
  its own plan-cost units; it does not use the planner's cost objective.
  The separate [pilot cost file](../../configs/pouring3d_pilot_costs.json) leaves the
  older smoke presets unchanged (those used reset price 1 and duration 100).
- Record all evaluation motion and seed-zero practice motion; defer rendering.
  Replay support actions at 4x and pours at 1x. Show representative failures too.

From the clean launch checkout, with its own populated dependency pins and scene assets:

```bash
scripts/with_env.sh python -m scripts.pouring3d_launch check
scripts/with_env.sh python -m scripts.pouring3d_launch check --stage long-pilot
scripts/with_env.sh python -m scripts.pouring3d_launch check --native
scripts/with_env.sh python -m scripts.pouring3d_launch manifest \
  --out artifacts/pouring-pilot/manifest.json \
  --results-root artifacts/pouring-pilot/results
```

Manifest generation writes planned argv, resolved defaults, full source/dependency
commits and cost content/hash, and starts nothing. The output file must be new.
After blockers are resolved, `run --manifest FILE --go --max-workers 1` executes via
the standard SweepRunner. `--task 1` through `--task 6` selects one array row, retaining
the same validation. Existing result directories are never overwritten. A copied
manifest is not portable to a different checkout path: regenerate it at the destination
and retain both manifests as preparation/deployment records.

Use a memory-capped service in agents.slice locally, with worker count chosen from current shared memory/load;
max-workers is explicitly bounded at six. On Della, stage the exact clean source,
dependencies and assets, generate the manifest there, and use one row per array task,
two CPUs/16 GB and an initial two-hour pilot ceiling. Those are provisional resource
limits, not measured ETAs. Use the scratch watchdog and a job-specific temporary
directory as in the existing tossing worker. No Slurm submission is performed here.

## Results

Initial preparation passed 43 launcher/SweepRunner tests before the shared-stack merge.
Parse the real CLI for every arm/seed at both budgets, verify
the balanced evaluation/cost contracts and refuse changed revisions, altered argv,
invalid task indices and existing output directories. Native preflight checks
Pour1/Pour2/Pour3 and all three pieces inside the pitcher in a ready scene.
Ruff, formatting, mypy (265 source files), architecture and documentation checks pass.

The eventual report must include initial/final pour-only exact-count scores by target,
full-task scores after recovery, every seed's learning curve, completed training labels
by skill, reset counts, robot ticks, runtime/memory, and STOP/Wait/crash reasons. Report
interrupted attempts separately. Keep successful fixed-recipe feasibility evidence
separate from learned-policy performance.

## Recommendation / blockers

1. **Shared baseline integrated:** the launch branch includes the latest fetched
   `codex/t3-side-independent-toss` stack through `d079f94b9`, including the newer
   generic skill-identity/constant-effect and DP fidelity changes. No pouring-specific
   learner or competence class is introduced. Regenerate manifests after the merge:
   the original `ff1d5ae5fe7a5db95e163c95a507ec4e386cb5c4` manifests are obsolete for
   these pilots and must not be launched. New shared-code updates require another
   validation and freeze rather than changing running workers.
2. **Pilot validation:** parsing/native initialization is not evidence that all three
   methods practice productively. Run the short pilots and inspect EES Wait behavior,
   planner STOPs, per-skill labels, initial headroom and runtime before the longer stage.
3. **Always-ask semantics:** the existing wrapper requests a full sampled-start reset,
   which differs from the agreed payload-only human skill. It is excluded from this
   three-arm matrix. Resolve and label that ablation before doubling to six arms.
4. **Deployment:** the launch PR must pass CI; Della needs the exact frozen checkout,
   native dependencies/assets, manifest and worker resource settings. Unmerged native
   dependencies are published and pinned, so merging is not intrinsically required
   for branch experiments. A later physics rebase requires fresh validation.

The current development search found exact-count witnesses for all targets on all
three initial seeds. **Revision note:** the numbers in the following paragraph
precede the entry-trajectory repair. Use the
[fresh witness report](../experiment-logs/2026-10-30-potsticker-repaired-entry.md)
for current settings; the historical paragraph is preserved below.

The earlier development search found exact-count witnesses for all targets on all
three initial seeds (0, 1, 2). The selected tilt values for Pour1/2/3 are respectively
89/89.5/90 degrees on seed 0, 89.5/89.75/90 on seed 1, and 89.5/89.6/90 on seed 2.
Amplitude is 8/8/15 mm and feeding duration 12 seconds in these witnesses. These
parameters are diagnostic evidence only: the launcher does not seed learned samplers
with them. Seeds 1 and 2 were searched from physically prepared checkpoints; they do
not establish full-task reliability or generalization. Repeated payload resets change
the arrangement and can invalidate a fixed recipe. Full-study claims remain blocked
on completed pilot learning curves and recovery validation.
