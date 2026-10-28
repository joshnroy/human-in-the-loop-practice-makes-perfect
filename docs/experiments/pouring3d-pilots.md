# Frozen-shrimp pouring pilots: preparation, not submission

The primary comparison is EES, Curtis and expectimax on the same native environment,
with separate Pour1/Pour2/Pour3 samplers and competences. This launcher freezes the
configuration and verifies the source and dependency revisions before execution.
The environment/controllers and method implementations are unchanged by this PR.
No pilot outcomes are claimed here.

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

- Three rigid frozen-shrimp proxies, contact assumption mu=0.1, normal lab background.
  Tilt 87–95 degrees, horizontal sideways peak amplitude 0–25 mm, feed duration 0–30 s.
- Initial evaluation and every 3,200 counted ticks: nine held-out tasks, three per count,
  identical across methods for each seed/checkpoint; 2,500 control ticks per task.
  Evaluation uses the ordinary robot-only deployment policy, including recovery.
- Twenty skill slots per interaction, no scheduled resets, goal pursuit off, no
  perfect-score early termination in pilots. Normal STOP/safety termination remains.
- Shared learner: epsilon 0.5, sampler refit limit 10,000, competence window/recency 5/5,
  planning timeout 10 s, target-only exploration, double observation off, target history
  and seen-task order on, random-when-stranded on, skip-perfect off.
- Expectimax: Model B/local_trend, depth 6, grid inference, 1,024 cost particles,
  observation weight 0. Curtis: same model, determinized_astar/curtis, 100 queue
  iterations, node draw scope, all outcomes, observation weight 0.01.
- EES: inherited priors and human scoring on. These are the implementation/flags in
  the launch branch; see the newer unmerged EES stack blocker below.
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

Use a memory-capped service in agents.slice locally, initially one worker until measured;
max-workers is explicitly bounded at three. On Della, stage the exact clean source,
dependencies and assets, generate the manifest there, and use one row per array task,
two CPUs/16 GB and an initial two-hour pilot ceiling. Those are provisional resource
limits, not measured ETAs. Use the scratch watchdog and a job-specific temporary
directory as in the existing tossing worker. No Slurm submission is performed here.

## Results

Preparation checks only: 43 launcher/SweepRunner tests pass. Parse the real CLI for every arm/seed at both budgets, verify
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

1. **Baseline version reconciliation:** newer unmerged EES fidelity work exists on
   `codex/t3-ees-fidelity-protocol` (PR #605) and the active tossing launch branch
   `codex/launch-t3-ees-fidelity-20261027` (launch tooling PR #608). It changes shared EES/planning/step-protocol
   code and introduces statistics/grounding choices. This preparation is based on
   the pouring environment branch, which does not contain that stack. Reconcile the
   shared baseline before claiming parity with the current Della experiments; rerun
   regressions and regenerate manifests after integrating it. Do not silently select
   a winner from its ongoing factorial experiment or add a pouring-specific variant.
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

The partial-count controller can still fail; especially target one has not been shown
feasible for every packing. That is a pilot question, not evidence that the software
cannot be launched. Full-study claims remain blocked on learning curves and the
baseline/version/reset decisions above.
