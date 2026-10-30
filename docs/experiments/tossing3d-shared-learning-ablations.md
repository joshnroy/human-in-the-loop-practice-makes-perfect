# Tossing3D shared-learning validation

This is a planned short validation, not a full experiment or a new cost calibration.
Fresh skill-prior collection is a separate prerequisite, not a reuse of stale source hashes.
`scripts/tossing3d_shared_learning_validation.py` writes a manifest and Slurm array
script plus byte-preserved copies of their three evidence files. It never submits
jobs, contacts Della, runs practice, calibrates priors, or creates a
preflight test job. A unit-test preflight job will be launched separately and must
pass on the frozen source before the pilot array is launched.

## Matrix and interpretation

| Variant | `--competence-sharing` | `--practice-groundings` | Interpretation |
| --- | --- | --- | --- |
| `baseline` | `grounded` | `observed` | Grounded models and actually observed targets |
| `shared_only` | `parent` | `observed` | Parent-pooled robot competence/forecasts, observed targets |
| `flexible_only` | `grounded` | `all` | All provider groundings, grounded models/forecasts |
| `shared_flexible` | `parent` | `all` | Parent pooling plus interchangeable parent fulfillment |
| `shared_rate` | `rate` | `all` | Separate current competence, pooled improvement forecast and flexible fulfillment |

Every variant runs EES, DP episodes/once with budget **350**, and DP cost/period
with budget **2000**, with paired seeds **0 and 1**: **30 jobs**. DP uses Gurobi.
The 350 completed-target-rollout cap comes from the existing plain-reset cost
calibration. Reusing it is not a newly calibrated physical-cost equivalence for
any sharing variant, nor a guarantee that the cap is spent. Report actual robot,
human and total physical costs, completed target rollouts, and counted steps.
Setup executions and human resets are priced but are not extra target episodes.

Each job has **6400 practice steps**, evaluation interval **3200**, **10 held-out
tasks**, **500 control ticks per task**, and no perfect-evaluation early stopping.
The existing step protocol can finish an interaction at a budget boundary; check
the recorded actual step counts. This generator exposes no full phase, enlarged
budget, algorithm overrides, or submission option. Subsets select variants and
seeds while retaining all three algorithms.

All arms use the same executable skills, predicates, controllers, and shared
parameter sampler. The sampler and its training/proposal settings are unchanged;
the sampler was already shared and is not the ablation. Human scoring is explicitly
True, plain resets only, and human reset models remain grounded in every variant.
Sharing is within each independent run, never between seeds or algorithms. EES
retains its optimistic extrapolation and DP retains its linear improvement model;
this changes their forecast identities, not their respective estimation formulas.
The existing `Tossing3DLaunch.arms()["ees"]` and `PILOT` supply the physical and
practice configuration: barrier/o1, movable 1 kg bin, 0.1 kg cube, reset policy
`never`, and the EES cost file for all arms. Every arm additionally appends
`--no-bin-pouring`, preserving exactly the original robot skill names
**PickCube**, **MoveToTossLocationAndToss**, and **OpenGripper**. Main
`70c3f63d202d2e6870e167d778615e61b077230d` includes #592, which enabled
PourCubeOutOfBin after the historical `84b183df4` cohort; leaving today's default
enabled would change the comparison's
skill set and require a missing pouring prior. No same-side forcing or physical
reset shortcuts are introduced. The original simulator pins remain:

- KINDER: `490831fa54a40c4b2b4570d45bb08e3dd63c24ef`.
- kinder-baselines / kinder_models: `032b42634a015c7f710667b3a791836fa839f6ff`.

Main also includes #590's post-toss settling changes. All 30 paired validation
jobs, including every baseline, must rerun on the same new clean source SHA.
This provides a matched comparison within this pilot, not a claim of identical
historical physics. The old 350 cap remains historical budget reuse, not a new
cost calibration under these code or sharing changes.

`all` makes all EES candidates available; it does not alone imply forecast
transfer. Both `parent` plus `all` and `rate` plus `all` allow a target to be
fulfilled by another grounding of the same robot parent. Any applicable same-parent
grounding is preferred. Otherwise the existing setup plans are compared using immediate
negative-log-competence robot costs plus human prices, with stable provider-order
ties. This may select either side; it does not force same-side practice.
`parent/observed` uses only actual observed groundings as candidates: enumerating
siblings to reprice a pooled forecast must not add them to the observed list.
`parent/observed` and `rate/observed` retain exact-target fulfillment.
`grounded/all` exposes candidates without transferring their forecasts or making
their fulfillment interchangeable. The interaction between pooling and candidate
availability is therefore part of the ablation.

Current performance can differ by side despite a shared executable controller
and sampler. Parent pooling assumes common competence; it does not establish
physical equality. Optional rate sharing pools the improvement forecast while
keeping current success probabilities separate: sharing a learning-rate model
does not equate current `p`. The `rate/all` control has the same flexible context
choice as `parent/all`, isolating separate current competence from full pooling.
Rate sharing also shares practice budget/count gains across contexts; its forecast
uses that shared improvement count even when current `p` differs by grounding.
The same immediate-cost ordering applies, using the respective current competence
estimates, so chosen contexts need not coincide between the two variants.

Defaults remain `grounded/observed`; ordinary reproduction is unchanged unless
the new flags are supplied. `--record-learning-forecasts` defaults False and
requires an output directory. This launcher enables it, sampler-draw logging and
skill-competence recording explicitly. Preserve `learning_forecasts.jsonl` from
period ends, skill-competence snapshots and DP allocation logs: all current-model,
practice-count, forecast/rate and transfer keys must be logged along with the
actual target and executed grounding. Inspect those keys in the separate unit
preflight before trusting cross-side transfer; candidate availability by itself
is not evidence of transferred forecasts. Preserve human grounded keys as well.

## Prepare artifacts locally

From the checkout, after the new EES/DP flags and logging are integrated:

```bash
python -m scripts.tossing3d_shared_learning_validation prepare \
  --phase validation --tag t3-shared-validation-20261026 \
  --sha FULL_40_CHARACTER_COMMIT_SHA \
  --calibration-report /path/to/existing/calibration-report.json \
  --prior-calibration /path/to/refreshed/source-bound-prior.json \
  --budget-prior-calibration /path/to/historical/budget-prior.json \
  --out /path/to/planned-validation
```

Use `--variants baseline shared_rate --seeds 0` for six jobs, or any nonempty,
distinct subset of the five variants and paired seeds. `--phase full` is rejected.
Outputs are `manifest-shared-learning.tsv`, `array-shared-learning.sbatch`,
`prior-calibration.json` (current executable prior), `budget-prior-calibration.json`
(historical budget evidence), and `calibration-report.json` (historical report).
Evidence copies preserve the input bytes, including when inputs are already in
the output directory. Rows use the existing six-field manifest format;
each row contains the full immutable source SHA and shell-quoted exact argv.
Workers keep Python/planner temporary storage on GPFS, but graphics runtime
storage in a private node-local `/tmp` directory: Mesa's anonymous graphics
allocations must not inherit a GPFS runtime path.
`Tossing3DSharedLearningValidation.records()` builds and parses the matrix without
writing files; `.prepare()` validates both evidence roles and writes the artifacts.
Its returned receipt records `current_prior_calibration_sha256` separately from
`budget_reference_prior_digest`, plus `calibration_report_sha256`, fixed
`episode_budget=350`, and the explicit nominal-budget interpretation. The existing
`prior_calibration_sha256` receipt field remains an alias for the current prior's
digest. `main(argv=...)` provides the same interface for local callers.
Both `prepare()` and `verify()` accept optional `budget_prior_calibration`; the CLI
spelling is `--budget-prior-calibration`. Omitting it uses `prior_calibration` for
both roles, preserving existing callers when a single file satisfies both.

The existing evidence is at
`artifacts/dp-fidelity-fix/repair-20261026/calibration-pilot/calibration-report.json`
under the main research checkout; its exact prior is available at
`artifacts/dp-fidelity-fix/repair-20261026/open-gripper/calibrated-priors.json`.
This historical report binds its historical prior SHA256
`aa0287c3fecc12149dcae8e1a07b5342287a89de53604b645a84cb0ecd1592a5`.
The generator verifies the report's 2000-cost target, plain 350 cap, no evaluation
score use, historical prior digest, schema/splits and original simulator pins.
This check deliberately does not load historical prior source hashes as executable
code. Independently it validates the current prior's schema, passed validation,
independent splits, original pins and complete three-skill coverage. The current
prior has its own digest and need not match the historical report's prior digest.
Never rewrite the historical report to substitute the current prior's digest.
Historical
algorithm hashes remain historical evidence; changed ablation code does not become
recalibrated merely by reusing that report.

## Deployment checks and hazards

An operator must deploy a clean committed checkout, its pinned submodules and
assets to `/scratch/gpfs/TSILVER/jr2860/code/<SHA>`. Preparation may be done while
reviewing local edits, but deployment requires all source, tests and documentation
committed; no branch name, short hash or mutable code key is accepted. No commit,
deployment, SSH connection or submission is performed by this generator.

Deploy the generated directory to
`/scratch/gpfs/TSILVER/jr2860/launch/<tag>/`, preserving all five files. The Slurm
script pins the historical report's SHA256 and the independent current prior's
SHA256; the report in turn binds `budget-prior-calibration.json` by its original
`prior_calibration_sha256` field. The selected DP manifest row must reference the
same digest-pinned current prior, never the historical budget reference. Changing
any evidence file requires regenerating and reviewing the artifacts, not editing
digests by hand. Local preparation validates evidence structure but does not certify
current-source freshness; runtime verification must pass before any pilot work.

The array reuses `ShelfLaunch.della_sbatch`'s account/resources, isolated import
check, zstandard check, library/thread environment and runtime directory. It loads
`gurobi/13.0.0` and requires `DP_GUROBI_PYTHONPATH` to name the isolated Gurobi Python
directory. It hands the manifest to `scripts.tossing3d_launch.run_worker`, retaining
its clean frozen-source/dependency checks, compiled-mass/movable-bin/pouring
preflight, per-run `launch.json`/`physics.json` provenance, and scratch watchdog.
Arguments are parsed as data by that worker. The read-only `verify` command also
checks the original gitlink pins, clean source SHA, evidence digests, source-bound
prior validation, simulator identity and complete robot-skill prior coverage.
It parses the selected manifest task and requires `args.enable_bin_pouring` to be
False, then checks that the provider constructed with an environment having
`enable_bin_pouring=False` exposes exactly the original three robot skill names.

## Refresh the executable prior before deployment

The historical prior is stale and cannot be deployed as the executable prior.
`DpPriorCalibration.load()` checks each recorded source SHA256; the initial audit
first failed at `src/hitl_pmp/environments/tossing3d/cli.py`. That audit found nine changed
files among its 37 hashed files: Tossing3D's `cli.py`, `environment.py`,
`kinder_backend.py`, `recording.py`, `recovery_skills.py`, `skill_provider.py`,
`skills.py`, `state_log.py`, and the shared learner's `ees_method.py`. The first
eight already differ on main; the learner also changes in this ablation branch.
Collector refresh changes add their own source hashes to this comparison.
Disabling pouring restores the three-skill set but does not restore those hashes
or undo #590's settling changes.

`calibration.verify_simulators()` is a separate check: it checks the imported
KINDER/kinder_models checkout commits and their dirty status against simulator
pins. It does not validate the repository's learner/environment source hashes.
Both checks remain required for the **current** prior. Historical budget evidence
is parsed and digest-verified only; it is never passed to `.load()` or used by DP.

The safe refresh sequence is:

1. Integrate the separately owned collector changes first: both `collect` and
   `collect-open` must explicitly use `enable_bin_pouring=False`, barrier/o1, bin
   mass 1.0 kg and cube mass 0.1 kg, assert the three robot skill names, and record
   that protocol. The collector at the initial audit constructed a default
   environment, enabling pouring and leaving mass overrides unset; use the
   refreshed collector's explicit protocol rather than that historical behavior.
2. Freeze the final clean source SHA after collector and shared-learning changes,
   with isolated imports, original simulator pins, compiled-mass checks and a
   fixed numerical environment. Use this source for collection, fitting and all
   30 pilot jobs. Do not change hashed code between collection and fitting.
3. Collect twelve reserved-seed jobs: six Pick/Toss `collect` jobs and six
   ClosedEmpty `collect-open` jobs. Fit seeds are 2000--2002; validation seeds are
   2010--2012. Keep paired pilot seeds 0/1 and their held-out evaluations out of
   calibration. Use the unchanged shared sampler and grounded/observed baseline
   collection policy, then apply the same resulting priors across all variants.
4. Retain the existing bounded protocol: 60 training toss rollouts per seed,
   greedy probes at 0/20/60 with 16 scenes each, and 16 independent OpenGripper
   probes per seed. These are 648 completed training/probe tosses and 96
   OpenGripper probes across six seeds, plus setup retries. They can run in
   parallel, but a wall-clock completion estimate is not established. Prior
   collection uses EES's sampler, not a Gurobi allocation solve.
5. Fit from the new raw outputs with `Tossing3DDpPriors.fit(root=..., open_root=...)`.
   Require complete data, matching collector/learner/environment provenance,
   clean original simulator pins, matching numerical stacks, independent splits,
   and `validation_passed=True`. If validation fails, stop; do not relabel old
   measurements, replace their hashes, or relax the validation threshold.
6. Pass that new report as `--prior-calibration`, retaining the old cost report
   and its exact old prior as the historical budget reference. Require current
   `.load()`, simulator and robot-coverage checks to pass on the deployed source.
   Run the separately scheduled licensed Gurobi unit job, then permit the
   30-job pilot array with concurrency four only after its preflight passes.

Refitting old raw observations cannot establish freshness here: the fitter
rejects changed shared-learner/environment source, even though it separately
records collection-driver provenance. The new report changes the **current prior**
digest; it does not retroactively change the old cost report or calibrate a new
350-to-2000 physical-cost correspondence. The 350 cap remains a fixed nominal
historical choice. Fresh-scene priors still do not validate reset-free-state
transfer or saturation beyond the measured 60-rollout range.

The separately scheduled preflight unit-test job should run the shared-learning
method tests and this generator's tests on the exact deployed SHA, with required
runtime dependencies, before any pilot array. Locally the focused launcher test is:

```bash
python -m pytest -q tests/scripts/test_tossing3d_shared_learning_validation.py
```

Review the 30 rows, forecast/count/transfer keys, observed-only candidate isolation,
stable fulfillment choices, source identity and prior compatibility before the
operator launches anything. This document and the generated files are plans only;
actual completion and measured transfer require run outputs.
