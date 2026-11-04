# Kitchen recycling launch packaging

Preparation is not permission to submit. `prepare.py` and `snapshot.py` never
contact Slurm. Only `submit.sh` calls `sbatch`, and it requires explicit
authorization plus matching evidence. No old Tossing3D/sweeptable launcher is used.

## Final experiment contract

- Exactly one object, `cube_control`, in both training and deployment, from main's
  `experiment-config.json`. Seeded kitchen scatter remains enabled.
- Seeds 0, 1, 2; arms EES, expectimax, Curtis: **nine jobs**, not an object sweep.
- **160,000 counted practice steps; physical cost logged, not hard-capped.**
  Historical 2,000 cost budget belongs to DP. There is no DP arm or new global
  cost-budget flag in this package.
  An iteration run may set another budget with `prepare.py --practice-step-budget`;
  the manifest then records `practice_step_budget` and `worker.py` holds every row
  and the run's endpoint to it. Left at the default, the manifest carries no such key
  and is byte-identical to the 160,000-step package.
- Both human resets cost 5 by default. `prepare.py --human-reset-costs C1 C2 ...` sweeps
  that cost: the nine runs are repeated once per cost, in the order given (manifest
  `human_reset_costs`; row index = seed-major, then cost in that order, then arm, so
  one seed's whole sweep finishes first). Each cost gets its own
  cost file, `costs/human-reset-cost-<C>.json` beside the manifest (the repo's
  `costs.json` with both resets repriced), passed by its deployment path; each row runs
  in `results/reset-cost-<C>/<arm>/<seed>`. The cost lives only in that file: every row
  keeps `--human-reset-practice-cost` at its default 5, because the POMDP refuses any
  skill priced above its cost ceiling (20) and both arms price resets from the cost file,
  as Tossing3D's own runs do (cost file 500, flag 5). `worker.py` checks each row's cost
  file against its bound `costs_sha256` and its cost, and the planner flag against that
  default. EES prices a reset from the cost file from
  its first decision; the POMDP arms learn it from charges observed (first-reset forecast
  `skill_cost_prior_steps` x robot price, as in Tossing3D's own cost sweep).
- `prepare.py --constraint <feature>` adds `#SBATCH --constraint` to both scripts.
- One evaluation process per checkpoint by default. `prepare.py --evaluation-workers N`
  (N at most the ten test tasks) gives every row `--evaluation-workers N
  --evaluation-task-rng per-task`, records `evaluation_workers` in the manifest, and
  makes the smoke evaluate N held-out tasks instead of one, so it runs N evaluation
  processes at once. Size `--cpus-per-task` and `--memory` to match (practice plus N
  evaluation processes).
- Measure initially and every 3,200 steps, ten held-out tasks, 500 evaluation
  control steps per task. `--no-early-stop` is mandatory for pilots and full runs:
  disable perfect-score, repeated-STOP and no-progress termination. STOP remains
  an actual planner decision with explicitly logged accounting; no robot action
  is forced. A run ending below its counted-step horizon is not a completed run.
- `--no-scene-bg --defer-rendering` for both pilots and full runs. Mandatory
  physical kitchen/desk/chair assets are still packaged; video is separate.
- Robot ticks cost 0.01; each of the two human resets costs 5 with duration 1;
  planner linear objective lambda is 0.0003. `costs.json` prices exactly
  `ResetCubeBinSameSide` and `ResetCubeBinOppositeSide`, the names the kitchen charges
  Tossing3D's two resets under, and the planner's `--human-reset-practice-cost`
  defaults to the same 5; `worker.py` checks both. There is no `Recover_*`,
  `RelocateBinNear` or `RestoreBinTarget`, and no bottle/can object identity.
- **The environment is Tossing3D re-flavoured** (#637): Tossing3D's skills, operators,
  predicates, samplers, bounds and reset costs on the kitchen scene; one belief per
  skill, no near/far copies; goal `InBin` by KINDER's classifier. The practice domain
  for expectimax and Curtis is `TOSSING3D_DOMAIN`, with one belief per reset.

## Canonical method provenance

Read-only source: original checkout's
`artifacts/dp-fidelity-fix/ees-sidefree-full-20261029-2133/selection.json` and
`artifacts/dp-fidelity-fix/ours-sidefree-full-20261030-0912/selection.json`.
The selected historical EES cell is
`ees-grounded-all-skipperfectfalse-goaloff`. The selected planner cells are
`expectimax-goaloff` and `determinized-curtis-goaloff`.

All arms retain epsilon 0.5, competence recency/window 5/5, planning timeout 10,
sampler train iterations 10,000, goal horizon 0, goal interval/init 1/1,
target-only exploration, target-history learning, seen-task ordering,
random-when-stranded, no double observation and no skip-perfect.

EES: `--method ees --practice-groundings all
--ees-score-human-skills --record-ees-decisions --no-ees-reset-gate`.
Competence is never shared across groundings; "expanded" means all groundings.

Ours: `--method pomdp --pomdp-inference-engine grid --pomdp-num-particles 1024
--pomdp-linear-cost-lambda 0.0003 --practice-groundings observed`.
Expectimax uses `--pomdp-solver expectimax --pomdp-search-depth 6
--pomdp-observation-probability-weight 0`.
Curtis uses `--pomdp-solver determinized_astar --pomdp-determinized-variant curtis
--pomdp-num-rollouts all --pomdp-observation-probability-weight 0.01
--pomdp-max-search-iterations 100 --pomdp-draw-scope node`.
100 is the latest selected Tossing3D budget, not the old kitchen pilot's 1,000.

The Tossing3D preset is expanded explicitly: its scene/layout/variant flags and
two-object human reset cost file must not leak into kitchen recycling. Kitchen
tasks currently implement the 10,000 held-out seed offset directly; the worker
checks that default instead of passing the unsupported Tossing3D-only CLI flag.

## Package after the CLI agent finishes

Run from the worktree root, using a NEW output directory/tag each time:

```bash
python scripts/kitchen_recycling/prepare.py \
  --out artifacts/kitchen-launch-preflight/FINAL-LAUNCH \
  --tag kitchen-recycling-UNIQUE-FINAL-TAG --max-parallel 9
python scripts/kitchen_recycling/snapshot.py \
  --out artifacts/kitchen-launch-preflight/FINAL-SNAPSHOT \
  --original-root /home/josh/Documents/repos/research/human-in-the-loop-practice-makes-perfect \
  --dependency-root /home/josh/Documents/repos/research/human-in-the-loop-practice-makes-perfect/artifacts/dp-fidelity-fix/pr-publish-20261027 \
  --include-assets --downward /home/josh/Documents/repos/research/downward
```

The snapshot includes actual worktree `src/` contents, including untracked kitchen
code, configs, these launch scripts, all tests and kitchen docs. It exports exact
canonical dependency commits, **not** dirty original-root source or whichever
gitlinks happen to be in this worktree:

- kindergarden: `490831fa54a40c4b2b4570d45bb08e3dd63c24ef`. The branch's gitlink is
  `c85dbbc5`, which is `490831fa` plus PitcherPour3D only (`pitcher_pour3d.py`, two
  PitcherPour3D task JSONs, its test, and six registration lines in
  `kinder/__init__.py`); every file the kitchen scene loads is byte-identical between
  the two, per GitHub's compare of `490831fa...c85dbbc5`.
- kinder-baselines: `032b42634a015c7f710667b3a791836fa839f6ff`
- Pin basis: `pr-publish-20261027`, HEAD `2724868a657dbadf3bad5948a01598a23fc75111`.

Separate `assets.tar` carries actual meshes/textures rather than remote symlinks.
The desk lab3 meshes were missing in the inspected remote d079 deployment.
The chair is in the pinned source export. Fast Downward source is archived at
its recorded commit; build a fresh release binary after extraction.
`snapshot.json` records file and archive SHA256 digests. A source change after
this freeze requires new snapshot/evidence/authorization, not an override.

## Isolated runtime and deployment

Never run pip against `/scratch/gpfs/TSILVER/jr2860/envs/hitl-pmp` itself. Create a
new dedicated conda environment at the manifest's `remote_root/env`, or clone
the existing environment there and validate its frozen package inventory.
Install the tested pybullet_helpers 0.1.2 / dexmate-urdf 0.8.4 overlay under
`remote_root/runtime-overlay`, ahead of site-packages. The shared environment
remains unchanged. No credentials or user caches belong in the package.

Extract `source.tar.gz`, `assets.tar`, and `downward.tar` into a fresh
`remote_root/source`; never merge over an old experiment directory. Transfer
launch files and snapshot metadata to `remote_root/launch`. Preserve archive
hashes, verify extracted source hashes, compile Fast Downward, and run `pip check`
with the overlay before marking `runtime-lock.json` as `RESOLVED`.
The observed inventory/lock is evidence, not a claim of a clean environment.

## Evidence and submission gates

`readiness.json` contains `binding` (SHA256 of `manifest.json`, `snapshot.json`,
`runtime-lock.json`), `selected_object: "cube_control"`, and an entry for each
check with `passed: true`, a launch-relative evidence `path`, and its `sha256`.
Before pilots: `selected_object_pickup`, `single_object_train_eval`,
`seeded_scatter_reproducible`, `scatter_across_kitchen`. Before full, additionally:
`training_deployment_isolation`, `deployment_no_human_actions`,
`dependency_resolution`, `final_refit_evaluation`, `kitchen_costs_approved`.
Evidence paths must stay within the launch directory.

Pilot authorization file `smoke-authorization.json` contains the same binding,
`authorized_by: "main"` (or explicit `"user"`) and `submit_smoke: true`.
Only then: `bash scripts/kitchen_recycling/submit.sh "$REMOTE_ROOT" smoke`.
This runs the real CLI for all three arms at seed 0, 3,200 steps, one held-out
task per evaluation worker, 500 evaluation steps, unchanged planner settings, at most 30 minutes/arm.
Process success alone is insufficient: final evaluation and budget endpoint
must complete. A 2h allocation cap bounds the entire smoke. The original 15-minute
per-arm limit interrupted a progressing Della expectimax pilot at 3,156/3,200
steps; extending this wall-clock limit does not change the experiment budget.

Full authorization file `authorization.json` must contain the matching binding,
`authorized_by: "main"`, `submit_full: true`; full submission also requires the
worker-generated matching `smoke-passed.json` and all readiness checks.
Only after main greenlights: `bash scripts/kitchen_recycling/submit.sh "$REMOTE_ROOT" full`.

The array is seed-major (0:EES/s0, 1:expectimax/s0, 2:Curtis/s0, then seeds 1,2),
`--array=0-8%9`, with a configurable concurrency cap and no per-seed barriers.
Slurm can reorder task starts; seed-major indexing is not a strict start-order
guarantee. Submission locks and exclusive result-directory creation prevent
accidental repeats. No command cancels other jobs.

Each task requests `--account=tsilver --partition=cpu --cpus-per-task=2 --mem=16G
--time=48:00:00`, no GPU, numerical threads=1. 48h is a ceiling, not an ETA.
Earlier Tossing3D EES runs took about 1h52–2h06, Curtis 6h58–7h04, expectimax
9h22–9h55, but those are different dynamics. Do not promise full kitchen results
by morning or extrapolate long-run learning from a startup smoke. Queue delay,
final refits/evaluations, stalls and deployment overhead remain unmeasured here.

## Unlaunched local fallback

`python -m scripts.kitchen_recycling.local_queue --launch PATH` only prints a plan.
Prepare a fresh extracted immutable snapshot and a separate locally validated
runtime lock. The Della CPU Torch lock is not the workstation's runtime lock.
Even with `--execute`, all normal full-run gates plus an additional
`local-authorization.json` are mandatory: matching binding, `authorized_by: "main"`,
`run_local: true`, and the exact `max_parallel` (default 3). Do not create that
authorization or execute local jobs unless main explicitly chooses the fallback.
Its rolling process queue submits seeds first and has no seed-level barriers.
