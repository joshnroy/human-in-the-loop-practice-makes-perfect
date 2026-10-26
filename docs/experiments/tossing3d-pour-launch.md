# Tossing3D relaunch with pouring and charged settling

Merge the settling, planner compatibility, pouring, and launch PRs in stack order,
then generate manifests from the merged, clean commit. Dependency revisions must
match that commit's gitlinks. The stack retains current main's kindergarden
`490831fa54a40c4b2b4570d45bb08e3dd63c24ef` and kinder-baselines
`032b42634a015c7f710667b3a791836fa839f6ff` pins.

| Stage | Arms | Seeds | Jobs | Practice ticks per job |
| --- | --- | --- | ---: | ---: |
| Short pilots | EES, expectimax, Curtis; plain and always-ask | 0, 2 | 12 | 6,400 |
| Longer planner probes | Plain expectimax and Curtis | 0 | 2 | 25,600 |
| Full comparison | Same six default arms | 0–5 | 36 | 160,000 |

Every arm uses a movable 1 kg bin, a 100 g cube, Tossing3D-o1, barrier practice
and evaluation layouts, pouring enabled, 20 action slots per practice period,
and no scheduled resets. Pouring remains available when the cube can also be
picked; feasibility preconditions must not prescribe which action the method
chooses. The action-availability audit in
[the pouring documentation](../tossing3d-bin-pouring.md#action-availability-audit)
records the accepted environment, skill and algorithm contracts.
The standard comparison holds the environment and pre-programmed skill layer
(including predicates) fixed while varying the algorithm. RoboCode also supplies
its own skill layer; that difference must be explicit in its experiment log.
The upright-bin validity gate and empty-hand OpenGripper semantics are accepted.
Suppression after a finite parameter search fails belongs to the algorithm and
prevents getting stuck on a single failing skill. These are not open audit issues.
Post-toss settling is charged as part of the robot
skill. Evaluations run initially and every 3,200 practice ticks on ten fixed
held-out tasks with a 500-control-tick cap. Full runs stop after three consecutive
perfect evaluations; pilots do not stop early. Epsilon is 0.5 and sampler refits
use 10,000 iterations.

Prices are 0.01 per robot control tick and 5 per human reset, with objective
lambda 0.0003. Always-ask arms also price `reset_to_sampled_start` at 5. Use
current main's arm-specific cost files and method defaults, including EES's
human-skill scoring. Expectimax uses depth six and observation weight zero;
Curtis uses 100 queue iterations, node draw scope, all outcomes, and observation
weight 0.01. Model B/local_trend uses grid 25×16, 1,024 cost particles and 100
theta samples, with non-epsilon competence evidence.

```bash
scripts/with_env.sh python -m scripts.tossing3d_launch check
scripts/with_env.sh python -m scripts.tossing3d_launch check --size pilot
scripts/with_env.sh python -m scripts.tossing3d_launch della \
  --size pilot --tag t3-pour-pilots --out /absolute/path/to/pilot-launch
scripts/with_env.sh python -m scripts.tossing3d_launch della \
  --size long-pilot --arms expectimax determinized-curtis --seeds 0 \
  --tag t3-pour-long --out /absolute/path/to/long-launch
scripts/with_env.sh python -m scripts.tossing3d_launch della \
  --tag t3-pour-full --out /absolute/path/to/full-launch
```

Generation submits nothing. Submit each generated array with its corresponding
manifest as the first argument. The default selects six arms; the historical
Algorithm 3 determinized variants remain available through `--arms`.

Workers request two CPUs and 16 GB and pin numerical-library threads to one.
Both pilot sizes request two hours. Full time limits remain eight hours for
EES, sixteen for Curtis, and forty-eight for expectimax; these are resource
ceilings, not runtime predictions. Each task exports TMP/TMPDIR/TEMP to its own
GPFS directory before starting Python. Locally, `with_env.sh` uses `artifacts/tmp`
or an explicit `HITL_PMP_TMPDIR`.

Before practice, the worker verifies the source commit, clean tracked code, no
untracked source files, local KINDER imports, and exact clean dependency revisions.
It compiles the scene and verifies actual masses, a free bin joint, and pouring
availability. `physics.json` records that evidence and the temporary paths. A
scratch watchdog supervises the actual CLI process. Manifest arguments are
parsed as data; no shell eval is used.

Review pilot stop reasons, task scores, reset counts, pour attempts/outcomes,
charged settling ticks, memory and runtime before full submission. A reset
chosen over an available pour can be a valid objective choice. Cubes supported
by the robot base intentionally remain excluded from picking. Tipped bins still
need human recovery.

The earlier six pilots at `0fa7c204209ad2f32102cd5b40b37cd785398c6c` predate
settling and used different dependency revisions. They do not validate this
merged configuration. All six arms must be rerun consistently; comparison with
those archived results cannot isolate the effect of pouring alone.

`--no-bin-pouring` supports a separate ablation while retaining the same movable
bin and masses. No ablation jobs are included in the default matrix.

Every log entry must include full run-code/dependency commits and exact launch
flags. See the [historical provenance backfill](../experiment-logs/2026-10-26-tossing3d-launch-provenance.md).
Workers preserve exact commands and manifest hashes in `launch.json`, and resolved
defaults, verified identity, and cost-file contents/hash in `physics.json`.
