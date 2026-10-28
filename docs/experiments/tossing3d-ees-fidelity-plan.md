# Tossing3D EES fidelity: preparation only

No submission, worker, cancellation, launch, branch, commit, or DP change is part of
this tool. `scripts/tossing3d_ees_fidelity_launch.py` imports the existing canonical
arm/argv constructors; it does not modify the existing launcher. Its output is a
deterministic JSON manifest with argv, parsed configurations, selected source hashes,
and explicitly pending provenance. A successful check is **not launch approval**.

## Matrix and order

- EES: statistics `parent` / `grounded` × eligible groundings `observed` / `all` ×
  skip-perfect `true` / `false` × goal pursuit `off` / `on`: 16 cells.
- Ours: canonical expectimax, goal pursuit off/on: 2 cells.
- Determinized Curtis, goal pursuit off/on: 2 cells.
- Full: seeds 0, 1, 2, 160,000 counted practice steps each, 60 runs total.
- Proposed pilots: seeds 0, 1, 6,400 counted practice steps each, 40 runs total.
- Seed-major: all 20 seed-0 arms, then all 20 seed-1 arms, then (full only) seed 2.
  Within a seed, use the factor order above, then expectimax off/on, Curtis off/on.
- No dependencies, barriers, or launcher concurrency cap. A future scheduler should
  permit its maximum available concurrency; manifest ordering is not a seed barrier.
- Three-perfect-evaluation early stopping is disabled for every run.

## Required freeze

### Shared OpenGripper release behavior (2026-10-27)

The user requested that OpenGripper move the hand forward clear of the robot base
before opening, then wait for the released cube to settle. This is a shared
environment/controller change for EES, expectimax, Curtis and DP, not an experimental
axis or a seed-specific correction. A hand already clear of the base need not move
farther. Keep the base fixed and the fingers closed during the clearance motion;
an unsafe or unreachable release pose must fail rather than open over the robot.
All actual motion and settling ticks count toward the existing budgets and costs.
Keep the existing settling thresholds and bounded wait. Clearing the base is not
a symbolic guarantee that the cube lands on the floor or becomes pickable.

Implementation and physical validation of this new release motion must complete
before the common-environment freeze. Calibration collected against the earlier
stationary-opening controller is not valid for the new DP comparison.

The user approved goal pursuit **on = 5 skill executions, off = 0**. Keep the setting
explicit with `--goal-horizon 5`; omission, zero, negative values, and nonintegers are
rejected rather than falling back to a preset. The API accepts other positive integer
freezes for future approved revisions, but changing 5 requires a new decision.

Both sizes use goal interval 1, initialization cycles 1, human scoring enabled,
no reset gate, no always-ask wrapper, no bin pouring, barrier layouts, bin mass 1 kg,
cube mass 0.1 kg, at most 20 skill-level actions per interaction (each may consume
many control ticks), and reset policy `never`.
Evaluation is identical in full and pilot: every 3,200 counted practice steps,
10 held-out tasks, at most 500 control ticks per task, test seed offset 10,000.

Human scoring is semantically enabled on every arm: it is explicit for EES, while
the planners' belief scoring already includes all non-deterministic human skills.
The EES-only scoring switch must remain unset for planners (their CLI rejects it);
the inherited method default is also checked as true. Runtime human/outcome semantics
and goal behavioral tests remain separate
readiness evidence, not something manifest parsing proves.

The shared inner learner uses epsilon 0.5, 10,000 sampler training iterations,
competence window/recency 5/5, and planning timeout 10 seconds. All arms explicitly
use target-only exploration, including the planners whose canonical preset alone
does **not** enable it: goal pursuit and setup prefixes must not epsilon-explore.
Double observation is off; target history, seen-task order and random-when-stranded
are on. Sampler-draw recording is on. EES arms explicitly enable
`--record-ees-decisions`, writing `ees_decisions.jsonl`; planner arms retain their own
decision logs and never receive this EES-only flag. Source checks verify inherited EES/planner
defaults, 100 sampler candidates, 100 proposals per candidate, and sampler learning
rate 0.001 without instantiating a simulator. These are port/canonical settings, not
a claim that every paper-specific environment used identical hyperparameters.

Expectimax retains `tossing3d-canonical`: depth 6, grid inference, 1,024 particles,
observation weight 0, linear lambda 0.0003. Curtis uses determinized A*, variant
`curtis`, 100 search iterations, node draw scope, all outcomes, observation weight
0.01. Cost files come from `Tossing3DArmCosts`; tick price 0.01 and objective lambda
0.0003 are checked. Planner-specific defaults are not silently replaced with EES
priors or statistics-sharing settings.

An optional `--freeze-overrides FILE.json` accepts only these shared numerical keys:
`exploration_epsilon`, `sampler_max_train_iters`, `competence_window_size`,
`competence_recency_size`, `planning_timeout`. Overrides must be finite and valid,
apply to **every** arm, and are recorded alongside resolved values. They are explicit
deviations from the baseline above and require approval/rationale before deployment.
Unknown keys and protocol/matrix overrides are rejected; this is not an arbitrary
CLI flag passthrough. Do not use learner freezes to change factorial axes or budgets.

## Exact preparation commands

Run from the existing checkout with the approved explicit horizon 5.
Set `RESULTS_ROOT` to an absolute intended output root; it is only recorded, never
created by this tool. `--out` must name a new file in an existing directory.

```bash
cd /home/josh/Documents/repos/research/human-in-the-loop-practice-makes-perfect/artifacts/dp-fidelity-fix/pr-stack-20261026
export PYTHONPATH=src:.
PYTHON=/home/josh/miniconda3/envs/hitl-pmp/bin/python
: "${RESULTS_ROOT:?Set the absolute intended results root}"
$PYTHON -m scripts.tossing3d_ees_fidelity_launch check --size full --goal-horizon 5 --results-root "$RESULTS_ROOT"
$PYTHON -m scripts.tossing3d_ees_fidelity_launch prepare --size pilot --goal-horizon 5 --results-root "$RESULTS_ROOT" --out tossing3d-ees-fidelity-pilot.json
$PYTHON -m scripts.tossing3d_ees_fidelity_launch prepare --size full --goal-horizon 5 --results-root "$RESULTS_ROOT" --out tossing3d-ees-fidelity-full.json
$PYTHON -m scripts.tossing3d_ees_fidelity_launch validate --manifest tossing3d-ees-fidelity-full.json
```

Append `--freeze-overrides /absolute/path/freeze.json` to `check` or `prepare` when
needed. `validate` reads the saved freeze. It rebuilds the entire matrix, reparses
all argv with the real CLI (including inherited method fields), and rejects changes
to ordering, uniqueness, budgets, flags, stored configs, selected sources, or metadata.
Focused tests also exercise all 20 real method CLI run/factory paths with only the
environment runner mocked, verifying constructor values and rejecting incompatible
EES-only options on planners. They do not start simulation.
The preparation tool refuses overwriting an existing manifest; choose a new
path after any approved source/freeze change.

## Pending provenance and gates

Record any deviations from the approved horizon-5/learner baseline; final clean deployed revision and
complete source/diff digest; reference/submodule versions; simulator assets and
dependency environment; deployed runtime configuration and cost checksums; goal/CLI
behavioral test evidence; diagnostic schema/flag review with agent A; pilot results,
measured resources, and explicit launch authorization. The recorded local HEAD plus
selected file hashes are **not** full-source or clean-deployment attestations.
The EES diagnostics flag is now wired and factory-tested; archive agent A's runtime
schema/behavior evidence before final deployment. Runtime diagnostics and simulator
behavior are not validated by this preparation tool.

Planning estimates supplied by the coordinating agent are roughly 20–36 hours for
full runs at about 20-way concurrency and 2–4 hours for pilots. These are unverified
estimates, not scheduler reservations, walltime settings, or completion promises.

Focused offline tests (no simulations or asset download):

```bash
DISABLE_AUTO_DYNAMIC3D_SCENES_DOWNLOAD=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. /home/josh/miniconda3/envs/hitl-pmp/bin/python -m pytest -q tests/scripts/test_tossing3d_ees_fidelity_launch.py
```
